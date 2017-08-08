// Header for CleanupWorkItemFuncCalls, an LLVM pass to optimize calls to work-item
// functions like get_local_size().
//
// Copyright (c) 2017 Pekka Jääskeläinen / Tampere University of Technology
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "config.h"

#include <set>
#include <iostream>

#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>

#include "OptimizeWorkItemFuncCalls.h"

namespace pocl {

using namespace llvm;

namespace {
  static
  RegisterPass<pocl::OptimizeWorkItemFuncCalls>
  X("optimize-wi-func-calls",
    "Optimize work-item function calls.");
}

char OptimizeWorkItemFuncCalls::ID = 0;

OptimizeWorkItemFuncCalls::OptimizeWorkItemFuncCalls() : FunctionPass(ID) {}

bool
OptimizeWorkItemFuncCalls::runOnFunction(Function &F) {

  // Find calls to WI functions and unify them to a single call in the
  // entry to avoid confusing LLVM later with the 'pseudo loads' and to
  // reduce the inlining bloat.

  typedef std::set<std::string> WIFuncNameVec;
  const WIFuncNameVec WIFuncNames = {
    "_Z13get_global_idj",
    "_Z17get_global_offsetj",
    "_Z15get_global_sizej",
    "_Z12get_group_idj",
    "_Z12get_local_idj",
    "_Z14get_local_sizej",
    "_Z14get_num_groupsj",
    "_Z12get_work_dimv"
  };

  Function::iterator I = F.begin();
  Instruction *FirstInsnPt = &*(I++)->getFirstInsertionPt();

  bool Changed = false;

  std::map<std::string, std::vector<CallInst*>> Calls;

  // First collect all calls of interest.
  for (Function::iterator E = F.end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE;) {

      CallInst *Call = dyn_cast<CallInst>(BI++);

      if (Call == nullptr) continue;

      auto FuncNameI = WIFuncNames.find(Call->getCalledFunction()->getName().str());
      if (FuncNameI == WIFuncNames.end())
        continue;

      Calls[*FuncNameI].push_back(Call);
    }
  }

  // Add single calls for the interesting functions.
  std::map<std::string, std::vector<CallInst*> > CallsInEntry;
  for (const auto &Call : Calls) {
    std::string FuncName = Call.first;
    auto CallInsts = Call.second;

    for (auto CallInst : CallInsts) {

      // Set to false in case the optimization is not applicable at
      // all to this call.
      bool Unsupported = false;

      // Try to find a previous call with the same parameters which
      // we can reuse.
      std::vector<llvm::CallInst*> &CallsMoved = CallsInEntry[FuncName];
      llvm::CallInst *PreviousCall = nullptr;
      for (auto &M : CallsMoved) {
        llvm::CallInst *MovedCall = dyn_cast<llvm::CallInst>(M);

        // WI functions do not have variable argument lists.
        assert (MovedCall->getNumArgOperands() ==
                CallInst->getNumArgOperands());

        bool IsApplicable = true;
        for (unsigned I = 0; I < MovedCall->getNumArgOperands(); ++I) {
          llvm::ConstantInt *CallOperand =
            dyn_cast<llvm::ConstantInt>(CallInst->getArgOperand(I));
          llvm::ConstantInt *PrevCallOperand =
            dyn_cast<llvm::ConstantInt>(MovedCall->getArgOperand(I));

          if (CallOperand == nullptr) {
            // We do not support moving WI func calls with non-const
            // arguments yet. It would require dflow analyzing whether
            // the argument Value can be moved too.
            CallInst->getArgOperand(I)->dump();
            Unsupported = false;
            break;
          }

          assert (isa<llvm::ConstantInt>(PrevCallOperand));

          if (CallOperand->getValue() != PrevCallOperand->getValue()) {
            IsApplicable = false;
            break;
          }
        }
        if (Unsupported) break;
        if (IsApplicable) {
          // Found a suitable previous call instruction we can reuse.
          PreviousCall = MovedCall;
          break;
        }
      }
      if (Unsupported) continue;

      if (PreviousCall == nullptr) {
        CallInst->moveBefore(FirstInsnPt);
        CallsInEntry[FuncName].push_back(CallInst);
        Changed = true;
      } else {
        // Not the first call.  Refer to the first call that was moved to
        // the entry.
        CallInst->replaceAllUsesWith(PreviousCall);
        CallInst->eraseFromParent();
        Changed = true;
      }
    }
  }
  return Changed;
}

}
