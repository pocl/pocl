// OptimizeWorkItemFuncCalls is an LLVM pass to optimize calls to work-item
// functions like get_local_size().
//
// Copyright (c) 2017-2019 Pekka Jääskeläinen
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
#include "pocl.h"

#include <set>
#include <iostream>

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>

#include "OptimizeWorkItemFuncCalls.h"

namespace pocl {

using namespace llvm;

namespace {
static RegisterPass<pocl::OptimizeWorkItemFuncCalls>
    X("optimize-wi-func-calls", "Optimize work-item function calls.");
}

char OptimizeWorkItemFuncCalls::ID = 0;

OptimizeWorkItemFuncCalls::OptimizeWorkItemFuncCalls() : FunctionPass(ID) {}

bool
OptimizeWorkItemFuncCalls::runOnFunction(Function &F) {

  // Let's avoid reoptimizing pocl_printf in the kernel compiler. It should
  // be optimized already in the bitcode library, and we do not want to
  // aggressively inline it to the kernel, causing compile time expansion.
  if (F.getName().startswith("__pocl_print") &&
      !F.hasFnAttribute(Attribute::OptimizeNone)) {
    F.addFnAttr(Attribute::OptimizeNone);
    F.addFnAttr(Attribute::NoInline);
  }

  if (F.getName().startswith("_") || F.hasFnAttribute(Attribute::OptimizeNone))
    return false;

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

      if (Call->getCalledFunction() == nullptr) {
        // The callee can be null in case of asm snippets (TCE).
        continue;
      }
      auto FuncNameI =
          WIFuncNames.find(Call->getCalledFunction()->getName().str());
      if (FuncNameI == WIFuncNames.end())
        continue;

      bool Unsupported = false;
      // Check that the argument list is something we can handle.

#ifndef LLVM_OLDER_THAN_8_0
      const unsigned CallNumArg = Call->arg_size();
#else
      const unsigned CallNumArg = Call->getNumArgOperands();
#endif
      for (unsigned I = 0; I < CallNumArg; ++I) {
        llvm::ConstantInt *CallOperand =
          dyn_cast<llvm::ConstantInt>(Call->getArgOperand(I));
        if (CallOperand == nullptr)
          Unsupported = true;
      }
      if (Unsupported) continue;
      Calls[*FuncNameI].push_back(Call);
    }
  }

  // Add single calls for the interesting functions.
  std::map<std::string, std::vector<CallInst*> > CallsInEntry;
  for (const auto &Call : Calls) {
    std::string FuncName = Call.first;
    auto CallInsts = Call.second;

    for (auto CallInst : CallInsts) {
      // Try to find a previous call with the same parameters which
      // we can reuse.
      std::vector<llvm::CallInst*> &CallsMoved = CallsInEntry[FuncName];
      llvm::CallInst *PreviousCall = nullptr;
      for (auto &M : CallsMoved) {
        llvm::CallInst *MovedCall = dyn_cast<llvm::CallInst>(M);

        // WI functions do not have variable argument lists.

#ifndef LLVM_OLDER_THAN_8_0
        const unsigned MovedCallNumArg = MovedCall->arg_size();
        assert(MovedCallNumArg == CallInst->arg_size());
#else
        const unsigned MovedCallNumArg = MovedCall->getNumArgOperands();
        assert(MovedCallNumArg == CallInst->getNumArgOperands());
#endif

        bool IsApplicable = true;

        for (unsigned I = 0; I < MovedCallNumArg; ++I) {
          llvm::ConstantInt *CallOperand =
            dyn_cast<llvm::ConstantInt>(CallInst->getArgOperand(I));
          llvm::ConstantInt *PrevCallOperand =
            dyn_cast<llvm::ConstantInt>(MovedCall->getArgOperand(I));

          assert (isa<llvm::ConstantInt>(PrevCallOperand));

          if (CallOperand->getValue() != PrevCallOperand->getValue()) {
            IsApplicable = false;
            break;
          }
        }
        if (IsApplicable) {
          // Found a suitable previous call instruction we can reuse.
          PreviousCall = MovedCall;
          break;
        }
      }

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
