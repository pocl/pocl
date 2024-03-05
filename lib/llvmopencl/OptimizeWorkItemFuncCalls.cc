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

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>

#include "LLVMUtils.h"
#include "OptimizeWorkItemFuncCalls.h"
#include "VariableUniformityAnalysis.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#include <iostream>
#include <map>
#include <set>

#define PASS_NAME "optimize-wi-func-calls"
#define PASS_CLASS pocl::OptimizeWorkItemFuncCalls
#define PASS_DESC "Optimize work-item function calls."

namespace pocl {

using namespace llvm;

static bool optimizeWorkItemFuncCalls(Function &F) {

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
      std::string FuncName = Call->getCalledFunction()->getName().str();
      auto FuncNameI =
          std::find(WIFuncNameVec.begin(), WIFuncNameVec.end(), FuncName);
      if (FuncNameI == WIFuncNameVec.end())
        continue;

      bool Unsupported = false;
      // Check that the argument list is something we can handle.

      const unsigned CallNumArg = Call->arg_size();
      for (unsigned I = 0; I < CallNumArg; ++I) {
        llvm::ConstantInt *CallOperand =
          dyn_cast<llvm::ConstantInt>(Call->getArgOperand(I));
        if (CallOperand == nullptr)
          Unsupported = true;
      }
      if (Unsupported) continue;
      Calls[FuncName].push_back(Call);
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

        const unsigned MovedCallNumArg = MovedCall->arg_size();
        assert(MovedCallNumArg == CallInst->arg_size());
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

#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
char OptimizeWorkItemFuncCalls::ID = 0;

bool OptimizeWorkItemFuncCalls::runOnFunction(Function &F) {
  return optimizeWorkItemFuncCalls(F);
}

void OptimizeWorkItemFuncCalls::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addPreserved<WorkitemHandlerChooser>();
}

REGISTER_OLD_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

#else

llvm::PreservedAnalyses
OptimizeWorkItemFuncCalls::run(llvm::Function &F,
                               llvm::FunctionAnalysisManager &AM) {
  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();

  return optimizeWorkItemFuncCalls(F) ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

#endif

} // namespace pocl
