// ConvertPoclExit is an LLVM pass to convert __pocl_exit() and __pocl_trap()
// calls on CPU backends, avoiding llvm.trap (which kills the host process).
//
// __pocl_exit: non-fatal, just returns (kernel stops executing).
// __pocl_trap: stores __pocl_context_unreachable flag + returns (CL_FAILED).
//
// For SPMD (GPU) backends, both are lowered in the linker instead
// (PTX "exit;" / llvm.trap).
//
// Copyright (c) 2026 Tim Besard
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
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>

#include "ConvertPoclExit.h"
#include "LLVMUtils.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#define PASS_NAME "pocl-exit"
#define PASS_CLASS pocl::ConvertPoclExit
#define PASS_DESC "convert __pocl_exit/__pocl_trap calls to return on CPU"

#define DEBUG_TYPE PASS_NAME

namespace pocl {

using namespace llvm;

// Erase instructions from CI to end of its basic block (inclusive).
static void eraseFromCallToEndOfBlock(CallInst *CI) {
  BasicBlock *BB = CI->getParent();
  SmallVector<Instruction *, 8> ToErase;
  bool Found = false;
  for (Instruction &I : *BB) {
    if (&I == CI)
      Found = true;
    if (Found)
      ToErase.push_back(&I);
  }
  for (auto It = ToErase.rbegin(); It != ToErase.rend(); ++It)
    (*It)->eraseFromParent();
}

// Collect calls to Fn within F.
static void collectCalls(Function &F, Function *Fn,
                         SmallVectorImpl<CallInst *> &Calls) {
  if (!Fn)
    return;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (CI->getCalledFunction() == Fn)
          Calls.push_back(CI);
}

llvm::PreservedAnalyses
ConvertPoclExit::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();

  Module *M = F.getParent();
  Function *ExitFn = M->getFunction("__pocl_exit");
  Function *TrapFn = M->getFunction("__pocl_trap");

  SmallVector<CallInst *, 4> ExitCalls, TrapCalls;
  collectCalls(F, ExitFn, ExitCalls);
  collectCalls(F, TrapFn, TrapCalls);

  if (ExitCalls.empty() && TrapCalls.empty())
    return PreservedAnalyses::all();

  bool IsKernel = isKernelToProcess(F);
  IRBuilder<> Builder(M->getContext());

  // For non-kernel functions: just erase the calls and let the function
  // return normally. The caller (kernel) has an unreachable after the call
  // which UTR will convert to flag-store + return.
  if (!IsKernel) {
    for (auto *CI : ExitCalls)
      CI->eraseFromParent();
    for (auto *CI : TrapCalls)
      CI->eraseFromParent();
  } else {
    // Find a single-instruction return block (like UTR does)
    BasicBlock *RetBB = nullptr;
    for (BasicBlock &BB : F) {
      assert(BB.getTerminator());
      if ((BB.sizeWithoutDebug() == 1) &&
          isa<ReturnInst>(BB.getTerminator())) {
        RetBB = &BB;
        break;
      }
    }

    // __pocl_exit: non-fatal, just return
    for (auto *CI : ExitCalls) {
#if LLVM_MAJOR < 20
      Builder.SetInsertPoint(CI);
#else
      Builder.SetInsertPoint(CI->getIterator());
#endif
      if (RetBB)
        Builder.CreateBr(RetBB);
      else
        Builder.CreateRetVoid();
      eraseFromCallToEndOfBlock(CI);
    }

    // __pocl_trap: fatal, store flag + return (reports CL_FAILED)
    if (!TrapCalls.empty()) {
      Type *I32Ty = Type::getInt32Ty(M->getContext());
      M->getOrInsertGlobal("__pocl_context_unreachable", I32Ty);
      GlobalVariable *UnreachGV =
          M->getNamedGlobal("__pocl_context_unreachable");
      Constant *ConstOne = ConstantInt::get(I32Ty, 1);

      for (auto *CI : TrapCalls) {
#if LLVM_MAJOR < 20
        Builder.SetInsertPoint(CI);
#else
        Builder.SetInsertPoint(CI->getIterator());
#endif
        Builder.CreateStore(ConstOne, UnreachGV);
        if (RetBB)
          Builder.CreateBr(RetBB);
        else
          Builder.CreateRetVoid();
        eraseFromCallToEndOfBlock(CI);
      }
    }
  }

  // Clean up if no more uses remain across the module
  if (ExitFn && ExitFn->use_empty())
    ExitFn->eraseFromParent();
  if (TrapFn && TrapFn->use_empty())
    TrapFn->eraseFromParent();

  return PAChanged;
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
