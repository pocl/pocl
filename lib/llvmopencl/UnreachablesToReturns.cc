// ConvertUnreachablesToReturns is an LLVM pass to convert unreachable inst
// to defined behavior. In particular, it stores a flag (1) into an
// external variable, and a Terminator instruction (either branch or ret void).
//
// The store to global variable (__pocl_context_unreachable) is then converted
// to store into the pocl_context argument of the kernel in Workgroup pass.
//
// Note that this handling is not recursive. Therefore all functions that
// have an unreachable inst, must be inlined before this Pass is run.
//
// Copyright (c) 2025 Michal Babej / Intel Finland Oy
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
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>

#include "LLVMUtils.h"
#include "UnreachablesToReturns.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#include <iostream>
#include <map>
#include <set>

#define PASS_NAME "unreachables-to-returns"
#define PASS_CLASS pocl::ConvertUnreachablesToReturns
#define PASS_DESC "convert unreachable instruction uses to flag-store & return"

// #define DEBUG_CONVERT_UNREACHABLE

namespace pocl {

using namespace llvm;

static bool convertUnreachablesToReturns(Function &F) {

  Module *M = F.getParent();

  SmallVector<Instruction *, 8> PendingUnreachableInst;
  SmallVector<BasicBlock *, 8> PendingDeletableBBs;
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    BasicBlock &BB = *I;
    assert(BB.getTerminator());
    if (auto UI = dyn_cast<UnreachableInst>(BB.getTerminator())) {
#ifdef DEBUG_CONVERT_UNREACHABLE
      LLVM_DEBUG(dbgs() << "UNREACHABLE found: replacing Inst in "
                        << F.getName().str() << "\n");
#endif
      // this can happen when inlining functions which have unreachable Inst
      // we end up with a BB with 0 predecessors and a single unreachable
      if (BB.hasNPredecessors(0))
        PendingDeletableBBs.push_back(&BB);
      else
        PendingUnreachableInst.push_back(UI);
    }
  }

  for (auto BB : PendingDeletableBBs)
    BB->eraseFromParent();

  if (PendingUnreachableInst.empty())
    return false;

  // Find basic block with return instruction
  BasicBlock *RetBB = nullptr;
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    BasicBlock &BB = *I;
    assert(BB.getTerminator());
    if ((BB.sizeWithoutDebug() == 1) && isa<ReturnInst>(BB.getTerminator())) {
      RetBB = &BB;
      break;
    }
  }

  Type *I32Ty = Type::getInt32Ty(M->getContext());
  M->getOrInsertGlobal("__pocl_context_unreachable", I32Ty);
  GlobalVariable *UnreachGV = M->getNamedGlobal("__pocl_context_unreachable");
  Constant *ConstOne = ConstantInt::get(I32Ty, 1);
  IRBuilder<> Builder(M->getContext());
  for (auto UI : PendingUnreachableInst) {
#if LLVM_MAJOR < 20
    Builder.SetInsertPoint(UI);
#else
    Builder.SetInsertPoint(UI->getIterator());
#endif
    Builder.CreateStore(ConstOne, UnreachGV);
    if (RetBB)
      Builder.CreateBr(RetBB);
    else
      Builder.CreateRetVoid();
  }

  for (auto UI : PendingUnreachableInst)
    UI->eraseFromParent();

  return true;
}

llvm::PreservedAnalyses
ConvertUnreachablesToReturns::run(llvm::Function &F,
                                  llvm::FunctionAnalysisManager &AM) {
  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();

  if (!isKernelToProcess(F))
    return PreservedAnalyses::all();

  return convertUnreachablesToReturns(F) ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
