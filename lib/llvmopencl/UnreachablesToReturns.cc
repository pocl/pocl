// UnreachablesToReturns is an LLVM pass to convert unreachable inst
// to defined behavior. The behavior depends on WI handler (CBS / LOOPVEC)
//
// for CBS handler, we convert the unreachable to store of flag (1) into an
// external variable, and a Terminator instruction (either branch or ret void).
// The store to global variable (__pocl_context_unreachable) is then converted
// to store into the pocl_context argument of the kernel in Workgroup pass.
// The new terminator (ret/branch) changes the CFG and has the potential
// to create illegal code (barriers are only partially taken), however the CBS
// is able to handle these.
//
// for LOOPVEC handler, we find all basicblocks which have an unreachable inst,
// and "disconnect" them from their predecessor basicblocks. If the predecessor
// has an unconditional jump, it is also deleted. If it has a conditional jump,
// it's made unconditional (to the other branch). This matches the behavior
// of prior versions of PoCL (6.0) where the optimization similarly deleted
// these blocks.
//
// Note that neither handling is recursive. Therefore all non-kernel functions
// that have an unreachable inst, must be inlined before this Pass is run.
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
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

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

#define DEBUG_TYPE PASS_NAME
// #define DEBUG_CONVERT_UNREACHABLE

namespace pocl {

using namespace llvm;

using SmallBBSet = llvm::SmallPtrSet<BasicBlock *, 8>;

// convert unreachable inst to a store of a flag + return instruction
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

// recursively fix predecessor BBs of BB which has unreachable terminator inst.
// if the predecessor has unconditional branch, replace the branch with
// unreachable; if the predecessor has conditional branch, make it unconditional
static void detachBBFromPredecessor(BasicBlock *BB,
                                    SmallBBSet &UnreachableBBs2) {
  if (BB->hasNPredecessors(0))
    return;

  // To avoid invalidating the predecessors iterator,
  // store replacement instructions and replace after the loop
  SmallMapVector<Instruction *, Instruction *, 8> Replacements;

  for (BasicBlock *Pred : predecessors(BB)) {
    assert(Pred);
    Instruction *I = Pred->getTerminator();
    assert(I);
    if (BranchInst *BI = dyn_cast<BranchInst>(I)) {
      if (BI->isUnconditional()) {
        detachBBFromPredecessor(Pred, UnreachableBBs2);
        // predecessor is unconditional branch - remove this block too
        UnreachableBBs2.insert(Pred);
        // replace unconditional branch with unreachable BB
        UnreachableInst *UI = new UnreachableInst(BB->getContext());
        Replacements.insert(std::make_pair(BI, UI));
      } else {
        // conditional branch. replace with unconditional branch to the other BB
        BasicBlock *Other = nullptr;
        if (BI->getSuccessor(0) == BB)
          Other = BI->getSuccessor(1);
        else
          Other = BI->getSuccessor(0);
        BranchInst *NewBI = BranchInst::Create(Other);
        Replacements.insert(std::make_pair(BI, NewBI));
      }
    } else {
      // TODO which basicblock terminators should we handle here?
      // switch is already handled earlier by removeUnreachableSwitchCases()
#ifdef DEBUG_CONVERT_UNREACHABLE
      LLVM_DEBUG(dbgs() << "Unhadled BB Terminator: \n";
      I->dump();
#endif
      assert(0 && "Error: unhandled case in detachBBFromPredecessor\n");
    }
  }

  for (auto [OldI, NewI] : Replacements) {
    ReplaceInstWithInst(OldI, NewI);
  }
}

static bool deleteBlocksWithUnreachable(Function &F) {

  SmallBBSet UnreachableBBs;

  bool Changed = removeUnreachableSwitchCases(F);

  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    BasicBlock &BB = *I;
    assert(BB.getTerminator());
    if (isa<UnreachableInst>(BB.getTerminator())) {
#ifdef DEBUG_CONVERT_UNREACHABLE
      LLVM_DEBUG(dbgs() << "UNREACHABLE found: deleting BB in "
                        << F.getName().str() << "\n");
#endif
      UnreachableBBs.insert(&BB);
    }
  }

  if (UnreachableBBs.empty())
    return Changed;

  // check BB predecessors recursively, and disconnect them
  // from blocks which contain an unreachable
  SmallBBSet UnreachableBBs2;
  for (auto *BB : UnreachableBBs) {
    detachBBFromPredecessor(BB, UnreachableBBs2);
  }

  // all relevant blocks should now be disconnected (have 0 predecessors)
  // delete them
  for (auto *BB : UnreachableBBs) {
    if (BB->hasNPredecessors(0)) {
#ifdef DEBUG_CONVERT_UNREACHABLE
      LLVM_DEBUG(dbgs() << "deleting BB: \n");
      BB->dump();
      BB->eraseFromParent();
#endif
    }
  }

  for (auto *BB : UnreachableBBs2) {
    if (BB->hasNPredecessors(0)) {
#ifdef DEBUG_CONVERT_UNREACHABLE
      LLVM_DEBUG(dbgs() << "deleting BB: \n";
      BB->dump();
      BB->eraseFromParent();
#endif
    }
  }

  return true;
}

llvm::PreservedAnalyses
ConvertUnreachablesToReturns::run(llvm::Function &F,
                                  llvm::FunctionAnalysisManager &AM) {
  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();

  if (!isKernelToProcess(F))
    return PreservedAnalyses::all();

  // for LOOPS, remove the blocks with unreachable inst.
  // for CBS, replace unreachable with ret void
  WorkitemHandlerType WIH = AM.getResult<WorkitemHandlerChooser>(F).WIH;
  bool Changed = (WIH == WorkitemHandlerType::LOOPS)
                     ? deleteBlocksWithUnreachable(F)
                     : convertUnreachablesToReturns(F);

  return Changed ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
