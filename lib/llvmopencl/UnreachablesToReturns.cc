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
// for LOOPVEC handler, we delete regions which can reach an unreachable inst
// but cannot reach the function's return. Branches from live blocks into the
// deleted region are made unconditional (to the live branch), and switch cases
// leading to it are removed. This preserves the PoCL 6.0 behavior that commit
// 89bc42187 sought to restore.
//
// Note that neither handling is recursive. Therefore all non-kernel functions
// that have an unreachable inst, must be inlined before this Pass is run.
//
// Copyright (c) 2025 Michal Babej / Intel Finland Oy
//                    Pekka Jääskeläinen / Intel Finland Oy
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
#include <llvm/IR/Verifier.h>
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
//#define DEBUG_CONVERT_UNREACHABLE

// Use the LLVM_DEBUG macros to gradually convert to LLVM-upstreamable
// code.
#ifdef LLVM_DEBUG
#undef LLVM_DEBUG
#endif

#ifdef DEBUG_CONVERT_UNREACHABLE
#define LLVM_DEBUG(X) X
#define dbgs() std::cerr << PASS_NAME << ": "
#else
#define LLVM_DEBUG(X)
#endif

namespace pocl {

using namespace llvm;

using SmallBBSet = llvm::SmallPtrSet<BasicBlock *, 8>;

// convert unreachable inst to a store of a flag + return instruction
static bool convertUnreachablesToReturns(Function &F) {

  Module *M = F.getParent();

  SmallVector<Instruction *, 8> PendingUnreachableInst;
  SmallVector<BasicBlock *, 8> PendingDeletableBBs;
  for (BasicBlock &BB : F) {
    assert(BB.getTerminator() != nullptr);
    if (auto UI = dyn_cast<UnreachableInst>(BB.getTerminator())) {
      LLVM_DEBUG(dbgs() << "UNREACHABLE found: replacing Inst in "
                        << F.getName().str() << "\n");
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
    if ((BB.size() == 1) && isa<ReturnInst>(BB.getTerminator())) {
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

// Delete regions which lead only to an unreachable instruction.
//
// These "doomed" blocks either end in an unreachable inst, or can only make
// progress towards one -- possibly iterating forever in a loop whose sole
// exit leads to an unreachable, e.g. the string-length loop of an inlined
// printf-then-abort sequence. Merely detaching the unreachable-terminated
// blocks would sever such a loop's exit edge and leave behind an infinite
// loop that never reaches the parallel region exit, breaking WorkitemLoops
// (issue #1958), so delete the doomed blocks wholesale.
static bool deleteBlocksWithUnreachable(Function &F) {

  SmallBBSet UnreachableBBs;
  for (BasicBlock &BB : F) {
    assert(BB.getTerminator());
    if (isa<UnreachableInst>(BB.getTerminator()))
      UnreachableBBs.insert(&BB);
  }

  if (UnreachableBBs.empty())
    return false;

  // Collect the live blocks, i.e. those that can reach a return.
  SmallBBSet LiveBBs;
  SmallVector<BasicBlock *, 16> WorkList;
  for (BasicBlock &BB : F) {
    if (isa<ReturnInst>(BB.getTerminator())) {
      LiveBBs.insert(&BB);
      WorkList.push_back(&BB);
    }
  }
  while (!WorkList.empty()) {
    BasicBlock *BB = WorkList.pop_back_val();
    for (BasicBlock *Pred : predecessors(BB))
      if (LiveBBs.insert(Pred).second)
        WorkList.push_back(Pred);
  }

  // A non-returning region is not necessarily doomed: an intentional infinite
  // loop need not lead to an unreachable instruction. Walk backwards from the
  // unreachables and restrict deletion to the intersection of both sets.
  SmallBBSet DoomedBBs = UnreachableBBs;
  WorkList.assign(UnreachableBBs.begin(), UnreachableBBs.end());
  while (!WorkList.empty()) {
    BasicBlock *BB = WorkList.pop_back_val();
    for (BasicBlock *Pred : predecessors(BB))
      if (!LiveBBs.count(Pred) && DoomedBBs.insert(Pred).second)
        WorkList.push_back(Pred);
  }

  // If not even the entry block can reach a return, deleting the doomed
  // blocks would delete the whole function body. Convert the unreachables
  // to returns instead.
  if (DoomedBBs.count(&F.getEntryBlock()))
    return convertUnreachablesToReturns(F);

  // Retarget terminators of live blocks to only branch to live blocks. Note
  // that an edge from a doomed block to a live block cannot exist, and that
  // a live block always has at least one live successor.
  for (BasicBlock &BB : F) {
    if (!LiveBBs.count(&BB))
      continue;

    Instruction *TI = BB.getTerminator();
    bool HasDoomedSuccessor = false;
    for (BasicBlock *Succ : successors(&BB))
      HasDoomedSuccessor |= DoomedBBs.count(Succ);
    if (!HasDoomedSuccessor)
      continue;

    LLVM_DEBUG(dbgs() << "Detaching doomed successors of:\n");
    LLVM_DEBUG(BB.dump());

#if LLVM_MAJOR >= 23
    if (auto *CBI = dyn_cast<CondBrInst>(TI)) {
      BasicBlock *Live = !DoomedBBs.count(CBI->getSuccessor(0))
                             ? CBI->getSuccessor(0)
                             : CBI->getSuccessor(1);
      assert(!DoomedBBs.count(Live));
      ReplaceInstWithInst(TI, UncondBrInst::Create(Live));
    }
#else
    if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
      assert(BI->isConditional());
      BasicBlock *Live = !DoomedBBs.count(BI->getSuccessor(0))
                             ? BI->getSuccessor(0)
                             : BI->getSuccessor(1);
      assert(!DoomedBBs.count(Live));
      ReplaceInstWithInst(TI, BranchInst::Create(Live));
    }
#endif
    else if (SwitchInst *SI = dyn_cast<SwitchInst>(TI)) {
      // Remove the cases leading to doomed blocks. RemoveCase invalidates
      // all iterators and might reorder the cases, so restart after each
      // removal.
      bool Removed = true;
      while (Removed) {
        Removed = false;
        for (SwitchInst::CaseIt C = SI->case_begin(); C != SI->case_end();
             ++C) {
          if (DoomedBBs.count(C->getCaseSuccessor())) {
            SI->removeCase(C);
            Removed = true;
            break;
          }
        }
      }
      if (DoomedBBs.count(SI->getDefaultDest())) {
        // Make one of the remaining cases the new default. Removing the
        // repurposed case keeps the number of edges to its successor, and
        // thus the successor's phi incoming counts, unchanged.
        assert(SI->getNumCases() > 0);
        SI->setDefaultDest(SI->case_begin()->getCaseSuccessor());
        SI->removeCase(SI->case_begin());
      }
    } else {
      LLVM_DEBUG(dbgs() << "Unhandled BB Terminator: \n");
      LLVM_DEBUG(TI->dump());
      assert(0 && "Error: unexpected BB terminator\n");
    }
  }

  // The doomed blocks are now unreachable from the entry.
  EliminateUnreachableBlocks(F);
  return true;
}

llvm::PreservedAnalyses
ConvertUnreachablesToReturns::run(llvm::Function &F,
                                  llvm::FunctionAnalysisManager &AM) {
  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();

  if (!isKernelToProcess(F))
    return PreservedAnalyses::all();

  // LOOPS: remove the blocks with unreachable inst.
  // CBS: replace unreachable with ret void
  WorkitemHandlerType WIH = AM.getResult<WorkitemHandlerChooser>(F).WIH;
  bool Changed = (WIH == WorkitemHandlerType::LOOPS)
                     ? deleteBlocksWithUnreachable(F)
                     : convertUnreachablesToReturns(F);

#ifdef DEBUG_CONVERT_UNREACHABLE
  for (BasicBlock &BB : F) {
    if (auto UI = dyn_cast<UnreachableInst>(BB.getTerminator())) {
      LLVM_DEBUG(dbgs() << "UnreachableInsts still found!\n");
      LLVM_DEBUG(BB.dump());
      LLVM_DEBUG(dbgs() << "\n");
      LLVM_DEBUG(F.dump());
      assert(false);
    }
  }
#endif
  return Changed ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
