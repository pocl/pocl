// LLVM loop pass that adds required barriers to loops.
//
// Copyright (c) 2011 Universidad Rey Juan Carlos
//               2012-2019 Pekka Jääskeläinen
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
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Transforms/Scalar/LoopPassManager.h>

#include "Barrier.h"
#include "LLVMUtils.h"
#include "LoopBarriers.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "Workgroup.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#include <iostream>

//#define DEBUG_LOOP_BARRIERS
#define PASS_NAME "loop-barriers"
#define PASS_CLASS pocl::LoopBarriers
#define PASS_DESC "Add needed barriers to loops"

namespace pocl {

using namespace llvm;

static bool processLoopWithBarriers(Loop &L, llvm::DominatorTree &DT,
                                    VariableUniformityAnalysisResult &VUA) {

  for (Loop::block_iterator i = L.block_begin(), e = L.block_end();
       i != e; ++i) {
    for (BasicBlock::iterator j = (*i)->begin(), e = (*i)->end();
         j != e; ++j) {
      if (isa<Barrier>(j)) {

        // Found a barrier in this loop:
        // 1) add a barrier in the loop header.
        // 2) add a barrier in the latches

        // Add a barrier on the preheader to ensure all WIs reach
        // the loop header with all the previous code already
        // executed.
        BasicBlock *Preheader = L.getLoopPreheader();
        assert((Preheader != NULL) && "Non-canonicalized loop found!\n");
#ifdef DEBUG_LOOP_BARRIERS
        std::cerr << "### adding to preheader BB" << std::endl;
        preheader->dump();
        std::cerr << "### before instr" << std::endl;
        preheader->getTerminator()->dump();
#endif
        Barrier::Create(Preheader->getTerminator());
        Preheader->setName(Preheader->getName() + ".loopbarrier");

        // Add a barrier after the PHI nodes on the header (the replicated
        // headers will be merged afterwards).
        BasicBlock *Header = L.getHeader();
        if (Header->getFirstNonPHI() != &Header->front()) {
          Barrier::Create(Header->getFirstNonPHI());
          Header->setName(Header->getName() + ".phibarrier");
          // Split the block to  create a replicable region of
          // the loop contents in case the phi node contains a
          // branch (that can be to inside the region).
          //          if (header->getTerminator()->getNumSuccessors() > 1)
          //    SplitBlock(header, header->getTerminator(), this);
        }

        // Add the barriers on the exiting block and the latches,
        // which might not always be the same if there is computation
        // after the exit decision.
        BasicBlock *BrExit = L.getExitingBlock();
        if (BrExit != NULL) {
          Barrier::Create(BrExit->getTerminator());
          BrExit->setName(BrExit->getName() + ".brexitbarrier");
        }

        BasicBlock *Latch = L.getLoopLatch();
        if (Latch != NULL && BrExit != Latch) {
          // This loop has only one latch. Do not check for dominance, we
          // are probably running before BTR.
          Barrier::Create(Latch->getTerminator());
          Latch->setName(Latch->getName() + ".latchbarrier");
          return true;
        }

        // Modified code from llvm::LoopBase::getLoopLatch to
        // go trough all the latches.
        BasicBlock *Header2 = L.getHeader();
        typedef GraphTraits<Inverse<BasicBlock *> > InvBlockTraits;
        InvBlockTraits::ChildIteratorType PI =
          InvBlockTraits::child_begin(Header2);
        InvBlockTraits::ChildIteratorType PE =
          InvBlockTraits::child_end(Header2);

        BasicBlock *Latch2 = nullptr;
        for (; PI != PE; ++PI) {
          BasicBlock *N = *PI;
          if (L.contains(N)) {
            Latch2 = N;
            // Latch found in the loop, see if the barrier dominates it
            // (otherwise if might no even belong to this "tail", see
            // forifbarrier1 graph test).
            if (DT.dominates(j->getParent(), Latch2)) {
              Barrier::Create(Latch2->getTerminator());
              Latch2->setName(Latch2->getName() + ".latchbarrier");
            }
          }
        }
        return true;
      }
    }
  }
  return false;
}

static bool processLoop(Loop &L, llvm::DominatorTree &DT,
                        VariableUniformityAnalysisResult &VUA) {

  if (Barrier::IsLoopWithBarrier(L))
    return processLoopWithBarriers(L, DT, VUA);

  bool Changed = false;

  // This is a loop without a barrier. Ensure we have a non-barrier
  // block as a preheader so we capture the loop as a whole to the
  // parallel region.
  //
  // If the block has proper instructions after the barrier, it
  // will be split in CanonicalizeBarriers.
  //
  // Also attempt to isolate the loop with barriers to create the
  // WI loop around it in order to produce a nicely well-formed
  // WI-loop + K-loop hierarchy for the loop-interchange to analyze.

  // Include all the layers in a multi-level loop without barriers
  // in the region. Thus, do the handling only for the outermost
  // loop without barriers. Otherwise we end up creating a b-loop
  // to the innermost loop if we process that first, ruining the
  // idea for multi-level loops.
  Loop *ParentLoop = L.getParentLoop();
  if (!(ParentLoop == nullptr || Barrier::IsLoopWithBarrier(*ParentLoop)))
    return false;

  BasicBlock *Preheader = L.getLoopPreheader();
  assert((Preheader != NULL) && "Non-canonicalized loop found!\n");

  Instruction *Inst = Preheader->getTerminator();
  Instruction *PrevInst = NULL;
  if (&Preheader->front() != Inst)
    PrevInst = Inst->getPrevNode();
  if (PrevInst && isa<Barrier>(PrevInst)) {
      BasicBlock *NewBB = SplitBlock(Preheader, Inst);
      NewBB->setName(Preheader->getName() + ".postbarrier_dummy");
      return true;
  }

  return Changed;
}

llvm::PreservedAnalyses LoopBarriers::run(llvm::Loop &L,
                                          llvm::LoopAnalysisManager &AM,
                                          llvm::LoopStandardAnalysisResults &AR,
                                          llvm::LPMUpdater &U) {

  Function *K = L.getHeader()->getParent();

  if (!isKernelToProcess(*K))
    return PreservedAnalyses::all();

  if (!hasWorkgroupBarriers(*K))
    return PreservedAnalyses::all();

  PreservedAnalyses PAChanged = PreservedAnalyses::none();

  VariableUniformityAnalysisResult *VUA = nullptr;

  auto &FAMP = AM.getResult<FunctionAnalysisManagerLoopProxy>(L, AR);
  if (FAMP.cachedResultExists<VariableUniformityAnalysis>(*K)) {
    VUA = FAMP.getCachedResult<VariableUniformityAnalysis>(*K);
  } else {
    assert(0 && "missing cached result VUA for ImplicitLoopBarriers");
  }

  return processLoop(L, AR.DT, *VUA) ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_LPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
