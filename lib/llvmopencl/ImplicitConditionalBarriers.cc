// LLVM function pass that adds implicit barriers to branches where it sees
// beneficial (and legal).
//
// Copyright (c) 2013 Pekka Jääskeläinen / TUT
//               2024 Pekka Jääskeläinen / Intel Finland Oy
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
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include "Barrier.h"
#include "ImplicitConditionalBarriers.h"
#include "LLVMUtils.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "Workgroup.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#include <iostream>

#include "pocl.h"

// #define DEBUG_COND_BARRIERS

#define PASS_NAME "implicit-cond-barriers"
#define PASS_CLASS pocl::ImplicitConditionalBarriers
#define PASS_DESC "Adds implicit barriers to branches."

namespace pocl {

using namespace llvm;

/**
 * Finds a predecessor that does not come from a back edge.
 *
 * This is used to include loops in the conditional parallel region.
 */
static BasicBlock *firstNonBackedgePredecessor(llvm::BasicBlock *BB,
                                               DominatorTree &DT) {

  pred_iterator I = pred_begin(BB), E = pred_end(BB);
  if (I == E)
    return NULL;
  while (DT.dominates(BB, *I) && I != E)
    ++I;
  if (I == E)
    return NULL;
  else
    return *I;
}

llvm::PreservedAnalyses
ImplicitConditionalBarriers::run(llvm::Function &F,
                                 llvm::FunctionAnalysisManager &FAM) {

  if (!isKernelToProcess(F))
    return PreservedAnalyses::all();

  if (!hasWorkgroupBarriers(F))
    return PreservedAnalyses::all();

  WorkitemHandlerType WIH = FAM.getResult<WorkitemHandlerChooser>(F).WIH;
  if (WIH == WorkitemHandlerType::CBS)
    return PreservedAnalyses::all();

  llvm::PostDominatorTree &PDT = FAM.getResult<PostDominatorTreeAnalysis>(F);
  llvm::DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  llvm::LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);

  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<VariableUniformityAnalysis>();
  PAChanged.preserve<WorkitemHandlerChooser>();
  PAChanged.preserve<LoopAnalysis>();

  typedef std::vector<BasicBlock*> BarrierBlockIndex;
  BarrierBlockIndex ConditionalBarriers;

  bool Changed = false;

  // Required workaround hack. See the documentation of
  // removeUnreachableSwitchCases().
  DT.reset();
  PDT.reset();
  removeUnreachableSwitchCases(F);
  DT.recalculate(F);
  PDT.recalculate(F);

  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
    BasicBlock *BB = &*FI;
    if (!Barrier::hasBarrier(BB)) continue;

    // Unconditional barrier postdominates the entry node.
    if (PDT.dominates(BB, &F.getEntryBlock())) {
#ifdef DEBUG_COND_BARRIERS
      std::cerr << "### BB postdominates the entry block" << std::endl;
      BB->dump();
#endif
      continue;
    }
    ConditionalBarriers.push_back(BB);
  }

  for (BarrierBlockIndex::const_iterator i = ConditionalBarriers.begin();
       i != ConditionalBarriers.end(); ++i) {
    BasicBlock *BB = *i;
#ifdef DEBUG_COND_BARRIERS
    std::cerr << "### handling a conditional barrier in basic block:\n";
    BB->dump();
#endif
    // Trace upwards from the barrier until one encounters another
    // barrier or the split point that makes the barrier conditional.
    // In case of the latter, add a new barrier to both branches of the split
    // point.

    // BB before which to inject the barrier.
    BasicBlock *Pos = BB;
    if (pred_begin(BB) == pred_end(BB)) {
#ifdef DEBUG_COND_BARRIERS
      std::cerr << "BB before which to inject the barrier:\n";
      BB->dump();
#endif
    }
    BasicBlock *Pred = firstNonBackedgePredecessor(BB, DT);

    while (!Barrier::hasOnlyBarrier(Pred) && PDT.dominates(BB, Pred)) {

#ifdef DEBUG_COND_BARRIERS
      std::cerr << "### looking at BB " << Pred->getName().str() << std::endl;
#endif
      Pos = Pred;
      // If our BB post dominates the given block, we know it is not the
      // branching block that makes the barrier conditional.
      Pred = firstNonBackedgePredecessor(Pred, DT);

      if (Pred == BB) break; // Traced across a loop edge, skip this case.
    }

    // If we encountered a loop on the way, we do not have to do anything
    // as loops are isolated separately. This is likely a barrier isolating
    // the loop regions.
    if (LI.getLoopFor(BB) != LI.getLoopFor(Pos))
      continue;

    if (Barrier::hasOnlyBarrier(Pos) || isa<Barrier>(Pos->getFirstNonPHI()))
      continue;

    // Inject a barrier at the end of the conditional BB and let the
    // CanonicalizeBarriers split it to a separate BB.
    Barrier::Create(Pos->getTerminator());

    Changed = true;
#ifdef DEBUG_COND_BARRIERS
    std::cerr << "### added an implicit barrier to the BB" << std::endl;
    Pos->dump();
#endif
    if (BasicBlock *Source = Pos->getSinglePredecessor()) {
      Barrier::Create(Source->getTerminator());
#ifdef DEBUG_COND_BARRIERS
      std::cerr << "### added an implicit barrier to a source of the BB as well"
                << std::endl;
      Source->dump();
#endif
    }
  }

#ifdef DEBUG_COND_BARRIERS
  std::cerr << "### After implicit conditional barrier handling:" << std::endl;
  F.dump();
#endif

  return Changed ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
