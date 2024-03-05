// LLVM function pass that adds implicit barriers to branches where it sees
// beneficial (and legal).
// 
// Copyright (c) 2013 Pekka Jääskeläinen / TUT
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

//#define DEBUG_COND_BARRIERS

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

static bool implicitConditionalBarriers(Function &F,
                                        llvm::PostDominatorTree &PDT,
                                        llvm::DominatorTree &DT) {

  typedef std::vector<BasicBlock*> BarrierBlockIndex;
  BarrierBlockIndex ConditionalBarriers;

  // Required hack. See the docs of removeUnreachableSwitchCases
  DT.reset();
  PDT.reset();
  removeUnreachableSwitchCases(F);
  DT.recalculate(F);
  PDT.recalculate(F);

  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
    BasicBlock *BB = &*FI;
    if (!Barrier::hasBarrier(BB)) continue;

    // Unconditional barrier postdominates the entry node.
    if (PDT.dominates(BB, &F.getEntryBlock()))
      continue;

#ifdef DEBUG_COND_BARRIERS
    std::cerr << "### found a conditional barrier:\n";
    BB->dump();
#endif
    ConditionalBarriers.push_back(BB);
  }

  if (ConditionalBarriers.size() == 0)
    return true;

  for (BarrierBlockIndex::const_iterator i = ConditionalBarriers.begin();
       i != ConditionalBarriers.end(); ++i) {
    BasicBlock *BB = *i;
    // Trace upwards from the barrier until one encounters another
    // barrier or the split point that makes the barrier conditional. 
    // In case of the latter, add a new barrier to both branches of the split point. 

    // BB before which to inject the barrier.
    BasicBlock *Pos = BB;
    if (pred_begin(BB) == pred_end(BB)) {
#ifdef DEBUG_COND_BARRIERS
      std::cerr << "BB before which to inject the barrier:\n";
      BB->dump();
#endif
      assert (pred_begin(BB) == pred_end(BB));
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

    if (Barrier::hasOnlyBarrier(Pos)) continue;
    // Inject a barrier at the beginning of the BB and let the
    // CanonicalizeBarrier to clean it up (split to a separate BB).

    // mri-q of parboil breaks in case injected at the beginning
    // TODO: investigate. It might related to the alloca-converted
    // PHIs. It has a loop that is autoconverted to a b-loop and the
    // conditional barrier is inserted after the loop short cut check.
    Barrier::Create(Pos->getFirstNonPHI());
#ifdef DEBUG_COND_BARRIERS
    std::cerr << "### added an implicit barrier to the BB" << std::endl;
    Pos->dump();
#endif

  }

  return true;
}

#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
char ImplicitConditionalBarriers::ID = 0;

bool ImplicitConditionalBarriers::runOnFunction(Function &F) {
  if (!pocl::isKernelToProcess(F))
    return false;

  if (!pocl::hasWorkgroupBarriers(F))
    return false;

  auto WIH = getAnalysis<WorkitemHandlerChooser>().chosenHandler();
  if (WIH == WorkitemHandlerType::CBS)
    return false;

  llvm::PostDominatorTree &PDT =
      getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();
  llvm::DominatorTree &DT =
      getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  return implicitConditionalBarriers(F, PDT, DT);
}

void ImplicitConditionalBarriers::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<PostDominatorTreeWrapperPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addPreserved<VariableUniformityAnalysis>();
  AU.addRequired<WorkitemHandlerChooser>();
  AU.addPreserved<WorkitemHandlerChooser>();
}

REGISTER_OLD_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

#else

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

  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<VariableUniformityAnalysis>();
  PAChanged.preserve<WorkitemHandlerChooser>();

  return implicitConditionalBarriers(F, PDT, DT) ? PAChanged
                                                 : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

#endif

} // namespace pocl
