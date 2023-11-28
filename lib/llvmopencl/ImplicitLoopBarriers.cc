// LLVM function pass that adds implicit barriers to loops if it sees
// beneficial.
// 
// Copyright (c) 2012-2013 Pekka Jääskeläinen / TUT
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
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"

#if LLVM_MAJOR >= MIN_LLVM_NEW_PASSMANAGER
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Transforms/Scalar/LoopPassManager.h>
#endif

#include "Barrier.h"
#include "ImplicitLoopBarriers.h"
#include "LLVMUtils.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "Workgroup.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#include "pocl_runtime_config.h"

#include <iostream>

//#define DEBUG_ILOOP_BARRIERS

#define PASS_NAME "implicit-loop-barriers"
#define PASS_CLASS pocl::ImplicitLoopBarriers
#define PASS_DESC "Adds implicit barriers to loops"

namespace pocl {

using namespace llvm;

/**
 * Adds a barrier to the beginning of the loop body to force its treatment 
 * similarly to a loop with work-group barriers.
 *
 * This allows parallelizing work-items across the work-group per kernel
 * for-loop iteration, potentially leading to easier horizontal vectorization.
 * The idea is similar to loop switching where the work-item loop is 
 * switched with the kernel for-loop.
 *
 * We need to make sure it is legal to add the barrier, though. The
 * OpenCL barrier semantics require either all or none of the WIs to
 * reach the barrier at each iteration. This is satisfied at least when
 *
 * a) loop exit condition does not depend on the WI and 
 * b) all or none of the WIs always enter the loop
 */
static bool addInnerLoopBarrier(llvm::Loop &L,
                                VariableUniformityAnalysisResult &VUA) {

  /* Only add barriers to the innermost loops. */

  if (L.getSubLoops().size() > 0)
    return false;

#ifdef DEBUG_ILOOP_BARRIERS
  std::cerr << "### trying to add a loop barrier to force horizontal parallelization" 
            << std::endl;
#endif

  BasicBlock *brexit = L.getExitingBlock();
  if (brexit == NULL) return false; /* Multiple exit points */

  llvm::BasicBlock *loopEntry = L.getHeader();
  if (loopEntry == NULL) return false; /* Multiple entries blocks? */

  llvm::Function *f = brexit->getParent();

  /* Check if the whole loop construct is executed by all or none of the
     work-items. */
  if (!VUA.isUniform(f, loopEntry)) {
#ifdef DEBUG_ILOOP_BARRIERS
    std::cerr << "### the loop is not uniform because loop entry '"
              << loopEntry->getName().str() << "' is not uniform; LOOP: \n"
              << std::endl;
    L.dump();
#endif
    return false;
  }

  /* Check the branch condition predicate. If it is uniform, we know the loop 
     is  executed the same number of times for all WIs. */
  llvm::BranchInst *br = dyn_cast<llvm::BranchInst>(brexit->getTerminator());  
  if (br && br->isConditional() &&
      VUA.isUniform(f, br->getCondition())) {

    /* Add a barrier both to the beginning of the entry and to the very end
       to nicely isolate the parallel region. */
    Barrier::Create(brexit->getTerminator());   
    Barrier::Create(loopEntry->getFirstNonPHI());

#ifdef DEBUG_ILOOP_BARRIERS
    std::cerr << "### added an inner-loop barrier to the loop" << std::endl << std::endl;
#endif
    return true;
  } else {
#ifdef DEBUG_ILOOP_BARRIERS
    if (br && br->isConditional() && !VUA.isUniform(f, br->getCondition())) {
      std::cerr << "### loop condition not uniform" << std::endl;
      br->getCondition()->dump();
    }
#endif

  }

#ifdef DEBUG_ILOOP_BARRIERS
  std::cerr << "### cannot add an inner-loop barrier to the loop" << std::endl << std::endl;
#endif
  
  return false;
}

/**
 * Adds a barrier to the first BB of each loop.
 *
 * Note: it's not safe to do this in case the loop is not executed
 * by all work items. Therefore this is not enabled by default.
 */
static bool implicitLoopBarriers(Loop &L,
                                 VariableUniformityAnalysisResult &VUA) {

  bool IsBLoop = false;
  for (Loop::block_iterator LI = L.block_begin(), LE = L.block_end();
       LI != LE && !IsBLoop; ++LI) {
    for (BasicBlock::iterator BBI = (*LI)->begin(), BBE = (*LI)->end();
         BBI != BBE; ++BBI) {
      if (isa<Barrier>(BBI)) {
        IsBLoop = true;
        break;
      }
    }
  }
  if (IsBLoop)
    return false;

  return addInnerLoopBarrier(L, VUA);
}

#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
char ImplicitLoopBarriers::ID = 0;

bool ImplicitLoopBarriers::runOnLoop(Loop *L, llvm::LPPassManager &LPM) {
  Function *K = L->getHeader()->getParent();
  if (!isKernelToProcess(*K))
    return false;

  auto WIH = getAnalysis<WorkitemHandlerChooser>().chosenHandler();
  if (WIH == WorkitemHandlerType::CBS)
    return false;

  if (!pocl_get_bool_option("POCL_FORCE_PARALLEL_OUTER_LOOP", 0) &&
      !hasWorkgroupBarriers(*K)) {
#ifdef DEBUG_ILOOP_BARRIERS
    std::cerr
        << "### ILB: The kernel has no barriers, let's not add implicit ones "
        << "either to avoid WI context switch overheads" << std::endl;
#endif
    return false;
  }

  VariableUniformityAnalysisResult &VUA =
      getAnalysis<VariableUniformityAnalysis>().getResult();

  return implicitLoopBarriers(*L, VUA);
}

void ImplicitLoopBarriers::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<VariableUniformityAnalysis>();
  AU.addPreserved<VariableUniformityAnalysis>();
  AU.addRequired<WorkitemHandlerChooser>();
  AU.addPreserved<WorkitemHandlerChooser>();
}

REGISTER_OLD_LPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

#else

llvm::PreservedAnalyses
ImplicitLoopBarriers::run(llvm::Loop &L, llvm::LoopAnalysisManager &AM,
                          llvm::LoopStandardAnalysisResults &AR,
                          llvm::LPMUpdater &U) {

  Function *K = L.getHeader()->getParent();

  auto &FAMP = AM.getResult<FunctionAnalysisManagerLoopProxy>(L, AR);

  if (!isKernelToProcess(*K))
    return PreservedAnalyses::all();

  if (FAMP.cachedResultExists<WorkitemHandlerChooser>(*K)) {
    auto Res = FAMP.getCachedResult<WorkitemHandlerChooser>(*K);
    if (Res->WIH == WorkitemHandlerType::CBS)
      return PreservedAnalyses::all();
  } else {
    assert(0 && "missing cached result WIH for ImplicitLoopBarriers");
  }

  if (!pocl_get_bool_option("POCL_FORCE_PARALLEL_OUTER_LOOP", 0) &&
      !hasWorkgroupBarriers(*K)) {
#ifdef DEBUG_ILOOP_BARRIERS
    std::cerr
        << "### ILB: The kernel has no barriers, let's not add implicit ones "
        << "either to avoid WI context switch overheads" << std::endl;
#endif
    return PreservedAnalyses::all();
  }

  VariableUniformityAnalysisResult *VUA = nullptr;
  if (FAMP.cachedResultExists<VariableUniformityAnalysis>(*K)) {
    VUA = FAMP.getCachedResult<VariableUniformityAnalysis>(*K);
  } else {
    assert(0 && "missing cached result VUA for ImplicitLoopBarriers");
  }

  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();
  PAChanged.preserve<VariableUniformityAnalysis>();
  return implicitLoopBarriers(L, *VUA) ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_LPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

#endif

} // namespace pocl
