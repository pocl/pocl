// UnrollBarrierLoops fully unrolls short or requested loops with barriers.
//
// Copyright (c) 2026 PoCL developers
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/Analysis/CodeMetrics.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/OptimizationRemarkEmitter.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/LoopUtils.h>
#include <llvm/Transforms/Utils/UnrollLoop.h>

#include <optional>

#include "Barrier.h"
#include "LLVMUtils.h"
#include "UnrollBarrierLoops.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#define PASS_NAME "unroll-barrier-loops"
#define PASS_CLASS pocl::UnrollBarrierLoops
#define PASS_DESC "Fully unroll short or requested loops with barriers"

namespace pocl {

using namespace llvm;

// A loop with a barrier (a b-loop) gets its body split into parallel regions
// that are each executed for all work-items in every iteration, with the
// values that are live across the barriers saved in per-work-item context
// arrays, and extra barriers isolating the loop control. LLVM's unroller
// can't unroll such a loop, as pocl.barrier is noduplicate, and often the
// trip count only became a constant when the stage 1 optimizations folded the
// local or sub-group size. This pass fully unrolls such loops if their trip
// count is a constant and either they carry an unroll request ('#pragma
// unroll', opencl_unroll_hint, or SPIR-V's Unroll loop control) or LLVM would
// fully unroll them if not for the barrier. '#pragma nounroll' is honored.

// LLVM's default -pragma-unroll-threshold
static constexpr unsigned MaxUnrolledSize = 16 * 1024;

static unsigned getUnrollCount(Loop &L, ScalarEvolution &SE,
                               const TargetTransformInfo &TTI,
                               AssumptionCache &AC,
                               OptimizationRemarkEmitter &ORE) {
  if (!L.isInnermost() || !L.isLoopSimplifyForm() ||
      L.getExitingBlock() == nullptr || !Barrier::isLoopWithBarrier(L))
    return 0;

  TransformationMode Mode = hasUnrollTransformation(&L);
  if (Mode & TM_Disable)
    return 0;
  bool Forced = Mode == TM_ForcedByUser;

  unsigned TripCount = SE.getSmallConstantTripCount(&L);
  if (TripCount == 0)
    return 0;

  // a partial unroll would leave a b-loop
  std::optional<int> Count =
      getOptionalIntLoopAttribute(&L, "llvm.loop.unroll.count");
  if (Count && *Count >= 0 && unsigned(*Count) < TripCount)
    return 0;

  SmallPtrSet<const Value *, 32> EphValues;
  if (!Forced)
    CodeMetrics::collectEphemeralValues(&L, &AC, EphValues);

  unsigned Size = 0;
  for (BasicBlock *BB : L.blocks()) {
    for (Instruction &I : *BB) {
      // Convergence control tokens can't be cloned freely, and other
      // noduplicate calls aren't ours to duplicate.
      if (I.getType()->isTokenTy())
        return 0;
      if (auto *Call = dyn_cast<CallBase>(&I))
        if (Call->cannotDuplicate() && !isa<Barrier>(Call))
          return 0;
      if (!I.isDebugOrPseudoInst())
        ++Size;
      // UnrollCostEstimator can't size a loop with an invalid cost
      if (!Forced && !EphValues.contains(&I) &&
          !TTI.getInstructionCost(&I, TargetTransformInfo::TCK_CodeSize)
               .isValid())
        return 0;
    }
  }

  if (Forced)
    return uint64_t(Size) * TripCount > MaxUnrolledSize ? 0 : TripCount;

  // The static full unroll size check of LoopUnrollPass (shouldFullUnroll),
  // without the bonus for the simplifications of the unrolled body. OptLevel
  // 3 because the pass runs in stage 2's O3 pipeline. UCE.canUnroll() would
  // reject the loop for its noduplicate barrier.
  TargetTransformInfo::UnrollingPreferences UP = gatherUnrollingPreferences(
      &L, SE, TTI, /*BFI=*/nullptr, /*PSI=*/nullptr, ORE, /*OptLevel=*/3,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt);
  UnrollCostEstimator UCE(&L, TTI, EphValues, UP.BEInsns);
  if (TripCount > UP.FullUnrollMaxCount ||
      UCE.getUnrolledLoopSize(UP, TripCount) >= UP.Threshold)
    return 0;

  return TripCount;
}

static bool unrollBarrierLoops(Function &F, FunctionAnalysisManager &AM) {
  Function *BarrierF = F.getParent()->getFunction(BARRIER_FUNCTION_NAME);
  if (BarrierF == nullptr)
    return false;

  LoopInfo &LI = AM.getResult<LoopAnalysis>(F);
  ScalarEvolution &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  AssumptionCache &AC = AM.getResult<AssumptionAnalysis>(F);
  TargetTransformInfo &TTI = AM.getResult<TargetIRAnalysis>(F);
  OptimizationRemarkEmitter ORE(&F);

  SmallVector<std::pair<Loop *, unsigned>, 4> Loops;
  for (Loop *L : LI.getLoopsInPreorder())
    if (unsigned Count = getUnrollCount(*L, SE, TTI, AC, ORE))
      Loops.push_back({L, Count});
  if (Loops.empty())
    return false;

  DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);

  // pocl.barrier is noduplicate so that the optimizations don't create
  // barriers that only some work-items reach. Fully unrolling a loop with a
  // constant trip count makes every work-item execute the same sequence of
  // barriers, as the loop did, so lift the attribute while doing only that.
  bool DeclNoDup = BarrierF->hasFnAttribute(Attribute::NoDuplicate);
  BarrierF->removeFnAttr(Attribute::NoDuplicate);
  bool CallNoDup = false;
  for (User *U : BarrierF->users()) {
    auto *Call = dyn_cast<CallInst>(U);
    if (Call == nullptr || Call->getFunction() != &F ||
        !Call->hasFnAttr(Attribute::NoDuplicate))
      continue;
    CallNoDup = true;
    Call->removeFnAttr(Attribute::NoDuplicate);
  }

  bool Changed = false;
  for (auto [L, Count] : Loops) {
    // UnrollLoop only redirects the uses outside the loop to the last
    // iteration through LCSSA phis, and expects the enclosing loops in LCSSA
    // as well.
    Changed |= formLCSSARecursively(*L->getOutermostLoop(), DT, &LI, &SE);

    UnrollLoopOptions ULO{};
    ULO.Count = Count;
    ULO.Force = false;
    ULO.Runtime = false;
    ULO.AllowExpensiveTripCount = false;
    ULO.UnrollRemainder = false;
    ULO.ForgetAllSCEV = false;
    LoopUnrollResult Result = UnrollLoop(L, ULO, &LI, &SE, &DT, &AC, &TTI, &ORE,
                                         /*PreserveLCSSA=*/true);
    Changed |= Result != LoopUnrollResult::Unmodified;
  }

  if (DeclNoDup)
    BarrierF->addFnAttr(Attribute::NoDuplicate);
  if (CallNoDup)
    for (User *U : BarrierF->users())
      if (auto *Call = dyn_cast<CallInst>(U))
        if (Call->getFunction() == &F)
          Call->addFnAttr(Attribute::NoDuplicate);

  return Changed;
}

llvm::PreservedAnalyses
UnrollBarrierLoops::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  if (!isKernelToProcess(F) || F.hasOptNone())
    return PreservedAnalyses::all();

  if (!unrollBarrierLoops(F, AM))
    return PreservedAnalyses::all();

  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();
  return PAChanged;
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
