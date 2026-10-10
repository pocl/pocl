// UnrollBarrierLoops fully unrolls small loops with barriers.
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
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/OptimizationRemarkEmitter.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/LoopUtils.h>
#include <llvm/Transforms/Utils/UnrollLoop.h>

#include "Barrier.h"
#include "LLVMUtils.h"
#include "UnrollBarrierLoops.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#define PASS_NAME "unroll-barrier-loops"
#define PASS_CLASS pocl::UnrollBarrierLoops
#define PASS_DESC "Fully unroll small loops with barriers"

namespace pocl {

using namespace llvm;

// A loop with a barrier (a b-loop) gets its body split into parallel regions
// that are each executed for all work-items in every iteration, with the
// values that are live across the barriers saved in per-work-item context
// arrays, and extra barriers isolating the loop control. Loops of a few
// iterations, like the butterflies of sub-group reductions written with
// shuffles, are cheaper to execute unrolled. The stage 2 optimizations would
// not do that: PoCL disables LLVM's automatic unrolling, and pocl.barrier is
// noduplicate.
//
// Eight iterations cover the butterflies of sub-groups of up to 256
// work-items, while limiting the number of parallel regions and of copies of
// the work-group storage of shuffles. Longer loops stay b-loops (e.g. the one
// in tests/workgroup/for_bug.cl).
static constexpr unsigned MaxTripCount = 8;
static constexpr unsigned MaxUnrolledSize = 2048;

// This also unrolls loops with llvm.loop.unroll.disable, including those of
// an explicit '#pragma nounroll': the OpenCL C front-end puts it on every loop
// (UnrollLoops = false in pocl_llvm_build.cc), so a user's request can't be
// told apart. -cl-opt-disable and optnone do disable the pass.
static unsigned getUnrollCount(Loop &L, ScalarEvolution &SE) {
  if (!L.isInnermost() || !L.isLoopSimplifyForm() ||
      L.getExitingBlock() == nullptr || !Barrier::isLoopWithBarrier(L))
    return 0;

  unsigned TripCount = SE.getSmallConstantTripCount(&L);
  if (TripCount == 0 || TripCount > MaxTripCount)
    return 0;

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
    }
  }
  if (Size * TripCount > MaxUnrolledSize)
    return 0;

  return TripCount;
}

static bool unrollBarrierLoops(Function &F, FunctionAnalysisManager &AM) {
  Function *BarrierF = F.getParent()->getFunction(BARRIER_FUNCTION_NAME);
  if (BarrierF == nullptr)
    return false;

  LoopInfo &LI = AM.getResult<LoopAnalysis>(F);
  ScalarEvolution &SE = AM.getResult<ScalarEvolutionAnalysis>(F);

  SmallVector<std::pair<Loop *, unsigned>, 4> Loops;
  for (Loop *L : LI.getLoopsInPreorder())
    if (unsigned Count = getUnrollCount(*L, SE))
      Loops.push_back({L, Count});
  if (Loops.empty())
    return false;

  DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
  AssumptionCache &AC = AM.getResult<AssumptionAnalysis>(F);
  TargetTransformInfo &TTI = AM.getResult<TargetIRAnalysis>(F);
  OptimizationRemarkEmitter ORE(&F);

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
    // iteration through LCSSA phis.
    Changed |= formLCSSA(*L, DT, &LI, &SE);

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
