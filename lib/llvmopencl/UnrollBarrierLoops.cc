// UnrollBarrierLoops fully unrolls requested loops with barriers.
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
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallPtrSet.h>
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

#include <optional>

#include "Barrier.h"
#include "LLVMUtils.h"
#include "UnrollBarrierLoops.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#define PASS_NAME "unroll-barrier-loops"
#define PASS_CLASS pocl::UnrollBarrierLoops
#define PASS_DESC "Fully unroll requested loops with barriers"

namespace pocl {

using namespace llvm;

// A loop with a barrier (a b-loop) can't stay inside one work-item loop: its
// body is split into parallel regions that each run for all work-items in
// every iteration. That costs more than the user's barriers. loop-barriers
// adds barriers around the loop control, which then forms regions of its own,
// and phistoallocas turns the loop's phis (the induction variable, a stride or
// a mask) into allocas that are privatized per work-item even though they are
// uniform. Each region then reloads them from a context array, which also
// hides the access pattern from the vectorizer.
//
// Fully unrolling such a loop leaves straight-line regions separated only by
// the user's barriers, with the per-iteration values constant in each copy.
// LLVM's unroller can't do this, as pocl.barrier is noduplicate, and often the
// trip count only became a constant when the stage 1 optimizations folded the
// local or sub-group size. This pass fully unrolls loops that carry an unroll
// request ('#pragma unroll', opencl_unroll_hint, or SPIR-V's Unroll loop
// control) and have a constant trip count. A partial request leaves the
// per-iteration values variable, so it is left to LLVM's unroller in stage 2,
// which applies it once the barriers are gone.

// LLVM's default -pragma-unroll-threshold
static constexpr unsigned MaxUnrolledSize = 16 * 1024;

// Returns the trip count of L if it should be fully unrolled, or 0.
static unsigned getUnrollCount(Loop &L, ScalarEvolution &SE) {
  if (!L.isInnermost() || !L.isLoopSimplifyForm() ||
      L.getExitingBlock() == nullptr || !Barrier::isLoopWithBarrier(L))
    return 0;

  if (hasUnrollTransformation(&L) != TM_ForcedByUser)
    return 0;

  unsigned TripCount = SE.getSmallConstantTripCount(&L);
  if (TripCount == 0)
    return 0;

  std::optional<int> Count =
      getOptionalIntLoopAttribute(&L, "llvm.loop.unroll.count");
  if (Count && *Count > 0 && unsigned(*Count) < TripCount)
    return 0;

  unsigned Size = 0;
  for (BasicBlock *BB : L.blocks()) {
    for (Instruction &I : *BB) {
      // other noduplicate calls aren't ours to duplicate
      if (auto *Call = dyn_cast<CallBase>(&I))
        if (Call->cannotDuplicate() && !isa<Barrier>(Call))
          return 0;
      if (!I.isDebugOrPseudoInst())
        ++Size;
    }
  }
  if (uint64_t(Size) * TripCount > MaxUnrolledSize)
    return 0;

  return TripCount;
}

using LoopsToUnroll = SmallVector<std::pair<Loop *, unsigned>, 4>;

static LoopsToUnroll findLoopsToUnroll(Function &F,
                                       FunctionAnalysisManager &FAM) {
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
  ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);

  LoopsToUnroll Loops;
  for (Loop *L : LI.getLoopsInPreorder())
    if (unsigned Count = getUnrollCount(*L, SE))
      Loops.push_back({L, Count});
  return Loops;
}

static bool unrollLoops(Function &F, const LoopsToUnroll &Loops,
                        FunctionAnalysisManager &FAM) {
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
  ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  AssumptionCache &AC = FAM.getResult<AssumptionAnalysis>(F);
  TargetTransformInfo &TTI = FAM.getResult<TargetIRAnalysis>(F);
  OptimizationRemarkEmitter ORE(&F);

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
    Changed |=
        UnrollLoop(L, ULO, &LI, &SE, &DT, &AC, &TTI, &ORE,
                   /*PreserveLCSSA=*/true) != LoopUnrollResult::Unmodified;
  }
  return Changed;
}

// A module pass, as it modifies the barrier declaration.
llvm::PreservedAnalyses
UnrollBarrierLoops::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {
  Function *BarrierF = M.getFunction(BARRIER_FUNCTION_NAME);
  if (BarrierF == nullptr)
    return PreservedAnalyses::all();

  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  SmallSetVector<Function *, 4> Callers;
  for (User *U : BarrierF->users())
    if (auto *Call = dyn_cast<CallInst>(U))
      if (isKernelToProcess(*Call->getFunction()))
        Callers.insert(Call->getFunction());

  SmallVector<std::pair<Function *, LoopsToUnroll>, 4> Kernels;
  for (Function *F : Callers) {
    LoopsToUnroll Loops = findLoopsToUnroll(*F, FAM);
    if (!Loops.empty())
      Kernels.push_back({F, std::move(Loops)});
  }
  if (Kernels.empty())
    return PreservedAnalyses::all();

  // pocl.barrier is noduplicate as a workaround for #2127, to keep the
  // optimizations from moving barriers where only some work-items reach them
  // (which it doesn't guarantee). Fully unrolling a loop with a constant trip
  // count keeps every work-item on the same sequence of barriers, so lift the
  // attribute while doing only that. Calls with the attribute in a kernel
  // get it back on all barriers in it, which includes the copies.
  bool DeclNoDup = BarrierF->hasFnAttribute(Attribute::NoDuplicate);
  BarrierF->removeFnAttr(Attribute::NoDuplicate);
  SmallPtrSet<Function *, 4> KernelSet;
  for (auto &KL : Kernels)
    KernelSet.insert(KL.first);
  SmallPtrSet<Function *, 4> CallNoDup;
  for (User *U : BarrierF->users()) {
    auto *Call = dyn_cast<CallInst>(U);
    if (Call == nullptr || !KernelSet.contains(Call->getFunction()) ||
        !Call->hasFnAttr(Attribute::NoDuplicate))
      continue;
    CallNoDup.insert(Call->getFunction());
    Call->removeFnAttr(Attribute::NoDuplicate);
  }

  bool Changed = false;
  for (auto &[F, Loops] : Kernels) {
    if (!unrollLoops(*F, Loops, FAM))
      continue;
    Changed = true;
    PreservedAnalyses PA = PreservedAnalyses::none();
    PA.preserve<WorkitemHandlerChooser>();
    FAM.invalidate(*F, PA);
  }

  if (DeclNoDup)
    BarrierF->addFnAttr(Attribute::NoDuplicate);
  for (User *U : BarrierF->users())
    if (auto *Call = dyn_cast<CallInst>(U))
      if (CallNoDup.contains(Call->getFunction()))
        Call->addFnAttr(Attribute::NoDuplicate);

  if (!Changed)
    return PreservedAnalyses::all();

  // the function analyses were invalidated above
  PreservedAnalyses PA = PreservedAnalyses::none();
  PA.preserveSet<AllAnalysesOn<Function>>();
  PA.preserve<FunctionAnalysisManagerModuleProxy>();
  return PA;
}

REGISTER_NEW_MPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
