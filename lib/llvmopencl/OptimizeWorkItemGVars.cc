// OptimizeWorkItemGVars is an LLVM pass to optimize loads from "magic" extern
// global variables that are WI function related (_local_id_x etc).
//
// Copyright (c) 2024 Michal Babej / Intel Finland Oy
//
// This is unfortunately not optional; in some cases, if this pass is not run,
// LLVM optimizes switch cases with three loads (@_local_id_x, @_local_id_y...)
// into a Phi followed by a single load:
//   %p = phi [ @_local_id_x, %sw.branch.0 ], [ @_local_id_y, %sw.branch.1 ]...
//   load i64, ptr %p
//
// the Phi is then replaced by the phis2allocas pass with an alloca + stores &
// loads. Since the workgroup pass can only deal with loads from the special WI
// variables, this ends up leaving unresolved symbols in the final binary.
//
//
// TODO: we should replace the magic global variables (work-item ID
// placeholders) with function calls or intrinsics.
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

#include "KernelCompilerUtils.h"
#include "LLVMUtils.h"
#include "OptimizeWorkItemGVars.h"
#include "VariableUniformityAnalysis.h"
#include "WorkitemHandlerChooser.h"
#include "pocl_llvm_api.h"
POP_COMPILER_DIAGS

#include <iostream>
#include <map>
#include <set>

#define PASS_NAME "optimize-wi-gvars"
#define PASS_CLASS pocl::OptimizeWorkItemGVars
#define PASS_DESC "Optimize work-item global variable loads"

// #define DEBUG_OPTIMIZE_WI_GVARS

namespace pocl {

using namespace llvm;

// Replace the work-item values that are known at compile time with constants.
// Workgroup and the work-item handlers would only do so after the barrier
// passes, which then see the loops bounded by them (e.g. a tree reduction over
// the local size, or the butterflies of sub-group shuffles) as loops of unknown
// trip count. The values are known:
// - the local size, if the work-group function is specialized for it. All
//   work-groups have that size, as non-uniform work-groups are unsupported.
//   A dimension of size 1 also has local id 0.
// - the sub-group size, from intel_reqd_sub_group_size, or else the local
//   size X (see Workgroup.cc).
static bool foldWorkItemValues(Function &F) {
  Module *M = F.getParent();

  // SPMD devices don't use the work-group functions these are specialized for
  bool SPMD = false;
  getModuleBoolMetadata(*M, "device_is_spmd", SPMD);
  if (SPMD)
    return false;

  bool DynamicLocalSize = true;
  unsigned long LocalSize[3] = {0, 0, 0};
  getModuleBoolMetadata(*M, "WGDynamicLocalSize", DynamicLocalSize);
  getModuleIntMetadata(*M, "WGLocalSizeX", LocalSize[0]);
  getModuleIntMetadata(*M, "WGLocalSizeY", LocalSize[1]);
  getModuleIntMetadata(*M, "WGLocalSizeZ", LocalSize[2]);
  bool StaticLocalSize = !DynamicLocalSize && LocalSize[0] != 0 &&
                         LocalSize[1] != 0 && LocalSize[2] != 0;

  std::vector<std::pair<Instruction *, uint64_t>> Replacements;
  auto foldLoads = [&](const char *Name, uint64_t Value) {
    GlobalVariable *GVar = M->getGlobalVariable(Name);
    if (GVar == nullptr)
      return;
    for (User *U : GVar->users())
      if (auto *LI = dyn_cast<LoadInst>(U))
        if (LI->getFunction() == &F)
          Replacements.push_back({LI, Value});
  };

  if (StaticLocalSize) {
    foldLoads("_local_size_x", LocalSize[0]);
    foldLoads("_local_size_y", LocalSize[1]);
    foldLoads("_local_size_z", LocalSize[2]);
  }

  uint64_t SubgroupSize = 0;
  if (ConstantInt *Required = getRequiredSubgroupSize(F))
    SubgroupSize = Required->getZExtValue();
  else if (StaticLocalSize)
    SubgroupSize = LocalSize[0];
  if (SubgroupSize != 0)
    foldLoads("_pocl_sub_group_size", SubgroupSize);

  // the work-item functions with a constant dimension, which the work-item
  // handlers otherwise expand
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *Call = dyn_cast<CallInst>(&I);
      if (Call == nullptr || Call->getCalledFunction() == nullptr ||
          Call->arg_size() != 1)
        continue;
      auto *DimArg = dyn_cast<ConstantInt>(Call->getArgOperand(0));
      if (DimArg == nullptr)
        continue;
      uint64_t Dim = DimArg->getZExtValue();
      StringRef Name = Call->getCalledFunction()->getName();
      if (StaticLocalSize &&
          (Name == LS_BUILTIN_NAME || Name == ENQUEUE_LS_BUILTIN_NAME))
        Replacements.push_back({Call, Dim < 3 ? LocalSize[Dim] : 1});
      else if (StaticLocalSize && Name == LID_BUILTIN_NAME &&
               (Dim >= 3 || LocalSize[Dim] == 1))
        Replacements.push_back({Call, 0});
    }
  }

  for (auto [I, Value] : Replacements) {
    I->replaceAllUsesWith(ConstantInt::get(I->getType(), Value));
    I->eraseFromParent();
  }
  return !Replacements.empty();
}

static bool optimizeWorkItemGVars(Function &F) {

  bool Changed = false;

  Module *M = F.getParent();
  for (auto GVarName : WorkgroupVariablesVector) {
    GlobalVariable *GVar = M->getGlobalVariable(GVarName);
    if (!GVar)
      continue;
    if (!isGVarUsedByFunction(GVar, &F))
      continue;

#ifdef DEBUG_OPTIMIZE_WI_GVARS
    std::cerr << "; ######### Optimizing GVAR: " << GVarName << "\n";
#endif
    std::vector<LoadInst *> GVUsers;
    for (auto U : GVar->users()) {
      LoadInst *LI = dyn_cast<LoadInst>(U);
      if (LI == nullptr) {
#ifdef DEBUG_OPTIMIZE_WI_GVARS
        std::cerr << "; ######### ERROR: User is not LOAD\n";
        LI->dump();
#endif
        continue;
      }
      if (LI->getFunction() == &F) {
        GVUsers.push_back(LI);
      }
    }

    if (GVUsers.size() > 1) {
      Changed = true;
      IRBuilder<> Builder(&*(F.getEntryBlock().getFirstInsertionPt()));
      LoadInst *ReplLoad = Builder.CreateLoad(GVar->getValueType(), GVar,
                                              Twine(GVarName, "_load"));
      for (auto U : GVUsers) {
        U->replaceAllUsesWith(ReplLoad);
        U->eraseFromParent();
      }
    }
  }

  return Changed;
}

llvm::PreservedAnalyses
OptimizeWorkItemGVars::run(llvm::Function &F,
                           llvm::FunctionAnalysisManager &AM) {
  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();

  if (!isKernelToProcess(F))
    return PreservedAnalyses::all();

  bool Changed = foldWorkItemValues(F);
  Changed |= optimizeWorkItemGVars(F);
  return Changed ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
