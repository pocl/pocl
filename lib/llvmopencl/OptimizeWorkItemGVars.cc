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

#include "LLVMUtils.h"
#include "OptimizeWorkItemGVars.h"
#include "VariableUniformityAnalysis.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#include <iostream>
#include <map>
#include <set>

#define PASS_NAME "optimize-wi-gvars"
#define PASS_CLASS pocl::OptimizeWorkItemGVars
#define PASS_DESC "Optimize work-item global variable loads"

//#define DEBUG_OPTIMIZE_WI_GVARS

namespace pocl {

using namespace llvm;

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

  return optimizeWorkItemGVars(F) ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
