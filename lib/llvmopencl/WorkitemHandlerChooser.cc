// LLVM function pass to select the best way to create a work group
// function for a kernel and work group size.
//
// Copyright (c) 2012-2019 Pekka Jääskeläinen
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
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/LoopInfo.h"

#include "CanonicalizeBarriers.h"
#include "Kernel.h"
#include "LLVMUtils.h"
#include "SubCFGFormation.h"
#include "Workgroup.h"
#include "WorkitemHandlerChooser.h"
#include "WorkitemLoops.h"
POP_COMPILER_DIAGS

#include "pocl_llvm_api.h"

#include <iostream>

#define DEBUG_TYPE "workitem-loops"

#define PASS_NAME "workitem-handler-chooser"
#define PASS_CLASS pocl::WorkitemHandlerChooser
#define PASS_DESC                                                              \
  "Finds the best way to handle work-items to produce a multi-WI function."

namespace pocl {

using namespace llvm;

/**
 * Selects the work-group generator to use for handling the given
 * kernel.
 */
WorkitemHandlerType ChooseWorkitemHandler(Function &F,
                                          llvm::FunctionAnalysisManager &AM) {
  if (!isKernelToProcess(F))
    return WorkitemHandlerType::INVALID;

  WorkitemHandlerType Result = WorkitemHandlerType::INVALID;
  bool WGDynamicLocalSize = false;

  getModuleBoolMetadata(*F.getParent(), "WGDynamicLocalSize",
                        WGDynamicLocalSize);

  std::string method = "auto";
  if (getenv("POCL_WORK_GROUP_METHOD") != NULL) {
    method = getenv("POCL_WORK_GROUP_METHOD");
    if (method == "loops" || method == "workitemloops" || method == "loopvec")
      Result = WorkitemHandlerType::LOOPS;
    else if (method == "cbs")
      Result = WorkitemHandlerType::CBS;
    else if (method != "auto") {
      std::cerr << "Unknown work group generation method. Using 'auto'."
                << std::endl;
      method = "auto";
    }
  }

  if (method == "auto") {
    Result = WorkitemHandlerType::LOOPS;
  }

  if (Result == WorkitemHandlerType::LOOPS &&
      !WorkitemLoops::canHandleKernel(F, AM))
    Result = WorkitemHandlerType::CBS;
  else if (Result == WorkitemHandlerType::CBS &&
           !SubCFGFormation::canHandleKernel(F, AM))
    Result = WorkitemHandlerType::LOOPS;

  return Result;
}

/**********************************************************************/

llvm::AnalysisKey WorkitemHandlerChooser::Key;

WorkitemHandlerResult
WorkitemHandlerChooser::run(llvm::Function &F,
                            llvm::FunctionAnalysisManager &AM) {
  return ChooseWorkitemHandler(F, AM);
}

bool WorkitemHandlerResult::invalidate(
    llvm::Function &F, const llvm::PreservedAnalyses PA,
    llvm::AnalysisManager<llvm::Function>::Invalidator &Inv) {
  // Check whether the analysis has been explicitly invalidated.
  // Otherwise, it stays valid
  auto PAC = PA.getChecker<WorkitemHandlerChooser>();
  return !PAC.preservedWhenStateless();
}

REGISTER_NEW_FANALYSIS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
