// LLVM function pass to select the best way to create a work group
// function for a kernel and work group size.
//
// Copyright (c) 2012-2019 Pekka Jääskeläinen
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

#include <iostream>

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/LoopInfo.h"

#define DEBUG_TYPE "workitem-loops"

#include "WorkitemHandlerChooser.h"
#include "WorkitemLoops.h"
#include "WorkitemReplication.h"
#include "Workgroup.h"
#include "CanonicalizeBarriers.h"
#include "Kernel.h"

using namespace llvm;
using namespace pocl;

namespace {
static RegisterPass<WorkitemHandlerChooser>
    X("workitem-handler-chooser",
      "Finds the best way to handle work-items to produce a multi-WG function.",
      false, false);
}

namespace pocl {

char WorkitemHandlerChooser::ID = 0;

void
WorkitemHandlerChooser::getAnalysisUsage(AnalysisUsage &AU) const
{
  AU.setPreservesAll();
}


bool
WorkitemHandlerChooser::runOnFunction(Function &F)
{
  if (!isKernelToProcess(F))
    return false;

  Kernel *K = cast<Kernel> (&F);

  /* FIXME: this is not thread safe. We cannot compile multiple kernels at
     the same time with the same instances of these passes as they store
     to private attributes and use global values to pass in the dimensions.
     In the LLVMAPI version this logic should be at higher-level when
     constructing the passes for the kernel, or done fully inside a single
     FunctionPass that delegates to other passes. */    
  Initialize(K);

  std::string method = "auto";
  if (getenv("POCL_WORK_GROUP_METHOD") != NULL)
    {
      method = getenv("POCL_WORK_GROUP_METHOD");
      if ((method == "repl" || method == "workitemrepl") && !WGDynamicLocalSize)
        chosenHandler_ = POCL_WIH_FULL_REPLICATION;
      else if (method == "loops" || method == "workitemloops" || method == "loopvec")
        chosenHandler_ = POCL_WIH_LOOPS;
      else if (method == "cbs")
        chosenHandler_ = POCL_WIH_CBS;
      else if (method != "auto")
        {
          std::cerr << "Unknown work group generation method. Using 'auto'." << std::endl;
          method = "auto";
        }
    }

  if (method == "auto") 
    {
      unsigned ReplThreshold = 2;
      if (getenv("POCL_FULL_REPLICATION_THRESHOLD") != NULL) 
      {
        ReplThreshold = atoi(getenv("POCL_FULL_REPLICATION_THRESHOLD"));
      }

      if (!WGDynamicLocalSize &&
          WGLocalSizeX * WGLocalSizeY * WGLocalSizeZ <= ReplThreshold) {
        chosenHandler_ = POCL_WIH_FULL_REPLICATION;
      } else {
        chosenHandler_ = POCL_WIH_LOOPS;
      }
    }

  return false;
}

}
