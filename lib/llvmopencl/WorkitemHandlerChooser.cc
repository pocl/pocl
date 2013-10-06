// LLVM function pass to select the best way to create a work group
// function for a kernel and work group size.
// 
// Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
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

#define DEBUG_TYPE "workitem-loops"

#include "WorkitemHandlerChooser.h"
#include "WorkitemLoops.h"
#include "WorkitemReplication.h"
#include "Workgroup.h"
#include "CanonicalizeBarriers.h"
#include "Kernel.h"

#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/LoopInfo.h"

#include <iostream>

using namespace llvm;
using namespace pocl;

namespace {
  static
  RegisterPass<WorkitemHandlerChooser> X(
      "workitem-handler-chooser", 
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
  if (!Workgroup::isKernelToProcess(F))
    return false;

  Kernel *K = cast<Kernel> (&F);
  Initialize(K);

  std::string method = "auto";
  if (getenv("POCL_WORK_GROUP_METHOD") != NULL)
    {
      method = getenv("POCL_WORK_GROUP_METHOD");
      if (method == "repl" || method == "workitemrepl")
        chosenHandler_ = POCL_WIH_FULL_REPLICATION;
      else if (method == "loops" || method == "workitemloops" || method == "loopvec")
        chosenHandler_ = POCL_WIH_LOOPS;
      else if (method != "auto")
        {
          std::cerr << "Unknown work group generation method. Using 'auto'." << std::endl;
          method = "auto";
        }
    }

  if (method == "auto") 
    {
      int ReplThreshold = 2;
      if (getenv("POCL_FULL_REPLICATION_THRESHOLD") != NULL) 
      {
        ReplThreshold = atoi(getenv("POCL_FULL_REPLICATION_THRESHOLD"));
      }
      
      if (LocalSizeX*LocalSizeY*LocalSizeZ <= ReplThreshold)
        {
          chosenHandler_ = POCL_WIH_FULL_REPLICATION;
        }
      else
        {
          chosenHandler_ = POCL_WIH_LOOPS;
        }
    }

  return false;
}

}
