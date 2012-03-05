// Header for IsolateRegions RegionPass.
// 
// Copyright (c) 2012 Pekka Jääskeläinen / TUT
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

#include "IsolateRegions.h"
#include "Workgroup.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "config.h"

#include <iostream>

//#define DEBUG_ISOLATE_REGIONS
using namespace llvm;
using namespace pocl;
 
namespace {
  static
  RegisterPass<IsolateRegions> X("isolate-regions",
					 "Single-Entry Single-Exit region isolation pass.");
}

char IsolateRegions::ID = 0;

void
IsolateRegions::getAnalysisUsage(AnalysisUsage &AU) const
{
}

/* Ensure Single-Entry Single-Exit Regions are isolated from the
   exit node so they won't get split illegally with tail replication. 

   This might happen in casen an if .. else .. structure is just 
   before an exit from kernel. Both branches are split even though
   we would like to replicate the structure as a whole to retain
   semantics. This adds dummy basic blocks to all Regions just for
   clarity. Cleanup with -simplifycfg.
*/
bool
IsolateRegions::runOnRegion(Region *R, llvm::RGPassManager&) 
{
  llvm::BasicBlock* exit = R->getExit();
  if (exit == NULL) return false;

#ifdef DEBUG_ISOLATE_REGIONS
  std::cerr << "### processing region:" << std::endl;
  R->dump();
  std::cerr << "### exit block:" << std::endl;
  exit->dump();
#endif

  std::vector< llvm::BasicBlock* > regionPreds;

  for (pred_iterator i = pred_begin(exit), e = pred_end(exit);
       i != e; ++i) {
    llvm::BasicBlock* pred = *i;
    if (R->contains(pred))
      regionPreds.push_back(pred);
  }
#ifdef LLVM_3_0
  llvm::BasicBlock* newExit = 
    SplitBlockPredecessors
    (exit, &regionPreds[0], regionPreds.size(), ".region_exit", this);
#else
  llvm::BasicBlock* newExit = 
    SplitBlockPredecessors(exit, regionPreds, ".region_exit", this);
#endif
  R->replaceExit(newExit);
  return true;
}
