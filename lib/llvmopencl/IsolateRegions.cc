// Header for IsolateRegions RegionPass.
// 
// Copyright (c) 2012-2015 Pekka Jääskeläinen / TUT
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

#include "config.h"
#include "pocl.h"

#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "IsolateRegions.h"
#include "Barrier.h"
#include "Workgroup.h"
#include "VariableUniformityAnalysis.h"
#include "WorkitemHandlerChooser.h"

POP_COMPILER_DIAGS

//#define DEBUG_ISOLATE_REGIONS
using namespace llvm;
using namespace pocl;
 
namespace {
  static
  RegisterPass<IsolateRegions> X("isolate-regions",
					 "Single-Entry Single-Exit region isolation pass.");
}

char IsolateRegions::ID = 0;

void IsolateRegions::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addPreserved<pocl::VariableUniformityAnalysis>();
  AU.addRequired<WorkitemHandlerChooser>();
  AU.addPreserved<WorkitemHandlerChooser>();
}

/* Ensure Single-Entry Single-Exit Regions are isolated from the
   exit node so they won't get split illegally with tail replication. 

   This might happen in case an if .. else .. structure is just 
   before an exit from kernel. Both branches are split even though
   we would like to replicate the structure as a whole to retain
   semantics. This adds dummy basic blocks to all Regions just for
   clarity. Cleanup with -simplifycfg.

   TODO: Also add a dummy BB in case the Region starts with a
   barrier. Such a Region might not get optimally replicated and
   can lead to problematic cases. E.g.:

   digraph G {
      BAR1 -> A;
      A -> X; 
      BAR1 -> X; 
      X -> BAR2;
   }

   (draw with "dot -Tpng -o graph.png"   + copy paste the above)

   Here you have a structure which should be replicated fully but
   it won't as the Region starts with a barrier at a split point
   BB, thus it tries to replicate both of the branches which lead
   to interesting errors and is not supported. Another option would
   be to tail replicate both of the branches, but currently tail
   replication is done only starting from the exit nodes.

   IsolateRegions "normalizes" the graph to:

   digraph G {
      BAR1 -> r_entry;
      r_entry -> A;
      A -> X; 
      r_entry -> X; 
      X -> BAR2;
   }

   
*/
bool IsolateRegions::runOnRegion(Region *R, llvm::RGPassManager&) {

  llvm::BasicBlock *exit = R->getExit();
  if (exit == NULL) return false;
  if (getAnalysis<pocl::WorkitemHandlerChooser>().chosenHandler() ==
      pocl::WorkitemHandlerChooser::POCL_WIH_CBS)
    return false;

#ifdef DEBUG_ISOLATE_REGIONS
  std::cerr << "### processing region:" << std::endl;
  R->dump();
  std::cerr << "### exit block:" << std::endl;
  exit->dump();
#endif
  bool isFunctionExit = exit->getTerminator()->getNumSuccessors() == 0;

  bool changed = false;

  if (Barrier::hasBarrier(exit) || isFunctionExit) {
      addDummyBefore(R, exit);
      changed = true;
  }

  llvm::BasicBlock *entry = R->getEntry();
  if (entry == NULL) return changed;

  bool isFunctionEntry = &entry->getParent()->getEntryBlock() == entry;

  if (Barrier::hasBarrier(entry) || isFunctionEntry) {
    addDummyAfter(R, entry);
    changed = true;
  }

  return changed;
}


/**
 * Adds a dummy node after the given basic block.
 */
void IsolateRegions::addDummyAfter(llvm::Region *R, llvm::BasicBlock *bb) {
  llvm::BasicBlock* newEntry = 
    SplitBlock(bb, bb->getTerminator());
  newEntry->setName(bb->getName() + ".r_entry");
  R->replaceEntry(newEntry);
}

/**
 * Adds a dummy node before the given basic block.
 *
 * The edges going in to the original BB are moved to go
 * in to the dummy BB in case the source BB is inside the
 * same region.
 */
void
IsolateRegions::addDummyBefore(llvm::Region *R, llvm::BasicBlock *bb) {
  std::vector< llvm::BasicBlock* > regionPreds;

  for (pred_iterator i = pred_begin(bb), e = pred_end(bb);
       i != e; ++i) {
    llvm::BasicBlock* pred = *i;
    if (R->contains(pred))
      regionPreds.push_back(pred);
  }
  llvm::BasicBlock* newExit = 
    SplitBlockPredecessors(bb, regionPreds, ".r_exit");
  R->replaceExit(newExit);
}
