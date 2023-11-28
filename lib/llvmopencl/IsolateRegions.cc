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


#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "Barrier.h"
#include "IsolateRegions.h"
#include "LLVMUtils.h"
#include "VariableUniformityAnalysis.h"
#include "Workgroup.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#include <iostream>

#include "pocl.h"

#include "DebugHelpers.h"

//#define DEBUG_ISOLATE_REGIONS

#define PASS_NAME "isolate-regions"
#define PASS_CLASS pocl::IsolateRegions
#define PASS_DESC "Single-Entry Single-Exit region isolation pass."

namespace pocl {

using namespace llvm;

static void addDummyBefore(Region &R, llvm::BasicBlock *BB);
static void addDummyAfter(Region &R, llvm::BasicBlock *BB);

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

static bool isolateRegions(Region &R, WorkitemHandlerType WIH) {

  llvm::BasicBlock *Exit = R.getExit();
  if (Exit == nullptr)
    return false;
  if (WIH == WorkitemHandlerType::CBS &&
      hasWorkgroupBarriers(*Exit->getParent()))
    return false;

#ifdef DEBUG_ISOLATE_REGIONS
  std::cerr << "### processing region:" << std::endl;
  R.dump();
  std::cerr << "### exit block:" << std::endl;
  Exit->dump();
#endif
  bool isFunctionExit = Exit->getTerminator()->getNumSuccessors() == 0;

  bool changed = false;

  if (Barrier::hasBarrier(Exit) || isFunctionExit) {
      addDummyBefore(R, Exit);
      changed = true;
  }

  llvm::BasicBlock *entry = R.getEntry();
  if (entry == NULL) return changed;

  bool isFunctionEntry = &entry->getParent()->getEntryBlock() == entry;

  if (Barrier::hasBarrier(entry) || isFunctionEntry) {
    addDummyAfter(R, entry);
    changed = true;
  }

#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
#ifdef DEBUG_ISOLATE_REGIONS
  Function *F = exit->getParent();
  dumpCFG(*F, F->getName().str() + "_after_isolateregs.dot", nullptr, nullptr);
#endif
#endif
  return changed;
}

/**
 * Adds a dummy node after the given basic block.
 */
static void addDummyAfter(Region &R, llvm::BasicBlock *BB) {
  llvm::BasicBlock *NewEntry = SplitBlock(BB, BB->getTerminator());
  NewEntry->setName(BB->getName() + ".r_entry");
  R.replaceEntry(NewEntry);
}

/**
 * Adds a dummy node before the given basic block.
 *
 * The edges going in to the original BB are moved to go
 * in to the dummy BB in case the source BB is inside the
 * same region.
 */
static void addDummyBefore(llvm::Region &R, llvm::BasicBlock *BB) {
  std::vector<llvm::BasicBlock*> RegionPreds;

  for (pred_iterator PI = pred_begin(BB), PE = pred_end(BB); PI != PE; ++PI) {
    llvm::BasicBlock* Pred = *PI;
    if (R.contains(Pred))
      RegionPreds.push_back(Pred);
  }
  llvm::BasicBlock *NewExit =
      SplitBlockPredecessors(BB, RegionPreds, ".r_exit");
  R.replaceExit(NewExit);
}

#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
char IsolateRegions::ID = 0;

bool IsolateRegions::runOnRegion(Region *R, RGPassManager &RGM) {
  WorkitemHandlerType WIH =
      getAnalysis<pocl::WorkitemHandlerChooser>().chosenHandler();
  return isolateRegions(*R, WIH);
}

void IsolateRegions::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addPreserved<pocl::VariableUniformityAnalysis>();
  AU.addRequired<WorkitemHandlerChooser>();
  AU.addPreserved<WorkitemHandlerChooser>();
}

REGISTER_OLD_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

#else

static void findRegionsDepthFirst(Region *Reg, std::vector<Region *> &Regions) {

  for (Region::iterator I = Reg->begin(); I != Reg->end(); ++I) {
    findRegionsDepthFirst(I->get(), Regions);
  }
  Regions.push_back(Reg);
}

llvm::PreservedAnalyses IsolateRegions::run(llvm::Function &F,
                                            llvm::FunctionAnalysisManager &AM) {
  WorkitemHandlerType WIH = AM.getResult<WorkitemHandlerChooser>(F).WIH;
  RegionInfo &RI = AM.getResult<RegionInfoAnalysis>(F);

  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();
  PAChanged.preserve<VariableUniformityAnalysis>();
  bool ChangedAny = false;

  std::vector<Region *> Regions;
  findRegionsDepthFirst(RI.getTopLevelRegion(), Regions);
  unsigned NumRegions = Regions.size();
  for (unsigned i = 0; i < NumRegions; ++i) {
    bool ChangedCurrent = isolateRegions(*Regions[i], WIH);
    // changing a Region changes the pointers of the loop; retrieve them again
    if (ChangedCurrent) {
      Regions.clear();
      findRegionsDepthFirst(RI.getTopLevelRegion(), Regions);
      assert(Regions.size() == NumRegions);
    }
    ChangedAny = ChangedAny || ChangedCurrent;
  }

#ifdef DEBUG_ISOLATE_REGIONS
  dumpCFG(F, F.getName().str() + "_after_isolateregs.dot", nullptr, nullptr);
#endif

  return ChangedAny ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

#endif

} // namespace pocl
