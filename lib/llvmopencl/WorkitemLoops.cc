// LLVM function pass to create a loop that runs all the work items 
// in a work group.
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

#define DEBUG_TYPE "workitem-loops"

#include "WorkitemLoops.h"
#include "Workgroup.h"
#include "Barrier.h"
#include "Kernel.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/ValueSymbolTable.h"
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#ifdef DUMP_RESULT_CFG
#include "llvm/Analysis/CFGPrinter.h"
#endif

using namespace llvm;
using namespace pocl;

namespace {
  static
  RegisterPass<WorkitemLoops> X("workitemloops", "Workitem loop generation pass");
}

char WorkitemLoops::ID = 0;

void
WorkitemLoops::getAnalysisUsage(AnalysisUsage &AU) const
{
  AU.addRequired<DominatorTree>();
  AU.addRequired<LoopInfo>();
  AU.addRequired<TargetData>();
}

bool
WorkitemLoops::runOnFunction(Function &F)
{
  if (!Workgroup::isKernelToProcess(F))
    return false;

  DT = &getAnalysis<DominatorTree>();
  LI = &getAnalysis<LoopInfo>();

  bool changed = ProcessFunction(F);
#ifdef DUMP_RESULT_CFG
  FunctionPass* cfgPrinter = createCFGPrinterPass();
  cfgPrinter->runOnFunction(F);
#endif

  changed |= fixUndominatedVariableUses(DT, F);
  return changed;
}

bool
WorkitemLoops::ProcessFunction(Function &F)
{
  Kernel *K = cast<Kernel> (&F);
  CheckLocalSize(K);

  // Allocate space for workitem reference maps. Workitem 0 does
  // not need it.
  unsigned workitem_count = LocalSizeZ * LocalSizeY * LocalSizeX;

  BasicBlockVector original_bbs;
  for (Function::iterator i = F.begin(), e = F.end(); i != e; ++i) {
      if (!Barrier::hasBarrier(i))
        original_bbs.push_back(i);
  }

  ParallelRegion::ParallelRegionVector* original_parallel_regions =
    K->getParallelRegions(LI);

  std::vector<SmallVector<ParallelRegion *, 8> > parallel_regions(workitem_count);

  assert (false && "Unimplemented.");
#if 0
  parallel_regions[0] = *original_parallel_regions;
  
  // Then replicate the ParallelRegions.  
  ValueToValueMapTy *const reference_map = new ValueToValueMapTy[workitem_count - 1];
  for (int z = 0; z < LocalSizeZ; ++z) {
    for (int y = 0; y < LocalSizeY; ++y) {
      for (int x = 0; x < LocalSizeX ; ++x) {
              
        int index = 
          (LocalSizeY * LocalSizeX * z + LocalSizeX * y + x);
	  
        if (index == 0)
          continue;
	  
        std::size_t regionCounter = 0;
        for (SmallVector<ParallelRegion *, 8>::iterator
               i = original_parallel_regions->begin(), 
               e = original_parallel_regions->end();
             i != e; ++i) {
          ParallelRegion *original = (*i);
          ParallelRegion *replicated =
            original->replicate
            (reference_map[index - 1],
             (".wi_" + Twine(x) + "_" + Twine(y) + "_" + Twine(z)));
          regionCounter++;
          parallel_regions[index].push_back(replicated);
#ifdef DEBUG_PR_REPLICATION
          std::cerr << "### new replica:" << std::endl;
          replicated->dump();
#endif
        }
      }
    }
  }
  
  for (int z = 0; z < LocalSizeZ; ++z) {
    for (int y = 0; y < LocalSizeY; ++y) {
      for (int x = 0; x < LocalSizeX ; ++x) {
	  
        int index = 
          (LocalSizeY * LocalSizeX * z + LocalSizeX * y + x);
        
        for (unsigned i = 0, e = parallel_regions[index].size(); i != e; ++i) {
          ParallelRegion *region = parallel_regions[index][i];
          if (index != 0) {
            region->remap(reference_map[index - 1]);
            region->chainAfter(parallel_regions[index - 1][i]);
            region->purge();
          }
          region->insertPrologue(x, y, z);
        }
      }
    }
  }
    
  // Try to merge all workitem first block of each region
  // together (for PHI predecessor correctness).
  for (int z = LocalSizeZ - 1; z >= 0; --z) {
    for (int y = LocalSizeY - 1; y >= 0; --y) {
      for (int x = LocalSizeX - 1; x >= 0; --x) {
          
        int index = 
          (LocalSizeY * LocalSizeX * z + LocalSizeX * y + x);

        if (index == 0)
          continue;
          
        for (unsigned i = 0, e = parallel_regions[index].size(); i != e; ++i) {
          ParallelRegion *region = parallel_regions[index][i];
          BasicBlock *entry = region->entryBB();

          assert (entry != NULL);
          BasicBlock *pred = entry->getUniquePredecessor();
          assert (pred != NULL && "No unique predecessor.");
#ifdef DEBUG_BB_MERGING
          std::cerr << "### pred before merge into predecessor " << std::endl;
          pred->dump();
          std::cerr << "### entry before merge into predecessor " << std::endl;
          entry->dump();
#endif
          movePhiNodes(entry, pred);
        }
      }
    }
  }

  // Add the suffixes to original (wi_0_0_0) basic blocks.
  for (BasicBlockVector::iterator i = original_bbs.begin(),
         e = original_bbs.end();
       i != e; ++i)
    (*i)->setName((*i)->getName() + ".wi_0_0_0");

  // Initialize local size variables (done at the end to avoid unnecessary
  // replication).
  K->addLocalSizeInitCode(LocalSizeX, LocalSizeY, LocalSizeZ);

  delete [] reference_map;
#endif
  return true;
}

