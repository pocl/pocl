// LLVM function pass to replicate the kernel body for all work items 
// in a work group.
// 
// Copyright (c) 2011-2012 Carlos Sánchez de La Lama / URJC and
//               2011-2015 Pekka Jääskeläinen / TUT
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
#include <map>
#include <sstream>
#include <vector>

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "pocl.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "workitem"

#include "WorkitemReplication.h"
#include "Workgroup.h"
#include "Barrier.h"
#include "Kernel.h"
#include "WorkitemHandlerChooser.h"
#include "DebugHelpers.h"
#include "VariableUniformityAnalysis.h"

//#define DEBUG_BB_MERGING
//#define DUMP_RESULT_CFG
//#define DEBUG_PR_REPLICATION

#ifdef DUMP_RESULT_CFG
#include "llvm/Analysis/CFGPrinter.h"
#endif

POP_COMPILER_DIAGS

using namespace llvm;
using namespace pocl;

STATISTIC(ContextValues, "Number of SSA values which have to be context-saved");
STATISTIC(ContextSize, "Context size per workitem in bytes");

namespace {
  static
  RegisterPass<WorkitemReplication> X("workitemrepl", "Workitem replication pass");
}

char WorkitemReplication::ID = 0;

void
WorkitemReplication::getAnalysisUsage(AnalysisUsage &AU) const
{
  AU.addRequired<DominatorTreeWrapperPass>();

  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<pocl::WorkitemHandlerChooser>();
  AU.addPreserved<pocl::VariableUniformityAnalysis>();
}

bool
WorkitemReplication::runOnFunction(Function &F)
{
  if (!Workgroup::isKernelToProcess(F))
    return false;

  if (getAnalysis<pocl::WorkitemHandlerChooser>().chosenHandler() != 
      pocl::WorkitemHandlerChooser::POCL_WIH_FULL_REPLICATION)
    return false;

  DTP = &getAnalysis<DominatorTreeWrapperPass>();
  DT = &DTP->getDomTree();

  LI = &getAnalysis<LoopInfoWrapperPass>();

  bool changed = ProcessFunction(F);
#ifdef DUMP_RESULT_CFG
  FunctionPass* cfgPrinter = createCFGPrinterPass();
  cfgPrinter->runOnFunction(F);
#endif

  changed |= fixUndominatedVariableUses(DTP, F);
  return changed;
}

bool
WorkitemReplication::ProcessFunction(Function &F)
{
  Module *M = F.getParent();

//  F.viewCFG();

  Kernel *K = cast<Kernel> (&F);
  Initialize(K);

  // Allocate space for workitem reference maps. Workitem 0 does
  // not need it.
  unsigned workitem_count = WGLocalSizeZ * WGLocalSizeY * WGLocalSizeX;

  BasicBlockVector original_bbs;
  for (Function::iterator i = F.begin(), e = F.end(); i != e; ++i) {
      if (!Barrier::hasBarrier(&*i))
        original_bbs.push_back(&*i);
  }

  ParallelRegion::ParallelRegionVector* original_parallel_regions =
    K->getParallelRegions(&LI->getLoopInfo());

  std::vector<ParallelRegion::ParallelRegionVector> parallel_regions(
      workitem_count);

  parallel_regions[0] = *original_parallel_regions;

  //pocl::dumpCFG(F, F.getName().str() + ".before_repl.dot", original_parallel_regions);

  /* Enable to get region identification printouts */
#if 0
  for (ParallelRegion::ParallelRegionVector::iterator
           i = original_parallel_regions->begin(), 
           e = original_parallel_regions->end();
       i != e; ++i) 
  {
    ParallelRegion *region = (*i);
    region->InjectRegionPrintF();
    region->InjectVariablePrintouts();
  }
#endif
  
  // Measure the required context (variables alive in more than one region).

  for (SmallVector<ParallelRegion *, 8>::iterator
         i = original_parallel_regions->begin(), 
           e = original_parallel_regions->end();
       i != e; ++i) {
    ParallelRegion *pr = (*i);
    
    for (ParallelRegion::iterator i2 = pr->begin(), e2 = pr->end();
         i2 != e2; ++i2) {
      BasicBlock *bb = (*i2);
      
      for (BasicBlock::iterator i3 = bb->begin(), e3 = bb->end();
           i3 != e3; ++i3) {
        for (Value::use_iterator i4 = i3->use_begin(), e4 = i3->use_end();
             i4 != e4; ++i4) {
          // Instructions can only be used by instructions.
          llvm::Instruction *user = cast<Instruction>(i4->getUser());
          
          if (find (pr->begin(), pr->end(), user->getParent()) ==
              pr->end()) {
            // User is not in the defining region.
            ++ContextValues;
            ContextSize += F.getParent()->getDataLayout().getTypeAllocSize(i3->getType());
            break;
          }
        }
      }
    }
  }

  // Then replicate the ParallelRegions.  
  ValueToValueMapTy *const reference_map = 
    new ValueToValueMapTy[workitem_count - 1];
  for (unsigned z = 0; z < WGLocalSizeZ; ++z) {
    for (unsigned y = 0; y < WGLocalSizeY; ++y) {
      for (unsigned x = 0; x < WGLocalSizeX ; ++x) {
              
        int index = 
          (WGLocalSizeY * WGLocalSizeX * z + WGLocalSizeX * y + x);
	  
        if (index == 0)
          continue;
	  
        for (SmallVector<ParallelRegion *, 8>::iterator
               i = original_parallel_regions->begin(), 
               e = original_parallel_regions->end();
             i != e; ++i) {
          ParallelRegion *original = (*i);
          ParallelRegion *replicated =
            original->replicate
            (reference_map[index - 1],
             (".wi_" + Twine(x) + "_" + Twine(y) + "_" + Twine(z)));
          if (AddWIMetadata)
            replicated->AddIDMetadata(M->getContext(), x, y, z);
          parallel_regions[index].push_back(replicated);
#ifdef DEBUG_PR_REPLICATION
          std::cerr << "### new replica:" << std::endl;
          replicated->dump();
#endif
        }
      }
    }
  }
  if (AddWIMetadata) {
    for (SmallVector<ParallelRegion *, 8>::iterator
          i = original_parallel_regions->begin(), 
           e = original_parallel_regions->end();
        i != e; ++i) {
      ParallelRegion *original = (*i);  
      original->AddIDMetadata(M->getContext(), 0, 0, 0);
    }
  }  
  
  for (unsigned z = 0; z < WGLocalSizeZ; ++z) {
    for (unsigned y = 0; y < WGLocalSizeY; ++y) {
      for (unsigned x = 0; x < WGLocalSizeX ; ++x) {
	  
        int index = 
          (WGLocalSizeY * WGLocalSizeX * z + WGLocalSizeX * y + x);
        
        for (unsigned i = 0, e = parallel_regions[index].size(); i != e; ++i) {
          ParallelRegion *region = parallel_regions[index][i];
          if (index != 0) {
            region->remap(reference_map[index - 1]);
            region->chainAfter(parallel_regions[index - 1][i]);
#ifdef DEBUG_PR_REPLICATION
            std::ostringstream s;
            s << F.getName().str() << ".before_purge_of_" << region->GetID()
              << ".dot";
            pocl::dumpCFG
                (F, s.str(), &parallel_regions[index]);
#endif
            region->purge();
          }
          region->insertPrologue(x, y, z);
        }
      }
    }
  }
    
  // Push the PHI nodes from the entry BBs of the chained
  // region copys to the entry BB of the first copy to retain
  // their semantics for branches from outside the parallel
  // region to the beginning of the region copy chain.
  for (int z = WGLocalSizeZ - 1; z >= 0; --z) {
    for (int y = WGLocalSizeY - 1; y >= 0; --y) {
      for (int x = WGLocalSizeX - 1; x >= 0; --x) {
          
        int index = 
          (WGLocalSizeY * WGLocalSizeX * z + WGLocalSizeX * y + x);

        if (index == 0)
          continue;

        for (unsigned i = 0, e = parallel_regions[index].size(); i != e; ++i) {
          ParallelRegion *firstCopy = parallel_regions[0][i];
          ParallelRegion *region = parallel_regions[index][i];
          BasicBlock *entry = region->entryBB();
#ifdef DEBUG_BB_MERGING
          std::cerr << "### first entry before hoisting the PHIs " << std::endl;
          firstCopy->entryBB()->dump();
#endif
          movePhiNodes(entry, firstCopy->entryBB());
#ifdef DEBUG_BB_MERGING
          std::cerr << "### first entry after hoisting the PHIs " << std::endl;
          firstCopy->entryBB()->dump();
#endif
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
  K->addLocalSizeInitCode(WGLocalSizeX, WGLocalSizeY, WGLocalSizeZ);

  delete [] reference_map;

  //pocl::dumpCFG(F, F.getName().str() + ".after_repl.dot", original_parallel_regions);

  for (size_t j = 0; j < parallel_regions.size(); ++j) {
    for (auto i = parallel_regions[j].begin(),
              e = parallel_regions[j].end();
              i != e; ++i) {
      ParallelRegion *p = *i;
      delete p;
    }
  }
  delete original_parallel_regions;
  original_parallel_regions = nullptr;

  return true;
}

