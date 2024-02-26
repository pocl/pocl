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

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/ADT/Statistic.h>
#include <llvm/Analysis/CFGPrinter.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/ValueSymbolTable.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include "Barrier.h"
#include "DebugHelpers.h"
#include "Kernel.h"
#include "LLVMUtils.h"
#include "VariableUniformityAnalysis.h"
#include "Workgroup.h"
#include "WorkitemHandler.h"
#include "WorkitemHandlerChooser.h"
#include "WorkitemReplication.h"
POP_COMPILER_DIAGS

#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#define DEBUG_TYPE "workitem"
//#define DEBUG_BB_MERGING
//#define DUMP_RESULT_CFG
//#define DEBUG_PR_REPLICATION

STATISTIC(ContextValues, "Number of SSA values which have to be context-saved");
STATISTIC(ContextSize, "Context size per workitem in bytes");

#define PASS_NAME "workitemrepl"
#define PASS_CLASS pocl::WorkitemReplication
#define PASS_DESC "Workitem replication pass"

namespace pocl {

using namespace llvm;

class WorkitemReplicationImpl : public WorkitemHandler {
public:
  WorkitemReplicationImpl(llvm::DominatorTree &DT, llvm::LoopInfo &LI,
                          llvm::PostDominatorTree &PDT)
      : DT(DT), LI(LI), PDT(PDT) {}
  bool runOnFunction(Function &F);

private:
  llvm::DominatorTree &DT;
  llvm::LoopInfo &LI;
  llvm::PostDominatorTree &PDT;

  typedef std::set<llvm::BasicBlock *> BasicBlockSet;
  typedef std::vector<llvm::BasicBlock *> BasicBlockVector;
  typedef std::map<llvm::Value *, llvm::Value *> ValueValueMap;

  virtual bool processFunction(llvm::Function &F);
};

bool WorkitemReplicationImpl::runOnFunction(Function &F) {
  bool Changed = processFunction(F);
#ifdef DUMP_RESULT_CFG
  FunctionPass* cfgPrinter = createCFGPrinterPass();
  cfgPrinter->runOnFunction(F);
#endif

  Changed |= fixUndominatedVariableUses(DT, F);
  return Changed;
}

bool WorkitemReplicationImpl::processFunction(Function &F) {
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

  ParallelRegion::ParallelRegionVector *OriginalParallelRegions =
      K->getParallelRegions(LI);

  std::vector<ParallelRegion::ParallelRegionVector> parallel_regions(
      workitem_count);

  parallel_regions[0] = *OriginalParallelRegions;

#if 0
  pocl::dumpCFG(F, F.getName().str() + ".before_repl.dot",
                nullptr, OriginalParallelRegions);

  /* Enable to get region identification printouts */
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
         i = OriginalParallelRegions->begin(),
           e = OriginalParallelRegions->end();
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
               i = OriginalParallelRegions->begin(),
               e = OriginalParallelRegions->end();
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
          i = OriginalParallelRegions->begin(),
           e = OriginalParallelRegions->end();
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

#if 0
  pocl::dumpCFG(F, F.getName().str() + ".after_repl.dot",
                OriginalParallelRegions);
#endif

  for (size_t j = 0; j < parallel_regions.size(); ++j) {
    for (auto i = parallel_regions[j].begin(),
              e = parallel_regions[j].end();
              i != e; ++i) {
      ParallelRegion *p = *i;
      delete p;
    }
  }
  delete OriginalParallelRegions;
  OriginalParallelRegions = nullptr;

  return true;
}

#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
char WorkitemReplication::ID = 0;

void WorkitemReplication::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<PostDominatorTreeWrapperPass>();
  AU.addRequired<LoopInfoWrapperPass>();

  AU.addRequired<WorkitemHandlerChooser>();
  AU.addPreserved<WorkitemHandlerChooser>();

  AU.addRequired<VariableUniformityAnalysis>();
  AU.addPreserved<VariableUniformityAnalysis>();
}

bool WorkitemReplication::runOnFunction(llvm::Function &F) {
  if (!isKernelToProcess(F))
    return false;

  auto WIH = getAnalysis<pocl::WorkitemHandlerChooser>().chosenHandler();
  if (WIH != WorkitemHandlerType::FULL_REPLICATION)
    return false;

  auto &DT = getAnalysis<llvm::DominatorTreeWrapperPass>().getDomTree();
  auto &PDT =
      getAnalysis<llvm::PostDominatorTreeWrapperPass>().getPostDomTree();
  auto &LI = getAnalysis<llvm::LoopInfoWrapperPass>().getLoopInfo();

  WorkitemReplicationImpl WIR(DT, LI, PDT);
  return WIR.runOnFunction(F);
}

#undef DEBUG_TYPE

REGISTER_OLD_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

#else

// enable new pass manager infrastructure
llvm::PreservedAnalyses
WorkitemReplication::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  if (!isKernelToProcess(F))
    return llvm::PreservedAnalyses::all();

  WorkitemHandlerType WIH = AM.getResult<WorkitemHandlerChooser>(F).WIH;
  if (WIH != WorkitemHandlerType::FULL_REPLICATION)
    return llvm::PreservedAnalyses::all();

  auto &DT = AM.getResult<llvm::DominatorTreeAnalysis>(F);
  auto &PDT = AM.getResult<llvm::PostDominatorTreeAnalysis>(F);
  auto &LI = AM.getResult<llvm::LoopAnalysis>(F);

  llvm::PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<VariableUniformityAnalysis>();
  PAChanged.preserve<WorkitemHandlerChooser>();

  WorkitemReplicationImpl WIR(DT, LI, PDT);
  return WIR.runOnFunction(F) ? PAChanged : PreservedAnalyses::all();
}

#undef DEBUG_TYPE

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

#endif

} // namespace pocl
