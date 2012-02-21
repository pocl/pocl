// LLVM function pass to replicate kernel body for several workitems.
// 
// Copyright (c) 2011 Universidad Rey Juan Carlos
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

#include "WorkitemReplication.h"
#include "Workgroup.h"
#include "Barrier.h"
#include "Kernel.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

//#define DEBUG_BB_MERGING
//#define DUMP_RESULT_CFG

#ifdef DUMP_RESULT_CFG
#include "llvm/Analysis/CFGPrinter.h"
#endif

using namespace llvm;
using namespace pocl;

static bool block_has_barrier(const BasicBlock *bb);

namespace {
  static
  RegisterPass<WorkitemReplication> X("workitem", "Workitem replication pass");
}

cl::list<int>
LocalSize("local-size",
	  cl::desc("Local size (x y z)"),
	  cl::multi_val(3));

char WorkitemReplication::ID = 0;

void
WorkitemReplication::getAnalysisUsage(AnalysisUsage &AU) const
{
  AU.addRequired<DominatorTree>();
  AU.addRequired<LoopInfo>();
}

bool
WorkitemReplication::runOnFunction(Function &F)
{
  if (!Workgroup::isKernelToProcess(F))
    return false;

  DT = &getAnalysis<DominatorTree>();
  LI = &getAnalysis<LoopInfo>();

  bool ok = ProcessFunction(F);
#ifdef DUMP_RESULT_CFG
  FunctionPass* cfgPrinter = createCFGPrinterPass();
  cfgPrinter->runOnFunction(F);
#endif
  return ok;
}

bool
WorkitemReplication::ProcessFunction(Function &F)
{
  Module *M = F.getParent();

  CheckLocalSize(&F);

  // Allocate space for workitem reference maps. Workitem 0 does
  // not need it.
  unsigned workitem_count = LocalSizeZ * LocalSizeY * LocalSizeX;

  BasicBlockVector original_bbs;
  for (Function::iterator i = F.begin(), e = F.end(); i != e; ++i) {
    if (!block_has_barrier(i))
      original_bbs.push_back(i);
  }

  Kernel *K = cast<Kernel> (&F);

  SmallVector<BarrierBlock *, 4> exit_blocks;
  K->getExitBlocks(exit_blocks);

  while(!exit_blocks.empty()) {
    // We need to keep track of traversed barriers to detect back edges.
    SmallPtrSet<BarrierBlock *, 8> found_barriers;
    
    // We start on an exit block and process the parallel regions upwards
    // (finding an execution trace).
    BarrierBlock *exit = exit_blocks.back();
    exit_blocks.pop_back();
    
    SmallVector<ParallelRegion *, 8> parallel_regions[workitem_count];
    ValueToValueMapTy reference_map[workitem_count - 1];

    while(ParallelRegion *PR = K->createParallelRegionBefore(exit)) {
      assert(!PR->empty() && "Empty parallel region in kernel (contiguous barriers)!");

      found_barriers.insert(exit);

      parallel_regions[0].push_back(PR);
      BasicBlock *entry = PR->front();
      int non_backedge_predecessors = 0;
      for (pred_iterator i = pred_begin(entry), e = pred_end(entry);
           i != e; ++i) {
        BarrierBlock *barrier = cast<BarrierBlock> (*i);
        if (!found_barriers.count(barrier)) {
          ++non_backedge_predecessors;
          exit = barrier;
        }
      }
      assert((non_backedge_predecessors == 1) &&
             "Could not determine parallel region entry!");
      assert ((exit != NULL) && "Parallel region without entry barrier!");
    }
    
    for (int z = 0; z < LocalSizeZ; ++z) {
      for (int y = 0; y < LocalSizeY; ++y) {
        for (int x = 0; x < LocalSizeX ; ++x) {
	  
          int index = 
            (LocalSizeY * LocalSizeX * z + LocalSizeX * y + x);
	  
          if (index == 0)
            continue;
	  
          for (SmallVector<ParallelRegion *, 8>::iterator
                 i = parallel_regions[0].begin(), e = parallel_regions[0].end();
               i != e; ++i) {
            ParallelRegion *original = (*i);
            ParallelRegion *replicated =
              original->replicate
              (reference_map[index - 1],
               (".wi_" + Twine(x) + "_" + Twine(y) + "_" + Twine(z)));
            parallel_regions[index].push_back(replicated);
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
            BasicBlock *entry = region->front();

#ifdef DEBUG_BB_MERGING
            std::cerr << "### before merge into predecessor " << std::endl;
            entry->dump();
#endif

            BasicBlock *pred = entry->getUniquePredecessor();
            assert (pred != NULL && "No unique predecessor.");
            movePhiNodes(entry, pred);
            MergeBlockIntoPredecessor(entry, this);

#ifdef DEBUG_BB_MERGING
            std::cerr << "### pred after merge " << std::endl;
            pred->dump();
#endif
          }
        }
      }
    }
  }

  // Add the suffixes to original (wi_0_0_0) basic blocks.
  for (BasicBlockVector::iterator i = original_bbs.begin(),
         e = original_bbs.end();
       i != e; ++i)
    (*i)->setName((*i)->getName() + ".wi_0_0_0");

  // Initialize local size (done at the end to avoid unnecessary
  // replication).
  IRBuilder<> builder(F.getEntryBlock().getFirstNonPHI());

  GlobalVariable *gv;

  int size_t_width = 32;
  if (M->getPointerSize() == llvm::Module::Pointer64)
    size_t_width = 64;

  gv = M->getGlobalVariable("_local_size_x");
  if (gv != NULL)
    builder.CreateStore
      (ConstantInt::get
       (IntegerType::get(M->getContext(), size_t_width),
        LocalSizeX), gv);
  gv = M->getGlobalVariable("_local_size_y");

  if (gv != NULL)
    builder.CreateStore
      (ConstantInt::get
       (IntegerType::get(M->getContext(), size_t_width),
        LocalSizeY), gv);
  gv = M->getGlobalVariable("_local_size_z");

  if (gv != NULL)
    builder.CreateStore
      (ConstantInt::get
       (IntegerType::get(M->getContext(), size_t_width),
        LocalSizeZ), gv);

  return true;
}

/**
 * Moves the phi nodes in the beginning of the src to the beginning of
 * the dst. 
 *
 * MergeBlockIntoPredecessor function from llvm discards the phi nodes
 * of the replicated BB because it has only one entry.
 */
void
WorkitemReplication::movePhiNodes(llvm::BasicBlock* src, llvm::BasicBlock* dst) 
{
  while (PHINode *PN = dyn_cast<PHINode>(src->begin())) 
    PN->moveBefore(dst->getFirstNonPHI());
}

void
WorkitemReplication::CheckLocalSize(Function *F)
{
  Module *M = F->getParent();
  
  LocalSizeX = LocalSize[0];
  LocalSizeY = LocalSize[1];
  LocalSizeZ = LocalSize[2];
  
  NamedMDNode *size_info = M->getNamedMetadata("opencl.kernel_wg_size_info");
  if (size_info) {
    for (unsigned i = 0, e = size_info->getNumOperands(); i != e; ++i) {
      MDNode *KernelSizeInfo = size_info->getOperand(i);
      if (KernelSizeInfo->getOperand(0) == F) {
        LocalSizeX = (cast<ConstantInt>(KernelSizeInfo->getOperand(1)))->getLimitedValue();
        LocalSizeY = (cast<ConstantInt>(KernelSizeInfo->getOperand(2)))->getLimitedValue();
        LocalSizeZ = (cast<ConstantInt>(KernelSizeInfo->getOperand(3)))->getLimitedValue();
      }
    }
  }
}

static bool
block_has_barrier(const BasicBlock *bb)
{
  for (BasicBlock::const_iterator i = bb->begin(), e = bb->end();
       i != e; ++i) {
    if (isa<Barrier>(i))
      return true;
  }
  return false;
}
