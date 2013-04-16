// Class for kernels, llvm::Functions that represent OpenCL C kernels.
// 
// Copyright (c) 2011 Universidad Rey Juan Carlos and
//               2012 Pekka Jääskeläinen / TUT
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

#include "Kernel.h"
#include "Barrier.h"
#include <iostream>

#include "config.h"
#ifdef LLVM_3_1
#include "llvm/Support/IRBuilder.h"
#elif defined LLVM_3_2
#include "llvm/IRBuilder.h"
#else
#include "llvm/IR/IRBuilder.h"
#endif

//#define DEBUG_PR_CREATION

using namespace llvm;
using namespace pocl;

static void add_predecessors(SmallVectorImpl<BasicBlock *> &v,
                             BasicBlock *b);
static bool verify_no_barriers(const BasicBlock *B);

void
Kernel::getExitBlocks(SmallVectorImpl<BarrierBlock *> &B) 
{
  for (iterator i = begin(), e = end(); i != e; ++i) {
    const TerminatorInst *t = i->getTerminator();
    if (t->getNumSuccessors() == 0) {
      // All exits must be barrier blocks.
      B.push_back(cast<BarrierBlock>(i));
    }
  }
}

ParallelRegion *
Kernel::createParallelRegionBefore(BarrierBlock *B) 
{
  SmallVector<BasicBlock *, 4> pending_blocks;
  SmallPtrSet<BasicBlock *, 8> blocks_in_region;
  BarrierBlock *region_entry_barrier = NULL;
  llvm::BasicBlock *entry = NULL;
  llvm::BasicBlock *exit = B->getSinglePredecessor();
  add_predecessors(pending_blocks, B);

#ifdef DEBUG_PR_CREATION
  std::cerr << "createParallelRegionBefore " << B->getName().str() << std::endl;
#endif
  
  while (!pending_blocks.empty()) {
    BasicBlock *current = pending_blocks.back();
    pending_blocks.pop_back();

#ifdef DEBUG_PR_CREATION
    std::cerr << "considering " << current->getName().str() << std::endl;
#endif
    
    // avoid infinite recursion of loops
    if (blocks_in_region.count(current) != 0)
      {
#ifdef DEBUG_PR_CREATION
        std::cerr << "already in the region!" << std::endl;
#endif
        continue;
      }
    
    // If we reach another barrier this must be the
    // parallel region entry.
    if (isa<BarrierBlock>(current)) {
      if (region_entry_barrier == NULL)
        region_entry_barrier = cast<BarrierBlock>(current);
#ifdef DEBUG_PR_CREATION
      std::cerr << "### it's a barrier!" << std::endl;        
#endif     
      continue;
    }
    

    if (!verify_no_barriers(current))
      {
        assert(verify_no_barriers(current) &&
               "Barrier found in a non-barrier block! (forgot barrier canonicalization?)");
      }

#ifdef DEBUG_PR_CREATION
    std::cerr << "added it to the region" << std::endl;
#endif        
    // Non-barrier block, this must be on the region.
    blocks_in_region.insert(current);
    
    // Add predecessors to pending queue.
    add_predecessors(pending_blocks, current);
  }

  if (blocks_in_region.empty())
    return NULL;

  // Find the entry node.
  assert (region_entry_barrier != NULL);
  for (unsigned suc = 0, num = region_entry_barrier->getTerminator()->getNumSuccessors(); 
       suc < num; ++suc) 
    {
      llvm::BasicBlock *entryCandidate = 
        region_entry_barrier->getTerminator()->getSuccessor(suc);
      if (blocks_in_region.count(entryCandidate) == 0)
        continue;
      entry = entryCandidate;
      break;
    }
  assert (blocks_in_region.count(entry) != 0);

  // We got all the blocks in a region, create it.
  return ParallelRegion::Create(blocks_in_region, entry, exit);
}

static void
add_predecessors(SmallVectorImpl<BasicBlock *> &v, BasicBlock *b)
{
  for (pred_iterator i = pred_begin(b), e = pred_end(b);
       i != e; ++i) {
    if ((isa<BarrierBlock> (*i)) && isa<BarrierBlock> (b)) {
      // Ignore barrier-to-barrier edges * Why? --Pekka
      add_predecessors(v, *i);
      continue;
    }
    v.push_back(*i);
  }
}

static bool
verify_no_barriers(const BasicBlock *B)
{
  for (BasicBlock::const_iterator i = B->begin(), e = B->end(); i != e; ++i) {
    if (isa<Barrier>(i))
      return false;
  }

  return true;
}

ParallelRegion::ParallelRegionVector *
Kernel::getParallelRegions(llvm::LoopInfo *LI) {
  ParallelRegion::ParallelRegionVector *parallel_regions =
    new ParallelRegion::ParallelRegionVector;

  SmallVector<BarrierBlock *, 4> exit_blocks;
  getExitBlocks(exit_blocks);

  // We need to keep track of traversed barriers to detect back edges.
  SmallPtrSet<BarrierBlock *, 8> found_barriers;

  // First find all the ParallelRegions in the Function.
  while (!exit_blocks.empty()) {
    
    // We start on an exit block and process the parallel regions upwards
    // (finding an execution trace).
    BarrierBlock *exit = exit_blocks.back();
    exit_blocks.pop_back();

    while (ParallelRegion *PR = createParallelRegionBefore(exit)) {
      assert(PR != NULL && !PR->empty() && 
             "Empty parallel region in kernel (contiguous barriers)!");

      found_barriers.insert(exit);
      exit = NULL;
      parallel_regions->push_back(PR);
      BasicBlock *entry = PR->entryBB();
      int found_predecessors = 0;
      BarrierBlock *loop_barrier = NULL;
      for (pred_iterator i = pred_begin(entry), e = pred_end(entry);
           i != e; ++i) {
        BarrierBlock *barrier = cast<BarrierBlock> (*i);
        if (!found_barriers.count(barrier)) {
          /* If this is a loop header block we might have edges from two 
             unprocessed barriers. The one inside the loop (coming from a 
             computation block after a branch block) should be processed 
             first. */
          std::string bbName = "";
          const bool IS_IN_THE_SAME_LOOP = 
              LI->getLoopFor(barrier) != NULL &&
              LI->getLoopFor(entry) != NULL &&
              LI->getLoopFor(entry) == LI->getLoopFor(barrier);

          if (IS_IN_THE_SAME_LOOP)
            {
#ifdef DEBUG_PR_CREATION
              std::cout << "### found a barrier inside the loop:" << std::endl;
              std::cout << barrier->getName().str() << std::endl;
#endif
              if (loop_barrier != NULL) {
                // there can be multiple latches and each have their barrier,
                // save the previously found inner loop barrier
                exit_blocks.push_back(loop_barrier);
              }
              loop_barrier = barrier;
            }
          else
            {
#ifdef DEBUG_PR_CREATION
              std::cout << "### found a barrier:" << std::endl;
              std::cout << barrier->getName().str() << std::endl;
#endif
              exit = barrier;
            }
          ++found_predecessors;
        }
      }

      if (loop_barrier != NULL)
        {
          /* The secondary barrier to process in case it was a loop
             header. Push it for later processing. */
          if (exit != NULL) 
            exit_blocks.push_back(exit);
          /* always process the inner loop regions first */
          if (!found_barriers.count(loop_barrier))
            exit = loop_barrier; 
        }

#ifdef DEBUG_PR_CREATION
      std::cout << "### created a ParallelRegion:" << std::endl;
      PR->dumpNames();
      std::cout << std::endl;
#endif

      if (found_predecessors == 0)
        {
          /* This path has been traversed and we encountered no more
             unprocessed regions. It means we have either traversed all
             paths from the exit or have transformed a loop and thus 
             encountered only a barrier that was seen (and thus
             processed) before. */
          break;
        }
      assert ((exit != NULL) && "Parallel region without entry barrier!");
    }
  }
  return parallel_regions;

}

void
Kernel::addLocalSizeInitCode(size_t LocalSizeX, size_t LocalSizeY, size_t LocalSizeZ) {
  
  IRBuilder<> builder(getEntryBlock().getFirstNonPHI());

  GlobalVariable *gv;

  llvm::Module* M = getParent();

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

}

