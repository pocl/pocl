// Class for kernels, llvm::Functions that represent OpenCL C kernels.
//
// Copyright (c) 2011 Universidad Rey Juan Carlos and
//               2012-2019 Pekka Jääskeläinen
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

#include "pocl.h"
#include "pocl_cl.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"

#include "Kernel.h"
#include "Barrier.h"
#include "DebugHelpers.h"

POP_COMPILER_DIAGS

using namespace llvm;
using namespace pocl;

extern cl_device_id currentPoclDevice;

static void add_predecessors(SmallVectorImpl<BasicBlock *> &v,
                             BasicBlock *b);
static bool verify_no_barriers(const BasicBlock *B);

void
Kernel::getExitBlocks(SmallVectorImpl<llvm::BasicBlock *> &B)
{
  for (iterator i = begin(), e = end(); i != e; ++i) {
    auto t = i->getTerminator();
    if (t->getNumSuccessors() == 0) {
      // All exits must be barrier blocks.
      llvm::BasicBlock *BB = cast<BasicBlock>(i);
      if (!Barrier::hasBarrier(BB))
        Barrier::Create(BB->getTerminator());
      B.push_back(BB);
    }
  }
}

ParallelRegion *
Kernel::createParallelRegionBefore(llvm::BasicBlock *B)
{
  SmallVector<BasicBlock *, 4> pending_blocks;
  SmallPtrSet<BasicBlock *, 8> blocks_in_region;
  BasicBlock *region_entry_barrier = NULL;
  BasicBlock *entry = NULL;
  BasicBlock *exit = B->getSinglePredecessor();
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
    if (Barrier::hasOnlyBarrier(current)) {
      if (region_entry_barrier == NULL)
        region_entry_barrier = current;
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

/**
 * The main entry to the "parallel region formation", phase which search
 * for the regions between barriers that can be freely parallelized 
 * across work-items in the work-group.
 */
ParallelRegion::ParallelRegionVector *
Kernel::getParallelRegions(llvm::LoopInfo *LI) {
  ParallelRegion::ParallelRegionVector *parallel_regions =
    new ParallelRegion::ParallelRegionVector;

  SmallVector<BasicBlock *, 4> exit_blocks;
  getExitBlocks(exit_blocks);

  // We need to keep track of traversed barriers to detect back edges.
  SmallPtrSet<BasicBlock *, 8> found_barriers;

  // First find all the ParallelRegions in the Function.
  while (!exit_blocks.empty()) {
    
    // We start on an exit block and process the parallel regions upwards
    // (finding an execution trace).
    BasicBlock *exit = exit_blocks.back();
    exit_blocks.pop_back();

    // already handled
    if (found_barriers.count(exit) != 0)
      continue;

    while (ParallelRegion *PR = createParallelRegionBefore(exit)) {
      assert(PR != NULL && !PR->empty() && 
             "Empty parallel region in kernel (contiguous barriers)!");

      found_barriers.insert(exit);
      exit = NULL;
      parallel_regions->push_back(PR);
      BasicBlock *entry = PR->entryBB();
      int found_predecessors = 0;
      BasicBlock *loop_barrier = NULL;
      for (pred_iterator i = pred_begin(entry), e = pred_end(entry);
           i != e; ++i) {
        BasicBlock *barrier = (*i);
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

#ifdef DEBUG_PR_CREATION
  pocl::dumpCFG(*this, this->getName().str() + ".pregions.dot", parallel_regions);
#endif
  return parallel_regions;

}

void
Kernel::addLocalSizeInitCode(size_t LocalSizeX, size_t LocalSizeY, size_t LocalSizeZ) {

  IRBuilder<> Builder(getEntryBlock().getFirstNonPHI());

  GlobalVariable *GV;

  llvm::Module* M = getParent();

  llvm::Type *SizeT =
    IntegerType::get(M->getContext(), currentPoclDevice->address_bits);

  GV = M->getGlobalVariable("_local_size_x");
  if (GV != NULL)
    Builder.CreateStore(ConstantInt::get(SizeT, LocalSizeX), GV);

  GV = M->getGlobalVariable("_local_size_y");
  if (GV != NULL)
    Builder.CreateStore(ConstantInt::get(SizeT, LocalSizeY), GV);

  GV = M->getGlobalVariable("_local_size_z");
  if (GV != NULL)
    Builder.CreateStore(ConstantInt::get(SizeT, LocalSizeZ), GV);
}

