// Class for kernels, a special kind of function.
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

#include "Kernel.h"
#include "Barrier.h"
#include <iostream>

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
    TerminatorInst *t = i->getTerminator();
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
  std::cerr << "createParallelRegionBefore:" << std::endl;
  B->dump();
#endif
  
  while (!pending_blocks.empty()) {
    BasicBlock *current = pending_blocks.back();
    pending_blocks.pop_back();

#ifdef DEBUG_PR_CREATION
    std::cerr << "considering " << current->getName().str() << std::endl;
#endif
    
    // If this block is already in the region, continue
    // (avoid infinite recursion of loops).
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

#if 0
      // This should be legal in case the barriers preceed the same
      // entry block.
      else {        
        B->getParent()->viewCFG();
        assert((region_entry_barrier == current) &&
               "Barrier is dominated by more than one barrier! (forgot BTR?)");
      }
#endif
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
      // Ignore barrier-to-barrier edges
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
