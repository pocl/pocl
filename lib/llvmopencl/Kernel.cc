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

using namespace llvm;
using namespace pocl;

static void add_predecessors(SmallVectorImpl<BasicBlock *> &v,
                             BasicBlock *b);
static bool verify_no_barriers(const BasicBlock *B);

void
Kernel::getBarrierBlocks(SmallVectorImpl<BarrierBlock *> &B)
{
  Module *M = getParent();

  assert((M != NULL) && "Kernel function has to be in a module");

  SmallVector<Barrier *, 4> barriers;
  Barrier::GetBarriers(barriers, *getParent());
  for (SmallVector<Barrier *, 4>::iterator i = barriers.begin(),
         e = barriers.end();
       i != e; ++i) {
    BasicBlock *block = (*i)->getParent();
    if (block->getParent() == this)
      B.push_back(cast<BarrierBlock>(block));
  }
}

void
Kernel::getParallelRegions(SmallVectorImpl<ParallelRegion *> &PR)
{
  SmallVector<BarrierBlock *, 4> barriers;

  getBarrierBlocks(barriers);

  for (SmallVector<BarrierBlock *, 4>::iterator i = barriers.begin(),
         e = barriers.end();
       i != e; ++i) {
    SmallVector<BasicBlock *, 4> pending_blocks;
    SmallPtrSet<BasicBlock *, 8> blocks_in_region;
    BarrierBlock *region_entry_barrier = NULL;
 
    add_predecessors(pending_blocks, *i);

    while (!pending_blocks.empty()) {
      BasicBlock *current = pending_blocks.back();
      pending_blocks.pop_back();

      // If this block is already in the region, continue
      // (avoid infinite recursion of loops).
      if (blocks_in_region.count(current) != 0)
        continue;

      // If we reach another barrier this must be the
      // parallel region entry.
      if (isa<BarrierBlock>(current)) {
        if (region_entry_barrier == NULL)
          region_entry_barrier = cast<BarrierBlock>(current);
        else {
          assert((region_entry_barrier == current) &&
                 "Barrier is dominated by more than one barrier! (forgot BTR?)");
          assert(verify_no_barriers(current) &&
                 "Barrier found in a non-barrier block! (forgot barrier canonicalization?)");
        }

        continue;
      }

      // Non-barrier block, this must be on the region.
      blocks_in_region.insert(current);

      // Add predecessors to pending queue.
      add_predecessors(pending_blocks, current);
    }

    // We got all the blocks in a region, create it.
    if (!blocks_in_region.empty())
      PR.push_back(ParallelRegion::Create(blocks_in_region.begin(),
                                          blocks_in_region.end()));
  }
}

static void
add_predecessors(SmallVectorImpl<BasicBlock *> &v, BasicBlock *b)
{
  for (pred_iterator i = pred_begin(b), e = pred_end(b);
       i != e; ++i)
    v.push_back(*i);
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
