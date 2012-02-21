// Class definition for parallel regions, a group of BasicBlocks that
// each kernel should run in parallel.
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

#include "ParallelRegion.h"
#include "Barrier.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <set>

#define LOCAL_ID_X "_local_id_x"
#define LOCAL_ID_Y "_local_id_y"
#define LOCAL_ID_Z "_local_id_z"

using namespace std;
using namespace llvm;
using namespace pocl;

//#define DEBUG_REMAP
//#define DEBUG_REPLICATE
//#define DEBUG_PURGE

#include <iostream>

// BarrierBlock *
// ParallelRegion::getEntryBarrier()
// {
//   BasicBlock *entry = front();
//   BasicBlock *barrier = entry->getSinglePredecessor();

//   return cast<BarrierBlock> (barrier);
// }

ParallelRegion *
ParallelRegion::replicate(ValueToValueMapTy &map,
                          const Twine &suffix = "")
{
  ParallelRegion *new_region = new ParallelRegion();

  for (iterator i = begin(), e = end(); i != e; ++i) {
    BasicBlock *block = *i;
    BasicBlock *new_block = CloneBasicBlock(block, map, suffix);
    // Insert the block itself into the map.
    map[block] = new_block;
    new_region->push_back(new_block);

#ifdef DEBUG_REPLICATE
    std::cerr << "### clonee block:" << std::endl;
    block->dump();
    std::cerr << endl << "### cloned block: " << std::endl;
    new_block->dump();
#endif
  }
  
  return new_region;
}

void
ParallelRegion::remap(ValueToValueMapTy &map)
{
  for (iterator i = begin(), e = end(); i != e; ++i) {

#ifdef DEBUG_REMAP
    std::cerr << "### block before remap:" << std::endl;
    (*i)->dump();
#endif

    for (BasicBlock::iterator ii = (*i)->begin(), ee = (*i)->end();
         ii != ee; ++ii)
      RemapInstruction(ii, map,
                       RF_IgnoreMissingEntries | RF_NoModuleLevelChanges);

#ifdef DEBUG_REMAP
    std::cerr << endl << "### block after remap: " << std::endl;
    (*i)->dump();
#endif
  }
}

void
ParallelRegion::chainAfter(ParallelRegion *region)
{
  TerminatorInst *t = region->back()->getTerminator();
  assert (t->getNumSuccessors() == 1);
  
  BasicBlock *successor = t->getSuccessor(0);
  Function::BasicBlockListType &bb_list = 
    successor->getParent()->getBasicBlockList();
  
  for (iterator i = begin(), e = end(); i != e; ++i)
    bb_list.insertAfter(region->back(), *i);
  
  t->setSuccessor(0, front());

  t = back()->getTerminator();
  assert (t->getNumSuccessors() == 1);
  t->setSuccessor(0, successor);
}

void
ParallelRegion::purge()
{
  SmallVector<BasicBlock *, 4> new_blocks;

  for (iterator i = begin(), e = end(); i != e; ++i) {

    // Exit block has a successor out of the region.
    if (*i == back())
      continue;

#ifdef DEBUG_PURGE
    std::cerr << "### block before purge:" << std::endl;
    (*i)->dump();
#endif
    TerminatorInst *t = (*i)->getTerminator();
    for (unsigned ii = 0, ee = t->getNumSuccessors(); ii != ee; ++ii) {
      BasicBlock *successor = t->getSuccessor(ii);
      if (count(begin(), end(), successor) == 0) {
        // This successor is not on the parallel region, purge.
        iterator next_block = i;
        ++next_block;
        BasicBlock *unreachable =
          BasicBlock::Create((*i)->getContext(),
                             (*i)->getName() + ".unreachable",
                             (*i)->getParent(),
                             *next_block);
        new UnreachableInst(unreachable->getContext(),
                            unreachable);
        t->setSuccessor(ii, unreachable);
        new_blocks.push_back(unreachable);
      }
    }
#ifdef DEBUG_PURGE
    std::cerr << std::endl << "### block after purge:" << std::endl;
    (*i)->dump();
#endif
  }

  // Add the new "unreachable" blocks to the
  // region. We cannot do in the loop as it
  // corrupts iterators.
  insert(end(), new_blocks.begin(), new_blocks.end());
}

void
ParallelRegion::insertPrologue(unsigned x,
                               unsigned y,
                               unsigned z)
{
  BasicBlock *entry = front();

  IRBuilder<> builder(entry, entry->getFirstInsertionPt());

  Module *M = entry->getParent()->getParent();

  int size_t_width = 32;
  if (M->getPointerSize() == llvm::Module::Pointer64)
    size_t_width = 64;

  GlobalVariable *gvx = M->getGlobalVariable(LOCAL_ID_X);
  if (gvx != NULL)
      builder.CreateStore(ConstantInt::get(IntegerType::
                                           get(M->getContext(), size_t_width), 
                                           x), gvx);

  GlobalVariable *gvy = M->getGlobalVariable(LOCAL_ID_Y);
  if (gvy != NULL)
    builder.CreateStore(ConstantInt::get(IntegerType::
                                         get(M->getContext(), size_t_width),
                                         y), gvy);

  GlobalVariable *gvz = M->getGlobalVariable(LOCAL_ID_Z);
  if (gvz != NULL)
    builder.CreateStore(ConstantInt::get(IntegerType::
                                         get(M->getContext(), size_t_width),
                                         z), gvz);
}

void
ParallelRegion::dump()
{
  for (iterator i = begin(), e = end(); i != e; ++i)
    (*i)->dump();
}

ParallelRegion *
ParallelRegion::Create(SmallPtrSetIterator<BasicBlock *> entry,
                       SmallPtrSetIterator<BasicBlock *> exit)
{
  ParallelRegion *new_region = new ParallelRegion();

  // This is done in two steps so order of the vector
  // is the same as original function order.
  Function *F = (*entry)->getParent();
  for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
    BasicBlock *b = i;
    for (SmallPtrSetIterator<BasicBlock *> j = entry; j != exit; ++j) {
      if (*j == b) {
        new_region->push_back(i);
        break;
      }
    }
  }

  assert(new_region->Verify());

  return new_region;
}

bool
ParallelRegion::Verify()
{
  // Parallel region conditions:
  // 1) Single entry, in entry block.
  // 2) Single outgoing edge from exit block
  //    (other outgoing edges allowed, will be purged in replicas).
  // 3) No barriers inside the region.
  
  int entry_edges = 0;

  for (iterator i = begin(), e = end(); i != e; ++i) {
    for (pred_iterator ii(*i), ee(*i, true); ii != ee; ++ii) {
      if (count(begin(), end(), *ii) == 0) {
        if ((*i) != front()) {
          assert(0 && "Incoming edges to non-entry block!");
          return false;
        }
        if (!isa<BarrierBlock>(*ii)) {
          assert (0 && "Entry has edges from non-barrier blocks!");
          return false;
        }
        ++entry_edges;
      }
    }
    
    // if (entry_edges != 1) {
    //   assert(0 && "Parallel regions must be single entry!");
    //   return false;
    // }

    if (back()->getTerminator()->getNumSuccessors() != 1) {
      assert(0 && "Multiple outgoing edges from exit block!");
      return false;
    }

    for (BasicBlock::iterator ii = (*i)->begin(), ee = (*i)->end();
           ii != ee; ++ii) {
      if (isa<Barrier> (ii)) {
        assert(0 && "Barrier found inside parallel region!");
        return false;
      }
    }
  }

  return true;
}
