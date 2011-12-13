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

static bool
basic_block_successor_dfs(std::set<const BasicBlock *> &set,
                          const BasicBlock *entry,
                          const BasicBlock *exit);

ParallelRegion *
ParallelRegion::replicate(ValueToValueMapTy &map,
                          const Twine &suffix = "")
{
  ParallelRegion *new_region = new ParallelRegion();

  for (iterator i = begin(), e = end(); i != e; ++i)
    new_region->push_back(CloneBasicBlock((*i), map, suffix));

  // Does it at BB to the map also?

  return new_region;
}

void
ParallelRegion::purge()
{
  for (iterator i = begin(), e = end(); i != e; ++i) {

    // Exit block has a successor out of the region.
    if (*i == back())
      continue;

    TerminatorInst *t = (*i)->getTerminator();
    for (unsigned ii = 0, ee = t->getNumSuccessors(); ii != ee; ++ii) {
      BasicBlock *successor = t->getSuccessor(ii);
      if (count(begin(), end(), successor) == 0) {
        // This successor is not on the parallel region, purge.
        iterator next_block = i;
        ++next_block;
        BasicBlock *unreachable =
          BasicBlock::Create(successor->getContext(),
                             successor->getName() + ".unreachable",
                             successor->getParent(),
                             *next_block);
        new UnreachableInst(unreachable->getContext(),
                            unreachable);
        t->setSuccessor(ii, unreachable);
        insert(next_block, unreachable);
      }
    }
  }
}

void
ParallelRegion::remap(ValueToValueMapTy &map)
{
  for (iterator i = begin(), e = end(); i != e; ++i) {
    for (BasicBlock::iterator ii = (*i)->begin(), ee = (*i)->end();
         ii != ee; ++ii)
      RemapInstruction(ii, map,
                       RF_IgnoreMissingEntries | RF_NoModuleLevelChanges);
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
ParallelRegion::insertPrologue(unsigned x,
                               unsigned y,
                               unsigned z)
{
  BasicBlock *entry = front();

  IRBuilder<> builder(entry, entry->getFirstInsertionPt());

  Module *M = entry->getParent()->getParent();

  GlobalVariable *gvx = M->getGlobalVariable(LOCAL_ID_X);
  if (gvx != NULL)
    builder.CreateStore(ConstantInt::get(IntegerType::
                                         get(M->getContext(),
                                             32), x), gvx);

  GlobalVariable *gvy = M->getGlobalVariable(LOCAL_ID_Y);
  if (gvy != NULL)
    builder.CreateStore(ConstantInt::get(IntegerType::
                                         get(M->getContext(),
                                             32), y), gvy);

  GlobalVariable *gvz = M->getGlobalVariable(LOCAL_ID_Z);
  if (gvz != NULL)
    builder.CreateStore(ConstantInt::get(IntegerType::
                                         get(M->getContext(),
                                             32), z), gvz);
}

void
ParallelRegion::dump()
{
  for (iterator i = begin(), e = end(); i != e; ++i)
    (*i)->dump();
}

static bool
basic_block_successor_dfs(std::set<const BasicBlock *> &set,
                          const BasicBlock *entry,
                          const BasicBlock *exit)
{
  if (entry == exit) {
    set.insert(entry);
    return true;
  }

  bool found = false;
  
  for (succ_const_iterator i(entry->getTerminator()), e(entry->getTerminator(), true);
         i != e; ++i) {
    // Check if the successor is in the set already.
    if (set.count(*i) != 0)
      continue;

    found |= basic_block_successor_dfs(set, *i, exit);
  }
  
  if (found) {
    set.insert(entry);
    return true;
  }

  return false;
}

ParallelRegion *
ParallelRegion::Create(BasicBlock *entry,
                       BasicBlock *exit)
{
  ParallelRegion *new_region = new ParallelRegion();

  std::set<const BasicBlock *> basic_blocks_in_region;

  if (!basic_block_successor_dfs(basic_blocks_in_region,
                                 entry, exit)) {
    // No path from entry to exit, empty region.
    assert(basic_blocks_in_region.empty());
    return NULL;;
  }

  // This is done in two steps so order of the vector
  // is the same as original function order.
  Function *F = entry->getParent();
  for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
    if (basic_blocks_in_region.count(i) != 0)
      new_region->push_back(i);
  }

  assert((entry == new_region->front()) && "entry must be first element!");
  assert((exit == new_region->back()) && "exit must be last element!");
  // assert(new_region->Verify());

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
        ++entry_edges;
      }
    }
    
    if (entry_edges != 1) {
      assert(0 && "Parallel regions must be single entry!");
      return false;
    }

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
