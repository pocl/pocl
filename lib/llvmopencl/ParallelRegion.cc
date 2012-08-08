// Class definition for parallel regions, a group of BasicBlocks that
// each kernel should run in parallel.
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

#include "ParallelRegion.h"
#include "Barrier.h"
#include "config.h"
#ifdef LLVM_3_2
#include "llvm/IRBuilder.h"
#else
#include "llvm/Support/IRBuilder.h"
#endif
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/ValueSymbolTable.h"

#include <set>
#include <sstream>
#include <map>

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

/**
 * Ensure all variables are named so they will be replicated and renamed
 * correctly.
 */
void
ParallelRegion::GenerateTempNames(llvm::BasicBlock *bb) 
{
  for (llvm::BasicBlock::iterator i = bb->begin(), e = bb->end(); i != e; ++i)
    {
      llvm::Instruction *instr = i;
      if (instr->hasName() || !instr->isUsedOutsideOfBlock(bb)) continue;
      int tempCounter = 0;
      std::string tempName = "";
      do {
          std::ostringstream name;
          name << ".pocl_temp." << tempCounter;
          ++tempCounter;
          tempName = name.str();
      } while (bb->getParent()->getValueSymbolTable().lookup(tempName) != NULL);
      instr->setName(tempName);
    }
}

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

  /* Because ParallelRegions are all replicated before they
     are attached to the function, it can happen that
     the same BB is replicated multiple times and it gets
     the same name (only the BB name will be autorenamed
     by LLVM). This causes the variable references to become
     broken. This hack ensures the BB suffices are unique
     before cloning so each path gets their own value
     names. Split points can be such paths.*/
  static std::map<std::string, int> cloneCounts;

  for (iterator i = begin(), e = end(); i != e; ++i) {
    BasicBlock *block = *i;
    GenerateTempNames(block);
    std::ostringstream suf;
    suf << suffix.str();
    std::string block_name = block->getName().str() + "." + suffix.str();
    if (cloneCounts[block_name] > 0)
      {
        suf << ".pocl_" << cloneCounts[block_name];
      }
    BasicBlock *new_block = CloneBasicBlock(block, map, suf.str());
    cloneCounts[block_name]++;
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
  
  new_region->exitIndex_ = exitIndex_;
  new_region->entryIndex_ = entryIndex_;
  /* Remap here to get local variables fixed before they
     are (possibly) overwritten by another clone of the 
     same BB. */
  new_region->remap(map); 
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
  /* If we are replicating a conditional barrier
     region, the last block can be an unreachable 
     block to mark the impossible path. Skip
     it and choose the correct branch instead. 

     TODO: why have the unreachable block there the
     first place? Could we just not add it and fix
     the branch? */
  BasicBlock *tail = region->exitBB();
  TerminatorInst *t = tail->getTerminator();
  if (isa<UnreachableInst>(t))
    {
      tail = region->at(region->size() - 2);
      t = tail->getTerminator();
    }
  if (t->getNumSuccessors() != 1)
    {
      std::cout << "!!! trying to chain region" << std::endl;
      this->dump();
      std::cout << "!!! after region" << std::endl;
      region->dump();
      assert (t->getNumSuccessors() == 1);
    }
  
  BasicBlock *successor = t->getSuccessor(0);
  Function::BasicBlockListType &bb_list = 
    successor->getParent()->getBasicBlockList();
  
  for (iterator i = begin(), e = end(); i != e; ++i)
    bb_list.insertAfter(tail, *i);
  
  t->setSuccessor(0, entryBB());

  t = exitBB()->getTerminator();
  assert (t->getNumSuccessors() == 1);
  t->setSuccessor(0, successor);
}

void
ParallelRegion::purge()
{
  SmallVector<BasicBlock *, 4> new_blocks;

  for (iterator i = begin(), e = end(); i != e; ++i) {

    // Exit block has a successor out of the region.
    if (*i == exitBB())
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
  BasicBlock *entry = entryBB();

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

void
ParallelRegion::dumpNames()
{
  for (iterator i = begin(), e = end(); i != e; ++i)
    std::cout << (*i)->getName().str() << " ";
  std::cout << std::endl;
}

ParallelRegion *
ParallelRegion::Create(const SmallPtrSet<BasicBlock *, 8>& bbs, BasicBlock *entry, BasicBlock *exit)
{
  ParallelRegion *new_region = new ParallelRegion();

  assert (entry != NULL);
  assert (exit != NULL);

  // This is done in two steps so order of the vector
  // is the same as original function order.
  Function *F = entry->getParent();
  for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
    BasicBlock *b = i;
    for (SmallPtrSetIterator<BasicBlock *> j = bbs.begin(); j != bbs.end(); ++j) {
      if (*j == b) {
        new_region->push_back(i);
        if (entry == *j)
            new_region->setEntryBBIndex(new_region->size() - 1);
        else if (exit == *j)
            new_region->setExitBBIndex(new_region->size() - 1);
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
        if ((*i) != entryBB()) {
          dumpNames();
          std::cerr << "suspicious block: " << (*i)->getName().str() << std::endl;
          std::cerr << "the entry is: " << entryBB()->getName().str() << std::endl;

#if 1
          (*i)->getParent()->viewCFG();
#endif
          assert(0 && "Incoming edges to non-entry block!");
          return false;
        } else if (!isa<BarrierBlock>(*ii)) {
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

    if (exitBB()->getTerminator()->getNumSuccessors() != 1) {
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

void
ParallelRegion::setID(
    llvm::LLVMContext& context, 
    std::size_t x, 
    std::size_t y, 
    std::size_t z,
    std::size_t regionID) {
  
    int counter = 1;
    for (iterator i = begin(), e = end(); i != e; ++i) {
      BasicBlock* bb= *i;      
      for (BasicBlock::iterator ii = bb->begin();
            ii != bb->end(); ii++) {
        Value *v[] = {
            MDString::get(context, "WI_id"),      
            ConstantInt::get(Type::getInt32Ty(context), regionID),
            ConstantInt::get(Type::getInt32Ty(context), x),
            ConstantInt::get(Type::getInt32Ty(context), y),      
            ConstantInt::get(Type::getInt32Ty(context), z),
            ConstantInt::get(Type::getInt32Ty(context), counter)};      
        MDNode* md = MDNode::get(context, v);  
        counter++;
        ii->setMetadata("wi",md);
      }
    }
}
