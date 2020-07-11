// Class definition for parallel regions, a group of BasicBlocks that
// each kernel should run in parallel.
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

#include <set>
#include <sstream>
#include <map>
#include <algorithm>

#include "pocl.h"
#include "pocl_cl.h"

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "ParallelRegion.h"
#include "Barrier.h"
#include "Kernel.h"
#include "DebugHelpers.h"

using namespace std;
using namespace llvm;
using namespace pocl;

//#define DEBUG_REMAP
//#define DEBUG_REPLICATE
//#define DEBUG_PURGE

#include <iostream>

int ParallelRegion::idGen = 0;

extern cl_device_id currentPoclDevice;

ParallelRegion::ParallelRegion(int forcedRegionId) : 
  std::vector<llvm::BasicBlock *>(), 
  LocalIDXLoadInstr(NULL), LocalIDYLoadInstr(NULL), LocalIDZLoadInstr(NULL),
  exitIndex_(0), entryIndex_(0), pRegionId(forcedRegionId)
{
  if (forcedRegionId == -1)
    pRegionId = idGen++;
}

/**
 * Ensure all variables are named so they will be replicated and renamed
 * correctly.
 */
void
ParallelRegion::GenerateTempNames(llvm::BasicBlock *bb) 
{
  for (llvm::BasicBlock::iterator i = bb->begin(), e = bb->end(); i != e; ++i)
    {
      llvm::Instruction *instr = &*i;
      if (instr->hasName() || !instr->isUsedOutsideOfBlock(bb)) continue;
      int tempCounter = 0;
      std::string tempName = "";
      do {
          std::ostringstream name;
          name << ".pocl_temp." << tempCounter;
          ++tempCounter;
          tempName = name.str();
      } while (bb->getParent()->getValueSymbolTable()->lookup(tempName) != NULL);
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
  ParallelRegion *new_region = new ParallelRegion(pRegionId);
  
  /* Because ParallelRegions are all replicated before they
     are attached to the function, it can happen that
     the same BB is replicated multiple times and it gets
     the same name (only the BB name will be autorenamed
     by LLVM). This causes the variable references to become
     broken. This hack ensures the BB suffixes are unique
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

#ifdef DEBUG_REPLICATE
  Verify();
#endif
  LocalizeIDLoads();

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
      RemapInstruction(&*ii, map,
                       RF_IgnoreMissingLocals | RF_NoModuleLevelChanges);

#ifdef DEBUG_REMAP
    std::cerr << endl << "### block after remap: " << std::endl;
    (*i)->dump();
#endif
  }
}

void
ParallelRegion::chainAfter(ParallelRegion *region)
{
  /* If we are replicating a conditional barrier region, the last block can be
     an unreachable block to mark the impossible path. Skip it and choose the
     correct branch instead.

     TODO: why have the unreachable block there the first place? Could we just
     not add it and fix the branch? */
  BasicBlock *tail = region->exitBB();
  auto t = tail->getTerminator();
  if (isa<UnreachableInst>(t))
    {
      tail = region->at(region->size() - 2);
      t = tail->getTerminator();
    }
#ifdef LLVM_BUILD_MODE_DEBUG
    if (t->getNumSuccessors() != 1) {
      std::cout << "!!! trying to chain region" << std::endl;
      this->dumpNames();
      std::cout << "!!! after region" << std::endl;
      region->dumpNames();
      t->getParent()->dump();

      assert (t->getNumSuccessors() == 1);
    }
#endif

  BasicBlock *successor = t->getSuccessor(0);
  Function::BasicBlockListType &bb_list = 
    successor->getParent()->getBasicBlockList();
  
  for (iterator i = begin(), e = end(); i != e; ++i)

    bb_list.insertAfter(tail->getIterator(), *i);
  t->setSuccessor(0, entryBB());

  t = exitBB()->getTerminator();
  assert (t->getNumSuccessors() == 1);
  t->setSuccessor(0, successor);
}

/**
 * Removes known dead side exits from parallel regions.
 *
 * These occur with conditional barriers. The head of the path
 * leading to the conditional barrier is shared by two PRs. The
 * first work-item defines which path is taken (by definition the
 * barrier is taken by all or none of the work-items). The blocks 
 * in the branches are in different regions which can contain branches 
 * to blocks that are in known non-taken path. This method replaces 
 * the targets of such branches with undefined BBs so they will be cleaned 
 * up by the optimizer.
 */
void
ParallelRegion::purge()
{
  SmallVector<BasicBlock *, 4> new_blocks;

  // Go through all the BBs in the region and check their branch
  // targets, looking for destinations that are outside the region.
  // Only the last block in the PR can now contain such branches.
  for (iterator i = begin(), e = end(); i != e; ++i) {

    // Exit block has a successor out of the region.
    if (*i == exitBB())
      continue;

#ifdef DEBUG_PURGE
    std::cerr << "### block before purge:" << std::endl;
    (*i)->dump();
#endif
    auto t = (*i)->getTerminator();
    for (unsigned ii = 0, ee = t->getNumSuccessors(); ii != ee; ++ii) {
      BasicBlock *successor = t->getSuccessor(ii);
      if (count(begin(), end(), successor) == 0) {
        // This successor is not on the parallel region, purge.
#ifdef DEBUG_PURGE
          std::cerr 
              << "purging a branch to a block " 
              << successor->getName().str() << " outside the region" 
              << std::endl;
#endif

        BasicBlock *unreachable =
          BasicBlock::Create((*i)->getContext(),
                             (*i)->getName() + ".unreachable",
                             (*i)->getParent(), back());
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
ParallelRegion::insertLocalIdInit(llvm::BasicBlock* Entry,
                                  unsigned X, unsigned Y, unsigned Z) {

  IRBuilder<> Builder(Entry, Entry->getFirstInsertionPt());

  Module *M = Entry->getParent()->getParent();

  llvm::Type *SizeT =
    IntegerType::get(M->getContext(), currentPoclDevice->address_bits);

  GlobalVariable *GVX = M->getGlobalVariable(POCL_LOCAL_ID_X_GLOBAL);
  if (GVX != NULL)
      Builder.CreateStore(ConstantInt::get(SizeT, X), GVX);

  GlobalVariable *GVY = M->getGlobalVariable(POCL_LOCAL_ID_Y_GLOBAL);
  if (GVY != NULL)
      Builder.CreateStore(ConstantInt::get(SizeT, Y), GVY);

  GlobalVariable *GVZ = M->getGlobalVariable(POCL_LOCAL_ID_Z_GLOBAL);
  if (GVZ != NULL)
      Builder.CreateStore(ConstantInt::get(SizeT, Z), GVZ);
}

void
ParallelRegion::insertPrologue(unsigned x,
                               unsigned y,
                               unsigned z)
{
  BasicBlock *entry = entryBB();
  ParallelRegion::insertLocalIdInit(entry, x, y, z);
}

void
ParallelRegion::dump()
{
#ifdef LLVM_BUILD_MODE_DEBUG
  for (iterator i = begin(), e = end(); i != e; ++i)
    (*i)->dump();
#endif
}

void
ParallelRegion::dumpNames()
{
  for (iterator i = begin(), e = end(); i != e; ++i)
    {
      std::cout << (*i)->getName().str();
      if (entryBB() == (*i)) 
        std::cout << "(EN)";
      if (exitBB() == (*i))
        std::cout << "(EX)";
      std::cout << " ";
    }
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
    BasicBlock *b = &*i;
    for (SmallPtrSetIterator<BasicBlock *> j = bbs.begin(); j != bbs.end(); ++j) {
      if (*j == b) {
        new_region->push_back(&*i);
        if (entry == *j)
            new_region->setEntryBBIndex(new_region->size() - 1);
        else if (exit == *j)
            new_region->setExitBBIndex(new_region->size() - 1);
        break;
      }
    }
  }

  new_region->LocalizeIDLoads();

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

          ParallelRegion::ParallelRegionVector prvec;
          prvec.push_back(this);
          std::set<llvm::BasicBlock*> highlights;
          highlights.insert(entryBB());
          highlights.insert(*i);
          pocl::dumpCFG(
            *(*i)->getParent(), (*i)->getParent()->getName().str() + ".dot",
            &prvec, &highlights);
          assert(0 && "Incoming edges to non-entry block!");
          return false;
        } else if (!Barrier::hasBarrier(*ii)) {
          (*i)->getParent()->viewCFG();
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
      ParallelRegion::ParallelRegionVector regions;
      regions.push_back(this);

#ifdef LLVM_BUILD_MODE_DEBUG
      std::set<llvm::BasicBlock*> highlights;
      highlights.insert((*i));
      highlights.insert(exitBB());
      exitBB()->dump();
      dumpNames();
      dumpCFG(*(*i)->getParent(), "broken.dot", &regions, &highlights);
#endif

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

#ifdef LLVM_OLDER_THAN_8_0
#define PARALLEL_MD_NAME "llvm.mem.parallel_loop_access"
#else
#define PARALLEL_MD_NAME "llvm.access.group"
#endif

/**
 * Adds metadata to all the memory instructions to denote
 * they originate from a parallel loop.
 *
 * Due to nested parallel loops, there can be multiple loop
 * references.
 *
 * Format (LLVM 8+):
 *
 *     !llvm.access.group !0
 *
 *     !0 distinct !{}
 *
 * In a 2-nested loop:
 *
 *     !llvm.access.group !0
 *
 *     !0 { !1, !2 }
 *     !1 distinct !{}
 *     !2 distinct !{}
 *
 * Parallel loop metadata on memory reads also implies that
 * if-conversion (i.e., speculative execution within a loop iteration)
 * is safe. Given an instruction reading from memory,
 * IsLoadUnconditionallySafe should return whether it is safe under
 * (unconditional, unpredicated) speculative execution.
 * See https://bugs.llvm.org/show_bug.cgi?id=46666
 */
void
ParallelRegion::AddParallelLoopMetadata(
    llvm::MDNode *Identifier,
    std::function<bool(llvm::Instruction *)> IsLoadUnconditionallySafe) {
  for (iterator i = begin(), e = end(); i != e; ++i) {
    BasicBlock* bb = *i;      
    for (BasicBlock::iterator ii = bb->begin(), ee = bb->end();
         ii != ee; ii++) {
      if (!ii->mayReadOrWriteMemory()) {
        continue;
      }

      if (ii->mayReadFromMemory() && !IsLoadUnconditionallySafe(&*ii)) {
        continue;
      }

      MDNode *NewMD = MDNode::get(bb->getContext(), Identifier);
      MDNode *OldMD = ii->getMetadata(PARALLEL_MD_NAME);
      if (OldMD != nullptr) {
        NewMD = llvm::MDNode::concatenate(OldMD, NewMD);
      }
      ii->setMetadata(PARALLEL_MD_NAME, NewMD);
    }
  }
}

void
ParallelRegion::AddIDMetadata(
    llvm::LLVMContext& context, 
    std::size_t x, 
    std::size_t y, 
    std::size_t z) {
    int counter = 1;
    Metadata *v1[] = {
        MDString::get(context, "WI_region"),      
        llvm::ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt32Ty(context), pRegionId))
    };
    MDNode* mdRegion = MDNode::get(context, v1);  
    Metadata *v2[] = {
        MDString::get(context, "WI_xyz"),      
        llvm::ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt32Ty(context), x)),
        llvm::ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt32Ty(context), y)),      
        llvm::ConstantAsMetadata::get(
          ConstantInt::get(Type::getInt32Ty(context), z))};
    MDNode* mdXYZ = MDNode::get(context, v2);  
    Metadata *v[] = {
        MDString::get(context, "WI_data"),      
        mdRegion,
        mdXYZ};
    MDNode* md = MDNode::get(context, v);              
    
    for (iterator i = begin(), e = end(); i != e; ++i) {
      BasicBlock* bb = *i;
      for (BasicBlock::iterator ii = bb->begin();
            ii != bb->end(); ii++) {
        Metadata *v3[] = {
            MDString::get(context, "WI_counter"),      
            llvm::ConstantAsMetadata::get(
              ConstantInt::get(Type::getInt32Ty(context), counter))};
        MDNode* mdCounter = MDNode::get(context, v3);  
        counter++;
        ii->setMetadata("wi", md);
        ii->setMetadata("wi_counter", mdCounter);
      }
    }
}


/**
 * Inserts a new basic block to the region, before an old basic block in
 * the region.
 *
 * Assumes the inserted block to be before the other block in control
 * flow, that is, there should be direct CFG edge from the block to the
 * other.
 */
void
ParallelRegion::AddBlockBefore(llvm::BasicBlock *block, llvm::BasicBlock *before)
{
    llvm::BasicBlock *oldExit = exitBB();
    ParallelRegion::iterator beforePos = find(begin(), end(), before);
    ParallelRegion::iterator oldExitPos = find(begin(), end(), oldExit);
    assert (beforePos != end());

    /* The old exit node might is now pushed further, at most one position. 
       Whether this is the case, depends if the node was inserted before or
       after that node in the vector. That is, if indexof(before) < indexof(oldExit). */
    if (beforePos < oldExitPos) ++exitIndex_;

    insert(beforePos, block);
    /* The entryIndex_ should be still correct. In case the 'before' block
       was an old entry node, the new one replaces it as an entry node at
       the same index and the old one gets pushed forward. */      
}


void
ParallelRegion::AddBlockAfter(llvm::BasicBlock *block, llvm::BasicBlock *after)
{
    llvm::BasicBlock *oldExit = exitBB();
    ParallelRegion::iterator afterPos = find(begin(), end(), after);
    ParallelRegion::iterator oldExitPos = find(begin(), end(), oldExit);
    assert (afterPos != end());

    /* The old exit node might be pushed further, at most one position. 
       Whether this is the case, depends if the node was inserted before or
       after that node in the vector. That is, if indexof(before) < indexof(oldExit). */
    if (afterPos < oldExitPos) ++exitIndex_;
    afterPos++;
    insert(afterPos, block);
}

bool 
ParallelRegion::HasBlock(llvm::BasicBlock *bb)
{
    return find(begin(), end(), bb) != end();
}

/**
 * Find the instruction that loads the Z dimension of the work item
 * in the beginning of the parallel region, if not found, creates it.
 */
llvm::Instruction*
ParallelRegion::LocalIDZLoad()
{
  if (LocalIDZLoadInstr != NULL) return LocalIDZLoadInstr;
  IRBuilder<> builder(&*(entryBB()->getFirstInsertionPt()));
  return LocalIDZLoadInstr =
    builder.CreateLoad
    (entryBB()->getParent()->getParent()->getGlobalVariable(POCL_LOCAL_ID_Z_GLOBAL));
}

/**
 * Find the instruction that loads the Y dimension of the work item
 * in the beginning of the parallel region, if not found, creates it.
 */
llvm::Instruction*
ParallelRegion::LocalIDYLoad()
{
  if (LocalIDYLoadInstr != NULL) return LocalIDYLoadInstr;
  IRBuilder<> builder(&*(entryBB()->getFirstInsertionPt()));
  return LocalIDYLoadInstr = 
    builder.CreateLoad
    (entryBB()->getParent()->getParent()->getGlobalVariable(POCL_LOCAL_ID_Y_GLOBAL));
}

/**
 * Find the instruction that loads the X dimension of the work item
 * in the beginning of the parallel region, if not found, creates it.
 */
llvm::Instruction*
ParallelRegion::LocalIDXLoad()
{
  if (LocalIDXLoadInstr != NULL) return LocalIDXLoadInstr;
  IRBuilder<> builder(&*(entryBB()->getFirstInsertionPt()));
  return LocalIDXLoadInstr = 
    builder.CreateLoad
    (entryBB()->getParent()->getParent()->getGlobalVariable(POCL_LOCAL_ID_X_GLOBAL));
}

void
ParallelRegion::InjectPrintF
(llvm::Instruction *before, std::string formatStr,
 std::vector<Value*>& params)
{
  IRBuilder<> builder(before);
  llvm::Module *M = before->getParent()->getParent()->getParent();

  llvm::Value *stringArg = 
    builder.CreateGlobalString(formatStr);
    
  /* generated with help from http://llvm.org/demo/index.cgi */
  Function* printfFunc = M->getFunction("printf");
  if (printfFunc == NULL) {
    PointerType* PointerTy_4 = PointerType::get(IntegerType::get(M->getContext(), 8), 0);
 
    std::vector<Type*> FuncTy_6_args;
    FuncTy_6_args.push_back(PointerTy_4);
    
    FunctionType* FuncTy_6 = 
      FunctionType::get
      (/*Result=*/IntegerType::get(M->getContext(), 32),
       /*Params=*/FuncTy_6_args,
       /*isVarArg=*/true);

    printfFunc = 
      Function::Create
      (/*Type=*/FuncTy_6,
       /*Linkage=*/GlobalValue::ExternalLinkage,
       /*Name=*/"printf", M); 
    printfFunc->setCallingConv(CallingConv::C);

    AttributeList func_printf_PAL =
      AttributeList()
      .addAttribute(M->getContext(), 1U, Attribute::NoCapture)
      .addAttribute(M->getContext(), 4294967295U, Attribute::NoUnwind);

    printfFunc->setAttributes(func_printf_PAL);
  }

  std::vector<Constant*> const_ptr_8_indices;

  ConstantInt* const_int64_9 = ConstantInt::get(M->getContext(), APInt(64, StringRef("0"), 10));
  const_ptr_8_indices.push_back(const_int64_9);
  const_ptr_8_indices.push_back(const_int64_9);
  assert (isa<Constant>(stringArg));
  Constant* const_ptr_8 =
    ConstantExpr::getGetElementPtr
    (PointerType::getUnqual(Type::getInt8Ty(M->getContext())), cast<Constant>(stringArg), const_ptr_8_indices);

  std::vector<Value*> args;
  args.push_back(const_ptr_8);
  args.insert(args.end(), params.begin(), params.end());

  CallInst::Create(printfFunc, args, "", before);
}

void
ParallelRegion::SetExitBB(llvm::BasicBlock *block)
{
  for (size_t i = 0; i < size(); ++i)
    {
      if (at(i) == block) 
        {
          setExitBBIndex(i);
          return;
        }
    }
  assert (false && "The block was not found in the PRegion!");
}

/**
 * Adds a printf to the end of the parallel region that prints the
 * region ID and the work item ID. 
 *
 * Useful for debugging control flow bugs.
 */
void
ParallelRegion::InjectRegionPrintF()
{
  llvm::Module *M = entryBB()->getParent()->getParent();

#if 0
  // it should reuse equal strings anyways
  const char* FORMAT_STR_VAR = ".pocl.pRegion_debug_str";
  llvm::Value *stringArg = M->getGlobalVariable(FORMAT_STR_VAR);
  if (stringArg == NULL)
    {
      IRBuilder<> builder(entryBB());
      stringArg = builder.CreateGlobalString("PR %d WI %u %u %u\n", FORMAT_STR_VAR);
    }
#endif

  ConstantInt* pRID = ConstantInt::get(M->getContext(), APInt(32, pRegionId, 10));
  std::vector<Value*> params;
  params.push_back(pRID);
  params.push_back(LocalIDXLoad());
  params.push_back(LocalIDYLoad());
  params.push_back(LocalIDZLoad());

  InjectPrintF(exitBB()->getTerminator(), "PR %d WI %u %u %u\n", params);

}

/**
 * Adds a printf to the end of the parallel region that prints the
 * hex contents of all named non-pointer variables.
 *
 * Useful for debugging data flow bugs.
 */
void
ParallelRegion::InjectVariablePrintouts()
{
  for (ParallelRegion::iterator i = begin();
       i != end(); ++i)
    {
      llvm::BasicBlock *bb = *i;
      for (llvm::BasicBlock::iterator instr = bb->begin();
           instr != bb->end(); ++instr) 
        {
          llvm::Instruction *instruction = &*instr;
          if (isa<PointerType>(instruction->getType()) ||
              !instruction->hasName()) continue;
          std::string name = instruction->getName().str();
          std::vector<Value*> args;
          IRBuilder<> builder(exitBB()->getTerminator());
          args.push_back(builder.CreateGlobalString(name));
          args.push_back(instruction);
          InjectPrintF(instruction->getParent()->getTerminator(), "variable %s == %x\n", args);
        }
    }
}

/**
 * Localizes all the loads to the the work-item identifiers.
 *
 * In case the code inside the region queries the WI id, it
 * should not (re)use one that is loaded in another region, but
 * one that is loaded in the same region. Otherwise, it ends
 * up using the last id the previous PR work-item loop got.
 * This caused problems in cases where the local id was stored
 * to a temporary variable in an earlier region and that temp
 * was reused later.
 *
 * The function scans for all loads from the local id variables
 * and converts them to loads inside the parallel region.
 */
void
ParallelRegion::LocalizeIDLoads() 
{
  /* The local id loads inside the parallel region. */
  llvm::Instruction* LocalIDXLoadInstr = LocalIDXLoad();
  llvm::Instruction* LocalIDYLoadInstr = LocalIDYLoad();
  llvm::Instruction* LocalIDZLoadInstr = LocalIDZLoad();
  llvm::Module *M = LocalIDXLoadInstr->getParent()->getParent()->getParent();
  llvm::Value *localIdZ = M->getNamedGlobal(POCL_LOCAL_ID_Z_GLOBAL);
  llvm::Value *localIdY = M->getNamedGlobal(POCL_LOCAL_ID_Y_GLOBAL);
  llvm::Value *localIdX = M->getNamedGlobal(POCL_LOCAL_ID_X_GLOBAL);

  assert (localIdZ != NULL && localIdY != NULL && localIdX != NULL &&
	  "The local id globals were not created.");

  for (ParallelRegion::iterator i = begin();
       i != end(); ++i)
    {
      llvm::BasicBlock *bb = *i;
      for (llvm::BasicBlock::iterator instrI = bb->begin();
           instrI != bb->end(); ++instrI) 
        {
          llvm::Instruction *instr = &*instrI;
	  if (instr == LocalIDXLoadInstr ||
	      instr == LocalIDYLoadInstr ||
	      instr == LocalIDZLoadInstr) continue;

	  /* Search all operands of the instruction. If any of them is
	     using a local id, replace it with the intra-PR load from the
	     id variable. */
          for (unsigned opr = 0; opr < instr->getNumOperands(); ++opr)
            {
	      llvm::LoadInst *load = 
		dyn_cast<llvm::LoadInst>(instr->getOperand(opr));
	      if (load == NULL) continue;
	      if (load == LocalIDXLoadInstr ||
		  load == LocalIDYLoadInstr ||
		  load == LocalIDZLoadInstr) continue;
	      
	      if (load->getPointerOperand() == localIdZ)
		instr->setOperand(opr, LocalIDZLoadInstr);
	      if (load->getPointerOperand() == localIdY)
		instr->setOperand(opr, LocalIDYLoadInstr);
	      if (load->getPointerOperand() == localIdX)
		instr->setOperand(opr, LocalIDXLoadInstr);
	    }
	}
    }
}
