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

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "Barrier.h"
#include "DebugHelpers.h"
#include "Kernel.h"
#include "KernelCompilerUtils.h"
#include "LLVMUtils.h"
#include "ParallelRegion.h"

POP_COMPILER_DIAGS

#include <algorithm>
#include <map>
#include <set>
#include <sstream>

#include "pocl_llvm_api.h"

using namespace std;
using namespace llvm;
using namespace pocl;

//#define DEBUG_REMAP
//#define DEBUG_REPLICATE
//#define DEBUG_PURGE
//#define DEBUG_CREATE

#include <iostream>

int ParallelRegion::idGen = 0;

ParallelRegion::ParallelRegion(int forcedRegionId)
    : exitIndex_(0), entryIndex_(0), pRegionId(forcedRegionId) {
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
  Function *F = successor->getParent();

#if LLVM_MAJOR < 16
  Function::BasicBlockListType &bb_list =
    F->getBasicBlockList();
  for (iterator i = begin(), e = end(); i != e; ++i)
    bb_list.insertAfter(tail->getIterator(), *i);
#else
  for (iterator i = begin(), e = end(); i != e; ++i)
    F->insert(tail->getIterator(), *i);
#endif

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

  GlobalVariable *GVX = M->getGlobalVariable(LID_G_NAME(0));
  if (GVX != NULL)
    Builder.CreateStore(ConstantInt::get(SizeT(M), X), GVX);

  GlobalVariable *GVY = M->getGlobalVariable(LID_G_NAME(1));
  if (GVY != NULL)
    Builder.CreateStore(ConstantInt::get(SizeT(M), Y), GVY);

  GlobalVariable *GVZ = M->getGlobalVariable(LID_G_NAME(2));
  if (GVZ != NULL)
    Builder.CreateStore(ConstantInt::get(SizeT(M), Z), GVZ);
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
    std::cerr << (*i)->getName().str();
    if (entryBB() == (*i))
      std::cerr << "(EN)";
    if (exitBB() == (*i))
      std::cerr << "(EX)";
    std::cerr << " ";
    }
    std::cerr << std::endl;
}

ParallelRegion *ParallelRegion::Create(const SmallPtrSet<BasicBlock *, 8> &BBs,
                                       BasicBlock *Entry, BasicBlock *Exit) {
  ParallelRegion *NewRegion = new ParallelRegion();

  assert(Entry != NULL);
  assert(Exit != NULL);

  // This is done in two steps so order of the vector
  // is the same as original function order.
  Function *F = Entry->getParent();
  for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
    BasicBlock *B = &*i;
    for (SmallPtrSetIterator<BasicBlock *> j = BBs.begin(); j != BBs.end();
         ++j) {
      if (*j == B) {
        NewRegion->push_back(&*i);
        if (Entry == *j)
          NewRegion->setEntryBBIndex(NewRegion->size() - 1);
        else if (Exit == *j)
          NewRegion->setExitBBIndex(NewRegion->size() - 1);
        break;
      }
    }
  }

  NewRegion->LocalizeIDLoads();
#ifdef DEBUG_CREATE
  assert(NewRegion->Verify());
#endif

  return NewRegion;
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
          pocl::dumpCFG(*(*i)->getParent(),
                        (*i)->getParent()->getName().str() + ".dot", nullptr,
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
      dumpCFG(*(*i)->getParent(), "broken.dot", nullptr,
              &regions, &highlights);
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

#define PARALLEL_MD_NAME "llvm.access.group"

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
 * Parallel loop metadata prior to LLVM 12.0.1 on memory reads also implies that
 * if-conversion (i.e., speculative execution within a loop iteration) is safe.
 * Given an instruction reading from memory, IsLoadUnconditionallySafe should
 * return whether it is safe under (unconditional, unpredicated) speculative
 * execution. See https://bugs.llvm.org/show_bug.cgi?id=46666 and
 * https://github.com/pocl/pocl/issues/757.
 *
 * From LLVM 12.0.1 onward parallel loop metadata does not imply if-conversion
 * safety anymore. This got fixed by this change:
 * https://reviews.llvm.org/D103907 for LLVM 13 which also got backported to
 * LLVM 12.0.1. In other words this means that before the fix, the loop
 * vectorizer was not able to vectorize some kernels because they would required
 * a huge runtime memory check code insertion. Leading to vectorizer to give up.
 * With above fix, we can add metadata to every load.  This will cause
 * vectorizer to skip runtime memory check code insertion part because it
 * indicates that iterations do not depend on each other. Which in turn makes
 * vectorization easier. In this case using of IsLoadUnconditionallySafe
 * parameter will be skipped.
 */
void ParallelRegion::addParallelLoopMetadata(
    llvm::MDNode *Identifier,
    std::function<bool(llvm::Instruction *)> IsLoadUnconditionallySafe) {
  for (iterator i = begin(), e = end(); i != e; ++i) {
    BasicBlock *BB = *i;
    for (BasicBlock::iterator ii = BB->begin(), ee = BB->end(); ii != ee;
         ii++) {
      if (!ii->mayReadOrWriteMemory()) {
        continue;
      }

      MDNode *NewMD = MDNode::get(BB->getContext(), Identifier);
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
    Metadata *v[] = {MDString::get(context, "WI_data"), mdRegion, mdXYZ};
    MDNode *md = MDNode::get(context, v);

    for (iterator i = begin(), e = end(); i != e; ++i) {
      BasicBlock *BB = *i;
      for (BasicBlock::iterator ii = BB->begin(); ii != BB->end(); ii++) {
        Metadata *v3[] = {MDString::get(context, "WI_counter"),
                          llvm::ConstantAsMetadata::get(ConstantInt::get(
                              Type::getInt32Ty(context), counter))};
        MDNode *mdCounter = MDNode::get(context, v3);
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

bool ParallelRegion::hasBlock(llvm::BasicBlock *Block) {
  return find(begin(), end(), Block) != end();
}

/// Finds the instruction that loads an id of the work item in the
/// beginning of the parallel region, if not found, creates it.
///
/// \param IDGlobalName The name of the (magic) GlobalVariable temporally
/// representing the id.
/// \param Before If given, finds one in the basic block of the given
/// instruction, or creates one just before it.
/// \returns The instruction loading the id.
llvm::Instruction *
ParallelRegion::getOrCreateIDLoad(std::string IDGlobalName,
                                  llvm::Instruction *Before) {

  Module *M = entryBB()->getParent()->getParent();

  llvm::Type *ST = SizeT(M);
  GlobalVariable *IDGlobal =
      cast<GlobalVariable>(M->getOrInsertGlobal(IDGlobalName, ST));

  if (Before != nullptr) {
    // Try to find one in the same BB.
    BasicBlock *BB = Before->getParent();
    for (auto &I : *BB) {
      Instruction *BBInst = &I;

      if (BBInst == Before) {
        // Didn't find one before it. Create one.
        IRBuilder<> Builder(Before);
        return Builder.CreateLoad(ST, IDGlobal);
      }

      LoadInst *Load = dyn_cast<LoadInst>(BBInst);
      if (Load == nullptr)
        continue;
      GlobalVariable *Global =
          dyn_cast<GlobalVariable>(Load->getPointerOperand());
      if (Global == IDGlobal)
        return Load;
    }
  }

  // Otherwise, create one to the parallel region entry.
  if (IDLoadInstrs.find(IDGlobalName) != IDLoadInstrs.end())
    return IDLoadInstrs[IDGlobalName];

  GlobalVariable *Ptr =
      cast<GlobalVariable>(M->getOrInsertGlobal(IDGlobalName, ST));

  IRBuilder<> Builder(entryBB()->getFirstNonPHI());

  Instruction *IDLoad = Builder.CreateLoad(ST, IDGlobal);
  IDLoadInstrs[IDGlobalName] = IDLoad;
  return IDLoad;
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
    
  /* generated with help from https://llvm.org/demo/index.cgi */
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
            .addAttributeAtIndex(M->getContext(), 1U, Attribute::NoCapture)
            .addAttributeAtIndex(M->getContext(), 4294967295U,
                                 Attribute::NoUnwind);

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
  params.push_back(getOrCreateIDLoad(LID_G_NAME(0)));
  params.push_back(getOrCreateIDLoad(LID_G_NAME(1)));
  params.push_back(getOrCreateIDLoad(LID_G_NAME(2)));

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
 * In case the code inside the region queries the WI id, it should not (re)use
 * one that is loaded in another region, but one that is loaded in the same
 * region. Otherwise, it ends up using the last id the previous PR work-item
 * loop got. This caused problems in cases where the local id was stored to a
 * temporary variable in an earlier region and that temp was reused later.
 *
 * The function scans for all accesses to the local and global ids and converts
 * them to loads inside the parallel region. Also converts calls to
 * get_global_id() declaration to the magic global variable calls. (TODO: Move
 * that functionality to a separate method).
 */
void ParallelRegion::LocalizeIDLoads() {

  // Replace get_global_id with loads from the _get_global_id magic
  // global.
  std::set<llvm::Instruction *> InstrsToDelete;
  for (ParallelRegion::iterator BBI = begin(); BBI != end(); ++BBI) {
    llvm::BasicBlock *BB = *BBI;
    for (llvm::BasicBlock::iterator II = BB->begin(); II != BB->end(); ++II) {
      llvm::Instruction *Instr = &*II;
      llvm::CallInst *Call = dyn_cast<llvm::CallInst>(Instr);
      if (Call == nullptr)
        continue;

      auto Callee = Call->getCalledFunction();
      if (Callee != nullptr && Callee->isDeclaration() &&
          Callee->getName() == GID_BUILTIN_NAME) {
        int Dim =
            cast<llvm::ConstantInt>(Call->getArgOperand(0))->getZExtValue();
        llvm::Instruction *GIDLoad = getOrCreateIDLoad(GID_G_NAME(Dim));
        Call->replaceAllUsesWith(GIDLoad);
        InstrsToDelete.insert(Call);
        continue;
      }
    }
  }
  for (auto I : InstrsToDelete)
    I->eraseFromParent();

  // The id loads inside the parallel region.
  std::array<llvm::Instruction *, 6> RegionIDLoads = {
      getOrCreateIDLoad(LID_G_NAME(0)), getOrCreateIDLoad(LID_G_NAME(1)),
      getOrCreateIDLoad(LID_G_NAME(2)), getOrCreateIDLoad(GID_G_NAME(0)),
      getOrCreateIDLoad(GID_G_NAME(1)), getOrCreateIDLoad(GID_G_NAME(2))};

  llvm::Module *M = RegionIDLoads[0]->getParent()->getParent()->getParent();

  std::array<llvm::Value *, 6> Globals = {
      M->getNamedGlobal(LID_G_NAME(0)), M->getNamedGlobal(LID_G_NAME(1)),
      M->getNamedGlobal(LID_G_NAME(2)), M->getNamedGlobal(GID_G_NAME(0)),
      M->getNamedGlobal(GID_G_NAME(1)), M->getNamedGlobal(GID_G_NAME(2))};

  for (ParallelRegion::iterator BBI = begin(); BBI != end(); ++BBI) {
    llvm::BasicBlock *BB = *BBI;
    for (llvm::BasicBlock::iterator II = BB->begin(); II != BB->end(); ++II) {
      llvm::Instruction *Instr = &*II;

      // If any of the operands is using an id, replace it with the
      // intra-PR load from the parallel region specific id variable.
      for (unsigned Opr = 0; Opr < Instr->getNumOperands(); ++Opr) {
        llvm::LoadInst *Load = dyn_cast<llvm::LoadInst>(Instr->getOperand(Opr));
        if (Load == NULL)
          continue;

        if (std::find(RegionIDLoads.begin(), RegionIDLoads.end(), Load) !=
            RegionIDLoads.end())
          continue; // Already converted.

        auto Pos = std::find(Globals.begin(), Globals.end(),
                             Load->getPointerOperand());
        if (Pos == Globals.end())
          continue;

        Instr->setOperand(Opr, RegionIDLoads[Pos - Globals.begin()]);
      }
    }
  }
}
