// Class for kernels, llvm::Functions that represent OpenCL C kernels.
//
// Copyright (c) 2011 Universidad Rey Juan Carlos and
//               2012-2019 Pekka Jääskeläinen
//               2024 Pekka Jääskeläinen / Intel Finland Oy
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
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS

IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include "Kernel.h"
#include "Barrier.h"
#include "DebugHelpers.h"

#include "pocl.h"
#include "pocl_llvm_api.h"

// #define DEBUG_PR_CREATION

POP_COMPILER_DIAGS

using namespace llvm;
using namespace pocl;

static void addPredecessors(SmallVectorImpl<BasicBlock *> &V, BasicBlock *BB);

void Kernel::getExitBlocks(SmallVectorImpl<llvm::BasicBlock *> &B) {
  for (iterator i = begin(), e = end(); i != e; ++i) {
    auto t = i->getTerminator();
    if (t->getNumSuccessors() == 0) {
      // All exits must be barrier blocks.
      llvm::BasicBlock *BB = cast<BasicBlock>(i);
      if (!Barrier::hasBarrier(BB))
        Barrier::create(BB->getTerminator());
      B.push_back(BB);
    }
  }
}

/* @todo Move to ParallelRegion.cc */
ParallelRegion *Kernel::createParallelRegionBefore(llvm::BasicBlock *B) {

  BasicBlock *RegionEntryBarrier = NULL;
  // The original entry basic block of the parallel region, before creating the
  // context restore entry.
  BasicBlock *OrigEntry = NULL;
  BasicBlock *Exit = B->getSinglePredecessor();

  // The special entry block is not treated as a parallel region.
  if (Exit == &getEntryBlock())
    return nullptr;

  SmallVector<BasicBlock *, 4> PendingBlocks;
  addPredecessors(PendingBlocks, B);

#ifdef DEBUG_PR_CREATION
  std::cerr << "createParallelRegionBefore " << B->getName().str() << std::endl;
#endif

  SmallPtrSet<BasicBlock *, 8> BlocksInRegion;
  while (!PendingBlocks.empty()) {
    BasicBlock *Current = PendingBlocks.back();
    PendingBlocks.pop_back();

#ifdef DEBUG_PR_CREATION
    std::cerr << "considering " << Current->getName().str() << std::endl;
#endif

    // avoid infinite recursion of loops
    if (BlocksInRegion.count(Current) != 0)
      continue;

    // If we reach another barrier this must be the parallel region's original
    // entry node.
    if (Barrier::hasOnlyBarrier(Current)) {
      if (RegionEntryBarrier == NULL)
        RegionEntryBarrier = Current;
#ifdef DEBUG_PR_CREATION
      std::cerr << "### it's a barrier!" << std::endl;
#endif
      continue;
    }

    // We expect no other instructions in barrier blocks expect the function
    // entry node where we push context data allocas.
    if (Barrier::hasBarrier(Current)) {
      assert(
          false &&
          "Barrier found in a non-barrier (non-entry) block! (forgot barrier "
          "canonicalization?)");
    }

#ifdef DEBUG_PR_CREATION
    std::cerr << "added it to the region" << std::endl;
#endif
    // Non-barrier block, this must be on the region.
    BlocksInRegion.insert(Current);

    // Add predecessors to pending queue.
    addPredecessors(PendingBlocks, Current);
  }

  if (BlocksInRegion.empty())
    return NULL;

  // Find the entry node.
  assert(RegionEntryBarrier != NULL);
  for (unsigned Suc = 0,
                Num = RegionEntryBarrier->getTerminator()->getNumSuccessors();
       Suc < Num; ++Suc) {
    llvm::BasicBlock *EntryCandidate =
        RegionEntryBarrier->getTerminator()->getSuccessor(Suc);
    if (BlocksInRegion.count(EntryCandidate) == 0)
      continue;
    OrigEntry = EntryCandidate;
    break;
  }
  assert(BlocksInRegion.count(OrigEntry) != 0);

  // We got all the blocks in a region, create it.
  return ParallelRegion::Create(BlocksInRegion, OrigEntry, Exit);
}

static void addPredecessors(SmallVectorImpl<BasicBlock *> &V, BasicBlock *BB) {
  for (pred_iterator i = pred_begin(BB), e = pred_end(BB); i != e; ++i) {
    V.push_back(*i);
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
 * The main entry to the "parallel region formation" which searches for regions
 * of basic blocks between barriers that can be freely parallelized across
 * work-items in the work-group.
 */
void Kernel::getParallelRegions(
    llvm::LoopInfo &LI,
    ParallelRegion::ParallelRegionVector *ParallelRegions) {

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
      ParallelRegions->push_back(PR);
      BasicBlock *Entry = PR->entryBB();
      int found_predecessors = 0;
      BasicBlock *loop_barrier = NULL;
      for (pred_iterator i = pred_begin(Entry), e = pred_end(Entry);
           i != e; ++i) {
        BasicBlock *Barrier = (*i);
        if (!found_barriers.count(Barrier)) {
          /* If this is a loop header block we might have edges from two 
             unprocessed barriers. The one inside the loop (coming from a 
             computation block after a branch block) should be processed 
             first. */
          std::string bbName = "";
          bool IsInTheSameLoop =
              LI.getLoopFor(Barrier) != NULL && LI.getLoopFor(Entry) != NULL &&
              LI.getLoopFor(Entry) == LI.getLoopFor(Barrier);

          if (IsInTheSameLoop)
            {
#ifdef DEBUG_PR_CREATION
            std::cout << "### found a barrier inside a loop:" << std::endl;
            std::cout << Barrier->getName().str() << std::endl;
#endif
              if (loop_barrier != NULL) {
                // there can be multiple latches and each have their barrier,
                // save the previously found inner loop barrier
                exit_blocks.push_back(loop_barrier);
              }
              loop_barrier = Barrier;
            }
          else
            {
#ifdef DEBUG_PR_CREATION
              std::cout << "### found a barrier:" << std::endl;
              std::cout << Barrier->getName().str() << std::endl;
#endif
              exit = Barrier;
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
  pocl::dumpCFG(*this, this->getName().str() + ".pregions.dot", nullptr,
                ParallelRegions);
#endif
}

ParallelRegion::ParallelRegionVector *
Kernel::getParallelRegions(llvm::LoopInfo &LI) {
  ParallelRegion::ParallelRegionVector *ParallelRegions =
      new ParallelRegion::ParallelRegionVector;

  getParallelRegions(LI, ParallelRegions);

  return ParallelRegions;
}

void Kernel::addLocalSizeInitCode(size_t LocalSizeX, size_t LocalSizeY,
                                  size_t LocalSizeZ) {

  IRBuilder<> Builder(getEntryBlock().getFirstNonPHI());

  GlobalVariable *GV;

  llvm::Module* M = getParent();

  unsigned long AddressBits;
  getModuleIntMetadata(*M, "device_address_bits", AddressBits);

  llvm::Type *SizeT = IntegerType::get(M->getContext(), AddressBits);

  GV = M->getGlobalVariable("_local_size_x");
  if (GV != NULL) {
    Builder.CreateStore(ConstantInt::get(SizeT, LocalSizeX), GV);
  }

  GV = M->getGlobalVariable("_local_size_y");
  if (GV != NULL)
    Builder.CreateStore(ConstantInt::get(SizeT, LocalSizeY), GV);

  GV = M->getGlobalVariable("_local_size_z");
  if (GV != NULL)
    Builder.CreateStore(ConstantInt::get(SizeT, LocalSizeZ), GV);
}
