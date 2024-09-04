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

static void AddPredecessors(SmallVectorImpl<BasicBlock *> &V, BasicBlock *BB);

void Kernel::getExitBlocks(SmallVectorImpl<llvm::BasicBlock *> &B) {
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
  AddPredecessors(PendingBlocks, B);

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
    AddPredecessors(PendingBlocks, Current);
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

  // Ensure we have a unique PR entry block without phis where all the context
  // restore code will be added and which acts as a unique landing pad from
  // other PRs.
  BasicBlock *PREntry = BasicBlock::Create(
      OrigEntry->getContext(),
      "parallel_region_" + std::to_string(ParallelRegion::getNextID()) +
          "_entry",
      OrigEntry->getParent(), OrigEntry);

  IRBuilder<> Builder(PREntry);
  Builder.CreateBr(OrigEntry);

  // Does it have a jump to the next block?
  BlocksInRegion.insert(PREntry);
  std::set<BasicBlock *> Preds;
  for (pred_iterator PI = pred_begin(OrigEntry), PE = pred_end(OrigEntry);
       PI != PE; ++PI) {
    Preds.insert(*PI);
  }
  // There can be many predecessor basic blocks to the region,
  // fix all the predecessor blocks from other regions to jump to the region
  // entry. Note: a special case is the intra-PR-loop case where the header node
  // is the loop header. In that case get incoming branches to the
  // entry from inside the PR.
  for (BasicBlock *PredBB : Preds) {
    if (BlocksInRegion.count(PredBB) > 0)
      continue; // Must be an intra-PR loop backedge source.

    // Fix the branch of the predecessor to point to the new entry.
    BranchInst *BR = cast<BranchInst>(PredBB->getTerminator());
    for (unsigned Suc = 0; Suc < BR->getNumSuccessors(); ++Suc)
      if (BR->getSuccessor(Suc) == OrigEntry)
        BR->setSuccessor(Suc, PREntry);
    OrigEntry->replacePhiUsesWith(PredBB, PREntry);
  }
  assert(PREntry != nullptr);

  return ParallelRegion::Create(BlocksInRegion, PREntry, Exit);
}

static void AddPredecessors(SmallVectorImpl<BasicBlock *> &V, BasicBlock *BB) {
  for (pred_iterator i = pred_begin(BB), e = pred_end(BB); i != e; ++i) {
    V.push_back(*i);
  }
}

/**
 * The main entry to the "parallel region formation" which searches for regions
 * of basic blocks between barriers that can be freely parallelized across
 * work-items in the work-group.
 *
 * @todo Move to ParallelRegion.
 */
void Kernel::getParallelRegions(
    llvm::LoopInfo &LI, ParallelRegion::ParallelRegionVector *ParallelRegions) {

  // Ensure we have a separate function entry node where we push context
  // array allocas and that it's separated from the very first use code
  // block with an isolated barrier node.
  BasicBlock *KernelEntry = &getEntryBlock();
  assert(Barrier::hasBarrier(KernelEntry));
  Barrier *Bar = Barrier::FindInBasicBlock(KernelEntry);
  llvm::SplitBlock(KernelEntry, Bar);

  // Unhandled parallel region exit blocks.
  SmallVector<BasicBlock *, 4> PRExitBlocks;

  // We start on a function exit block and process the parallel regions upwards
  // (finding an execution trace).
  getExitBlocks(PRExitBlocks);

  // We need to keep track of traversed barriers to detect back edges.
  SmallPtrSet<BasicBlock *, 8> ProcessedPRExits;

  // First find all the ParallelRegions in the Function.
  while (!PRExitBlocks.empty()) {

    BasicBlock *PRExitToProcess = PRExitBlocks.back();
    PRExitBlocks.pop_back();

    // Already handled.
    if (ProcessedPRExits.count(PRExitToProcess) > 0)
      continue;

    while (ParallelRegion *PR = createParallelRegionBefore(PRExitToProcess)) {
      assert(PR != NULL && !PR->empty() &&
             "Empty parallel region in kernel (contiguous barriers)!");

      ProcessedPRExits.insert(PRExitToProcess);
      PRExitToProcess = NULL;
      ParallelRegions->push_back(PR);
      BasicBlock *PREntry = PR->entryBB();
      int FoundPredecessors = 0;
      BasicBlock *LoopBarrier = NULL;

      // Find the other parallel regions that flow into this one.
      for (pred_iterator i = pred_begin(PREntry), e = pred_end(PREntry); i != e;
           ++i) {
        BasicBlock *Barrier = (*i);
        if (ProcessedPRExits.count(Barrier) > 0)
          continue;

        if (!Barrier::hasBarrier(Barrier) && PR->HasBlock(Barrier)) {
#ifdef DEBUG_PR_CREATION
          std::cout << "### a block that branches to the entry node, must be a "
                       "loop header:"
                    << std::endl;
          std::cout << Barrier->getName().str() << std::endl;
#endif
          continue;
        }

        // If this is a loop header block we might have edges from two
        // unprocessed barriers. The one inside the loop (coming from a
        // computation block after a branch block) should be processed
        // first.

        // Do we need to recompute LoopInfo here since we've added the entry
        // blocks to PRs?
        bool IsInTheSameLoop = LI.getLoopFor(Barrier) != NULL &&
                               LI.getLoopFor(PREntry) != NULL &&
                               LI.getLoopFor(PREntry) == LI.getLoopFor(Barrier);

        if (IsInTheSameLoop) {
#ifdef DEBUG_PR_CREATION
          std::cout << "### found a barrier inside a loop:" << std::endl;
          std::cout << Barrier->getName().str() << std::endl;
#endif
          if (LoopBarrier != NULL) {
            // There can be multiple latches and each have their barrier,
            // save the previously found inner loop barrier.
            PRExitBlocks.push_back(LoopBarrier);
          }
          LoopBarrier = Barrier;
        } else {
#ifdef DEBUG_PR_CREATION
          std::cout << "### found a barrier:" << std::endl;
          std::cout << Barrier->getName().str() << std::endl;
#endif
          PRExitToProcess = Barrier;
          PRExitBlocks.push_back(Barrier);
        }
        ++FoundPredecessors;
      }

      // Always process the inner loop regions first.
      if (LoopBarrier != NULL && !ProcessedPRExits.count(LoopBarrier))
        PRExitToProcess = LoopBarrier;

#ifdef DEBUG_PR_CREATION
      std::cout << "### created a ParallelRegion:" << std::endl;
      PR->dumpNames();
      std::cout << std::endl;
#endif

      if (FoundPredecessors == 0) {
        // This path has been traversed and we encountered no more
        // unprocessed regions. It means we have either traversed all
        // paths from the exit or have transformed a loop and thus
        // encountered only a barrier that was seen (and thus
        // processed) before.
        break;
      }
      assert(PRExitToProcess != NULL &&
             "Parallel region without entry barrier!");
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
