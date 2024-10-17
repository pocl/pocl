/*
 * Adapted from
 * https://github.com/OpenSYCL/OpenSYCL/blob/develop/include/hipSYCL/compiler/cbs/SubCfgFormation.hpp
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay and contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef POCL_SUBCFGFORMATION_H
#define POCL_SUBCFGFORMATION_H

#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Pass.h>
#include <llvm/Passes/PassBuilder.h>

#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "WorkitemHandler.h"

namespace pocl {

// TO CLEAN: Merge with / derive from ParallelRegion.
class SubCFG {
public:
  using BlockVector = llvm::SmallVector<llvm::BasicBlock *, 8>;
  SubCFG(llvm::BasicBlock *EntryBarrier, llvm::AllocaInst *LastBarrierIdStorage,
         const llvm::DenseMap<llvm::BasicBlock *, size_t> &BarrierIds,
         llvm::Value *IndVar, size_t Dim);

  SubCFG(const SubCFG &) = delete;
  SubCFG &operator=(const SubCFG &) = delete;

  SubCFG(SubCFG &&) = default;
  SubCFG &operator=(SubCFG &&) = default;

  BlockVector &getBlocks() noexcept { return Blocks_; }
  const BlockVector &getBlocks() const noexcept { return Blocks_; }

  BlockVector &getNewBlocks() noexcept { return NewBlocks_; }
  const BlockVector &getNewBlocks() const noexcept { return NewBlocks_; }

  size_t getEntryId() const noexcept { return EntryId_; }

  llvm::BasicBlock *getEntry() noexcept { return EntryBB_; }
  llvm::BasicBlock *getExit() noexcept { return ExitBB_; }
  llvm::BasicBlock *getLoadBB() noexcept { return LoadBB_; }
  llvm::Value *getContiguousIdx() noexcept { return ContIdx_; }

  void replicate(llvm::Function &F,
                 const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *>
                     &InstAllocaMap,
                 llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *>
                     &BaseInstAllocaMap,
                 llvm::DenseMap<llvm::Instruction *,
                                llvm::SmallVector<llvm::Instruction *, 8>>
                     &ContInstReplicaMap,
                 llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *>
                     &RemappedInstAllocaMap,
                 llvm::BasicBlock *AfterBB,
                 llvm::ArrayRef<llvm::Value *> LocalSize);

  void arrayifyMultiSubCfgValues(
      llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *>
          &BaseInstAllocaMap,
      llvm::DenseMap<llvm::Instruction *,
                     llvm::SmallVector<llvm::Instruction *, 8>>
          &ContInstReplicaMap,
      llvm::ArrayRef<SubCFG> SubCFGs, llvm::Instruction *AllocaIP,
      llvm::Value *ReqdArrayElements,
      pocl::VariableUniformityAnalysisResult &VecInfo);
  void fixSingleSubCfgValues(
      llvm::DominatorTree &DT,
      const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *>
          &RemappedInstAllocaMap,
      llvm::Value *ReqdArrayElements,
      pocl::VariableUniformityAnalysisResult &VecInfo);

  void print() const;
  void removeDeadPhiBlocks(
      llvm::SmallVector<llvm::BasicBlock *, 8> &BlocksToRemap) const;
  llvm::SmallVector<llvm::Instruction *, 16> topoSortInstructions(
      const llvm::SmallPtrSet<llvm::Instruction *, 16> &UniquifyInsts) const;

private:
  BlockVector Blocks_;
  BlockVector NewBlocks_;
  size_t EntryId_;
  llvm::BasicBlock *EntryBarrier_;
  llvm::SmallDenseMap<llvm::BasicBlock *, size_t> ExitIds_;
  llvm::AllocaInst *LastBarrierIdStorage_;
  llvm::Value *ContIdx_;
  llvm::BasicBlock *EntryBB_;
  llvm::BasicBlock *ExitBB_;
  llvm::BasicBlock *LoadBB_;
  llvm::BasicBlock *PreHeader_;
  size_t Dim;

  llvm::BasicBlock *createExitWithID(
      llvm::detail::DenseMapPair<llvm::BasicBlock *, size_t> BarrierPair,
      llvm::BasicBlock *After, llvm::BasicBlock *TargetBB);

  void loadMultiSubCfgValues(
      const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *>
          &InstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *>
          &BaseInstAllocaMap,
      llvm::DenseMap<llvm::Instruction *,
                     llvm::SmallVector<llvm::Instruction *, 8>>
          &ContInstReplicaMap,
      llvm::BasicBlock *UniformLoadBB, llvm::ValueToValueMapTy &VMap);
  void loadUniformAndRecalcContValues(
      llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *>
          &BaseInstAllocaMap,
      llvm::DenseMap<llvm::Instruction *,
                     llvm::SmallVector<llvm::Instruction *, 8>>
          &ContInstReplicaMap,
      llvm::BasicBlock *UniformLoadBB, llvm::ValueToValueMapTy &VMap);
  llvm::BasicBlock *createLoadBB(llvm::ValueToValueMapTy &VMap);
  llvm::BasicBlock *createUniformLoadBB(llvm::BasicBlock *OuterMostHeader);
};

class SubCFGFormation : public llvm::PassInfoMixin<SubCFGFormation>,
                        WorkitemHandler {
public:
  static void registerWithPB(llvm::PassBuilder &B);
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }

  static bool canHandleKernel(llvm::Function &K,
                              llvm::FunctionAnalysisManager &AM);

protected:
  llvm::Value *getLinearWIIndexInRegion(llvm::Instruction *Instr);
  llvm::Instruction *getLocalIdInRegion(llvm::Instruction *Instr, size_t Dim);

private:
  void formSubCfgs(llvm::Function &F, llvm::LoopInfo &LI,
                   llvm::DominatorTree &DT, llvm::PostDominatorTree &PDT,
                   pocl::VariableUniformityAnalysisResult &VUA);
  void arrayifyAllocas(llvm::BasicBlock *EntryBlock, llvm::DominatorTree &DT,
                       std::vector<SubCFG> &SubCfgs,
                       llvm::Value *ReqdArrayElements,
                       pocl::VariableUniformityAnalysisResult &VecInfo);
  SubCFG *regionOfBlock(llvm::BasicBlock *BB);

  // The sub CFGs in the currently handled kernel.
  std::vector<SubCFG> SubCFGs;
};

} // namespace pocl

#endif // POCL_SUBCFGFORMATION_H
