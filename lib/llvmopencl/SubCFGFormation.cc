/*
 * Adapted from
 * https://github.com/OpenSYCL/OpenSYCL/blob/develop/src/compiler/cbs/SubCfgFormation.cpp
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

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Regex.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/Transforms/Utils/LoopSimplify.h>
POP_COMPILER_DIAGS

#include "Barrier.h"
#include "KernelCompilerUtils.h"
#include "LLVMUtils.h"
#include "SubCFGFormation.h"
#include "Workgroup.h"
#include "WorkitemHandlerChooser.h"

#include "pocl_llvm_api.h"

#include <cstddef>
#include <functional>
#include <iostream>
#include <numeric>

// #define DEBUG_SUBCFG_FORMATION

#define PASS_NAME "subcfgformation"
#define PASS_CLASS pocl::SubCFGFormation
#define PASS_DESC "Form SubCFGs according to CBS"

namespace pocl {

using namespace llvm;

constexpr size_t EntryBarrierId = 0;
constexpr size_t ExitBarrierId = -1;

static size_t DefaultAlignment = 64;
static constexpr const char LoopStateMD[] = "poclLoopState";

namespace PoclMDKind {
  static constexpr const char Arrayified[] = "pocl.arrayified";
  static constexpr const char InnerLoop[] = "pocl.loop.inner";
  static constexpr const char WorkItemLoop[] = "pocl.loop.workitem";
};

static constexpr const char LocalIdGlobalNameX[] = "_local_id_x";
static constexpr const char LocalIdGlobalNameY[] = "_local_id_y";
static constexpr const char LocalIdGlobalNameZ[] = "_local_id_z";
static const std::array<const char *, 3> LocalIdGlobalNames{
    LocalIdGlobalNameX, LocalIdGlobalNameY, LocalIdGlobalNameZ};

static const std::array<char, 3> DimName{'x', 'y', 'z'};

llvm::Loop *updateDtAndLi(llvm::LoopInfo &LI, llvm::DominatorTree &DT,
                          const llvm::BasicBlock *B, llvm::Function &F) {
  DT.reset();
  DT.recalculate(F);
  LI.releaseMemory();
  LI.analyze(DT);
  return LI.getLoopFor(B);
}

template <class UserType, class Func>
bool anyOfUsers(llvm::Value *V, Func &&L) {
  for (auto *U : V->users())
    if (UserType *UT = llvm::dyn_cast<UserType>(U))
      if (L(UT))
        return true;
  return false;
}

/// Arrayification of work item private values

// Create a new alloca of size \a NumElements at \a IPAllocas.
// The type is taken from \a ToArrayify.
// At \a InsertionPoint, a store is added that stores the \a ToArrayify
// value to the alloca element at \a Idx.
llvm::AllocaInst *arrayifyValue(llvm::Instruction *IPAllocas,
                                llvm::Value *ToArrayify,
                                llvm::Instruction *InsertionPoint,
                                llvm::Value *Idx, llvm::Value *NumElements,
                                llvm::MDTuple *MDAlloca = nullptr) {
  assert(Idx && "Valid WI-Index required");

  if (!MDAlloca)
    MDAlloca = llvm::MDNode::get(
        IPAllocas->getContext(),
        {llvm::MDString::get(IPAllocas->getContext(), LoopStateMD)});

  auto *T = ToArrayify->getType();
  llvm::IRBuilder<> AllocaBuilder{IPAllocas};
  auto *Alloca = AllocaBuilder.CreateAlloca(T, NumElements,
                                            ToArrayify->getName() + "_alloca");
  if (NumElements)
    Alloca->setAlignment(llvm::Align{DefaultAlignment});
  Alloca->setMetadata(PoclMDKind::Arrayified, MDAlloca);

  llvm::IRBuilder<> WriteBuilder{InsertionPoint};
  llvm::Value *StoreTarget = Alloca;
  if (NumElements) {
    auto *GEP = llvm::cast<llvm::GetElementPtrInst>(
        WriteBuilder.CreateInBoundsGEP(Alloca->getAllocatedType(), Alloca, Idx,
                                       ToArrayify->getName() + "_gep"));
    GEP->setMetadata(PoclMDKind::Arrayified, MDAlloca);
    StoreTarget = GEP;
  }
  WriteBuilder.CreateStore(ToArrayify, StoreTarget);
  return Alloca;
}

// see arrayifyValue. The store is inserted after the \a ToArrayify instruction
llvm::AllocaInst *arrayifyInstruction(llvm::Instruction *IPAllocas,
                                      llvm::Instruction *ToArrayify,
                                      llvm::Value *Idx,
                                      llvm::Value *NumElements,
                                      llvm::MDTuple *MDAlloca = nullptr) {
  llvm::Instruction *InsertionPoint = &*(++ToArrayify->getIterator());
  if (llvm::isa<llvm::PHINode>(ToArrayify))
    InsertionPoint = ToArrayify->getParent()->getFirstNonPHI();

  return arrayifyValue(IPAllocas, ToArrayify, InsertionPoint, Idx, NumElements,
                       MDAlloca);
}

// load from the \a Alloca at \a Idx, if array alloca, otherwise just load the
// alloca value
llvm::LoadInst *loadFromAlloca(llvm::AllocaInst *Alloca, llvm::Value *Idx,
                               llvm::Instruction *InsertBefore,
                               const llvm::Twine &NamePrefix = "") {
  assert(Idx && "Valid WI-Index required");
  auto *MDAlloca = Alloca->getMetadata(PoclMDKind::Arrayified);

  llvm::IRBuilder<> LoadBuilder{InsertBefore};
  llvm::Value *LoadFrom = Alloca;
  if (Alloca->isArrayAllocation()) {
    auto *GEP =
        llvm::cast<llvm::GetElementPtrInst>(LoadBuilder.CreateInBoundsGEP(
            Alloca->getAllocatedType(), Alloca, Idx, NamePrefix + "_lgep"));
    GEP->setMetadata(PoclMDKind::Arrayified, MDAlloca);
    LoadFrom = GEP;
  }
  auto *Load = LoadBuilder.CreateLoad(Alloca->getAllocatedType(), LoadFrom,
                                      NamePrefix + "_load");
  return Load;
}

// get the work-item state alloca a load reads from (through GEPs..)
llvm::AllocaInst *getLoopStateAllocaForLoad(llvm::LoadInst &LInst) {
  llvm::AllocaInst *Alloca = nullptr;
  if (auto *GEPI =
          llvm::dyn_cast<llvm::GetElementPtrInst>(LInst.getPointerOperand())) {
    Alloca = llvm::dyn_cast<llvm::AllocaInst>(GEPI->getPointerOperand());
  } else {
    Alloca = llvm::dyn_cast<llvm::AllocaInst>(LInst.getPointerOperand());
  }
  if (Alloca && Alloca->hasMetadata(PoclMDKind::Arrayified))
    return Alloca;
  return nullptr;
}

// bring along the llvm.dbg.value intrinsics when cloning values
void copyDgbValues(llvm::Value *From, llvm::Value *To,
                   llvm::Instruction *InsertBefore) {
  llvm::SmallVector<llvm::DbgValueInst *, 1> DbgValues;
  llvm::findDbgValues(DbgValues, From);
  if (!DbgValues.empty()) {
    auto *DbgValue = DbgValues.back();
    llvm::DIBuilder DbgBuilder{
        *InsertBefore->getParent()->getParent()->getParent()};
    DbgBuilder.insertDbgValueIntrinsic(To, DbgValue->getVariable(),
                                       DbgValue->getExpression(),
                                       DbgValue->getDebugLoc(), InsertBefore);
  }
}

// gets the load inside F from the global variable called VarName
llvm::Instruction *getLoadForGlobalVariable(llvm::Function &F,
                                            llvm::StringRef VarName) {
  auto SizeT =
      F.getParent()->getDataLayout().getLargestLegalIntType(F.getContext());
  auto *GV = F.getParent()->getOrInsertGlobal(
      VarName, SizeT); // getGlobalVariable(VarName)
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *LoadI = llvm::dyn_cast<llvm::LoadInst>(&I)) {
        if (LoadI->getPointerOperand() == GV)
          return &I;
      }
    }
  }
  llvm::IRBuilder<> Builder{F.getEntryBlock().getTerminator()};
  return Builder.CreateLoad(SizeT, GV);
}

// init local id to 0
void insertLocalIdInit(llvm::BasicBlock *Entry) {

  llvm::IRBuilder<> Builder(Entry, Entry->getFirstInsertionPt());

  llvm::Module *M = Entry->getParent()->getParent();

  unsigned long address_bits;
  getModuleIntMetadata(*M, "device_address_bits", address_bits);

  llvm::Type *SizeT = llvm::IntegerType::get(M->getContext(), address_bits);

  llvm::GlobalVariable *GVX = M->getGlobalVariable(LocalIdGlobalNameX);
  if (GVX != NULL)
    Builder.CreateStore(llvm::ConstantInt::getNullValue(SizeT), GVX);

  llvm::GlobalVariable *GVY = M->getGlobalVariable(LocalIdGlobalNameY);
  if (GVY != NULL)
    Builder.CreateStore(llvm::ConstantInt::getNullValue(SizeT), GVY);

  llvm::GlobalVariable *GVZ = M->getGlobalVariable(LocalIdGlobalNameZ);
  if (GVZ != NULL)
    Builder.CreateStore(llvm::ConstantInt::getNullValue(SizeT), GVZ);
}

// get the wg size values for the loop bounds
llvm::SmallVector<llvm::Value *, 3>
getLocalSizeValues(llvm::Function &F, llvm::ArrayRef<unsigned long> LocalSizes,
                   bool DynSizes, int Dim) {
  auto &DL = F.getParent()->getDataLayout();
  llvm::IRBuilder<> Builder{F.getEntryBlock().getTerminator()};

  llvm::SmallVector<llvm::Value *, 3> LocalSize(Dim);
  for (int D = 0; D < Dim; ++D) {
    if (DynSizes) {
      auto I =
          getLoadForGlobalVariable(F, std::string{"_local_size_"} + DimName[D]);
      LocalSize[D] = I;

      if (I->getParent() != &F.getEntryBlock()) {
        // must be in entry block. move.
        if (F.getEntryBlock().size() == 1)
          I->moveBefore(F.getEntryBlock().getFirstNonPHI());
        else
          I->moveAfter(F.getEntryBlock().getFirstNonPHI());
      }
    } else
      LocalSize[D] = llvm::ConstantInt::get(
          DL.getLargestLegalIntType(F.getContext()), LocalSizes[D]);
  }

  return LocalSize;
}

// create the wi-loops around a kernel or subCFG, LastHeader input should be the
// load block, ContiguousIdx may be any identifyable value (load from undef)
void createLoopsAround(llvm::Function &F, llvm::BasicBlock *AfterBB,
                       const llvm::ArrayRef<llvm::Value *> &LocalSize,
                       int EntryId, llvm::ValueToValueMapTy &VMap,
                       llvm::SmallVector<llvm::BasicBlock *, 3> &Latches,
                       llvm::BasicBlock *&LastHeader,
                       llvm::Value *&ContiguousIdx) {
  const auto &DL = F.getParent()->getDataLayout();
  auto *LoadBB = LastHeader;
  llvm::IRBuilder<> Builder{LoadBB, LoadBB->getFirstInsertionPt()};

  const size_t Dim = LocalSize.size();

  // from innermost to outermost: create loops around the LastHeader and use
  // AfterBB as dummy exit to be replaced by the outer latch later
  llvm::SmallVector<llvm::PHINode *, 3> IndVars;
  for (size_t D = 0; D < Dim; ++D) {
    const std::string Suffix =
        (llvm::Twine{DimName[D]} + ".subcfg." + llvm::Twine{EntryId}).str();

    auto *Header = llvm::BasicBlock::Create(
        LastHeader->getContext(), "header." + Suffix + "b",
        LastHeader->getParent(), LastHeader);

    Builder.SetInsertPoint(Header, Header->getFirstInsertionPt());

    auto *WIIndVar = Builder.CreatePHI(
        DL.getLargestLegalIntType(F.getContext()), 2, "indvar." + Suffix);
    WIIndVar->addIncoming(
        Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), 0),
        &F.getEntryBlock());
    IndVars.push_back(WIIndVar);

    // Create the global id calculations.
    llvm::GlobalVariable *LocalIDG =
        F.getParent()->getGlobalVariable(LID_G_NAME(D));
    assert(LocalIDG != nullptr);
    Builder.CreateStore(WIIndVar, LocalIDG);
    Builder.CreateBr(LastHeader);

    auto *Latch =
        llvm::BasicBlock::Create(F.getContext(), "latch." + Suffix + "b", &F);
    Builder.SetInsertPoint(Latch, Latch->getFirstInsertionPt());
    auto *IncIndVar = Builder.CreateAdd(
        WIIndVar, Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), 1),
        "addInd." + Suffix, true, false);
    WIIndVar->addIncoming(IncIndVar, Latch);

    auto *LoopCond =
        Builder.CreateICmpULT(IncIndVar, LocalSize[D], "exit.cond." + Suffix);
    Builder.CreateCondBr(LoopCond, Header, AfterBB);
    Latches.push_back(Latch);
    LastHeader = Header;
  }

  for (size_t D = 1; D < Dim; ++D) {
    Latches[D - 1]->getTerminator()->replaceSuccessorWith(AfterBB, Latches[D]);
    IndVars[D - 1]->replaceIncomingBlockWith(&F.getEntryBlock(),
                                             IndVars[D]->getParent());
  }

  auto *MDWorkItemLoop = llvm::MDNode::get(
      F.getContext(),
      {llvm::MDString::get(F.getContext(), PoclMDKind::WorkItemLoop)});
  auto *LoopID = llvm::makePostTransformationMetadata(F.getContext(), nullptr,
                                                      {}, {MDWorkItemLoop});
  Latches[0]->getTerminator()->setMetadata("llvm.loop", LoopID);
  VMap[AfterBB] = Latches[0];

  // add contiguous ind var calculation to load block
  Builder.SetInsertPoint(IndVars[0]->getParent(), ++IndVars[0]->getIterator());
  llvm::Value *Idx = IndVars[Dim - 1];
  for (size_t D = Dim - 1; D > 0; --D) {
    size_t DD = D - 1;
    const std::string Suffix =
        (llvm::Twine{DimName[DD]} + ".subcfg." + llvm::Twine{EntryId}).str();

    Idx = Builder.CreateMul(Idx, LocalSize[DD], "idx.mul." + Suffix, true);
    Idx = Builder.CreateAdd(IndVars[DD], Idx, "idx.add." + Suffix, true);

    VMap[getLoadForGlobalVariable(F, LocalIdGlobalNames[DD])] = IndVars[DD];
  }

  // todo: replace `ret` with branch to innermost latch

  VMap[getLoadForGlobalVariable(F, LocalIdGlobalNames[Dim - 1])] =
      IndVars[Dim - 1];

  VMap[ContiguousIdx] = Idx;
  ContiguousIdx = Idx;
}


// create new exiting block writing the exit's id to LastBarrierIdStorage_
llvm::BasicBlock *SubCFG::createExitWithID(
    llvm::detail::DenseMapPair<llvm::BasicBlock *, size_t> BarrierPair,
    llvm::BasicBlock *After, llvm::BasicBlock *TargetBB) {
#ifdef DEBUG_SUBCFG_FORMATION
  llvm::errs() << "Create new exit with ID: " << BarrierPair.second << " at "
               << After->getName() << "\n";
#endif

  auto *Exit = llvm::BasicBlock::Create(
      After->getContext(),
      After->getName() + ".subcfg.exit" + llvm::Twine{BarrierPair.second} + "b",
      After->getParent(), TargetBB);

  auto &DL = Exit->getParent()->getParent()->getDataLayout();
  llvm::IRBuilder<> Builder{Exit, Exit->getFirstInsertionPt()};
  Builder.CreateStore(Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(),
                                      BarrierPair.second),
                      LastBarrierIdStorage_);
  Builder.CreateBr(TargetBB);

  After->getTerminator()->replaceSuccessorWith(BarrierPair.first, Exit);
  return Exit;
}

// identify a new SubCFG using DFS starting at EntryBarrier
SubCFG::SubCFG(llvm::BasicBlock *EntryBarrier,
               llvm::AllocaInst *LastBarrierIdStorage,
               const llvm::DenseMap<llvm::BasicBlock *, size_t> &BarrierIds,
               llvm::Value *IndVar, size_t Dim)
    : EntryId_(BarrierIds.lookup(EntryBarrier)), EntryBarrier_(EntryBarrier),
      LastBarrierIdStorage_(LastBarrierIdStorage), ContIdx_(IndVar),
      EntryBB_(EntryBarrier->getSingleSuccessor()), LoadBB_(nullptr),
      PreHeader_(nullptr), Dim(Dim) {
  assert(ContIdx_ && "Must have found _local_id_{x,y,z}");

  llvm::SmallVector<llvm::BasicBlock *, 4> WL{EntryBarrier};
  while (!WL.empty()) {
    auto *BB = WL.pop_back_val();

    llvm::SmallVector<llvm::BasicBlock *, 2> Succs{llvm::succ_begin(BB),
                                                   llvm::succ_end(BB)};
    for (auto *Succ : Succs) {
      if (std::find(Blocks_.begin(), Blocks_.end(), Succ) != Blocks_.end())
        continue;

      if (!Barrier::hasOnlyBarrier(Succ)) {
        WL.push_back(Succ);
        Blocks_.push_back(Succ);
      } else {
        size_t BId = BarrierIds.lookup(Succ);
        assert(BId != 0 && "Exit barrier block not found in map");
        ExitIds_.insert({Succ, BId});
      }
    }
  }
}

void SubCFG::print() const {
#ifdef DEBUG_SUBCFG_FORMATION
  llvm::errs() << "SubCFG entry barrier: " << EntryId_ << "\n";
  llvm::errs() << "SubCFG block names: ";
  for (auto *BB : Blocks_) {
    llvm::errs() << BB->getName() << ", ";
  }
  llvm::errs() << "\n";
  llvm::errs() << "SubCFG exits: ";
  for (auto ExitIt : ExitIds_) {
    llvm::errs() << ExitIt.first->getName() << " (" << ExitIt.second << "), ";
  }
  llvm::errs() << "\n";
  llvm::errs() << "SubCFG new block names: ";
  for (auto *BB : NewBlocks_) {
    llvm::errs() << BB->getName() << ", ";
  }
  llvm::errs() << "\n";
#endif
}

void addRemappedDenseMapKeys(
    const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *>
        &OrgInstAllocaMap,
    const llvm::ValueToValueMapTy &VMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &NewInstAllocaMap) {
  for (auto &InstAllocaPair : OrgInstAllocaMap) {
    if (auto *NewInst = llvm::dyn_cast_or_null<llvm::Instruction>(
            VMap.lookup(InstAllocaPair.first)))
      NewInstAllocaMap.insert({NewInst, InstAllocaPair.second});
  }
}

// clone all BBs of the subcfg, create wi-loop structure around and fixup values
void SubCFG::replicate(
    llvm::Function &F,
    const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *>
        &InstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *,
                   llvm::SmallVector<llvm::Instruction *, 8>>
        &ContInstReplicaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *>
        &RemappedInstAllocaMap,
    llvm::BasicBlock *AfterBB, llvm::ArrayRef<llvm::Value *> LocalSize) {
  llvm::ValueToValueMapTy VMap;

  // clone blocks
  for (auto *BB : Blocks_) {
    auto *NewBB = llvm::CloneBasicBlock(
        BB, VMap, ".subcfg." + llvm::Twine{EntryId_} + "b", &F);
    VMap[BB] = NewBB;
    NewBlocks_.push_back(NewBB);
    for (auto *Succ : llvm::successors(BB)) {
      auto ExitIt = ExitIds_.find(Succ);
      if (ExitIt != ExitIds_.end()) {
        NewBlocks_.push_back(createExitWithID(*ExitIt, NewBB, AfterBB));
      }
    }
  }

  LoadBB_ = createLoadBB(VMap);

  VMap[EntryBarrier_] = LoadBB_;

  llvm::SmallVector<llvm::BasicBlock *, 3> Latches;
  llvm::BasicBlock *LastHeader = LoadBB_;
  llvm::Value *Idx = ContIdx_;

  createLoopsAround(F, AfterBB, LocalSize, EntryId_, VMap, Latches, LastHeader,
                    Idx);

  PreHeader_ = createUniformLoadBB(LastHeader);
  LastHeader->replacePhiUsesWith(&F.getEntryBlock(), PreHeader_);

  print();

  addRemappedDenseMapKeys(InstAllocaMap, VMap, RemappedInstAllocaMap);
  loadMultiSubCfgValues(InstAllocaMap, BaseInstAllocaMap, ContInstReplicaMap,
                        PreHeader_, VMap);
  loadUniformAndRecalcContValues(BaseInstAllocaMap, ContInstReplicaMap,
                                 PreHeader_, VMap);

  llvm::SmallVector<llvm::BasicBlock *, 8> BlocksToRemap{NewBlocks_.begin(),
                                                         NewBlocks_.end()};
  llvm::remapInstructionsInBlocks(BlocksToRemap, VMap);

  removeDeadPhiBlocks(BlocksToRemap);

  EntryBB_ = PreHeader_;
  ExitBB_ = Latches[Dim - 1];
  ContIdx_ = Idx;
}

// remove incoming PHI blocks that no longer actually have an edge to the PHI
void SubCFG::removeDeadPhiBlocks(
    llvm::SmallVector<llvm::BasicBlock *, 8> &BlocksToRemap) const {
  for (auto *BB : BlocksToRemap) {
    llvm::SmallPtrSet<llvm::BasicBlock *, 4> Predecessors{llvm::pred_begin(BB),
                                                          llvm::pred_end(BB)};
    for (auto &I : *BB) {
      if (auto *Phi = llvm::dyn_cast<llvm::PHINode>(&I)) {
        llvm::SmallVector<llvm::BasicBlock *, 4> IncomingBlocksToRemove;
        for (size_t IncomingIdx = 0; IncomingIdx < Phi->getNumIncomingValues();
             ++IncomingIdx) {
          auto *IncomingBB = Phi->getIncomingBlock(IncomingIdx);
          if (Predecessors.find(IncomingBB) == Predecessors.end())
            IncomingBlocksToRemove.push_back(IncomingBB);
        }
        for (auto *IncomingBB : IncomingBlocksToRemove) {
#ifdef DEBUG_SUBCFG_FORMATION
          llvm::errs() << "[SubCFG] Remove incoming block "
                       << IncomingBB->getName() << " from PHI " << *Phi << "\n";
#endif
          Phi->removeIncomingValue(IncomingBB);
#ifdef DEBUG_SUBCFG_FORMATION
          llvm::errs() << "[SubCFG] Removed incoming block "
                       << IncomingBB->getName() << " from PHI " << *Phi << "\n";
#endif
        }
      }
    }
  }
}

// Requires a uniformity analysis that is able to determine contiguous values
#if 0
// check if a contiguous value can be tracked back to only uniform values and
// the wi-loop indvar currently cannot track back the value through PHI nodes.
bool dontArrayifyContiguousValues(
    llvm::Instruction &I,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *,
                   llvm::SmallVector<llvm::Instruction *, 8>>
        &ContInstReplicaMap,
    llvm::Instruction *AllocaIP, llvm::Value* ReqdArrayElements, llvm::Value *IndVar,
    pocl::VariableUniformityAnalysisResult &VecInfo) {
  // is cont indvar
  if (VecInfo.isPinned(I))
    return true;

  llvm::SmallVector<llvm::Instruction *, 4> WL;
  llvm::SmallPtrSet<llvm::Instruction *, 8> UniformValues;
  llvm::SmallVector<llvm::Instruction *, 8> ContiguousInsts;
  llvm::SmallPtrSet<llvm::Value *, 8> LookedAt;
  llvm::errs() << "[SubCFG] IndVar: " << *IndVar << "\n";
  WL.push_back(&I);
  while (!WL.empty()) {
    auto *WLValue = WL.pop_back_val();
    if (auto *WLI = llvm::dyn_cast<llvm::Instruction>(WLValue))
      for (auto *V : WLI->operand_values()) {
        llvm::errs() << "[SubCFG] Considering: " << *V << "\n";

        if (V == IndVar || VecInfo.isPinned(*V))
          continue;
        // todo: fix PHIs
        if (LookedAt.contains(V))
          return false;
        LookedAt.insert(V);

        // collect cont and uniform source values
        if (auto *OpI = llvm::dyn_cast<llvm::Instruction>(V)) {
          if (VecInfo.getVectorShape(*OpI).isContiguous()) {
            WL.push_back(OpI);
            ContiguousInsts.push_back(OpI);
          } else if (!UniformValues.contains(OpI))
            UniformValues.insert(OpI);
        }
      }
  }
  for (auto *UI : UniformValues) {
    llvm::errs() << "[SubCFG] UniValue to store: " << *UI << "\n";
    if (BaseInstAllocaMap.lookup(UI))
      continue;
    llvm::errs()
        << "[SubCFG] Store required uniform value to single element alloca "
        << I << "\n";
    auto *Alloca = arrayifyInstruction(AllocaIP, UI, IndVar, nullptr);
    BaseInstAllocaMap.insert({UI, Alloca});
    VecInfo.setVectorShape(*Alloca, pocl::VectorShape::uni());
  }
  ContInstReplicaMap.insert({&I, ContiguousInsts});
  return true;
}
#endif

// creates array allocas for values that are identified as spanning multiple
// subcfgs
void SubCFG::arrayifyMultiSubCfgValues(
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *,
                   llvm::SmallVector<llvm::Instruction *, 8>>
        &ContInstReplicaMap,
    llvm::ArrayRef<SubCFG> SubCFGs, llvm::Instruction *AllocaIP,
    llvm::Value *ReqdArrayElements,
    pocl::VariableUniformityAnalysisResult &VecInfo) {
  llvm::SmallPtrSet<llvm::BasicBlock *, 16> OtherCFGBlocks;
  for (auto &Cfg : SubCFGs) {
    if (&Cfg != this)
      OtherCFGBlocks.insert(Cfg.Blocks_.begin(), Cfg.Blocks_.end());
  }

  for (auto *BB : Blocks_) {
    for (auto &I : *BB) {
      if (&I == ContIdx_)
        continue;
      if (InstAllocaMap.lookup(&I))
        continue;
      // if any use is in another subcfg
      if (anyOfUsers<llvm::Instruction>(&I, [&OtherCFGBlocks, this,
                                             &I](auto *UI) {
            return UI->getParent() != I.getParent() &&
                   OtherCFGBlocks.find(UI->getParent()) != OtherCFGBlocks.end();
          })) {
        // load from an alloca, just widen alloca
        if (auto *LInst = llvm::dyn_cast<llvm::LoadInst>(&I))
          if (auto *Alloca = getLoopStateAllocaForLoad(*LInst)) {
            InstAllocaMap.insert({&I, Alloca});
            continue;
          }
        // GEP from already widened alloca: reuse alloca
        if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(&I))
          if (GEP->hasMetadata(PoclMDKind::Arrayified)) {
            InstAllocaMap.insert(
                {&I, llvm::cast<llvm::AllocaInst>(GEP->getPointerOperand())});
            continue;
          }

#ifndef CBS_NO_PHIS_IN_SPLIT
        // if value is uniform, just store to 1-wide alloca
        if (VecInfo.isUniform(I.getFunction(), &I)) {
#ifdef DEBUG_SUBCFG_FORMATION
          llvm::errs()
              << "[SubCFG] Value uniform, store to single element alloca " << I
              << "\n";
#endif
          auto *Alloca = arrayifyInstruction(AllocaIP, &I, ContIdx_, nullptr);
          InstAllocaMap.insert({&I, Alloca});
          VecInfo.setUniform(I.getFunction(), Alloca);
          continue;
        }
#endif

// Requires a uniformity analysis that is able to determine contiguous values
#if 0
        // if contiguous, and can be recalculated, don't arrayify but store
        // uniform values and insts required for recalculation
        if (Shape.isContiguous()) {
          if (dontArrayifyContiguousValues(
                  I, BaseInstAllocaMap, ContInstReplicaMap, AllocaIP,
                  ReqdArrayElements, ContIdx_, VecInfo)) {
#ifdef DEBUG_SUBCFG_FORMATION
            llvm::errs() << "[SubCFG] Not arrayifying " << I << "\n";
#endif
            continue;
          }
        }
#endif
        // create wide alloca and store the value
        auto *Alloca =
            arrayifyInstruction(AllocaIP, &I, ContIdx_, ReqdArrayElements);
        InstAllocaMap.insert({&I, Alloca});
      }
    }
  }
}

void remapInstruction(llvm::Instruction *I, llvm::ValueToValueMapTy &VMap) {
  llvm::SmallVector<llvm::Value *, 8> WL{I->value_op_begin(),
                                         I->value_op_end()};
  for (auto *V : WL) {
    if (VMap.count(V))
      I->replaceUsesOfWith(V, VMap[V]);
  }
#ifdef DEBUG_SUBCFG_FORMATION
  llvm::errs() << "[SubCFG] remapped Inst " << *I << "\n";
#endif
}

// inserts loads from the loop state allocas for varying values that were
// identified as multi-subcfg values
void SubCFG::loadMultiSubCfgValues(
    const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *>
        &InstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *,
                   llvm::SmallVector<llvm::Instruction *, 8>>
        &ContInstReplicaMap,
    llvm::BasicBlock *UniformLoadBB, llvm::ValueToValueMapTy &VMap) {
  llvm::Value *NewContIdx = VMap[ContIdx_];
  auto *LoadTerm = LoadBB_->getTerminator();
  auto *UniformLoadTerm = UniformLoadBB->getTerminator();
  llvm::IRBuilder<> Builder{LoadTerm};

  for (auto &InstAllocaPair : InstAllocaMap) {
    // If def not in sub CFG but a use of it is in the sub CFG
    if (std::find(Blocks_.begin(), Blocks_.end(),
                  InstAllocaPair.first->getParent()) == Blocks_.end()) {
      if (anyOfUsers<llvm::Instruction>(
              InstAllocaPair.first, [this](llvm::Instruction *UI) {
                return std::find(NewBlocks_.begin(), NewBlocks_.end(),
                                 UI->getParent()) != NewBlocks_.end();
              })) {
        if (auto *GEP =
                llvm::dyn_cast<llvm::GetElementPtrInst>(InstAllocaPair.first))
          if (auto *MDArrayified = GEP->getMetadata(PoclMDKind::Arrayified)) {
            auto *NewGEP =
                llvm::cast<llvm::GetElementPtrInst>(Builder.CreateInBoundsGEP(
                    GEP->getType(), GEP->getPointerOperand(), NewContIdx,
                    GEP->getName() + "c"));
            NewGEP->setMetadata(PoclMDKind::Arrayified, MDArrayified);
            VMap[InstAllocaPair.first] = NewGEP;
            continue;
          }
        auto *IP = LoadTerm;
        if (!InstAllocaPair.second->isArrayAllocation())
          IP = UniformLoadTerm;
#ifdef DEBUG_SUBCFG_FORMATION
        llvm::errs() << "[SubCFG] Load from Alloca " << *InstAllocaPair.second
                     << " in " << IP->getParent()->getName() << "\n";
#endif
        auto *Load = loadFromAlloca(InstAllocaPair.second, NewContIdx, IP,
                                    InstAllocaPair.first->getName());
        copyDgbValues(InstAllocaPair.first, Load, IP);
        VMap[InstAllocaPair.first] = Load;
      }
    }
  }
}

// Inserts loads for the multi-subcfg values that were identified as uniform
// inside the wi-loop preheader. Additionally clones the instructions that were
// identified as contiguous \a ContInstReplicaMap inside the LoadBB_ to restore
// the contiguous value just from the uniform values and the wi-idx.
void SubCFG::loadUniformAndRecalcContValues(
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *,
                   llvm::SmallVector<llvm::Instruction *, 8>>
        &ContInstReplicaMap,
    llvm::BasicBlock *UniformLoadBB, llvm::ValueToValueMapTy &VMap) {
  llvm::ValueToValueMapTy UniVMap;
  auto *LoadTerm = LoadBB_->getTerminator();
  auto *UniformLoadTerm = UniformLoadBB->getTerminator();
  llvm::Value *NewContIdx = VMap[this->ContIdx_];
  UniVMap[this->ContIdx_] = NewContIdx;

  // copy local id load value to univmap
  for (size_t D = 0; D < this->Dim; ++D) {
    auto *Load = getLoadForGlobalVariable(*this->LoadBB_->getParent(),
                                          LocalIdGlobalNames[D]);
    UniVMap[Load] = VMap[Load];
  }

  // load uniform values from allocas
  for (auto &InstAllocaPair : BaseInstAllocaMap) {
    auto *IP = UniformLoadTerm;
#ifdef DEBUG_SUBCFG_FORMATION
    llvm::errs() << "[SubCFG] Load base value from Alloca "
                 << *InstAllocaPair.second << " in "
                 << IP->getParent()->getName() << "\n";
#endif
    auto *Load = loadFromAlloca(InstAllocaPair.second, NewContIdx, IP,
                                InstAllocaPair.first->getName());
    copyDgbValues(InstAllocaPair.first, Load, IP);
    UniVMap[InstAllocaPair.first] = Load;
  }

  // get a set of unique contiguous instructions
  llvm::SmallPtrSet<llvm::Instruction *, 16> UniquifyInsts;
  for (auto &Pair : ContInstReplicaMap) {
    UniquifyInsts.insert(Pair.first);
    for (auto &Target : Pair.second)
      UniquifyInsts.insert(Target);
  }

  auto OrderedInsts = topoSortInstructions(UniquifyInsts);

  llvm::SmallPtrSet<llvm::Instruction *, 16> InstsToRemap;
  // clone the contiguous instructions to restore the used values
  for (auto *I : OrderedInsts) {
    if (UniVMap.count(I))
      continue;

#ifdef DEBUG_SUBCFG_FORMATION
    llvm::errs() << "[SubCFG] Clone cont instruction and operands of: " << *I
                 << " to " << LoadTerm->getParent()->getName() << "\n";
#endif
    auto *IClone = I->clone();
    IClone->insertBefore(LoadTerm);
    InstsToRemap.insert(IClone);
    UniVMap[I] = IClone;
    if (VMap.count(I) == 0)
      VMap[I] = IClone;
#ifdef DEBUG_SUBCFG_FORMATION
    llvm::errs() << "[SubCFG] Clone cont instruction: " << *IClone << "\n";
#endif
  }

  // finally remap the singular instructions to use the other cloned contiguous
  // instructions / uniform values
  for (auto *IToRemap : InstsToRemap)
    remapInstruction(IToRemap, UniVMap);
}
llvm::SmallVector<llvm::Instruction *, 16> SubCFG::topoSortInstructions(
    const llvm::SmallPtrSet<llvm::Instruction *, 16> &UniquifyInsts) const {
  llvm::SmallVector<llvm::Instruction *, 16> OrderedInsts(UniquifyInsts.size());
  std::copy(UniquifyInsts.begin(), UniquifyInsts.end(), OrderedInsts.begin());

  auto IsUsedBy = [](llvm::Instruction *LHS, llvm::Instruction *RHS) {
    for (auto *U : LHS->users()) {
      if (U == RHS)
        return true;
    }
    return false;
  };
  for (size_t I = 0; I < OrderedInsts.size(); ++I) {
    size_t InsertAt = I;
    for (size_t J = OrderedInsts.size() - 1; J > I; --J) {
      if (IsUsedBy(OrderedInsts[J], OrderedInsts[I])) {
        InsertAt = J;
        break;
      }
    }
    if (InsertAt != I) {
      auto *Tmp = OrderedInsts[I];
      for (size_t J = I + 1; J <= InsertAt; ++J) {
        OrderedInsts[J - 1] = OrderedInsts[J];
      }
      OrderedInsts[InsertAt] = Tmp;
      --I;
    }
  }
  return OrderedInsts;
}

llvm::BasicBlock *
SubCFG::createUniformLoadBB(llvm::BasicBlock *OuterMostHeader) {
  auto *LoadBB = llvm::BasicBlock::Create(
      OuterMostHeader->getContext(),
      "uniloadblock.subcfg." + llvm::Twine{EntryId_} + "b",
      OuterMostHeader->getParent(), OuterMostHeader);
  llvm::IRBuilder<> Builder{LoadBB, LoadBB->getFirstInsertionPt()};
  Builder.CreateBr(OuterMostHeader);
  return LoadBB;
}

llvm::BasicBlock *SubCFG::createLoadBB(llvm::ValueToValueMapTy &VMap) {
  auto *NewEntry =
      llvm::cast<llvm::BasicBlock>(static_cast<llvm::Value *>(VMap[EntryBB_]));
  auto *LoadBB = llvm::BasicBlock::Create(
      NewEntry->getContext(), "loadblock.subcfg." + llvm::Twine{EntryId_} + "b",
      NewEntry->getParent(), NewEntry);
  llvm::IRBuilder<> Builder{LoadBB, LoadBB->getFirstInsertionPt()};
  Builder.CreateBr(NewEntry);
  return LoadBB;
}

// if the kernel contained a loop, it is possible, that values inside a single
// subcfg don't dominate their uses inside the same subcfg. This function
// identifies and fixes those values.
void SubCFG::fixSingleSubCfgValues(
    llvm::DominatorTree &DT,
    const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *>
        &RemappedInstAllocaMap,
    llvm::Value *ReqdArrayElements,
    pocl::VariableUniformityAnalysisResult &VecInfo) {

  auto *AllocaIP = LoadBB_->getParent()->getEntryBlock().getTerminator();
  auto *LoadIP = LoadBB_->getTerminator();
  auto *UniLoadIP = PreHeader_->getTerminator();
  llvm::IRBuilder<> Builder{LoadIP};

  llvm::DenseMap<llvm::Instruction *, llvm::Instruction *> InstLoadMap;

  for (auto *BB : NewBlocks_) {
    llvm::SmallVector<llvm::Instruction *, 16> Insts{};
    std::transform(BB->begin(), BB->end(), std::back_inserter(Insts),
                   [](auto &I) { return &I; });
    for (auto *Inst : Insts) {
      auto &I = *Inst;
      for (auto *OPV : I.operand_values()) {
        // check if all operands dominate the instruction -> otherwise we have
        // to fix it
        auto *OPI = llvm::dyn_cast<llvm::Instruction>(OPV);
        if (OPI && !DT.dominates(OPI, &I)) {
          if (auto *Phi = llvm::dyn_cast<llvm::PHINode>(Inst)) {
            // if a PHI node, we have to check that the incoming values dominate
            // the terminators of the incoming block..
            bool FoundIncoming = false;
            for (auto &Incoming : Phi->incoming_values()) {
              if (OPV == Incoming.get()) {
                auto *IncomingBB = Phi->getIncomingBlock(Incoming);
                if (DT.dominates(OPI, IncomingBB->getTerminator())) {
                  FoundIncoming = true;
                  break;
                }
              }
            }
            if (FoundIncoming)
              continue;
          }
#ifdef DEBUG_SUBCFG_FORMATION
          llvm::errs() << "Instruction not dominated " << I
                       << " operand: " << *OPI << "\n";
#endif

          if (auto *Load = InstLoadMap.lookup(OPI))
            // if the already inserted Load does not dominate I, we must create
            // another load.
            if (DT.dominates(Load, &I)) {
              I.replaceUsesOfWith(OPI, Load);
              continue;
            }

          if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(OPI))
            if (auto *MDArrayified =
                    GEP->getMetadata(PoclMDKind::Arrayified)) {
              auto *NewGEP =
                  llvm::cast<llvm::GetElementPtrInst>(Builder.CreateInBoundsGEP(
                      GEP->getType(), GEP->getPointerOperand(), ContIdx_,
                      GEP->getName() + "c"));
              NewGEP->setMetadata(PoclMDKind::Arrayified, MDArrayified);
              I.replaceUsesOfWith(OPI, NewGEP);
              InstLoadMap.insert({OPI, NewGEP});
              continue;
            }

          llvm::AllocaInst *Alloca = nullptr;
          if (auto *RemAlloca = RemappedInstAllocaMap.lookup(OPI))
            Alloca = RemAlloca;
          if (auto *LInst = llvm::dyn_cast<llvm::LoadInst>(OPI))
            Alloca = getLoopStateAllocaForLoad(*LInst);
          if (!Alloca) {
#ifdef DEBUG_SUBCFG_FORMATION
            llvm::errs() << "[SubCFG] No alloca, yet for " << *OPI << "\n";
#endif
            Alloca =
                arrayifyInstruction(AllocaIP, OPI, ContIdx_, ReqdArrayElements);
          }

#ifdef CBS_NO_PHIS_IN_SPLIT
          // in split loop, OPI might be used multiple times, get the user,
          // dominating this user and insert load there
          llvm::Instruction *NewIP = &I;
          for (auto *U : OPI->users()) {
            if (auto *UI = llvm::dyn_cast<llvm::Instruction>(U);
                UI && DT.dominates(UI, NewIP)) {
              NewIP = UI;
            }
          }
#else
          // doesn't happen if we keep the PHIs
          auto *NewIP = LoadIP;
          if (!Alloca->isArrayAllocation())
            NewIP = UniLoadIP;
#endif

          auto *Load = loadFromAlloca(Alloca, ContIdx_, NewIP, OPI->getName());
          copyDgbValues(OPI, Load, NewIP);

#ifdef CBS_NO_PHIS_IN_SPLIT
          I.replaceUsesOfWith(OPI, Load);
          InstLoadMap.insert({OPI, Load});
#else
          // if a loop is conditionally split, the first block in a subcfg might
          // have another incoming edge, need to insert a PHI node then
          const auto NumPreds =
              std::distance(llvm::pred_begin(BB), llvm::pred_end(BB));
          if (!llvm::isa<llvm::PHINode>(I) && NumPreds > 1 &&
              std::find(llvm::pred_begin(BB), llvm::pred_end(BB), LoadBB_) !=
                  llvm::pred_end(BB)) {
            Builder.SetInsertPoint(BB, BB->getFirstInsertionPt());
            auto *PHINode =
                Builder.CreatePHI(Load->getType(), NumPreds, I.getName());
            for (auto *PredBB : llvm::predecessors(BB))
              if (PredBB == LoadBB_)
                PHINode->addIncoming(Load, PredBB);
              else
                PHINode->addIncoming(OPV, PredBB);

            I.replaceUsesOfWith(OPI, PHINode);
            InstLoadMap.insert({OPI, PHINode});
          } else {
            I.replaceUsesOfWith(OPI, Load);
            InstLoadMap.insert({OPI, Load});
          }
#endif
        }
      }
    }
  }
}

llvm::BasicBlock *createUnreachableBlock(llvm::Function &F) {
  auto *Default =
      llvm::BasicBlock::Create(F.getContext(), "cbs.while.default", &F);
  llvm::IRBuilder<> Builder{Default, Default->getFirstInsertionPt()};
  Builder.CreateUnreachable();
  return Default;
}

// create the actual while loop around the subcfgs and the switch instruction to
// select the next subCFG based on the value in \a LastBarrierIdStorage
llvm::BasicBlock *
generateWhileSwitchAround(llvm::BasicBlock *PreHeader,
                          llvm::BasicBlock *OldEntry, llvm::BasicBlock *Exit,
                          llvm::AllocaInst *LastBarrierIdStorage,
                          std::vector<SubCFG> &SubCFGs) {
  auto &F = *PreHeader->getParent();
  auto &M = *F.getParent();
  const auto &DL = M.getDataLayout();

  auto *WhileHeader =
      llvm::BasicBlock::Create(PreHeader->getContext(), "cbs.while.header",
                               PreHeader->getParent(), OldEntry);
  llvm::IRBuilder<> Builder{WhileHeader, WhileHeader->getFirstInsertionPt()};
  auto *LastID =
      Builder.CreateLoad(LastBarrierIdStorage->getAllocatedType(),
                         LastBarrierIdStorage, "cbs.while.last_barr.load");
  auto *Switch =
      Builder.CreateSwitch(LastID, createUnreachableBlock(F), SubCFGs.size());
  for (auto &Cfg : SubCFGs) {
    Switch->addCase(Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(),
                                    Cfg.getEntryId()),
                    Cfg.getEntry());
    Cfg.getEntry()->replacePhiUsesWith(PreHeader, WhileHeader);
    Cfg.getExit()->getTerminator()->replaceSuccessorWith(Exit, WhileHeader);
  }
  Switch->addCase(
      Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), ExitBarrierId),
      Exit);

  Builder.SetInsertPoint(PreHeader->getTerminator());
  Builder.CreateStore(
      llvm::ConstantInt::get(LastBarrierIdStorage->getAllocatedType(),
                             EntryBarrierId),
      LastBarrierIdStorage);
  PreHeader->getTerminator()->replaceSuccessorWith(OldEntry, WhileHeader);
  return WhileHeader;
}

// drops all lifetime intrinsics - they are misinforming ASAN otherwise (and are
// not really fixable at the right scope..)
void purgeLifetime(SubCFG &Cfg) {
  llvm::SmallVector<llvm::Instruction *, 8> ToDelete;
  for (auto *BB : Cfg.getNewBlocks())
    for (auto &I : *BB)
      if (auto *CI = llvm::dyn_cast<llvm::CallInst>(&I))
        if (CI->getCalledFunction())
          if (CI->getCalledFunction()->getIntrinsicID() ==
                  llvm::Intrinsic::lifetime_start ||
              CI->getCalledFunction()->getIntrinsicID() ==
                  llvm::Intrinsic::lifetime_end)
            ToDelete.push_back(CI);

  for (auto *I : ToDelete)
    I->eraseFromParent();
}

// fills \a Hull with all transitive users of \a Alloca
void fillUserHull(llvm::AllocaInst *Alloca,
                  llvm::SmallVectorImpl<llvm::Instruction *> &Hull) {
  llvm::SmallVector<llvm::Instruction *, 8> WL;
  std::transform(Alloca->user_begin(), Alloca->user_end(),
                 std::back_inserter(WL),
                 [](auto *U) { return llvm::cast<llvm::Instruction>(U); });
  llvm::SmallPtrSet<llvm::Instruction *, 32> AlreadySeen;
  while (!WL.empty()) {
    auto *I = WL.pop_back_val();
    AlreadySeen.insert(I);
    Hull.push_back(I);
    for (auto *U : I->users()) {
      if (auto *UI = llvm::dyn_cast<llvm::Instruction>(U)) {
        if (AlreadySeen.find(UI) == AlreadySeen.end())
          if (UI->mayReadOrWriteMemory() || UI->getType()->isPointerTy())
            WL.push_back(UI);
      }
    }
  }
}

template <class PtrSet> struct PtrSetWrapper {
  explicit PtrSetWrapper(PtrSet &PtrSetArg) : Set(PtrSetArg) {}
  PtrSet &Set;
  using iterator = typename PtrSet::iterator;
  using value_type = typename PtrSet::value_type;
  template <class IT, class ValueT> IT insert(IT, const ValueT &Value) {
    return Set.insert(Value).first;
  }
};

// checks if all uses of an alloca are in just a single subcfg (doesn't have to
// be arrayified!).
// TO CLEAN: merge with WorkitemLoopsImpl::shouldNotBeContextSaved().
bool isAllocaSubCfgInternal(llvm::AllocaInst *Alloca,
                            const std::vector<SubCFG> &SubCfgs,
                            const llvm::DominatorTree &DT) {
  llvm::SmallPtrSet<llvm::BasicBlock *, 16> UserBlocks;
  {
    llvm::SmallVector<llvm::Instruction *, 32> Users;
    fillUserHull(Alloca, Users);
    PtrSetWrapper<decltype(UserBlocks)> Wrapper{UserBlocks};
    std::transform(Users.begin(), Users.end(),
                   std::inserter(Wrapper, UserBlocks.end()),
                   [](auto *I) { return I->getParent(); });
  }

  for (auto &SubCfg : SubCfgs) {
    llvm::SmallPtrSet<llvm::BasicBlock *, 8> SubCfgSet{
        SubCfg.getNewBlocks().begin(), SubCfg.getNewBlocks().end()};
    if (std::any_of(UserBlocks.begin(), UserBlocks.end(),
                    [&SubCfgSet](auto *BB) {
                      return SubCfgSet.find(BB) != SubCfgSet.end();
                    }) &&
        !std::all_of(UserBlocks.begin(), UserBlocks.end(),
                     [&SubCfgSet, Alloca](auto *BB) {
                       if (SubCfgSet.find(BB) != SubCfgSet.end()) {
                         return true;
                       }
#ifdef DEBUG_SUBCFG_FORMATION
                       llvm::errs()
                           << "[SubCFG] BB not in subcfgset: " << BB->getName()
                           << " for alloca: ";
                       Alloca->print(llvm::outs());
                       llvm::outs() << "\n";
#endif
                       return false;
                     }))
      return false;
  }

  return true;
}

// Widens the allocas in the entry block to array allocas.
// Replace uses of the original alloca with a GEP that indexes the new alloca
// with
// \a Idx.
void SubCFGFormation::arrayifyAllocas(
    llvm::BasicBlock *EntryBlock, llvm::DominatorTree &DT,
    std::vector<SubCFG> &SubCfgs, llvm::Value *ReqdArrayElements,
    pocl::VariableUniformityAnalysisResult &VecInfo) {
  auto *MDAlloca = llvm::MDNode::get(
      EntryBlock->getContext(),
      {llvm::MDString::get(EntryBlock->getContext(), "poclLoopState")});

  llvm::SmallPtrSet<llvm::BasicBlock *, 32> SubCfgsBlocks;
  for (auto &SubCfg : SubCfgs)
    SubCfgsBlocks.insert(SubCfg.getNewBlocks().begin(),
                         SubCfg.getNewBlocks().end());

  llvm::SmallVector<llvm::AllocaInst *, 8> WL;
  for (auto &I : *EntryBlock) {
    if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
      if (Alloca->hasMetadata(PoclMDKind::Arrayified))
        continue; // already arrayified
      if (anyOfUsers<llvm::Instruction>(Alloca, [&SubCfgsBlocks](
                                                    llvm::Instruction *UI) {
            return SubCfgsBlocks.find(UI->getParent()) == SubCfgsBlocks.end();
          }))
        continue;
      if (!isAllocaSubCfgInternal(Alloca, SubCfgs, DT))
        WL.push_back(Alloca);
    }
  }

  BasicBlock &Entry = K->getEntryBlock();
  for (auto *I : WL) {
    bool PaddingAdded = false;
    llvm::AllocaInst *Alloca = createAlignedAndPaddedContextAlloca(
        I, &*Entry.getFirstInsertionPt(), std::string(I->getName()) + "_alloca",
        PaddingAdded);
    Alloca->setMetadata(PoclMDKind::Arrayified, MDAlloca);

    for (auto &SubCfg : SubCfgs) {
      auto *Before = SubCfg.getLoadBB()->getFirstNonPHIOrDbgOrLifetime();

      auto GEP = createContextArrayGEP(Alloca, Before, PaddingAdded);
      GEP->setMetadata(PoclMDKind::Arrayified, MDAlloca);

      llvm::replaceDominatedUsesWith(I, GEP, DT, SubCfg.getLoadBB());
    }
    I->eraseFromParent();
  }
}

llvm::Value *
SubCFGFormation::getLinearWIIndexInRegion(llvm::Instruction *Instr) {
  SubCFG *Region = regionOfBlock(Instr->getParent());
  assert(Region != nullptr);
  return Region->getContiguousIdx();
}

llvm::Value *SubCFGFormation::getLocalIdInRegion(llvm::Instruction *Instr,
                                                 size_t Dim) {
  SubCFG *Region = regionOfBlock(Instr->getParent());

  std::string VarName = LID_G_NAME(Dim);
  // Find a load in the region load block to ensure it's defined before the
  // referred instruction.
  BasicBlock *LoadBB = Region->getLoadBB();
  auto *GV = K->getParent()->getOrInsertGlobal(VarName, ST);

  for (auto &I : *LoadBB) {
    if (auto *LoadI = llvm::dyn_cast<llvm::LoadInst>(&I)) {
      if (LoadI->getPointerOperand() == GV)
        return &I;
    }
  }
  llvm::IRBuilder<> Builder(LoadBB->getFirstNonPHI());
  return Builder.CreateLoad(ST, GV);
}

void moveAllocasToEntry(llvm::Function &F,
                        llvm::ArrayRef<llvm::BasicBlock *> Blocks) {
  llvm::SmallVector<llvm::AllocaInst *, 4> AllocaWL;
  for (auto *BB : Blocks)
    for (auto &I : *BB)
      if (auto *AllocaInst = llvm::dyn_cast<llvm::AllocaInst>(&I))
        AllocaWL.push_back(AllocaInst);
  for (auto *I : AllocaWL)
    I->moveBefore(F.getEntryBlock().getTerminator());
}

llvm::DenseMap<llvm::BasicBlock *, size_t>
getBarrierIds(llvm::BasicBlock *Entry,
              llvm::SmallPtrSetImpl<llvm::BasicBlock *> &ExitingBlocks,
              llvm::ArrayRef<llvm::BasicBlock *> Blocks) {
  llvm::DenseMap<llvm::BasicBlock *, size_t> Barriers;
  // mark exit barrier with the corresponding id:
  for (auto *BB : ExitingBlocks)
    Barriers[BB] = ExitBarrierId;
  // mark entry barrier with the corresponding id:
  Barriers[Entry] = EntryBarrierId;

  // store all other barrier blocks with a unique id:
  size_t BarrierId = 1;
  for (auto *BB : Blocks)
    if (Barriers.find(BB) == Barriers.end() && Barrier::hasOnlyBarrier(BB))
      Barriers.insert({BB, BarrierId++});
  return Barriers;
}

void SubCFGFormation::formSubCfgs(llvm::Function &F, llvm::LoopInfo &LI,
                                  llvm::DominatorTree &DT,
                                  llvm::PostDominatorTree &PDT,
                                  pocl::VariableUniformityAnalysisResult &VUA) {
#ifdef DEBUG_SUBCFG_FORMATION
  F.viewCFG();
#endif

  std::array<unsigned long, 3> LocalSizes;
  getModuleIntMetadata(*F.getParent(), "WGLocalSizeX", LocalSizes[0]);
  getModuleIntMetadata(*F.getParent(), "WGLocalSizeY", LocalSizes[1]);
  getModuleIntMetadata(*F.getParent(), "WGLocalSizeZ", LocalSizes[2]);
  bool WGDynamicLocalSize{};
  getModuleBoolMetadata(*F.getParent(), "WGDynamicLocalSize",
                        WGDynamicLocalSize);

  std::size_t Dim = 3;
  if (LocalSizes[2] == 1 && !WGDynamicLocalSize) {
    if (LocalSizes[1] == 1)
      Dim = 1;
    else
      Dim = 2;
  }

  const auto LocalSize =
      getLocalSizeValues(F, LocalSizes, WGDynamicLocalSize, Dim);

  auto *Entry = &F.getEntryBlock();

  insertLocalIdInit(Entry);
  llvm::IRBuilder<> Builder{Entry->getTerminator()};
  llvm::Value *ReqdArrayElements =
      WGDynamicLocalSize
          ? Builder.CreateMul(LocalSize[0],
                              Builder.CreateMul(LocalSize[1], LocalSize[2]))
          : Builder.getInt32(std::accumulate(LocalSizes.cbegin(),
                                             LocalSizes.cend(), 1,
                                             std::multiplies<>{}));

  std::vector<llvm::BasicBlock *> Blocks;
  Blocks.reserve(std::distance(F.begin(), F.end()));
  std::transform(F.begin(), F.end(), std::back_inserter(Blocks),
                 [](auto &BB) { return &BB; });

  // non-entry block Allocas are considered broken, move to entry.
  // TODO: Unify with the AllocasToEntry pass. Perhaps convert to
  // a WorkitemHandler helper function.
  moveAllocasToEntry(F, Blocks);

// kept for simple reenabling of more advanced uniformity analysis
#if 0
  auto RImpl = getRegion(F, LI, Blocks);
  pocl::Region R{*RImpl};
  auto VecInfo = getVectorizationInfo(F, R, LI, DT, PDT, Dim);
#endif

  llvm::SmallPtrSet<llvm::BasicBlock *, 2> ExitingBlocks;
  for (auto *BB : Blocks) {
    if (BB->getTerminator()->getNumSuccessors() == 0)
      ExitingBlocks.insert(BB);
  }

  if (ExitingBlocks.empty()) {
    llvm::errs() << "[SubCFG] Invalid kernel! No kernel exits!\n";
    llvm_unreachable("[SubCFG] Invalid kernel! No kernel exits!\n");
  }

  auto Barriers = getBarrierIds(Entry, ExitingBlocks, Blocks);

  const llvm::DataLayout &DL = F.getParent()->getDataLayout();
  auto *LastBarrierIdStorage = Builder.CreateAlloca(
      DL.getLargestLegalIntType(F.getContext()), nullptr, "LastBarrierId");

  // get a common (pseudo) index value to be replaced by the actual index later
  Builder.SetInsertPoint(F.getEntryBlock().getTerminator());
  auto *IndVarT =
      getLoadForGlobalVariable(F, LocalIdGlobalNames[Dim - 1])->getType();
  llvm::Instruction *IndVar = Builder.CreateLoad(
      IndVarT, llvm::UndefValue::get(llvm::PointerType::get(IndVarT, 0)));
  // kept for simple reenabling of more advanced uniformity analysis
#if 0
  VecInfo.setPinnedShape(*IndVar, pocl::VectorShape::cont());
#endif

  SubCFGs.clear();
  for (auto &BIt : Barriers) {
#ifdef DEBUG_SUBCFG_FORMATION
    llvm::errs() << "Create SubCFG from " << BIt.first->getName() << "("
                 << BIt.first << ") id: " << BIt.second << "\n";
#endif
    if (BIt.second != ExitBarrierId)
      SubCFGs.emplace_back(BIt.first, LastBarrierIdStorage, Barriers, IndVar,
                           Dim);
  }

  llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> InstAllocaMap;
  llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> BaseInstAllocaMap;
  llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
      InstContReplicaMap;

  for (auto &Cfg : SubCFGs)
    Cfg.arrayifyMultiSubCfgValues(
        InstAllocaMap, BaseInstAllocaMap, InstContReplicaMap, SubCFGs,
        F.getEntryBlock().getTerminator(), ReqdArrayElements, VUA);

  llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> RemappedInstAllocaMap;
  for (auto &Cfg : SubCFGs) {
    Cfg.print();
    Cfg.replicate(F, InstAllocaMap, BaseInstAllocaMap, InstContReplicaMap,
                  RemappedInstAllocaMap, *ExitingBlocks.begin(), LocalSize);
    purgeLifetime(Cfg);
  }

  llvm::BasicBlock *WhileHeader = nullptr;
  WhileHeader = generateWhileSwitchAround(
      &F.getEntryBlock(), F.getEntryBlock().getSingleSuccessor(),
      *ExitingBlocks.begin(), LastBarrierIdStorage, SubCFGs);

  llvm::removeUnreachableBlocks(F);

  DT.recalculate(F);
  arrayifyAllocas(&F.getEntryBlock(), DT, SubCFGs, ReqdArrayElements, VUA);

  for (auto &Cfg : SubCFGs) {
    Cfg.fixSingleSubCfgValues(DT, RemappedInstAllocaMap, ReqdArrayElements,
                              VUA);
  }

  IndVar->eraseFromParent();

#ifdef DEBUG_SUBCFG_FORMATION
  F.viewCFG();
#endif
  assert(!llvm::verifyFunction(F, &llvm::errs()) &&
         "Function verification failed");

  // simplify while loop to get single latch that isn't marked as wi-loop to
  // prevent misunderstandings.
  auto *WhileLoop = updateDtAndLi(LI, DT, WhileHeader, F);
  llvm::simplifyLoop(WhileLoop, &DT, &LI, nullptr, nullptr, nullptr, false);
}

// Finds the SubCFG in the currently found SubCFGs which has the given BB.
SubCFG *SubCFGFormation::regionOfBlock(llvm::BasicBlock *BB) {
  for (auto &Region : SubCFGs) {
    if (Region.getLoadBB() == BB)
      return &Region;
    SubCFG::BlockVector &BlocksInRegion = Region.getBlocks();
    for (auto *Block : BlocksInRegion)
      if (Block == BB)
        return &Region;
  }
  return nullptr;
}

void createParallelAccessesMdOrAddAccessGroup(const llvm::Function *F,
                                              llvm::Loop *const &L,
                                              llvm::MDNode *MDAccessGroup) {
  // findOptionMDForLoopID also checks if there's a loop id, so this is fine
  if (auto *ParAccesses = llvm::findOptionMDForLoopID(
          L->getLoopID(), "llvm.loop.parallel_accesses")) {
    llvm::SmallVector<llvm::Metadata *, 4> AccessGroups{
        ParAccesses->op_begin(),
        ParAccesses->op_end()}; // contains .parallel_accesses
    AccessGroups.push_back(MDAccessGroup);
    auto *NewParAccesses = llvm::MDNode::get(F->getContext(), AccessGroups);

    const auto *const PIt = std::find(L->getLoopID()->op_begin(),
                                      L->getLoopID()->op_end(), ParAccesses);
    auto PIdx = std::distance(L->getLoopID()->op_begin(), PIt);
    L->getLoopID()->replaceOperandWith(PIdx, NewParAccesses);
  } else {
    auto *NewParAccesses = llvm::MDNode::get(
        F->getContext(),
        {llvm::MDString::get(F->getContext(), "llvm.loop.parallel_accesses"),
         MDAccessGroup});
    L->setLoopID(llvm::makePostTransformationMetadata(
        F->getContext(), L->getLoopID(), {}, {NewParAccesses}));
  }
}

void addAccessGroupMD(llvm::Instruction *I, llvm::MDNode *MDAccessGroup) {
  if (auto *PresentMD = I->getMetadata(llvm::LLVMContext::MD_access_group)) {
    llvm::SmallVector<llvm::Metadata *, 4> MDs;
    if (PresentMD->getNumOperands() == 0)
      MDs.push_back(PresentMD);
    else
      MDs.append(PresentMD->op_begin(), PresentMD->op_end());
    MDs.push_back(MDAccessGroup);
    auto *CombinedMDAccessGroup =
        llvm::MDNode::getDistinct(I->getContext(), MDs);
    I->setMetadata(llvm::LLVMContext::MD_access_group, CombinedMDAccessGroup);
  } else
    I->setMetadata(llvm::LLVMContext::MD_access_group, MDAccessGroup);
}

void markLoopParallel(llvm::Function &F, llvm::Loop *L) {
  // LLVM < 12.0.1 might miscompile if conditionals in "parallel" loop
  // (https://llvm.org/PR46666)

  // Mark memory accesses with access group
  auto *MDAccessGroup = llvm::MDNode::getDistinct(F.getContext(), {});
  for (auto *BB : L->blocks()) {
    for (auto &I : *BB) {
      if (I.mayReadOrWriteMemory() &&
          !I.hasMetadata(llvm::LLVMContext::MD_access_group)) {
        addAccessGroupMD(&I, MDAccessGroup);
      }
    }
  }

  // make the access group parallel w.r.t the WI loop
  createParallelAccessesMdOrAddAccessGroup(&F, L, MDAccessGroup);
}

// enable new pass manager infrastructure
llvm::PreservedAnalyses
SubCFGFormation::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  if (!isKernelToProcess(F))
    return PreservedAnalyses::all();

  WorkitemHandlerType WIH = AM.getResult<pocl::WorkitemHandlerChooser>(F).WIH;
  if (WIH != WorkitemHandlerType::CBS)
    return PreservedAnalyses::all();

  if (!hasWorkgroupBarriers(F))
    return PreservedAnalyses::all();

  Initialize(cast<pocl::Kernel>(&F));

#ifdef DEBUG_SUBCFG_FORMATION
  llvm::errs() << "[SubCFG] Form SubCFGs in " << F.getName() << "\n";
#endif

  auto &DT = AM.getResult<llvm::DominatorTreeAnalysis>(F);
  auto &PDT = AM.getResult<llvm::PostDominatorTreeAnalysis>(F);
  auto &LI = AM.getResult<llvm::LoopAnalysis>(F);
  auto &VUA = AM.getResult<pocl::VariableUniformityAnalysis>(F);

  formSubCfgs(F, LI, DT, PDT, VUA);

  if (canAnnotateParallelLoops())
    for (auto *SL : LI.getLoopsInPreorder())
      if (llvm::findOptionMDForLoop(SL, PoclMDKind::WorkItemLoop))
        markLoopParallel(F, SL);

  handleLocalMemAllocas();

  GenerateGlobalIdComputation();
  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();
  return PAChanged;
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
