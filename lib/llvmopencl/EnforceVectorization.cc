// LLVM function pass to create loops that run all the work items
// in a work group while respecting barrier synchronization points.
//
// Copyright (c) 2012-2019 Pekka Jääskeläinen / Tampere University
//               2022-2025 Pekka Jääskeläinen / Intel Finland Oy
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
#include <llvm/ADT/Statistic.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/ValueSymbolTable.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/IR/PatternMatch.h>

#include "Kernel.h"
#include "KernelCompilerUtils.h"
#include "LLVMUtils.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "WorkitemHandlerChooser.h"
#include "EnforceVectorization.h"

POP_COMPILER_DIAGS

#include <array>
#include <map>
#include <vector>

#define DEBUG_TYPE "WIL"

#define PASS_NAME "enforce-vectorization"
#define PASS_CLASS pocl::EnforceVectorization
#define PASS_DESC "Forces vectorization through a simple walk over the data flow graph"

//#define DEBUG_WORK_ITEM_LOOPS
//#define POCL_KERNEL_COMPILER_DUMP_CFGS

#define HAS_VALUE_OUTPUT(x) (x.valueOutput != nullptr)
#define HAS_ARRAY_OUTPUT(x) (!x.arrayOutput.empty())

namespace pocl {

using namespace llvm;

class EnforceVectorizationImpl : public pocl::WorkitemHandler {
public:
  EnforceVectorizationImpl(llvm::DominatorTree &DT, llvm::LoopInfo &LI,
                    llvm::PostDominatorTree &PDT,
                    VariableUniformityAnalysisResult &VUA)
      : WorkitemHandler(), DT(DT), LI(LI), PDT(PDT), VUA(VUA) {}
  virtual bool runOnFunction(llvm::Function &F);

// protected:
  // llvm::Value *getLinearWIIndexInRegion(llvm::Instruction *Instr) override;
  // llvm::Instruction *getLocalIdInRegion(llvm::Instruction *Instr,
  //                                       size_t Dim) override;

private:
  struct VectorizationTarget {
    llvm::Instruction* target;
    std::vector<llvm::Instruction*> replacers;
    llvm::Value *valueOutput = nullptr;
    std::vector<llvm::Value *> arrayOutput;
    std::vector<llvm::Instruction *> children;
  };

  using BasicBlockVector = std::vector<llvm::BasicBlock *>;
  using InstructionIndex = std::set<llvm::Instruction *>;
  using InstructionVec = std::vector<llvm::Instruction *>;
  using StrInstructionMap = std::map<std::string, llvm::AllocaInst *>;

  llvm::DominatorTree &DT;
  llvm::LoopInfo &LI;
  llvm::PostDominatorTree &PDT;
  VariableUniformityAnalysisResult &VUA;
  llvm::Module *M;
  llvm::Function *F;

  llvm::GlobalVariable *VectorizedGlobalIdVar;
  llvm::GlobalVariable *VectorizedLocalIdVar;
  int GangSize;
  int VectorizationDim;
  llvm::Type *VectorST;
  llvm::Constant *SequentialVec;
  llvm::Constant *StepVec;
  // first is the load inst, second is the (vectorized) replacement value
  std::vector<std::pair<llvm::Instruction *, llvm::Value *>> LoadInsts;

  std::array<llvm::GlobalVariable *, 3> GlobalIdIterators;
  std::array<llvm::GlobalVariable *, 3> LocalIdIterators;

  // contains all instructions that reference get_global/local_id(vectorization_dim)
  // pair.first is the instruction, pair.second is a replacement value that loads
  // from the vectorized ptr.
  std::vector<std::pair<llvm::Value *, llvm::Value *>> LocalIdInsts;
  std::vector<std::pair<llvm::Value *, llvm::Value *>> GlobalIdInsts;

  std::map<llvm::Instruction *, VectorizationTarget> Remapper;

  bool processFunction(llvm::Function &F);

  Value *createVectorScalarAdd(IRBuilder<> &builder, Value *Vec, Value *Scalar);
  void vectorizedHandleWorkitemFunctions();

  void vectorizeInstruction(llvm::Instruction* I, llvm::Instruction* oldVal);
  void vectorizedReplace(llvm::Instruction* I, llvm::Instruction* oldVal);
  void unvectorizedReplace(llvm::Instruction* I, llvm::Instruction* oldVal);
  bool isVectorizableInstruction(Instruction *I);
  void traverseInstructionTree(Instruction *I);

  void fixMultiRegionVariables(ParallelRegion *Region);
  void addContextSaveRestore(llvm::Instruction *instruction);
  void releaseParallelRegions();

  Constant *makeSequentialVector();

  void transformForInit(BasicBlock *BB);
  void transformForInc(BasicBlock *BB);
  void transformForBody(BasicBlock *BB);
  Instruction *findIncrementOfGlobal(BasicBlock *BB, GlobalVariable *GV);
  void findLoadsOfGlobal(BasicBlock *Start, GlobalVariable *GV, std::vector<Instruction *> &Loads);

  bool privatizeContext();
};

bool EnforceVectorizationImpl::runOnFunction(Function &Func) {
  M = Func.getParent();
  F = &Func;
  Initialize(cast<Kernel>(&Func));
  // initialize vectorization member vars
  // TODO add proper inference of gang size
  GangSize = GANG_SIZE;
  if (WGLocalSizeX >= WGLocalSizeY && WGLocalSizeX >= WGLocalSizeZ) {
    VectorizationDim = 0;
  } else if (WGLocalSizeY >= WGLocalSizeX && WGLocalSizeY >= WGLocalSizeZ) {
    VectorizationDim = 1;
  } else {
    VectorizationDim = 2;
  }
  VectorST = VectorType::get(ST, GangSize, false);

  SequentialVec = makeSequentialVector();
  StepVec = ConstantVector::getSplat(ElementCount::getFixed(GangSize), ConstantInt::get(ST, GangSize));

  GlobalIdIterators = {
    M->getGlobalVariable(GID_G_NAME(0), ST),
    M->getGlobalVariable(GID_G_NAME(1), ST),
    M->getGlobalVariable(GID_G_NAME(2), ST)
  };

  LocalIdIterators = {
    M->getGlobalVariable(LID_G_NAME(0), ST),
    M->getGlobalVariable(LID_G_NAME(1), ST),
    M->getGlobalVariable(LID_G_NAME(2), ST)
  };

  VectorizedGlobalIdVar = cast<GlobalVariable>(M->getOrInsertGlobal(GID_G_NAME_VECTORIZED, VectorST));
  VectorizedLocalIdVar = cast<GlobalVariable>(M->getOrInsertGlobal(LID_G_NAME_VECTORIZED, VectorST));

  bool Changed = processFunction(Func);

  Changed |= handleLocalMemAllocas();

  Changed |= fixUndominatedVariableUses(DT, Func);

  Changed |= privatizeContext();
  return Changed;
}

Constant *EnforceVectorizationImpl::makeSequentialVector() {
    std::vector<Constant*> Elements;
    Elements.reserve(GangSize);
    for (unsigned i = 0; i < GangSize; i++) {
        Elements.push_back(ConstantInt::get(ST, i));
    }
    return ConstantVector::get(Elements);
}

void EnforceVectorizationImpl::traverseInstructionTree(Instruction *I) {
  for (User *U : I->users()) {
    Instruction *UserInst = dyn_cast<Instruction>(U);
    if (UserInst) {
      Remapper[I].children.push_back(UserInst);
      traverseInstructionTree(UserInst);
    }
  }
}

void EnforceVectorizationImpl::vectorizeInstruction(llvm::Instruction* I, llvm::Instruction* oldVal) {
  if (isVectorizableInstruction(I)) {
    vectorizedReplace(I, oldVal);
  } else {
    unvectorizedReplace(I, oldVal);
  }

  for (Instruction *nextI : Remapper[I].children) {
    vectorizeInstruction(nextI, I);
  }
}

// assumes the opcode is vectorizable, and that the operands are either vectors or vectorizable
// returns new instruction
void EnforceVectorizationImpl::vectorizedReplace(llvm::Instruction* I, llvm::Instruction* oldVal) {
  Instruction *RI;
  bool shouldDeleteRI = false;
  if (HAS_VALUE_OUTPUT(Remapper[I])) {
    assert(Remapper[I].replacers.size() == 1);
    RI = Remapper[I].replacers[0];
    shouldDeleteRI = true;
  } else {
    RI = I;
  }
  llvm::IRBuilder<> Builder(I);

  if (!HAS_VALUE_OUTPUT(Remapper[oldVal])) {
    // construct vectorized data out of the array in Remapper[oldVal].valueOutput
    std::vector<Value *>& oldValOutputs = Remapper[oldVal].arrayOutput;
    Type *elemTy = oldValOutputs[0]->getType();

    // Create the vector type <N x elemTy>
    VectorType *vecTy = VectorType::get(elemTy, GangSize, false);

    // Start with an undef vector
    Value *vec = UndefValue::get(vecTy);

    // Insert each element
    for (unsigned i = 0; i < GangSize; i++) {
      vec = Builder.CreateInsertElement(vec, oldValOutputs[i],
                                        Builder.getInt32(i));
    }
    Remapper[oldVal].valueOutput = vec;
    // Remapper[I].outputType = VectorizationTarget::OutputType::VALUE;
    // Remapper[I].valueOutput = vec;
  }
  std::vector<Value *> newOperands;
  for (unsigned i = 0; i < RI->getNumOperands(); ++i) {
    if (RI->getOperand(i) == oldVal) {
      newOperands.push_back(Remapper[oldVal].valueOutput);
    } else if (!RI->getOperand(i)->getType()->isVectorTy()) {
      Value *NewSplatInst = Builder.CreateVectorSplat(GangSize, RI->getOperand(i));
      newOperands.push_back(NewSplatInst);
    } else {
      newOperands.push_back(RI->getOperand(i));
    }
  }

  llvm::Instruction *newInst = nullptr;
  if (Instruction::isBinaryOp(RI->getOpcode())) {
    newInst = cast<Instruction>(Builder.CreateBinOp((llvm::Instruction::BinaryOps)RI->getOpcode(), newOperands[0], newOperands[1]));
  } else if (Instruction::isUnaryOp(RI->getOpcode())) {
    newInst = cast<Instruction>(Builder.CreateUnOp((llvm::Instruction::UnaryOps)RI->getOpcode(), newOperands[0]));
  } else if (Instruction::isCast(RI->getOpcode())) {
    newInst = cast<Instruction>(Builder.CreateCast((llvm::Instruction::CastOps)RI->getOpcode(), newOperands[0], RI->getType()));
  }
  if (shouldDeleteRI) {
    RI->eraseFromParent();
    Remapper[I].replacers.pop_back();
  }
  
  Remapper[I].replacers.push_back(newInst);
  Remapper[I].valueOutput = newInst;
}

// When you
void EnforceVectorizationImpl::unvectorizedReplace(llvm::Instruction* I, llvm::Instruction* oldVal) {
  Instruction *PositionTarget = Remapper[I].replacers.empty() ? I : Remapper[I].replacers[0];
  IRBuilder<> Builder(PositionTarget);

  unsigned changedOpIdx = 0;
  for (unsigned i = 0; i < I->getNumOperands(); ++i) {
    if (I->getOperand(i) == oldVal) {
      changedOpIdx = i;
      break;
    }
  }

  std::vector<Value *> newOperandUnpacked;

  if (!HAS_ARRAY_OUTPUT(Remapper[oldVal])) {
    // VectorType *VecTy = cast<VectorType>(Remapper[I].valueOutput->getType());
    for (unsigned i = 0; i < GangSize; ++i) {
      Value *ExtractedElem = Builder.CreateExtractElement(Remapper[oldVal].valueOutput, i);
      newOperandUnpacked.push_back(ExtractedElem);
    }
    Remapper[oldVal].arrayOutput = newOperandUnpacked;
  } else {
    newOperandUnpacked = Remapper[oldVal].arrayOutput;
  }

  if (!Remapper[I].replacers.empty()) {
    for (int i = 0; i < GangSize; ++i) {
      Instruction *newInst = Remapper[I].replacers[i];
      newInst->setOperand(changedOpIdx, newOperandUnpacked[i]);
    }
  } else {
    for (int i = 0; i < GangSize; ++i) {
      Instruction *newInst = I->clone();
      newInst->setOperand(changedOpIdx, newOperandUnpacked[i]);
      Remapper[I].arrayOutput.push_back(newInst);
      Remapper[I].replacers.push_back(newInst);
      Builder.Insert(newInst);
    }
  }
}

bool EnforceVectorizationImpl::isVectorizableInstruction(Instruction *I) {
  switch (I->getOpcode()) {
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
    case Instruction::Shl:
    case Instruction::AShr:
    case Instruction::LShr:
      break;
    default:
      return false;
  }

  for (Value *Op : I->operands()) {
    Type *OpType = Op->getType();
    if (!OpType->isIntegerTy() && !OpType->isFloatingPointTy() && !OpType->isVectorTy()) {
      return false;
    }
  }
  return true;
}

Value *EnforceVectorizationImpl::createVectorScalarAdd(IRBuilder<> &builder, Value *Vec, Value *Scalar) {
    auto *VecTy = cast<VectorType>(Vec->getType());
    unsigned NumElems = VecTy->getElementCount().getFixedValue();

    Value *Splat = builder.CreateVectorSplat(NumElems, Scalar);
    return builder.CreateAdd(Vec, Splat);
}

// Set up the vectorized global and local id. That is, initialize the vectors <0, 1, 2,.., GangSize-1>.
void EnforceVectorizationImpl::transformForInit(BasicBlock *BB) {
  // find init of global variable (which contains the offset as an operand)
  IRBuilder<> Builder(BB);
  Builder.SetInsertPoint(BB->getFirstInsertionPt());

  std::vector<StoreInst*> Results;
  for (Instruction &I : *BB) {
    if (auto *SI = dyn_cast<StoreInst>(&I)) {
      if (SI->getPointerOperand() == GlobalIdGlobals[VectorizationDim]) {
        Builder.SetInsertPoint(SI);
        Builder.CreateStore(createVectorScalarAdd(Builder, SequentialVec, SI->getValueOperand()), VectorizedGlobalIdVar);
        break;
      }
    }
  }
  Builder.CreateStore(SequentialVec, VectorizedLocalIdVar);
}

Instruction *EnforceVectorizationImpl::findIncrementOfGlobal(BasicBlock *BB, GlobalVariable *GV) {
  for (Instruction &I : *BB) {
    // Match: %x = load i64, ptr @GV
    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      if (LI->getPointerOperand()->stripPointerCasts() != GV)
        continue;

      // Next instruction must be %y = add i64 %x, 1
      if (auto *AI = dyn_cast<BinaryOperator>(LI->getNextNode())) {
        Value *X;
        if (!PatternMatch::match(AI, PatternMatch::m_Add(PatternMatch::m_Value(X), PatternMatch::m_ConstantInt<1>())))
          continue;

        if (X != LI) // ensure add uses the load
          continue;
        // Next instruction must be store %y, ptr @GV
        if (auto *SI = dyn_cast<StoreInst>(AI->getNextNode())) {
          if (SI->getValueOperand() == AI &&
            SI->getPointerOperand()->stripPointerCasts() == GV) {
            return AI;
          }
        }
      }
    }
  }
  return nullptr;
}

// Increment the iterator vector by GangSize
void EnforceVectorizationImpl::transformForInc(BasicBlock *BB) {
  IRBuilder<> Builder(BB);
  Builder.SetInsertPoint(BB->getFirstInsertionPt());
  llvm::GlobalVariable *LocalIdVar = LocalIdIterators[VectorizationDim];
  llvm::GlobalVariable *GlobalIdVar = GlobalIdIterators[VectorizationDim];

  Instruction *LocalIteratorInc = findIncrementOfGlobal(BB, LocalIdVar);
  if (LocalIteratorInc) {
    for (int i = 0; i < 2; i++) {
      if (ConstantInt *CI = dyn_cast<ConstantInt>(LocalIteratorInc->getOperand(i))) {
        LocalIteratorInc->setOperand(i, ConstantInt::get(ST, GangSize));
        break;
      }
    }
    Builder.CreateStore(Builder.CreateAdd(Builder.CreateLoad(VectorST, VectorizedLocalIdVar),
                                          StepVec),
                        VectorizedLocalIdVar);
  }

  Instruction *GlobalIteratorInc = findIncrementOfGlobal(BB, GlobalIdVar);
  if (GlobalIteratorInc) {
    for (int i = 0; i < 2; i++) {
      if (ConstantInt *CI = dyn_cast<ConstantInt>(GlobalIteratorInc->getOperand(i))) {
        GlobalIteratorInc->setOperand(i, ConstantInt::get(ST, GangSize));
        break;
      }
    }
    Builder.CreateStore(Builder.CreateAdd(Builder.CreateLoad(VectorST, VectorizedGlobalIdVar),
                                          StepVec),
                        VectorizedGlobalIdVar);
  }
}

void EnforceVectorizationImpl::findLoadsOfGlobal(BasicBlock *Start, GlobalVariable *GV, std::vector<Instruction *> &Loads) {
  SmallPtrSet<BasicBlock *, 16> Visited;
  SmallVector<BasicBlock *, 16> Worklist;

  Worklist.push_back(Start);

  while (!Worklist.empty()) {
    BasicBlock *BB = Worklist.pop_back_val();

    if (!Visited.insert(BB).second)
      continue; // already visited
    
    // Skip other parts of the for loop
    if (auto *M = BB->getTerminator()->getMetadata("myrole")) {
      auto *S = dyn_cast<MDString>(M->getOperand(0));
      if (S && 
        (S->getString() == "pregion_for_inc" || S->getString() == "pregion_for_init" || S->getString() == "pregion_for_cond")) {
          continue;
      }
    }

    // Look for load instructions in this block
    for (Instruction &I : *BB) {
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        if (LI->getPointerOperand()->stripPointerCasts() == GV) {
          Loads.push_back(LI);
        }
      }
    }

    // Add successors to the worklist
    for (BasicBlock *Succ : successors(BB)) {
      if (!Visited.count(Succ))
        Worklist.push_back(Succ);
    }
  }
}

// Starting with every load of the global and local id values, recursively walk the
// data flow graph, applying vectorization to every instruction found.
void EnforceVectorizationImpl::transformForBody(BasicBlock *BB) {
  IRBuilder<> Builder(BB);
  Builder.SetInsertPoint(BB->getFirstInsertionPt());

  std::vector<Instruction *> AllVectorizableLoads;
  std::vector<Instruction *> GlobalIteratorLoads;
  findLoadsOfGlobal(BB, GlobalIdIterators[VectorizationDim], GlobalIteratorLoads);
  for (Instruction *LoadInst : GlobalIteratorLoads) {
    Instruction *globalLoad = Builder.CreateLoad(VectorST, VectorizedGlobalIdVar);
    Remapper[LoadInst].replacers = {globalLoad};
    Remapper[LoadInst].valueOutput = globalLoad;
    // Remapper[LoadInst].outputType = VectorizationTarget::OutputType::VALUE;
    AllVectorizableLoads.push_back(LoadInst);
    traverseInstructionTree(LoadInst);
  }

  std::vector<Instruction *> LocalIteratorLoads;
  findLoadsOfGlobal(BB, LocalIdIterators[VectorizationDim], LocalIteratorLoads);
  for (Instruction *LoadInst : LocalIteratorLoads) {
    Instruction *globalLoad = Builder.CreateLoad(VectorST, VectorizedLocalIdVar);
    Remapper[LoadInst].replacers = {globalLoad};
    Remapper[LoadInst].valueOutput = globalLoad;
    AllVectorizableLoads.push_back(LoadInst);
    traverseInstructionTree(LoadInst);
  }

  for (Instruction *I : AllVectorizableLoads) {
    for (Instruction *NextI : Remapper[I].children) {
      vectorizeInstruction(NextI, I);
    }
  }
}

bool EnforceVectorizationImpl::privatizeContext() {
  CreateBuilder(Builder, F->getEntryBlock());
  if (VectorizedGlobalIdVar != nullptr) {
    AllocaInst *VectorizedGlobalIdAlloca = Builder.CreateAlloca(VectorST, 0, GID_G_NAME_VECTORIZED);
    for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
      for (BasicBlock::iterator ii = i->begin(), ee = i->end(); ii != ee; ++ii) {
        ii->replaceUsesOfWith(VectorizedGlobalIdVar,
                              VectorizedGlobalIdAlloca);
      }
    }
  }

  if (VectorizedLocalIdVar != nullptr) {
    AllocaInst *VectorizedLocalIdAlloca = Builder.CreateAlloca(VectorST, 0, LID_G_NAME_VECTORIZED);
    for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
      for (BasicBlock::iterator ii = i->begin(), ee = i->end(); ii != ee; ++ii) {
        ii->replaceUsesOfWith(VectorizedLocalIdVar,
                              VectorizedLocalIdAlloca);
      }
    }
  }
  M->eraseGlobalVariable(VectorizedGlobalIdVar);
  M->eraseGlobalVariable(VectorizedLocalIdVar);

  return true;
}

bool EnforceVectorizationImpl::processFunction(Function &F) {
  // vectorizedHandleWorkitemFunctions();
  std::vector<BasicBlock *> forInitBlocks;
  std::vector<BasicBlock *> forBodyBlocks; 
  std::vector<BasicBlock *> forIncBlocks;
  for (auto &BB : F) {
    if (auto *M = BB.getTerminator()->getMetadata("myrole")) {
      auto *S = dyn_cast<MDString>(M->getOperand(0));
      if (!S) {
        continue;
      }
      if (S->getString() == "pregion_for_entry") {
        // Note to self: search for other linked blocks when dealing with this block
        forBodyBlocks.push_back(&BB);
      } else if (S->getString() == "pregion_for_init") {
        forInitBlocks.push_back(&BB);
      } else if (S->getString() == "pregion_for_inc") {
        forIncBlocks.push_back(&BB);
      }
    }
  }

  for (BasicBlock *BB : forInitBlocks) {
    transformForInit(BB);
  }

  for (BasicBlock *BB : forIncBlocks) {
    transformForInc(BB);
  }

  for (BasicBlock *BB : forBodyBlocks) {
    transformForBody(BB);
  }
  return true;
}

llvm::PreservedAnalyses EnforceVectorization::run(llvm::Function &F,
                                           llvm::FunctionAnalysisManager &AM) {
  if (!isKernelToProcess(F))
    return llvm::PreservedAnalyses::all();

  WorkitemHandlerType WIH = AM.getResult<WorkitemHandlerChooser>(F).WIH;
  if (WIH != WorkitemHandlerType::LOOPS)
    return llvm::PreservedAnalyses::all();

  auto &DT = AM.getResult<llvm::DominatorTreeAnalysis>(F);
  auto &PDT = AM.getResult<llvm::PostDominatorTreeAnalysis>(F);
  auto &LI = AM.getResult<llvm::LoopAnalysis>(F);
  auto &VUA = AM.getResult<VariableUniformityAnalysis>(F);

  llvm::PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<VariableUniformityAnalysis>();
  PAChanged.preserve<WorkitemHandlerChooser>();

  EnforceVectorizationImpl WIL(DT, LI, PDT, VUA);
  // llvm::verifyFunction(F);

  return WIL.runOnFunction(F) ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
