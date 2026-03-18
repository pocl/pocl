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
#include <llvm/ADT/APInt.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Intrinsics.h>
#include <unordered_map>
#include <unordered_set>
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
#include <llvm/Transforms/Utils/Local.h>

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
  struct GraphNode {
    llvm::Instruction *valueOutput = nullptr;
    std::vector<llvm::Instruction *> arrayOutput;
    std::unordered_set<llvm::Instruction *> children;

    // Used for Kahn's algorithm to generate a topological sort
    int in_degree = 0;

    // Determines whether loads should be converted to wide loads or gathers
    bool isContiguous = true;
    llvm::Value *mask;
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
  std::unordered_set<llvm::Instruction *> markedForDeletion;

  // contains all instructions that reference get_global/local_id(vectorization_dim)
  // pair.first is the instruction, pair.second is a replacement value that loads
  // from the vectorized ptr.
  std::vector<std::pair<llvm::Value *, llvm::Value *>> LocalIdInsts;
  std::vector<std::pair<llvm::Value *, llvm::Value *>> GlobalIdInsts;

  std::map<llvm::Instruction *, GraphNode> InstGraph;
  std::map<llvm::BasicBlock *, llvm::Value *> BlockMasks;

  bool processFunction(llvm::Function &F);

  // void findBackedges(Loop *L);
  // void handleLatchMask(BasicBlock *Header, BasicBlock *Latch);
  Constant *createConstantMask(IRBuilder<> &Builder, int N);
  // void initializeMasks();

  Value *createVectorScalarAdd(IRBuilder<> &builder, Value *Vec, Value *Scalar);
  void vectorizedHandleWorkitemFunctions();

  Instruction *ConvertOpToMaskedIntrinsic(IRBuilder<> &Builder, Instruction *I, Value *op1, Value *op2);
  bool checkContiguous(llvm::Instruction *I, llvm::Instruction *oldVal);
  void vectorizeInstruction(llvm::Instruction* I);
  void vectorizedReplace(llvm::Instruction* I);
  void unvectorizedReplace(llvm::Instruction* I);
  bool isVectorizableInstruction(Instruction *I);
  void traverseInstructionTree(Instruction *I, std::unordered_set<llvm::Instruction *> &visited);

  void fixMultiRegionVariables(ParallelRegion *Region);
  void addContextSaveRestore(llvm::Instruction *instruction);
  void releaseParallelRegions();

  Constant *makeSequentialVector();

  void transformIdStores(BasicBlock *BB);
  void transformForInc(BasicBlock *BB);
  void transformIdLoads(BasicBlock *BB);
  Instruction *findIncrementOfGlobal(BasicBlock *BB, GlobalVariable *GV);
  void findLoadsOfGlobal(GlobalVariable *GV, std::vector<Instruction *> &Loads);

  bool privatizeContext();
};

bool EnforceVectorizationImpl::runOnFunction(Function &Func) {
  M = Func.getParent();
  F = &Func;
  Initialize(cast<Kernel>(&Func));
  // WriteGraph(llvm::errs(), M, false);
  // initialize vectorization member vars
  // TODO add proper inference of gang size

  GangSize = 4;
  VectorizationDim = 0;
  // if (WGLocalSizeX >= WGLocalSizeY && WGLocalSizeX >= WGLocalSizeZ) {
  //   VectorizationDim = 0;
  // } else if (WGLocalSizeY >= WGLocalSizeX && WGLocalSizeY >= WGLocalSizeZ) {
  //   VectorizationDim = 1;
  // } else {
  //   VectorizationDim = 2;
  // }
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

  // initializeMasks();

  bool Changed = processFunction(Func);

  Changed |= handleLocalMemAllocas();

  Changed |= fixUndominatedVariableUses(DT, Func);

  Changed |= privatizeContext();
  return Changed;
}

Constant *EnforceVectorizationImpl::createConstantMask(IRBuilder<> &Builder, int N) {
  Type *I1Ty = Builder.getInt1Ty();
  VectorType *MaskTy = FixedVectorType::get(I1Ty, GangSize);

  SmallVector<Constant*, 16> Elems;

  unsigned Active = std::min(N, GangSize);

  for (unsigned i = 0; i < GangSize; ++i) {
    bool Bit = (i < Active);
    Elems.push_back(ConstantInt::get(I1Ty, Bit));
  }

  return ConstantVector::get(Elems);
}

// void EnforceVectorizationImpl::handleLatchMask(BasicBlock *Header, BasicBlock *Latch) {
//   auto *TI = Latch->getTerminator();
//   auto *BI = dyn_cast<BranchInst>(TI);

//   if (!BI) {
//     // Could be indirectbr, switch, unreachable, etc.
//     return;
//   }

//   if (!BI->isConditional()) {
//     // Unconditional backedge (e.g., do { ... } while (true))
//     return;
//   }

//   Value *Cond = BI->getCondition();
//   ICmpInst *Cmp = dyn_cast<ICmpInst>(Cond);
//   if (!Cmp) {
//     return;
//   }
//   if (Cmp->getParent() != BI->getParent()) {
//     return;
//   }

//   Value *Op0 = Cmp->getOperand(0);
//   Value *Op1 = Cmp->getOperand(1);

//   LoadInst *LI = nullptr;
//   ConstantInt *CI = nullptr;

//   if ((LI = dyn_cast<LoadInst>(Op0)))
//     CI = dyn_cast<ConstantInt>(Op1);
//   else if ((LI = dyn_cast<LoadInst>(Op1)))
//     CI = dyn_cast<ConstantInt>(Op0);

//   if (!LI || !CI)
//     return; // pattern doesn't match

//   Value *Ptr = LI->getPointerOperand();
//   auto *GV = dyn_cast<GlobalVariable>(Ptr);
//   if (!GV || (GV != GlobalIdIterators[VectorizationDim] && GV != LocalIdIterators[VectorizationDim])) {
//     return;
//   }
  
//   // Set up Dynamic Mask (the mask used and updated each loop)
//   IRBuilder<> LatchBuilder(Cmp);
//   Value *NumLeft = LatchBuilder.CreateSub(LI, CI);
//   Constant *Sequential = makeSequentialVector();
//   Value *TrailingSplat = LatchBuilder.CreateVectorSplat(GangSize, NumLeft);
//   Value *DynMask = LatchBuilder.CreateICmpULT(Sequential, TrailingSplat);

//   // Set up Initial Mask (the mask used for the first loop)
//   Constant *InitialMask = createConstantMask(LatchBuilder, WGLocalSizeX);
  
//   // Set up mask PHI node
//   IRBuilder<> HeaderBuilder(Header, Header->begin());
//   unsigned NumPreds = std::distance(pred_begin(Header),
//                                     pred_end(Header));
//   PHINode *Phi = HeaderBuilder.CreatePHI(DynMask->getType(), NumPreds);
//   for (BasicBlock *Pred : predecessors(Header)) {
//     if (Pred == Latch)
//       Phi->addIncoming(DynMask, Pred);
//     else
//       Phi->addIncoming(InitialMask, Pred);
//   }
//   // BlockMasks[Header] = Phi;
//   BlockMasks[Header] = InitialMask;
// }

// void EnforceVectorizationImpl::findBackedges(Loop *L) {
//   BasicBlock *Header = L->getHeader();
//   for (BasicBlock *Pred : predecessors(Header)) {
//     if (L->contains(Pred)) {
//       handleLatchMask(Header, Pred);
//     }
//   }

//   for (Loop *SubL : L->getSubLoops()) {
//     findBackedges(SubL);
//   }
// }

// void EnforceVectorizationImpl::initializeMasks() {
//   for (auto &L : LI) {
//     findBackedges(L);
//   }

// }

Constant *EnforceVectorizationImpl::makeSequentialVector() {
    std::vector<Constant*> Elements;
    Elements.reserve(GangSize);
    for (unsigned i = 0; i < GangSize; i++) {
        Elements.push_back(ConstantInt::get(ST, i));
    }
    return ConstantVector::get(Elements);
}

void EnforceVectorizationImpl::traverseInstructionTree(Instruction *I, std::unordered_set<llvm::Instruction *> &visited) {
  visited.insert(I);
  for (User *U : I->users()) {
    Instruction *ChildInst = dyn_cast<Instruction>(U);
    if (ChildInst) {
      InstGraph[ChildInst].in_degree += 1;
      InstGraph[I].children.insert(ChildInst);
      if (visited.count(ChildInst) == 0) {
        traverseInstructionTree(ChildInst, visited);
      }
    }
  }
}

bool EnforceVectorizationImpl::checkContiguous(Instruction *I, Instruction *oldVal) {
  auto IOpcode = I->getOpcode();
  if (IOpcode == Instruction::Add || IOpcode == Instruction::Sub || IOpcode == Instruction::GetElementPtr) {
    return InstGraph[oldVal].isContiguous;
  }

  Instruction *Grandparent;
  const APInt *Shift;
  if (PatternMatch::match(I,
    PatternMatch::m_AShr(
      PatternMatch::m_Shl(
        PatternMatch::m_Instruction(Grandparent),
        PatternMatch::m_APInt(Shift)),
      PatternMatch::m_APInt(Shift))
    )
  ) {
    return InstGraph[Grandparent].isContiguous;
  }
  return false;
}

void EnforceVectorizationImpl::vectorizeInstruction(llvm::Instruction* I) {
  // InstGraph[I].mask = InstGraph[oldVal].mask;
  // InstGraph[I].isContiguous = checkContiguous(I, oldVal);

  if (isVectorizableInstruction(I)) {
    vectorizedReplace(I);
  } else {
    unvectorizedReplace(I);
  }
}

// Only convert load, store, gather, and scatter
// Instruction *EnforceVectorizationImpl::ConvertOpToMaskedIntrinsic(IRBuilder<> &Builder, Instruction *I, Value *op1, Value *op2) {
//   VectorType *newType = dyn_cast<VectorType>(I->getType());
//   if (!newType) {
//     newType = VectorType::get(I->getType(), GangSize, false);
//   }

//   // Value *Operation = Builder.createO
//   return Builder.CreateIntrinsic(vp_opcode, newType, {op1, op2, InstGraph[I].mask, Builder.getInt32(newType->getElementCount().getFixedValue())});

// }

#define SPLAT_OPERAND(idx) \
if (!newOperands[idx]->getType()->isVectorTy()) { \
  newOperands[idx] = Builder.CreateVectorSplat(GangSize, newOperands[idx]); \
}
// assumes the opcode is vectorizable, and that the operands are either vectors or vectorizable
// returns new instruction
void EnforceVectorizationImpl::vectorizedReplace(llvm::Instruction* I) {
  llvm::IRBuilder<> Builder(I);

  std::vector<Value *> newOperands;
  for (unsigned i = 0; i < I->getNumOperands(); ++i) {
    Instruction *operandInst = dyn_cast<Instruction>(I->getOperand(i));
    if (operandInst && InstGraph.count(operandInst) > 0) {
      // convert arrayOutput to valueOutput
      if (!HAS_VALUE_OUTPUT(InstGraph[operandInst])) {
        std::vector<Instruction *>& oldValOutputs = InstGraph[operandInst].arrayOutput;
        Type *elemTy = oldValOutputs[0]->getType();
        VectorType *vecTy = VectorType::get(elemTy, GangSize, false);
        Value *vec = UndefValue::get(vecTy);
        for (unsigned i = 0; i < GangSize; i++) {
          vec = Builder.CreateInsertElement(vec, oldValOutputs[i],
                                            Builder.getInt32(i));
        }
        // In theory, all masks of operands should be the same, so we can pick
        // one arbitrarily to propagate
        InstGraph[I].mask = InstGraph[operandInst].mask;
        InstGraph[operandInst].valueOutput = cast<Instruction>(vec);
      }
      newOperands.push_back(InstGraph[operandInst].valueOutput);
    } else {
      newOperands.push_back(I->getOperand(i));
    }
  }

  llvm::Instruction *newInst = nullptr;
  if (I->getOpcode() == Instruction::GetElementPtr) {
    // Again, I can't figure out the correct semantics for GEP, so it's disabled for now
    assert(false);
    // VectorType *vectorizedType = VectorType::get(I->getType(), GangSize, false);
    // SPLAT_OPERAND(0);
    // SPLAT_OPERAND(1);
    // newInst = cast<Instruction>(Builder.CreateGEP(I->getType(), newOperands[0], newOperands[1]));
  } else if (I->getOpcode() == Instruction::Load) {
    VectorType *vectorizedType = VectorType::get(I->getType(), GangSize, false);
    SPLAT_OPERAND(0);
    newInst = cast<Instruction>(Builder.CreateMaskedGather(vectorizedType, newOperands[0], Align(4), InstGraph[I].mask, PoisonValue::get(vectorizedType)));
  } else if (I->getOpcode() == Instruction::Store) {
    SPLAT_OPERAND(0);
    SPLAT_OPERAND(1);
    newInst = cast<Instruction>(Builder.CreateMaskedScatter(newOperands[0], newOperands[1], Align(4), InstGraph[I].mask)); 
  } else if (Instruction::isBinaryOp(I->getOpcode())) {
    SPLAT_OPERAND(0);
    SPLAT_OPERAND(1);
    newInst = cast<Instruction>(Builder.CreateBinOp((llvm::Instruction::BinaryOps)I->getOpcode(), newOperands[0], newOperands[1]));
  } else if (Instruction::isUnaryOp(I->getOpcode())) {
    SPLAT_OPERAND(0);
    newInst = cast<Instruction>(Builder.CreateUnOp((llvm::Instruction::UnaryOps)I->getOpcode(), newOperands[0]));
  } else if (Instruction::isCast(I->getOpcode())) {
    SPLAT_OPERAND(0);
    newInst = cast<Instruction>(Builder.CreateCast((llvm::Instruction::CastOps)I->getOpcode(), newOperands[0], I->getType()));
  }

  markedForDeletion.insert(I);
  InstGraph[I].valueOutput = newInst;
}

void EnforceVectorizationImpl::unvectorizedReplace(llvm::Instruction* I) {
  IRBuilder<> Builder(I);

  std::unordered_map<int, std::vector<Instruction *>> changedOperands;

  for (unsigned i = 0; i < I->getNumOperands(); ++i) {
    Instruction *operandInst = dyn_cast<Instruction>(I->getOperand(i));
    if (operandInst && InstGraph.count(operandInst) > 0) {
      // convert valueOutput to vectorOutput
      std::vector<Instruction *> unpackedOperand;
      if (!HAS_ARRAY_OUTPUT(InstGraph[operandInst])) {
        for (unsigned i = 0; i < GangSize; ++i) {
          Instruction *ExtractedElem = cast<Instruction>(Builder.CreateExtractElement(InstGraph[operandInst].valueOutput, i));
          unpackedOperand.push_back(ExtractedElem);
        }
        InstGraph[operandInst].arrayOutput = unpackedOperand;
      }

      changedOperands[i] = InstGraph[operandInst].arrayOutput;
    }
  }

  for (int i = 0; i < GangSize; ++i) {
    Instruction *newInst = I->clone();
    for (auto &[j, changedOperandArr] : changedOperands) {
      newInst->setOperand(j, changedOperandArr[i]);
    }
    InstGraph[I].arrayOutput.push_back(newInst);
    Builder.Insert(newInst);
  }

  markedForDeletion.insert(I);
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
    case Instruction::Load:
    case Instruction::Store:
    // In theory, there should be a vectorized version of GEP, but I can't figure out the semantics
    // case Instruction::GetElementPtr:
      break;
    default:
      return false;
  }

  for (Value *Op : I->operands()) {
    Type *OpType = Op->getType();
    if (!OpType->isIntegerTy() && !OpType->isFloatingPointTy() && !OpType->isVectorTy() && !OpType->isPointerTy()) {
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
void EnforceVectorizationImpl::transformIdStores(BasicBlock *BB) {
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

  Value *newGlobalValue;
  Instruction *GlobalIteratorInc = findIncrementOfGlobal(BB, GlobalIdVar);
  if (GlobalIteratorInc) {
    for (int i = 0; i < 2; i++) {
      if (ConstantInt *CI = dyn_cast<ConstantInt>(GlobalIteratorInc->getOperand(i))) {
        GlobalIteratorInc->setOperand(i, ConstantInt::get(ST, GangSize));
        break;
      }
    }
    Builder.CreateStore(Builder.CreateAdd(Builder.CreateLoad(VectorST, VectorizedGlobalIdVar), StepVec),
                        VectorizedGlobalIdVar);
  }

  // Generate new mask
  if (LocalIteratorInc) {
    // Value *sequential = makeSequentialVector();
    // Value *trailing = Builder.CreateSub(Builder.getInt32(WGLocalSizeX), LocalIteratorInc);
    // Value *trailingSplat = Builder.CreateVectorSplat(GangSize, trailing);
    // Value *mask = Builder.CreateICmpULT(sequential, trailingSplat);
  }

  
}

void EnforceVectorizationImpl::findLoadsOfGlobal(GlobalVariable *GV, std::vector<Instruction *> &Loads) {
  for (User *U : GV->users()) {
    if (auto *LI = dyn_cast<LoadInst>(U)) {
      BasicBlock *BB = LI->getParent();
      if (auto *M = BB->getTerminator()->getMetadata("myrole")) {
        auto *S = dyn_cast<MDString>(M->getOperand(0));
        if (!S || (S->getString() != "pregion_for_inc" && S->getString() != "pregion_for_init" && S->getString() != "pregion_for_cond")) {
          Loads.push_back(LI);
        }
      }
    }
  }
}

// Starting with every load of the global and local id values, recursively walk the
// data flow graph, applying vectorization to every instruction found.
void EnforceVectorizationImpl::transformIdLoads(BasicBlock *BB) {
  IRBuilder<> Builder(BB);
  Builder.SetInsertPoint(BB->getFirstInsertionPt());

  std::unordered_set<Instruction *> AllVectorizableLoads;
  std::vector<Instruction *> GlobalIteratorLoads;
  findLoadsOfGlobal(GlobalIdIterators[VectorizationDim], GlobalIteratorLoads);
  std::unordered_set<Instruction *> visited;
  for (Instruction *LoadInst : GlobalIteratorLoads) {
    Instruction *globalLoad = Builder.CreateLoad(VectorST, VectorizedGlobalIdVar);
    InstGraph[LoadInst].valueOutput = globalLoad;
    AllVectorizableLoads.insert(LoadInst);
    traverseInstructionTree(LoadInst, visited);
  }

  std::vector<Instruction *> LocalIteratorLoads;
  findLoadsOfGlobal(LocalIdIterators[VectorizationDim], LocalIteratorLoads);
  for (Instruction *LoadInst : LocalIteratorLoads) {
    Instruction *globalLoad = Builder.CreateLoad(VectorST, VectorizedLocalIdVar);
    InstGraph[LoadInst].valueOutput = globalLoad;
    AllVectorizableLoads.insert(LoadInst);
    traverseInstructionTree(LoadInst, visited);
  }
  
  // Topological sort. Currently, we assume no loops via PHI nodes. TODO: add explicit loop
  // checks when handling masking and other control flow stuff

  // start with nodes that have no dependencies
  SmallVector<Instruction*> Worklist;
  for (auto &LoadInst : AllVectorizableLoads) {
    Worklist.push_back(LoadInst);
  }
  SmallVector<Instruction*> TopoOrder;
  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();
    TopoOrder.push_back(I);
    for (Instruction *User : InstGraph[I].children) {
      if (--(InstGraph[User].in_degree) == 0) {
        Worklist.push_back(User);
      }
    }
  }

  // Value *initialMask = Constant::getAllOnesValue(VectorType::get(Builder.getInt1Ty(), GangSize, false));
  // if (WGLocalSizeX < GangSize) {
  //   SmallVector<Constant *, 16> MaskEls;
  //   for (unsigned lane = 0; lane < GangSize; ++lane) {
  //     MaskEls.push_back(Builder.getInt1(lane < WGLocalSizeX));
  //   }
  //   initialMask = ConstantVector::get(MaskEls);
  // }

  for (Instruction *I : AllVectorizableLoads) {
    if (BlockMasks.find(I->getParent()) != BlockMasks.end()) {
      InstGraph[I].mask = BlockMasks[I->getParent()];
    } else {
      InstGraph[I].mask = createConstantMask(Builder, GangSize);
    }
  }

  for (Instruction *I : TopoOrder) {
    // Don't try to vectorize the initial loads from the global/local IDs.
    if (AllVectorizableLoads.count(I) == 0) {
      vectorizeInstruction(I);
    }
  }

  for (Instruction *inst : markedForDeletion) {
    inst->replaceAllUsesWith(PoisonValue::get(inst->getType()));
    inst->eraseFromParent();
    // InstGraph[I].markedForDeletion.pop_back();
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
    transformIdStores(BB);
  }

  for (BasicBlock *BB : forIncBlocks) {
    transformForInc(BB);
  }

  for (BasicBlock *BB : forBodyBlocks) {
    transformIdLoads(BB);
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
