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
#include "llvm/Support/FileSystem.h"
#include <deque>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/CFG.h>
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
#include <llvm/Analysis/CFG.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PatternMatch.h>
#include <llvm/IR/ValueSymbolTable.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/IR/InstIterator.h>

#include "WholeFunctionVectorization.h"
#include "Kernel.h"
#include "KernelCompilerUtils.h"
#include "LLVMUtils.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "WorkitemHandlerChooser.h"

POP_COMPILER_DIAGS

#include <array>
#include <map>
#include <vector>

#define DEBUG_TYPE "WIL"

#define PASS_NAME "whole-function-vectorization"
#define PASS_CLASS pocl::WholeFunctionVectorization
#define PASS_DESC                                                              \
  "Forces vectorization through a simple walk over the data flow graph"

// #define DEBUG_WORK_ITEM_LOOPS
// #define POCL_KERNEL_COMPILER_DUMP_CFGS

#define HAS_VALUE_OUTPUT(x) (x.valueOutput != nullptr)
#define HAS_ARRAY_OUTPUT(x) (!x.arrayOutput.empty())

namespace pocl {

using namespace llvm;

static int modulesPrinted = 0;
static void printModule(llvm::Module &Mod, const std::string &filename) {
  for (Function &F : Mod) {
    if (F.isDeclaration()) {
      continue;
    }

    std::error_code EC;
    std::string Filename = filename + std::to_string(modulesPrinted++) + ".dot";

    llvm::errs() << "Writing " << Filename << "\n";
    raw_fd_ostream File(Filename, EC, sys::fs::OF_Text);
    if (EC) {
      errs() << "Failed to open " << Filename << ": "
             << EC.message() << "\n";
      return;
    }
    Mod.print(File, nullptr);
    return;
  }
}

class WholeFunctionVectorizationImpl : public pocl::WorkitemHandler {
public:
  WholeFunctionVectorizationImpl(llvm::DominatorTree &DT, llvm::LoopInfo &LI,
                           llvm::PostDominatorTree &PDT,
                           VariableUniformityAnalysisResult &VUA)
      : WorkitemHandler(), DT(DT), LI(LI), PDT(PDT), VUA(VUA) {}
  virtual bool runOnFunction(llvm::Function &F);

private:
  struct GraphNode {
    llvm::Instruction *valueOutput = nullptr;
    std::vector<llvm::Instruction *> arrayOutput;
    std::vector<llvm::Instruction *> children;
    bool isContiguous = false;
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

  std::array<llvm::GlobalVariable *, 3> GlobalIdIterators;
  std::array<llvm::GlobalVariable *, 3> LocalIdIterators;
  std::unordered_set<llvm::Instruction *> markedForDeletion;

  std::map<llvm::Instruction *, GraphNode> InstGraph;

  // Associates a basic block with its execution mask
  std::map<llvm::BasicBlock *, llvm::Value *> BlockMasks;

  // associates each loop with the mask before the loop began (since the mask
  // needs to reset to that value afterwards)
  std::map<llvm::Loop *, llvm::Value *> LoopMasks;

  // Store all created calls that require a mask.
  std::map<llvm::CallInst *, int> maskedCalls;

  std::unordered_set<llvm::BranchInst *> branchInsts;

  // set of blocks with loop exiting branches
  std::set<llvm::BasicBlock *> loopBranches;

  bool processFunction(llvm::Function &F);

  bool forceInlineAllCalls(Function &F);

  bool markLoop(Loop *L);
  Constant *createConstantMask(IRBuilder<> &Builder, int N);

  Value *createVectorScalarAdd(IRBuilder<> &builder, Value *Vec, Value *Scalar);

  void vectorizeInstruction(llvm::Instruction *I);
  void branchReplace(llvm::BranchInst *I);
  void vectorizedReplace(llvm::Instruction *I);
  void unvectorizedReplace(llvm::Instruction *I);
  bool isVectorizableInstruction(Instruction *I);
  void
  traverseInstructionTree(Instruction *I,
                          std::unordered_set<llvm::Instruction *> &visited);

  Constant *makeSequentialVector();

  void transformIdStores(BasicBlock *BB);
  void transformForInc(BasicBlock *BB);
  void transformIdLoads();
  void generateMasks();
  void transformControlFlow();
  void applyMasks();
  Instruction *findIncrementOfGlobal(BasicBlock *BB, GlobalVariable *GV);
  void findLoadsOfGlobal(GlobalVariable *GV, std::vector<Instruction *> &Loads);

  bool blockHasTag(const BasicBlock *BB, StringRef Role);

  // Given a typical mask, it constructs the prefix from the mask
  // mask. That is, a mask of 1101 becomes 1100, to represent the fact that
  // the loop should exit after the first 0.
  Value *constructLoopPredicatePrefix(IRBuilder<> &Builder, Value *predicate);

  bool privatizeContext();
};

bool WholeFunctionVectorizationImpl::runOnFunction(Function &Func) {
  M = Func.getParent();

  F = &Func;
  Initialize(cast<Kernel>(&Func));

  // TODO add proper inference of gang size
  GangSize = 8;
  VectorizationDim = 0;

  VectorST = VectorType::get(ST, GangSize, false);

  SequentialVec = makeSequentialVector();
  StepVec = ConstantVector::getSplat(ElementCount::getFixed(GangSize),
                                     ConstantInt::get(ST, GangSize));

  GlobalIdIterators = {M->getGlobalVariable(GID_G_NAME(0), ST),
                       M->getGlobalVariable(GID_G_NAME(1), ST),
                       M->getGlobalVariable(GID_G_NAME(2), ST)};

  LocalIdIterators = {M->getGlobalVariable(LID_G_NAME(0), ST),
                      M->getGlobalVariable(LID_G_NAME(1), ST),
                      M->getGlobalVariable(LID_G_NAME(2), ST)};

  VectorizedGlobalIdVar = cast<GlobalVariable>(
      M->getOrInsertGlobal(GID_G_NAME_VECTORIZED, VectorST));
  VectorizedLocalIdVar = cast<GlobalVariable>(
      M->getOrInsertGlobal(LID_G_NAME_VECTORIZED, VectorST));

  bool Changed = false;

  // Changed |= forceInlineAllCalls(Func);

  // printModule(*M, "wfv_inlined");
  
  for (Loop *L : LI.getLoopsInPreorder()) {
    Changed |= markLoop(L);
  }

  if (Changed) {
    DT.recalculate(Func);
    PDT.recalculate(Func);
  }

  bool functionChanged = processFunction(Func);

  if (!functionChanged) {
    return Changed;
  }
  Changed |= functionChanged;

  Changed |= handleLocalMemAllocas();

  Changed |= fixUndominatedVariableUses(DT, Func);

  Changed |= privatizeContext();

  return Changed;
}

bool WholeFunctionVectorizationImpl::forceInlineAllCalls(Function &F) {
    bool Changed = false;
    bool Progress = true;

    while (Progress) {
        Progress = false;

        for (BasicBlock &BB : F) {
            for (Instruction &I : BB) {
                auto *CB = dyn_cast<CallBase>(&I);
                if (!CB)
                    continue;

                Function *Callee = CB->getCalledFunction();

                // Skip indirect calls, declarations, and recursive calls.
                if (!Callee || Callee->isDeclaration() || Callee == &F)
                    continue;

                // InlineFunction supports CallInst and InvokeInst,
                // but not CallBrInst.
                if (!isa<CallInst>(CB) && !isa<InvokeInst>(CB))
                    continue;

                InlineFunctionInfo IFI;
                InlineResult Result = InlineFunction(*CB, IFI);

                if (Result.isSuccess()) {
                    Changed = true;
                    Progress = true;
                    break; // Iterator invalidated; restart traversal.
                }
            }

            if (Progress)
                break;
        }
    }

    return Changed;
}


Constant *WholeFunctionVectorizationImpl::createConstantMask(IRBuilder<> &Builder,
                                                       int N) {
  Type *I1Ty = Builder.getInt1Ty();
  VectorType *MaskTy = FixedVectorType::get(I1Ty, GangSize);

  SmallVector<Constant *, 16> Elems;

  unsigned Active = std::min(N, GangSize);

  for (unsigned i = 0; i < GangSize; ++i) {
    bool Bit = (i < Active);
    Elems.push_back(ConstantInt::get(I1Ty, Bit));
  }

  return ConstantVector::get(Elems);
}

bool WholeFunctionVectorizationImpl::markLoop(Loop *L) {
  bool changed = false;
  SmallVector<BasicBlock *, 4> exitingBlocks;
  L->getExitingBlocks(exitingBlocks);

  for (BasicBlock *BB : exitingBlocks) {
    BranchInst *br = dyn_cast<BranchInst>(BB->getTerminator());
    assert(br && br->isConditional());
    Value *cond = br->getCondition();

    bool trueContinuesLoop = L->contains(br->getSuccessor(0));
    if (!trueContinuesLoop) {
      IRBuilder<> Builder(br->getParent());
      Builder.SetInsertPoint(br);
      Value *condInv = Builder.CreateNot(cond);

      BasicBlock *tmp = br->getSuccessor(0);
      br->setSuccessor(0, br->getSuccessor(1));
      br->setSuccessor(1, tmp);
      br->setCondition(condInv);
      changed = true;
    }
    loopBranches.insert(br->getParent());
  }
  return changed;
}

Constant *WholeFunctionVectorizationImpl::makeSequentialVector() {
  std::vector<Constant *> Elements;
  Elements.reserve(GangSize);
  for (unsigned i = 0; i < GangSize; i++) {
    Elements.push_back(ConstantInt::get(ST, i));
  }
  return ConstantVector::get(Elements);
}

void WholeFunctionVectorizationImpl::traverseInstructionTree(
    Instruction *I, std::unordered_set<llvm::Instruction *> &visited) {
  visited.insert(I);
  for (User *U : I->users()) {
    Instruction *ChildInst = dyn_cast<Instruction>(U);
    if (ChildInst) {
      InstGraph[I].children.push_back(ChildInst);
      if (visited.count(ChildInst) == 0) {
        traverseInstructionTree(ChildInst, visited);
      }
    }
  }
}

void WholeFunctionVectorizationImpl::vectorizeInstruction(llvm::Instruction *I) {
  // Handle branches
  if (BranchInst *br = dyn_cast<BranchInst>(I)) {
    branchReplace(br);
  } else if (isVectorizableInstruction(I)) {
      vectorizedReplace(I);
  } else {
    // assert(false);
    if (I->getOpcode() == Instruction::Call) {
      if (auto *II = dyn_cast<IntrinsicInst>(I)) {
        if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
            II->getIntrinsicID() == Intrinsic::lifetime_end) {
          return;
        }
      }
    }
    // I->dump();
    unvectorizedReplace(I);
  }
}

bool WholeFunctionVectorizationImpl::blockHasTag(const BasicBlock *BB,
                                           StringRef Role) {
  auto *M = BB->getTerminator()->getMetadata("myrole");
  auto *S = M ? dyn_cast<MDString>(M->getOperand(0)) : nullptr;
  return S && S->getString() == Role;
}

void WholeFunctionVectorizationImpl::branchReplace(llvm::BranchInst *br) {
  if (!br->isConditional()) {
    return;
  }

  BasicBlock *BB = br->getParent();

  Value *cond = br->getCondition();
  Instruction *condInst = dyn_cast<Instruction>(cond);
  if (condInst && InstGraph.count(condInst) > 0) {
    // conversion from array output to value output
    if (!HAS_VALUE_OUTPUT(InstGraph[condInst])) {
      std::vector<Instruction *> &oldValOutputs =
          InstGraph[condInst].arrayOutput;
      IRBuilder<> Builder(oldValOutputs.back()->getNextNode());
      Type *elemTy = oldValOutputs[0]->getType();
      VectorType *vecTy = VectorType::get(elemTy, GangSize, false);
      Value *vec = UndefValue::get(vecTy);
      for (unsigned i = 0; i < GangSize; i++) {
        vec = Builder.CreateInsertElement(vec, oldValOutputs[i],
                                          Builder.getInt32(i));
      }
      InstGraph[condInst].valueOutput = cast<Instruction>(vec);
    }
    condInst = InstGraph[condInst].valueOutput;
    IRBuilder<> Builder(condInst->getParent());

    // TODO: fix this. in the case of for_cond blocks, maintaining the masks in
    // this way is massively overkill if we switch equality comps to geq.
    // Even in other cases, we don't need nearly as much overhead. Just zero out all masks
    // after the first false.
    if (loopBranches.count(br->getParent()) > 0) {
      Builder.SetInsertPoint(br);
      // For loop predicates in the pregion_for_cond blocks, any false means that all future mask values must
      // also be false.
      if (blockHasTag(br->getParent(), "pregion_for_cond")) {
        // Value *newLoopPredicate =
        //     constructLoopPredicatePrefix(Builder, condInst);
        cast<PHINode>(BlockMasks[br->getSuccessor(0)])
            ->addIncoming(condInst, BB);
        Value *newBrCond = Builder.CreateOrReduce(condInst);
        br->setCondition(newBrCond);
      } else {
        Value *newMask = Builder.CreateAnd(BlockMasks[BB], condInst);
        // Value *newMask = constructLoopPredicatePrefix(Builder, condInst);
        cast<PHINode>(BlockMasks[br->getSuccessor(0)])
            ->addIncoming(newMask, BB);
        Value *newBrCond = Builder.CreateOrReduce(newMask);
        br->setCondition(newBrCond);
      }

    } else {
      // Note that we don't change the condition of the branch here,
      // as there is a good chance this branch becomes unconditional for masking reasons.
      Builder.SetInsertPoint(condInst->getNextNode());
      Value *inv = Builder.CreateNot(condInst);
      cast<PHINode>(BlockMasks[br->getSuccessor(0)])->addIncoming(condInst, BB);
      cast<PHINode>(BlockMasks[br->getSuccessor(1)])->addIncoming(inv, BB);
      branchInsts.insert(br);
    }
  }
}


// Assumes the opcode is vectorizable and that operands are either vectors
// or can be vectorized. Creates and records the vectorized replacement.
void WholeFunctionVectorizationImpl::vectorizedReplace(llvm::Instruction *I) {
  std::vector<Value *> newOperands;
  newOperands.reserve(I->getNumOperands());

  Instruction *insertPoint = I->getNextNode();
  IRBuilder<> Builder(insertPoint);

  // Convert operands to their vectorized value outputs when necessary.
  for (unsigned i = 0; i < I->getNumOperands(); ++i) {
    Value *operand = I->getOperand(i);

    if (auto *operandInst = dyn_cast<Instruction>(operand);
        operandInst && InstGraph.count(operandInst)) {
      auto &operandGraph = InstGraph[operandInst];

      // Convert arrayOutput -> valueOutput.
      if (!HAS_VALUE_OUTPUT(operandGraph)) {
        auto &oldValOutputs = operandGraph.arrayOutput;

        Type *elementType = oldValOutputs.front()->getType();
        VectorType *vectorType =
            VectorType::get(elementType, GangSize, false);

        Value *vector = UndefValue::get(vectorType);

        Builder.SetInsertPoint(oldValOutputs.back()->getNextNode());

        for (unsigned lane = 0; lane < GangSize; ++lane) {
          vector = Builder.CreateInsertElement(
              vector, oldValOutputs[lane], Builder.getInt32(lane));
        }

        operandGraph.valueOutput = cast<Instruction>(vector);
        Builder.SetInsertPoint(insertPoint);
      }

      newOperands.push_back(operandGraph.valueOutput);
    } else {
      newOperands.push_back(operand);
    }
  }

  // Convert an operand to a vector by splatting it when necessary.
  auto splat = [&](unsigned index) -> Value * {
    Value *&operand = newOperands[index];
    if (!operand->getType()->isVectorTy()) {
      operand = Builder.CreateVectorSplat(GangSize, operand);
    }
    return operand;
  };

  auto vectorizedType = [&]() -> VectorType * {
    return VectorType::get(I->getType(), GangSize, false);
  };

  Instruction *newInst = nullptr;

  if (I->getOpcode() == Instruction::Call) {
    auto *intrinsic = dyn_cast<IntrinsicInst>(I);

    if (intrinsic) {
      for (int i = 0; i < newOperands.size()-1; i++) {
        splat(i);
      }

      newInst = cast<Instruction>(Builder.CreateIntrinsic(
          intrinsic->getIntrinsicID(),
          vectorizedType(),
          llvm::ArrayRef<llvm::Value *>(newOperands).drop_back()));
    }
  } else if (I->getOpcode() == Instruction::GetElementPtr) {
    auto *GEP = cast<GetElementPtrInst>(I);

    splat(0);

    SmallVector<Value *> indices(
        newOperands.begin() + 1, newOperands.end());

    newInst = cast<Instruction>(Builder.CreateGEP(
        GEP->getSourceElementType(),
        newOperands[0],
        indices,
        "",
        GEP->isInBounds()));
  } else if (I->getOpcode() == Instruction::Load) {
    splat(0);

    auto *load = cast<LoadInst>(I);
    auto *newLoad = Builder.CreateMaskedGather(
        vectorizedType(),
        newOperands[0],
        load->getAlign(),
        nullptr,
        PoisonValue::get(vectorizedType()));

    maskedCalls[newLoad] = 2;
    newInst = newLoad;
  } else if (I->getOpcode() == Instruction::Store) {
    splat(0);
    splat(1);

    auto *store = cast<StoreInst>(I);

    auto *newStore = Builder.CreateMaskedScatter(
        newOperands[0],
        newOperands[1],
        store->getAlign(),
        nullptr);

    maskedCalls[newStore] = 3;
    newInst = newStore;
  } else if (I->getOpcode() == Instruction::Select) {
    splat(0);
    splat(1);
    splat(2);

    newInst = cast<Instruction>(Builder.CreateSelect(
        newOperands[0],
        newOperands[1],
        newOperands[2]));
  } else if (auto *cmp = dyn_cast<CmpInst>(I)) {
    splat(0);
    splat(1);

    newInst = cast<Instruction>(Builder.CreateCmp(
        cmp->getPredicate(),
        newOperands[0],
        newOperands[1]));
  } else if (Instruction::isBinaryOp(I->getOpcode())) {
    splat(0);
    splat(1);

    newInst = cast<Instruction>(Builder.CreateBinOp(
        static_cast<Instruction::BinaryOps>(I->getOpcode()),
        newOperands[0],
        newOperands[1]));
  } else if (Instruction::isUnaryOp(I->getOpcode())) {
    splat(0);

    newInst = cast<Instruction>(Builder.CreateUnOp(
        static_cast<Instruction::UnaryOps>(I->getOpcode()),
        newOperands[0]));
  } else if (Instruction::isCast(I->getOpcode())) {
    splat(0);

    newInst = cast<Instruction>(Builder.CreateCast(
        static_cast<Instruction::CastOps>(I->getOpcode()),
        newOperands[0],
        vectorizedType()));
  } else {
    llvm_unreachable("Unsupported opcode in vectorizedReplace");
  }

  markedForDeletion.insert(I);
  InstGraph[I].valueOutput = newInst;
}

void WholeFunctionVectorizationImpl::unvectorizedReplace(llvm::Instruction *I) {
  Instruction *insertPoint = I->getNextNode();
  IRBuilder<> Builder(insertPoint);

  std::unordered_map<int, std::vector<Instruction *>> changedOperands;

  for (unsigned i = 0; i < I->getNumOperands(); ++i) {
    Instruction *operandInst = dyn_cast<Instruction>(I->getOperand(i));
    if (operandInst && InstGraph.count(operandInst) > 0) {
      // convert valueOutput to vectorOutput
      std::vector<Instruction *> unpackedOperand;
      if (!HAS_ARRAY_OUTPUT(InstGraph[operandInst])) {
        Builder.SetInsertPoint(
            InstGraph[operandInst].valueOutput->getNextNode());
        for (unsigned i = 0; i < GangSize; ++i) {
          Instruction *ExtractedElem =
              cast<Instruction>(Builder.CreateExtractElement(
                  InstGraph[operandInst].valueOutput, i));
          unpackedOperand.push_back(ExtractedElem);
        }
        InstGraph[operandInst].arrayOutput = unpackedOperand;
        Builder.SetInsertPoint(insertPoint);
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

bool WholeFunctionVectorizationImpl::isVectorizableInstruction(Instruction *I) {
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
  case Instruction::ICmp:
  case Instruction::FCmp:
  case Instruction::Load:
  case Instruction::Store:
  case Instruction::Trunc:
  case Instruction::Select:
  case Instruction::FNeg:
  case Instruction::GetElementPtr:
  case Instruction::SExt:
  case Instruction::ZExt:
    break;
  case Instruction::Call:
    if (auto *II = dyn_cast<IntrinsicInst>(I)) {
      if (II->getIntrinsicID() == Intrinsic::fmuladd ||
          II->getIntrinsicID() == Intrinsic::sqrt) {
        break;
      }
    }
  default:
    return false;
  }

  for (Value *Op : I->operands()) {
    Type *OpType = Op->getType();
    if (!OpType->isIntegerTy() && !OpType->isFloatingPointTy() && !OpType->isPointerTy()) {
      return false;
    }
  }
  if (I->getType()->isVectorTy()) {
    return false;
  }
  return true;
}

Value *WholeFunctionVectorizationImpl::createVectorScalarAdd(IRBuilder<> &builder,
                                                       Value *Vec,
                                                       Value *Scalar) {
  auto *VecTy = cast<VectorType>(Vec->getType());
  unsigned NumElems = VecTy->getElementCount().getFixedValue();

  Value *Splat = builder.CreateVectorSplat(NumElems, Scalar);
  return builder.CreateAdd(Vec, Splat);
}

// Set up the vectorized global and local id. That is, initialize the vectors
// as <0, 1, 2,.., GangSize-1>.
void WholeFunctionVectorizationImpl::transformIdStores(BasicBlock *BB) {
  // find init of global variable (which contains the offset as an operand)
  IRBuilder<> Builder(BB);
  Builder.SetInsertPoint(BB->getFirstInsertionPt());

  std::vector<StoreInst *> Results;
  for (Instruction &I : *BB) {
    if (auto *SI = dyn_cast<StoreInst>(&I)) {
      if (SI->getPointerOperand() == GlobalIdGlobals[VectorizationDim]) {
        Builder.SetInsertPoint(SI);
        Builder.CreateStore(createVectorScalarAdd(Builder, SequentialVec,
                                                  SI->getValueOperand()),
                            VectorizedGlobalIdVar);
        break;
      } else if (SI->getPointerOperand() == LocalIdGlobals[VectorizationDim]) {
        Builder.SetInsertPoint(SI);
        Builder.CreateStore(createVectorScalarAdd(Builder, SequentialVec,
                                                  SI->getValueOperand()),
                            VectorizedLocalIdVar);
      }
    }
  }
}

Instruction *
WholeFunctionVectorizationImpl::findIncrementOfGlobal(BasicBlock *BB,
                                                GlobalVariable *GV) {
  for (Instruction &I : *BB) {
    // Match: %x = load i64, ptr @GV
    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      if (LI->getPointerOperand()->stripPointerCasts() != GV)
        continue;

      // Next instruction must be %y = add i64 %x, 1
      if (auto *AI = dyn_cast<BinaryOperator>(LI->getNextNode())) {
        Value *X;
        if (!PatternMatch::match(
                AI, PatternMatch::m_Add(PatternMatch::m_Value(X),
                                        PatternMatch::m_ConstantInt<1>())))
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
void WholeFunctionVectorizationImpl::transformForInc(BasicBlock *BB) {
  IRBuilder<> Builder(BB);
  Builder.SetInsertPoint(BB->getFirstInsertionPt());
  llvm::GlobalVariable *LocalIdVar = LocalIdIterators[VectorizationDim];
  llvm::GlobalVariable *GlobalIdVar = GlobalIdIterators[VectorizationDim];

  Instruction *LocalIteratorInc = findIncrementOfGlobal(BB, LocalIdVar);
  if (LocalIteratorInc) {
    for (int i = 0; i < 2; i++) {
      if (ConstantInt *CI =
              dyn_cast<ConstantInt>(LocalIteratorInc->getOperand(i))) {
        LocalIteratorInc->setOperand(i, ConstantInt::get(ST, GangSize));
        break;
      }
    }
    Builder.CreateStore(
        Builder.CreateAdd(Builder.CreateLoad(VectorST, VectorizedLocalIdVar),
                          StepVec),
        VectorizedLocalIdVar);
  }

  Value *newGlobalValue;
  Instruction *GlobalIteratorInc = findIncrementOfGlobal(BB, GlobalIdVar);
  if (GlobalIteratorInc) {
    for (int i = 0; i < 2; i++) {
      if (ConstantInt *CI =
              dyn_cast<ConstantInt>(GlobalIteratorInc->getOperand(i))) {
        GlobalIteratorInc->setOperand(i, ConstantInt::get(ST, GangSize));
        break;
      }
    }
    Builder.CreateStore(
        Builder.CreateAdd(Builder.CreateLoad(VectorST, VectorizedGlobalIdVar),
                          StepVec),
        VectorizedGlobalIdVar);
  }
}

void WholeFunctionVectorizationImpl::findLoadsOfGlobal(
    GlobalVariable *GV, std::vector<Instruction *> &Loads) {
  for (User *U : GV->users()) {
    if (auto *LI = dyn_cast<LoadInst>(U)) {
      BasicBlock *BB = LI->getParent();
      if (auto *M = BB->getTerminator()->getMetadata("myrole")) {
        auto *S = dyn_cast<MDString>(M->getOperand(0));
        if (S && (S->getString() == "pregion_for_inc" ||
                  S->getString() == "pregion_for_init")) {
          continue;
        }
      }
      Loads.push_back(LI);
    }
  }
}

Value *
WholeFunctionVectorizationImpl::constructLoopPredicatePrefix(IRBuilder<> &Builder,
                                                       Value *predicate) {
  // auto *VTy = VectorType::get(Builder.getInt1Ty(), GangSize, false);
  
  Value *Prefix = predicate;
  Value *TrueVec =
    ConstantVector::getSplat(ElementCount::getFixed(GangSize), Builder.getTrue());

  for (unsigned Shift = 1; Shift < GangSize; Shift <<= 1) {
    SmallVector<int> Mask;

    for (unsigned I = 0; I < GangSize; ++I) {
      if (I < Shift)
        Mask.push_back(I);                  // take "true"
      else
        Mask.push_back(GangSize + I - Shift); // take predicate
    }

    Value *Shifted =
        Builder.CreateShuffleVector(TrueVec, Prefix, Mask);

    Prefix = Builder.CreateAnd(Prefix, Shifted);
  }

  return Prefix;
}

// Starting with every load of the global and local id values, recursively walk
// the data flow graph, applying vectorization to every instruction found.
void WholeFunctionVectorizationImpl::transformIdLoads() {
  IRBuilder<> Builder(M->getContext());

  // initialize masks with blank PHI nodes
  for (BasicBlock &BB : *F) {
    Builder.SetInsertPoint(BB.getFirstInsertionPt());
    unsigned numPreds = std::distance(pred_begin(&BB), pred_end(&BB));
    if (numPreds == 0) {
      // If WGLocalSizeX >= GangSize, then the mask is all ones, otherwise [0..WGLocalSizeX] = 1
      BlockMasks[&BB] = createConstantMask(Builder, WGLocalSizeX);
    } else {
      BlockMasks[&BB] = Builder.CreatePHI(
          VectorType::get(Builder.getInt1Ty(), GangSize, false), numPreds);
    }
  }

  std::unordered_set<Instruction *> AllVectorizableLoads;
  std::vector<Instruction *> GlobalIteratorLoads;
  std::vector<Instruction *> LocalIteratorLoads;
  std::unordered_set<Instruction *> visited;
  findLoadsOfGlobal(LocalIdIterators[VectorizationDim], LocalIteratorLoads);
  for (Instruction *LoadInst : LocalIteratorLoads) {
    Builder.SetInsertPoint(LoadInst);
    Instruction *globalLoad =
        Builder.CreateLoad(VectorST, VectorizedLocalIdVar);
    InstGraph[LoadInst].valueOutput = globalLoad;
    AllVectorizableLoads.insert(LoadInst);
    traverseInstructionTree(LoadInst, visited);
  }

  findLoadsOfGlobal(GlobalIdIterators[VectorizationDim], GlobalIteratorLoads);
  for (Instruction *LoadInst : GlobalIteratorLoads) {
    Builder.SetInsertPoint(LoadInst);
    Instruction *globalLoad =
        Builder.CreateLoad(VectorST, VectorizedGlobalIdVar);
    InstGraph[LoadInst].valueOutput = globalLoad;
    AllVectorizableLoads.insert(LoadInst);
    traverseInstructionTree(LoadInst, visited);
  }

  DenseMap<Instruction *, int> in_degree;
  for (Instruction *I : visited)
    in_degree[I] = 0;

  for (Instruction *I : visited) {
    for (User *U : I->users()) {
      Instruction *UI = dyn_cast<Instruction>(U);
      if (UI && visited.count(UI) > 0)
          in_degree[UI]++;
    }
  }

  // Topological sort
  SmallVector<Instruction *> Worklist;
  for (auto &LoadInst : AllVectorizableLoads) {
    Worklist.push_back(LoadInst);
  }
  SmallVector<Instruction *> TopoOrder;
  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();
    TopoOrder.push_back(I);
    for (Instruction *User : InstGraph[I].children) {
      if (--(in_degree[User]) == 0) {
        Worklist.push_back(User);
      }
    }
  }

  for (Instruction *I : TopoOrder) {
    // Don't try to vectorize the initial loads from the global/local IDs.
    if (AllVectorizableLoads.count(I) == 0) {
      vectorizeInstruction(I);
    }
  }

  // TODO: is this redundant with eraseUsersRecursively?
  // for (Instruction *inst : markedForDeletion) {
  //   inst->replaceAllUsesWith(PoisonValue::get(inst->getType()));
  //   inst->eraseFromParent();
  // }
}

void WholeFunctionVectorizationImpl::generateMasks() {
  IRBuilder<> Builder(M->getContext());

  // This is the second stage, which traverses the blocks in approximate order,
  // and fills out the masking. Generally, just use a phi node set to inherit
  // the right mask from the predecessor. In the case of loops, the
  // preprocessing in the first stage should avoid dependency cycles.
  auto updatePHI = [](PHINode *Phi, BasicBlock *BB, Value *V) -> bool {
    int idx = Phi->getBasicBlockIndex(BB);
    if (!V) {
      V = PoisonValue::get(Phi->getType());
    }

    if (idx < 0) {
      Phi->addIncoming(V, BB);
      return V != PoisonValue::get(Phi->getType());
    }
    if (Phi->getIncomingValue(idx) == V)
      return false;

    Phi->setIncomingValue(idx, V);
    return true;
  };

  Builder.SetInsertPoint(F->getEntryBlock().begin());

  std::deque<BasicBlock *> Worklist;
  for (BasicBlock *BB : successors(&F->getEntryBlock())) {
    Worklist.push_back(BB);
  }
  std::unordered_set<BasicBlock *> Visited;

  while (!Worklist.empty()) {
    BasicBlock *BB = Worklist.front();
    Worklist.pop_front();
    bool changed = false;

    IRBuilder<> Builder(&*BB->begin());
    PHINode *phi = dyn_cast<PHINode>(BlockMasks[BB]);
    if (!phi) {
      continue;
    }

    // For each predecessor, if it is loop exiting, then override the
    // inherited mask value to be the mask of the loop header (i.e. reset the
    // mask to the start of the loop) Otherwise, simply inherit the last
    // block's mask if not already set.

    for (BasicBlock *Pred : predecessors(BB)) {
      Value *Incoming = nullptr;
      for (Loop *L = LI.getLoopFor(Pred); L; L = L->getParentLoop()) {
        if (L->isLoopExiting(Pred) && !L->contains(BB) && LoopMasks[L]) {
          Incoming = LoopMasks[L];
          break;
        }
      }
      if (Incoming) {
        changed |= updatePHI(phi, Pred, Incoming);
      }

      if (phi->getBasicBlockIndex(Pred) >= 0) {
        continue;
      }
      changed |= updatePHI(phi, Pred, BlockMasks[Pred]);
    }

    // Handle loop mask. At the exit of the loop, the mask should reset to the
    // mask value before the loop, requiring this additional work to maintain
    // that value
    for (Loop *L = LI.getLoopFor(BB); L; L = L->getParentLoop()) {
      // Assumes that each loop header only has one direct predecessor. If this
      // assumption breaks, change this to regenerate the phi clone if changed
      // == true instead.
      if (LoopMasks.count(L) == 0 && L->getHeader() == BB) {
        PHINode *phiClone = cast<PHINode>(phi->clone());
        for (BasicBlock *Pred : predecessors(BB)) {
          if (L->contains(Pred)) {
            phiClone->setIncomingValueForBlock(Pred, phiClone);
          }
        }
        LoopMasks[L] = phiClone;
        Builder.Insert(phiClone);
        changed = true;
        for (BasicBlock *LoopBB : L->getBlocks()) {
          Worklist.push_back(LoopBB);
        }
      }
    }
    // BlockMasks[BB] = phi;
    if (changed || Visited.count(BB) == 0) {
      for (BasicBlock *succ : successors(BB)) {
        Worklist.push_back(succ);
      }
    }
    Visited.insert(BB);
  }
}

void WholeFunctionVectorizationImpl::transformControlFlow() {
  IRBuilder<> Builder(M->getContext());

  // After setting up all the masks with the phi, we need to merge branches into
  // a single straight line path.

  for (BranchInst *br : branchInsts) {
    BasicBlock *trueBB = br->getSuccessor(0);
    BasicBlock *falseBB = br->getSuccessor(1);

    BasicBlock *mergeBB = PDT.findNearestCommonDominator(trueBB, falseBB);

    // In the case of divergent control flow conditioned on something that is dependent on a vectorized value,
    // we just take the value of the last loop (which is what pocl was already doing).
    if (!mergeBB) {
      Instruction *condInst = dyn_cast<Instruction>(br->getCondition());
      assert(condInst && InstGraph.count(condInst) > 0 && "Condition must be an instruction in the graph");
      Builder.SetInsertPoint(br);

      Value *cond = InstGraph[condInst].valueOutput;

      auto *VecTy = cast<FixedVectorType>(cond->getType());
      unsigned N = GangSize;

      // Convert the mask to an integer bitmask.
      Type *IntTy = Builder.getIntNTy(N);
      Value *Bits = Builder.CreateBitCast(BlockMasks[br->getParent()], IntTy);

      // Find the highest set bit.
      Value *LeadingZeros = Builder.CreateBinaryIntrinsic(
          Intrinsic::ctlz, Bits, Builder.getFalse());

      Value *LastIdx = Builder.CreateSub(
          ConstantInt::get(IntTy, N - 1),
          LeadingZeros);

      // Extract the boolean at that index.
      Value *newCond = Builder.CreateExtractElement(cond, LastIdx);
      br->setCondition(newCond);
      continue;
    }

    std::unordered_set<BasicBlock *> trueBBMergePreds;
    std::unordered_set<BasicBlock *> falseBBMergePreds;
    // find all immediate predecessors to the merge point that are on the path
    // from each branch
    for (BasicBlock *Pred : predecessors(mergeBB)) {
      if (Pred == mergeBB)
        continue;

      SmallPtrSet<BasicBlock *, 1> exclusionSet({mergeBB});
      // TODO: isPotentiallyReachable is not strict, which can lead to cases where a predecessor is believed to be reachable from both branches.
      // when it isn't. Note that in the case of a true positive, that a merge block predecessor is actually reachable from both branches, the
      // entire masking system is fundamentally incapable of handling it.
      if (isPotentiallyReachable(trueBB, Pred, &exclusionSet)) {
        trueBBMergePreds.insert(Pred);
      }
      if (isPotentiallyReachable(falseBB, Pred, &exclusionSet)) {
        falseBBMergePreds.insert(Pred);
      }
    }

    // For the false branch, we need to change the mask to the false predicate
    // mask.
    PHINode *falsePhiNode = cast<PHINode>(BlockMasks[falseBB]);
    Value *falseMask = falsePhiNode->getIncomingValueForBlock(br->getParent());
    falsePhiNode->removeIncomingValue(br->getParent(), false);

    // Then, we set up the true branch by rerouting the merge block predecessor
    // terminators to the beginning of the false branch. Then, we set the masks
    // for the false branch.

    // If the true branch has no blocks, then the branch block should point directly to the falsePhiNodes instead
    if (trueBBMergePreds.empty()) {
      trueBBMergePreds.insert(br->getParent());
      trueBB = falseBB;
    }
    PHINode *MergeMask = cast<PHINode>(BlockMasks[mergeBB]);

    for (BasicBlock *trueBBMergePred : trueBBMergePreds){
      BranchInst *mergePredBr =
          cast<BranchInst>(trueBBMergePred->getTerminator());
      for (int i = 0; i < mergePredBr->getNumSuccessors(); i++) {
        if (mergePredBr->getSuccessor(i) == mergeBB) {
          mergePredBr->setSuccessor(i, falseBB);
        }
      }
      MergeMask->removeIncomingValue(trueBBMergePred, false);
      falsePhiNode->addIncoming(falseMask, trueBBMergePred);
    }

    // Finally, for the mask for the merge block, we need to set them to the
    // correct values (that is, the mask before the predicate). If the false
    // branch has no blocks (i.e. points directly to the merge block), we set
    // the mask differently
    if (falseBBMergePreds.size() == 0) {
      for (BasicBlock *trueBBMergePred : trueBBMergePreds) {
        MergeMask->setIncomingValueForBlock(trueBBMergePred,
                                            BlockMasks[br->getParent()]);
      }
    } else {
      for (BasicBlock *falseBBMergePred : falseBBMergePreds) {
        MergeMask->setIncomingValueForBlock(falseBBMergePred,
                                            BlockMasks[br->getParent()]);
      }
    }

    Builder.SetInsertPoint(br);
    Builder.CreateBr(trueBB);
    br->eraseFromParent();
  }


}

void WholeFunctionVectorizationImpl::applyMasks() {
  // Force mask all stores
  llvm::SmallVector<Instruction *> queue;
  for (Instruction &I : instructions(F)) {
    queue.push_back(&I);
  }

  for (Instruction *I : queue) {
    // If a store instruction is not dependent on the vectorized global/local id,
    // then we don't mask it, but we still need to check if it should be executed
    // at all (i.e, when the mask has any true values).
    if (auto *SI = dyn_cast<StoreInst>(I)) {
      BasicBlock *BB = SI->getParent();
      if (auto *M = BB->getTerminator()->getMetadata("myrole")) {
        // auto *S = dyn_cast<MDString>(M->getOperand(0));
        // if (S && (S->getString() == "pregion_for_inc" ||
        //           S->getString() == "pregion_for_init")) {
        //   continue;
        // }
        continue;
      }

      if (SI->getParent()->getName() == "entry.barrier") {
        continue;
      }

      IRBuilder<> Builder(SI);

      // Reduce <8 x i1> -> i1
      Value *AnyMask = Builder.CreateOrReduce(BlockMasks[BB]);
      Instruction *ThenTerm = SplitBlockAndInsertIfThen(AnyMask, SI, false);

      BasicBlock *StoreBB = ThenTerm->getParent();
      
      SI->removeFromParent();
      Builder.SetInsertPoint(ThenTerm);
      Builder.Insert(SI);

      BlockMasks[StoreBB] = BlockMasks[BB];
      BlockMasks[StoreBB->getNextNode()] = BlockMasks[BB];
    }
  }

  for (auto &[maskedCall, maskIdx] : maskedCalls) {
    maskedCall->setArgOperand(maskIdx, BlockMasks[maskedCall->getParent()]);
  }
}

bool WholeFunctionVectorizationImpl::privatizeContext() {
  CreateBuilder(Builder, F->getEntryBlock());
  if (VectorizedGlobalIdVar != nullptr) {
    AllocaInst *VectorizedGlobalIdAlloca =
        Builder.CreateAlloca(VectorST, 0, GID_G_NAME_VECTORIZED);
    for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
      for (BasicBlock::iterator ii = i->begin(), ee = i->end(); ii != ee;
           ++ii) {
        ii->replaceUsesOfWith(VectorizedGlobalIdVar, VectorizedGlobalIdAlloca);
      }
    }
  }

  if (VectorizedLocalIdVar != nullptr) {
    AllocaInst *VectorizedLocalIdAlloca =
        Builder.CreateAlloca(VectorST, 0, LID_G_NAME_VECTORIZED);
    for (Function::iterator i = F->begin(), e = F->end(); i != e; ++i) {
      for (BasicBlock::iterator ii = i->begin(), ee = i->end(); ii != ee;
           ++ii) {
        ii->replaceUsesOfWith(VectorizedLocalIdVar, VectorizedLocalIdAlloca);
      }
    }
  }
  M->eraseGlobalVariable(VectorizedGlobalIdVar);
  M->eraseGlobalVariable(VectorizedLocalIdVar);

  return true;
}

static void eraseUsersRecursively(Value *V) {
  SmallPtrSet<Instruction *, 32> ToErase;

  SmallVector<Value *, 32> Worklist{V};
  SmallPtrSet<Value *, 32> Visited;

  while (!Worklist.empty()) {
    Value *Cur = Worklist.pop_back_val();

    if (!Visited.insert(Cur).second)
      continue;

    for (User *U : Cur->users()) {
      if (auto *I = dyn_cast<Instruction>(U)) {
        Worklist.push_back(I);
        ToErase.insert(I);
      }
    }
  }

  for (Instruction *I : ToErase) {
    if (!I->use_empty())
      I->replaceAllUsesWith(PoisonValue::get(I->getType()));

    I->eraseFromParent();
  }
}

bool WholeFunctionVectorizationImpl::processFunction(Function &F) {
  std::vector<BasicBlock *> forInitBlocks;
  std::vector<BasicBlock *> forBodyBlocks;
  std::vector<BasicBlock *> forIncBlocks;

  bool shouldVectorize = false;
  for (auto &BB : F) {
    if (blockHasTag(&BB, "pregion_for_inc")) {
      shouldVectorize = true;
      break;
    }
  }
  if (!shouldVectorize) {
    return false;
  }

  for (auto &BB : F) {
    if (blockHasTag(&BB, "pregion_for_inc")) {
      // forIncBlocks.push_back(&BB);
      transformForInc(&BB);
    } else {
      transformIdStores(&BB);
    }
  }

  transformIdLoads();
  generateMasks();
  transformControlFlow();

  eraseUsersRecursively(GlobalIdIterators[VectorizationDim]);
  eraseUsersRecursively(LocalIdIterators[VectorizationDim]);

  applyMasks();

  return true;
}

llvm::PreservedAnalyses
WholeFunctionVectorization::run(llvm::Function &F,
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

  WholeFunctionVectorizationImpl WIL(DT, LI, PDT, VUA);
  // llvm::verifyFunction(F);

  return WIL.runOnFunction(F) ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
