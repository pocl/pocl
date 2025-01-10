// LLVM function pass that ensures corner cases of integer division
// do not trigger undefined behavior. The generated code is branchless,
// and identical for scalar & vector div/rem instructions.
//
// Copyright (c) 2024 Michal Babej / Intel Finland Oy
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Module.h>
#include <llvm/Pass.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

#include "LLVMUtils.h"
#include "SanitizeUBofDivRem.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#include <iostream>
#include <set>
#include <string>

#include "pocl_llvm_api.h"

// #define DEBUG_SANITIZE_UB_DIV_REM

#define PASS_NAME "sanitize-ub-of-div-rem"
#define PASS_CLASS pocl::SanitizeUBofDivRem
#define PASS_DESC "Handle integer division&remainder UB in kernel code"

namespace pocl {

using namespace llvm;

static void fixUnsignedDivRem(IRBuilder<> &Builder, BinaryOperator *BinI,
                              Value *Dividend, Value *Divisor,
                              ValueToValueMapTy &ZeroCheckedDivisorCache) {
  auto It = ZeroCheckedDivisorCache.find(Divisor);
  if (It != ZeroCheckedDivisorCache.end()) {
    BinI->setOperand(1, It->second);
    return;
  }

  // divisor == 0
  Value *Condition = Builder.CreateCmp(CmpInst::Predicate::ICMP_EQ, Divisor,
                                       ConstantInt::get(Divisor->getType(), 0));
  Value *FixedDivisor = Builder.CreateSelect(
      Condition, ConstantInt::get(Divisor->getType(), 1), // true
      Divisor);                                           // false

#ifdef DEBUG_SANITIZE_UB_DIV_REM
  std::cerr << "UNSIGNED fixing OP2: \n";
  Fixed->dump();
#endif

  BinI->setOperand(1, FixedDivisor);
  ZeroCheckedDivisorCache.insert(std::make_pair(Divisor, FixedDivisor));
}

static void fixSignedDivRem(IRBuilder<> &Builder, BinaryOperator *BinI,
                            Value *Dividend, Value *Divisor,
                            ValueToValueMapTy &ZeroCheckedCache,
                            ValueToValueMapTy &IntMinCheckedCache) {
  fixUnsignedDivRem(Builder, BinI, Dividend, Divisor, ZeroCheckedCache);

  auto It = IntMinCheckedCache.find(Divisor);
  if (It != IntMinCheckedCache.end()) {
    BinI->setOperand(0, It->second);
    return;
  }

  Type *T = Divisor->getType();
  // for signed div/rem, we need to fix also the dividend (INT_MIN / -1)
  Value *Condition1 = Builder.CreateCmp(CmpInst::Predicate::ICMP_EQ, Divisor,
                                        ConstantInt::getSigned(T, -1));
  // smallest negative value representable in given integer bit width
  int64_t IntMin = (int64_t)1 << (T->getScalarSizeInBits() - 1);
  Value *Condition2 = Builder.CreateCmp(CmpInst::Predicate::ICMP_EQ, Dividend,
                                        ConstantInt::getSigned(T, IntMin));
  Value *BothConds = Builder.CreateAnd(Condition1, Condition2);
  Value *ConditionZExt = Builder.CreateZExt(BothConds, Dividend->getType());
  Value *FixedDividend = Builder.CreateAdd(Dividend, ConditionZExt);

#ifdef DEBUG_SANITIZE_UB_DIV_REM
  std::cerr << "SIGNED fixing OP1: \n";
  Fixed2->dump();
#endif

  BinI->setOperand(0, FixedDividend);
  IntMinCheckedCache.insert(std::make_pair(Divisor, FixedDividend));
}

static bool sanitizeUBofDivRem(llvm::Function &F) {
  IRBuilder<> Builder(F.getContext());

#ifdef DEBUG_SANITIZE_UB_DIV_REM
  std::cerr << "BEFORE:\n";
  F.dump();
#endif

  // save already fixed values
  ValueToValueMapTy ZeroCheckedCache;
  ValueToValueMapTy IntMinCheckedCache;

  bool Changed = false;
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE;) {
      BinaryOperator *BinI = dyn_cast<BinaryOperator>(BI++);
      if (BinI == nullptr)
        continue;
      if (!BinI->isIntDivRem())
        continue;
#ifdef DEBUG_SANITIZE_UB_DIV_REM
      std::cerr << "Fixing division/remainder in Inst: \n";
      BinI->dump();
#endif
      Value *Dividend = BinI->getOperand(0);
      Value *Divisor = BinI->getOperand(1);
      if (Dividend == Divisor) {
        Value *One = ConstantInt::get(BinI->getType(), 0);
        BinI->replaceAllUsesWith(One);
        BinI->eraseFromParent();
        Changed = true;
        continue;
      }

      // skip non-problematic cases for constant divisors
      if (ConstantInt *ConstDiv = dyn_cast<ConstantInt>(Divisor)) {
        if ((BinI->getOpcode() == Instruction::UDiv ||
            BinI->getOpcode() == Instruction::URem) && (!ConstDiv->isZero())) {
          continue;
        }
        if ((BinI->getOpcode() == Instruction::SDiv ||
            BinI->getOpcode() == Instruction::SRem)
            && (!ConstDiv->isZero()) && (!ConstDiv->isMinusOne())) {
          continue;
        }
      }

      Builder.SetInsertPoint(BinI);
      if (BinI->getOpcode() == Instruction::SDiv ||
          BinI->getOpcode() == Instruction::SRem) {
        fixSignedDivRem(Builder, BinI, Dividend, Divisor, ZeroCheckedCache,
                        IntMinCheckedCache);
      } else {
        fixUnsignedDivRem(Builder, BinI, Dividend, Divisor, ZeroCheckedCache);
      }

      Changed = true;
    }
  }

#ifdef DEBUG_SANITIZE_UB_DIV_REM
  if (Changed) {
    std::cerr << "AFTER:\n";
    F.dump();
  }
#endif
  return Changed;
}

llvm::PreservedAnalyses SanitizeUBofDivRem::run(llvm::Function &F,
                                           llvm::FunctionAnalysisManager &AM) {
  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();
  return sanitizeUBofDivRem(F) ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
