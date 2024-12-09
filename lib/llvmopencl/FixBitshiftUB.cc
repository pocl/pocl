// LLVM module pass that ensures bitshift operators conform to OpenCL behavior
//
// on x86-64 CPU, shifting a scalar by a value N will result in actual shift by
// log2(N) least significant bits of N. IOW, (int32 << 33) is the same as
// (int32_t << 1). Shifting with AVX registers however behaves differently;
// shifting by too many bits results in zero.
//
// According to the C standard, this is fine; shifting by >type-width bits
// is undefined behavior, so both ways are acceptable. But in OpenCL C, it is
// not undefined behavior ; OpenCL C requires *all* shifts follow the "log2(N)
// least significant bits of N" rule.
// https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/shiftOperators.html
//
// Copyright (c) 2024 Michal Babej / Intel Finland Oy
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
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Module.h>
#include <llvm/Pass.h>

#include "FixBitshiftUB.h"
#include "LLVMUtils.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#include <iostream>

#include "pocl_llvm_api.h"

// #define DEBUG_FIX_BITSHIFT_UB

#define PASS_NAME "fix-bitshift-ub"
#define PASS_CLASS pocl::FixBitshiftUB
#define PASS_DESC "ensure bitshift operators conform to OpenCL behavior"

namespace pocl {

using namespace llvm;

static bool handleBitshiftUB(llvm::Function &F) {
  IRBuilder<> Builder(F.getContext());

  bool Changed = false;
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE;) {
      BinaryOperator *BinI = dyn_cast<BinaryOperator>(BI++);
      if (BinI == nullptr)
        continue;
      if (!BinI->isShift())
        continue;
      Value *ShiftBy = BinI->getOperand(1);
#ifdef DEBUG_FIX_BITSHIFT_UB
      std::cerr << "Fixing BitSHift in second operand of Inst: \n";
      BinI->dump();
#endif
      Type *ShiftByType = ShiftBy->getType();
      assert(ShiftByType->isIntOrIntVectorTy());
      Type *ElemT = ShiftByType->getScalarType();
      assert(ElemT->isIntegerTy());
      IntegerType *IntT = dyn_cast<IntegerType>(ElemT);
      unsigned ShiftByMaskInt = IntT->getBitWidth() - 1;

      ConstantInt *ConstShift = dyn_cast<ConstantInt>(ShiftBy);
      if (ConstShift) {
        // constant value. Either it's small enough & we can skip, or
        // we can adjust it
        if (ConstShift->uge(IntT->getBitWidth())) {
          uint64_t Val = ConstShift->getLimitedValue();
          Value *ShiftByMask =
              ConstantInt::get(ShiftByType, (Val & ShiftByMaskInt));
          BinI->setOperand(1, ShiftByMask);
#ifdef DEBUG_FIX_BITSHIFT_UB
          std::cerr << "Fixed BitSHift with constant shift: \n";
          BinI->dump();
#endif
          Changed = true;
        } else {
#ifdef DEBUG_FIX_BITSHIFT_UB
          std::cerr << "Fix BitSHift: SKIPPED, constant shift \n";
#endif
          continue;
        }
      } else {
        // non-constant bitshift
        Value *ShiftByMask = ConstantInt::get(ShiftByType, ShiftByMaskInt);
        Builder.SetInsertPoint(BinI);
        Value *MaxShiftBy = Builder.CreateAnd(ShiftBy, ShiftByMask);
        BinI->setOperand(1, MaxShiftBy);
#ifdef DEBUG_FIX_BITSHIFT_UB
        std::cerr << "Fixed BitSHift Inst: \n";
        BinI->dump();
#endif
        Changed = true;
      }
    }
  }

  return Changed;
}

llvm::PreservedAnalyses FixBitshiftUB::run(llvm::Function &F,
                                           llvm::FunctionAnalysisManager &AM) {
  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();
  return handleBitshiftUB(F) ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
