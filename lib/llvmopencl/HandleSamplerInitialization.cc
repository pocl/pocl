// LLVM function pass to convert the sampler initializer calls to a target
// desired format
//
// Copyright (c) 2017 Pekka Jääskeläinen / TUT
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
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IRBuilder.h>

#include "HandleSamplerInitialization.h"
#include "LLVMUtils.h"
#include "VariableUniformityAnalysis.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#include <iostream>
#include <set>

#include "pocl.h"

#define PASS_NAME "handle-samplers"
#define PASS_CLASS pocl::HandleSamplerInitialization
#define PASS_DESC "Handles sampler initialization calls."

namespace pocl {

using namespace llvm;

static bool handleSamplerInitialization(Function &F) {

  // Collect the sampler init calls to handle first, do not remove them
  // instantly as it'd invalidate the iterators.
  std::set<CallInst*> Calls;

  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE; ++BI) {
      Instruction *Instr = dyn_cast<Instruction>(BI);
      if (!llvm::isa<CallInst>(Instr)) continue;
      CallInst *CallInstr = dyn_cast<CallInst>(Instr);
      if (CallInstr->getCalledFunction() != nullptr &&
	  CallInstr->getCalledFunction()->getName() ==
	  "__translate_sampler_initializer") {
	Calls.insert(CallInstr);
      }
    }
  }

  bool Changed = false;
  for (CallInst *C : Calls) {
    llvm::IRBuilder<> Builder(C);

    // get the type of the return value of __translate_sampler
    // this may not always be opencl.sampler_t, it could be a remapped type.
#if LLVM_MAJOR < 15
    Type *type = C->getCalledOperand()->getType();
    PointerType *pt = dyn_cast<PointerType>(type);
    FunctionType *ft = dyn_cast<FunctionType>(pt->getPointerElementType());
    Type *rettype = ft->getReturnType();
#else
    Function *SF = C->getCalledFunction();
    assert(SF);
    Type *rettype = SF->getReturnType();
#endif

    ConstantInt *SamplerValue = dyn_cast<ConstantInt>(C->arg_begin()->get());

    IntegerType *it;
    if (F.getParent()->getDataLayout().getPointerSizeInBits() == 64)
      it = Builder.getInt64Ty();
    else
      it = Builder.getInt32Ty();
    // cast the sampler value to intptr_t
    Constant *sampler = ConstantInt::get(it, SamplerValue->getZExtValue());
    // then bitcast it to return value (opencl.sampler_t*)
    Value *val = Builder.CreateBitOrPointerCast(sampler, rettype);

    C->replaceAllUsesWith(val);
    C->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
char HandleSamplerInitialization::ID = 0;

bool HandleSamplerInitialization::runOnFunction(Function &F) {
  return handleSamplerInitialization(F);
}

void
HandleSamplerInitialization::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addPreserved<pocl::VariableUniformityAnalysis>();
  AU.addPreserved<WorkitemHandlerChooser>();
}

REGISTER_OLD_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

#else

llvm::PreservedAnalyses
HandleSamplerInitialization::run(llvm::Function &F,
                                 llvm::FunctionAnalysisManager &AM) {
  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();
  PAChanged.preserve<VariableUniformityAnalysis>();
  return handleSamplerInitialization(F) ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

#endif

} // namespace pocl
