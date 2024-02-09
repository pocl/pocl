// LLVM module pass that adds a constant offset to all addressess that could
// access SVM regions.
//
// Copyright (c) 2024 Pekka Jääskeläinen / Intel Finland Oy
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
#include <llvm/IR/Module.h>
#include <llvm/Pass.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/CommandLine.h>
POP_COMPILER_DIAGS

#include "LLVMUtils.h"
#include "SVMOffset.hh"
#include "WorkitemHandlerChooser.h"

#include <iostream>
#include <set>
#include <string>

//#include "pocl_llvm_api.h"

#define PASS_NAME "svm-offset"
#define PASS_CLASS pocl::SVMOffset
#define PASS_DESC "Adds an address offset to any potential SVM access"

using namespace llvm;

namespace pocl {

static cl::opt<uint64_t> SVMOffsetValue(
    "svm-offset-value", cl::init(0), cl::Hidden,
    cl::desc("The unsigned SVM offset value to add (wraparound for negative offsets)."));

llvm::PreservedAnalyses SVMOffset::run(llvm::Module &M,
                                       llvm::ModuleAnalysisManager &AM) {
  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();
  bool Changed = false;

  //M.dump();

  if (SVMOffsetValue == 0)
    return PreservedAnalyses::all();

  for (llvm::Module::iterator MI = M.begin(), ME = M.end(); MI != ME; ++MI) {

    llvm::Function *F = &*MI;
    const bool isKernel = isKernelToProcess(*F);
    if (isKernel) {
      // Add a negative offset to the global buffers pointers since they will
      // be adjusted (also) by the runtime at kernel argument setting. The
      // rest of the adjustments will then reverse this offset, leaning to
      // the standard compiler optimizations to remove unnecessary
      // back-and-forth offsetting.
      for (Function::arg_iterator Arg = F->arg_begin(), E = F->arg_end();
           Arg != E; ++Arg) {
        if (!Arg->getType()->isPointerTy() ||
            Arg->getType()->getPointerAddressSpace() !=
                SPIR_ADDRESS_SPACE_GLOBAL)
          continue;
        IRBuilder<> IRBuilder(F->getEntryBlock().getFirstNonPHI());
        Value *NegOffsettedPtr = IRBuilder.CreateGEP(
            IRBuilder.getInt8Ty(), Arg, IRBuilder.getInt64(-SVMOffsetValue));

        // Replace all uses of the old non-offsetted arg (except the offsetting
        // instr itself) with the offsetted one.
        Arg->replaceUsesWithIf(
            NegOffsettedPtr, [NegOffsettedPtr](Use &U) -> bool {
              if (Instruction *UseInst = dyn_cast<Instruction>(U.getUser()))
                return UseInst != NegOffsettedPtr;
              return false;
            });
      }
    }
    for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
      for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE;) {

        Instruction *Inst = dyn_cast<Instruction>(BI++);
        if (Inst == nullptr) continue;

        LoadInst *Load = dyn_cast<LoadInst>(Inst);
        StoreInst *Store = dyn_cast<StoreInst>(Inst);

        if (Load == nullptr && Store == nullptr) continue;

        unsigned AddressSpace =
          Load == nullptr ? Store->getPointerAddressSpace() : Load->getPointerAddressSpace();

        if (AddressSpace != SPIR_ADDRESS_SPACE_GLOBAL)
          continue;

        Value *PtrOperand =
          Load == nullptr ? Store->getPointerOperand() : Load->getPointerOperand();

        llvm::IRBuilder<> IRBuilder(Inst);
        Value *OffsettedPtr =
          IRBuilder.CreateGEP(IRBuilder.getInt8Ty(),
                              PtrOperand, IRBuilder.getInt64(SVMOffsetValue));
        Inst->replaceUsesOfWith(PtrOperand, OffsettedPtr);
        Changed = true;
      }
    }
  }

  //M.dump();
  return Changed ? PAChanged : PreservedAnalyses::all();
}


REGISTER_NEW_MPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl


extern "C" POCL_EXPORT ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "pocl-passes", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == PASS_NAME) {
                    MPM.addPass(PASS_CLASS());
                    return true;
                  }
                  return false;
                });
          }};
}
