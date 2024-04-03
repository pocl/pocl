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

/* An LLVM pass to adjust LLVM IR's memory accessess with a fixed offset.

   The adjustment is needed in case the host's SVM region's start address
   differs from the device's.

   Input: A non-adjusted kernel with all global pointers assumed to be
   pre-adjusted by the runtime to point to the SVM region of the targeted
   device.

   The pass works as follows with the general principle that the pointer
   addressess are adjusted to the correct offset at the point of a memory
   access. Due to generic pointers it is not always possible to figure out
   if the address space of a pointer is global or not, thus we must ensure
   all pointers can be adjusted at their usage time, thus they have to be
   negatively adjusted at the pointer creation or "import time".

   This means that all pointers that are created by

   - allocas,
   - when taking an address of a global variable or are
   - input to the kernel as arguments

   are negatively adjusted so we can adjust them back when accessing the
   memory. It leans heavily to compiler to remove the unnecessary
   negative/positive adjustment pairs.

   Pointer arguments to calls to defined functions are not adjusted, but
   the functions itself are handled separately. Arguments to undefined
   functions are assumed to be builtin functions which expect valid fixed
   pointers, thus they are adjusted at the call site.

   TODO: Indirect accesses? They are pointers loaded from another buffer
   or such. The pointers are global SVM so they should be fixed as well.
*/

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
    cl::desc("The unsigned SVM offset value to add (wraparound for negative "
             "offsets)."));

llvm::PreservedAnalyses SVMOffset::run(llvm::Module &M,
                                       llvm::ModuleAnalysisManager &AM) {
  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();
  bool Changed = false;

  // std::cerr << "SVMOffset adding region offset " << SVMOffsetValue << std::endl;
  // M.dump();

  if (SVMOffsetValue == 0)
    return PreservedAnalyses::all();

  for (llvm::Module::iterator MI = M.begin(), ME = M.end(); MI != ME; ++MI) {

    llvm::Function *F = &*MI;
    const bool isKernel = F->getCallingConv() == llvm::CallingConv::SPIR_KERNEL;
    if (isKernel) {
      // Add a negative offset to the arg pointers since they will
      // be adjusted (also) by the runtime at kernel argument setting. The
      // runtime has to adjust them to make them valid SVM pointers for the
      // Subbuffers.
      for (Function::arg_iterator Arg = F->arg_begin(), E = F->arg_end();
           Arg != E; ++Arg) {
        if (!Arg->getType()->isPointerTy())
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

        // Negatively adjust created new pointers.
        if ((isa<BitCastInst>(Inst) && dyn_cast<GlobalValue>(Inst->getOperand(0))) ||
            (isa<GetElementPtrInst>(Inst) &&
             dyn_cast<GlobalValue>(
               cast<GetElementPtrInst>(Inst)->getPointerOperand())) ||
            isa<AllocaInst>(Inst)) {

          llvm::IRBuilder<> IRBuilder(dyn_cast<Instruction>(BI));
          Value *NegOffsettedPtr = IRBuilder.CreateGEP(
              IRBuilder.getInt8Ty(), Inst, IRBuilder.getInt64(-SVMOffsetValue));

          // Replace all uses of the old non-offsetted alloc (except the
          // offsetting instr itself) with the offsetted one.
          Inst->replaceUsesWithIf(
              NegOffsettedPtr, [NegOffsettedPtr](Use &U) -> bool {
                if (Instruction *UseInst = dyn_cast<Instruction>(U.getUser()))
                  return UseInst != NegOffsettedPtr;
                return false;
              });
          Changed = true;
          continue;
        }

        // Fix built-in calls: If the argument is a pointer, offset it.
        if (CallInst *Call = dyn_cast<CallInst>(Inst)) {
          std::vector<Value *> InstrsToFix;
          // Fix only the args to non-visible (built-in) functions. The rest of
          // the functions we will fix in their definition.
          if (!Call->getCalledFunction()->isDeclaration())
            continue;

          // Treat __to_global as a special case. It's an address space cast;
          // inputs a pointer and returns a pointer.
          // Should we actually deoffset all ptr return values
          // from builtins instead. Likely yes... are there more of
          // such builtins?
          if (Call->getCalledFunction()->getName().str() == "__to_global")
            continue;

          for (const auto &Arg : Call->args()) {
            // There can be multiple uses of the same pointer in the arg list.
            // Ensure we fix them only once.
            if (std::find(InstrsToFix.begin(), InstrsToFix.end(), Arg.get()) ==
                    InstrsToFix.end() &&
                Arg.get()->getType()->isPointerTy())
              InstrsToFix.push_back(Arg.get());
          }
          for (llvm::Value *VToFix : InstrsToFix) {
            llvm::IRBuilder<> IRBuilder(Inst);
            Value *OffsettedPtr =
                IRBuilder.CreateGEP(IRBuilder.getInt8Ty(), VToFix,
                                    IRBuilder.getInt64(SVMOffsetValue));
            Call->replaceUsesOfWith(VToFix, OffsettedPtr);
            Changed = true;
          }
          continue;
        }

        // TODO: Other memory operations such as atomics? Are they covered here,
        // or can we assume all atomics are done via builtin functions?
        LoadInst *Load = dyn_cast<LoadInst>(Inst);
        StoreInst *Store = dyn_cast<StoreInst>(Inst);

        if (Load == nullptr && Store == nullptr) continue;

        unsigned AddressSpace =
          Load == nullptr ? Store->getPointerAddressSpace() : Load->getPointerAddressSpace();

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

  // M.dump();
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
