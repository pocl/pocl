// LLVM pass to recursively inline kernels which are called by other kernels
//
// Copyright (c) 2020 Michal Babej / Tampere University
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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "InlineKernels.hh"
#include "LLVMUtils.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

//#define DEBUG_INLINE_KERNELS

#include "pocl_llvm_api.h"

#include <iostream>
#include <string>

#define PASS_NAME "inline-kernels"
#define PASS_CLASS pocl::InlineKernels
#define PASS_DESC                                                              \
  "Inline kernels which are called from other kernels (only at the callsite)"

namespace pocl {

using namespace llvm;

static bool inlineKernelCalls(Function &F, NamedMDNode *KernelMDs) {

#ifdef DEBUG_INLINE_KERNELS
  std::cerr << "checking " << F.getName().str() << " for inline-able kernels\n";
#endif

  bool ChangedIter = false;
  bool ChangedAny = false;
  do {
    ChangedIter = false;
    Function::iterator I = F.begin();
    for (Function::iterator E = F.end(); I != E; ++I) {
      for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE;) {

        CallInst *CInstr = dyn_cast<CallInst>(BI++);
        if (CInstr == nullptr)
          continue;

        Function *Callee = CInstr->getCalledFunction();
        if (Callee == nullptr)
          continue;

        if (!pocl::isKernelToProcess(*Callee)) {
#ifdef DEBUG_INLINE_KERNELS
          std::cerr << "NOT a kernel, NOT Inlining call "
                    << Callee->getName().str() << "\n";
#endif
          if (inlineKernelCalls(*Callee, KernelMDs)) {
            ChangedIter = true;
            ChangedAny = true;
            break;
          } else
            continue;
        }

#ifdef DEBUG_INLINE_KERNELS
        std::cerr << "Inlining kernel call " << Callee->getName().str() << "\n";
#endif
        InlineFunctionInfo IFI;
        llvm::InlineFunction(*CInstr, IFI);
        ChangedIter = true;
        ChangedAny = true;
        break;
      }
      if (ChangedIter)
        break;
    }
  } while (ChangedIter);

  return ChangedAny;
}

static bool inlineKernels(Function &F) {
  SmallPtrSet<Function *, 8> functions_to_inline;
  SmallVector<Value *, 8> pending;

  Module *M = F.getParent();
  std::string KernelName;
  getModuleStringMetadata(*M, "KernelName", KernelName);

  if (F.getName().str() != KernelName)
    return false;

  if (F.isDeclaration())
    return false;

  const Module *m = F.getParent();
  NamedMDNode *kernels = m->getNamedMetadata("opencl.kernels");

  bool changed = inlineKernelCalls(F, kernels);

#ifdef DEBUG_INLINE_KERNELS
  if (changed) {
    std::cerr << "******************************************************\n";
    std::cerr << "After inlineKernelCalls:\n";
    m->dump();
    std::cerr << "******************************************************\n";
  }
#endif

  return changed;
}

#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
char InlineKernels::ID = 0;

bool InlineKernels::runOnFunction(Function &F) { return inlineKernels(F); }

void InlineKernels::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addPreserved<WorkitemHandlerChooser>();
}

REGISTER_OLD_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

#else

llvm::PreservedAnalyses InlineKernels::run(llvm::Function &F,
                                           llvm::FunctionAnalysisManager &AM) {
  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();
  return inlineKernels(F) ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

#endif

} // namespace pocl
