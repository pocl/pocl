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

#include <iostream>
#include <string>

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "config.h"
#define CLANG_MAJOR LLVM_MAJOR
#include "_libclang_versions_checks.h"

POP_COMPILER_DIAGS

using namespace llvm;

namespace {
class InlineKernels : public FunctionPass {

public:
  static char ID;
  InlineKernels() : FunctionPass(ID) {}

  virtual bool runOnFunction(Function &F);
};
} // namespace

extern cl::opt<std::string> KernelName;

char InlineKernels::ID = 0;
static RegisterPass<InlineKernels> X("inline-kernels",
                                     "Inline kernels which are called from "
                                     "other kernels (only at the callsite)");

//#define DEBUG_INLINE_KERNELS

// Returns true in case the given function is a kernel
static bool isOpenCLKernel(const Function *F, NamedMDNode *KernelMDs) {

  if (F->getMetadata("kernel_arg_access_qual"))
    return true;

  if (KernelMDs) {
    for (unsigned i = 0, e = KernelMDs->getNumOperands(); i != e; ++i) {
      if (KernelMDs->getOperand(i)->getOperand(0) == nullptr)
        continue; // globaldce might have removed uncalled kernels

      Function *k = cast<Function>(
          dyn_cast<ValueAsMetadata>(KernelMDs->getOperand(i)->getOperand(0))
              ->getValue());
      if (F == k)
        return true;
    }
  }

  return false;
}

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

        if (!isOpenCLKernel(Callee, KernelMDs)) {
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
#ifdef LLVM_OLDER_THAN_11_0
        llvm::InlineFunction(CInstr, IFI);
#else
        llvm::InlineFunction(*CInstr, IFI);
#endif
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

bool InlineKernels::runOnFunction(Function &F) {
  SmallPtrSet<Function *, 8> functions_to_inline;
  SmallVector<Value *, 8> pending;

  if (F.getName() != KernelName)
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
