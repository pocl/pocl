// LLVM module pass to process automatic locals.
//
// Copyright (c) 2011 Universidad Rey Juan Carlos
//               2012-2019 Pekka Jääskeläinen
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
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include "AutomaticLocals.h"
#include "LLVMUtils.h"
#include "Workgroup.h"
#include "pocl_llvm_api.h"
POP_COMPILER_DIAGS

#define PASS_NAME "automatic-locals"
#define PASS_CLASS pocl::AutomaticLocals
#define PASS_DESC "Processes automatic locals"

namespace pocl {

using namespace llvm;

using FunctionVec = std::vector<llvm::Function *>;

static Function *processAutomaticLocals(Function *F, unsigned long Strategy) {

  Module *M = F->getParent();

  SmallVector<GlobalVariable *, 8> Locals;

  SmallVector<Type *, 8> Parameters;
  for (Function::const_arg_iterator i = F->arg_begin(),
         e = F->arg_end(); i != e; ++i)
    Parameters.push_back(i->getType());

  for (Module::global_iterator i = M->global_begin(),
         e = M->global_end(); i != e; ++i) {

    if (isAutomaticLocal(F, *i)) {
      Locals.push_back(&*i);

      // Add the parameters to the end of the function parameter list.
      Parameters.push_back(i->getType());

      // Replace any constant expression users with an equivalent instruction.
      // Otherwise, the IR breaks when we replace the local with an argument.
      breakConstantExpressions(&*i, F);
    }
  }

  if (Locals.empty()) {
    // This kernel fingerprint has not changed.
    return F;
  }

  if (Strategy == POCL_AUTOLOCALS_TO_ARGS_ONLY_IF_DYNAMIC_LOCALS_PRESENT) {
    bool NeedsArgOffsets = false;
    for (auto &Arg : F->args()) {
      // Check for local memory pointer.
      llvm::Type *ArgType = Arg.getType();
      if (ArgType->isPointerTy() && ArgType->getPointerAddressSpace() == 3) {
        NeedsArgOffsets = true;
        break;
      }
    }
    if (!NeedsArgOffsets)
      return F;
  }

  // Create the new function.
  FunctionType *FT =
    FunctionType::get(F->getReturnType(), Parameters, F->isVarArg());
  Function *NewKernel = Function::Create(FT, F->getLinkage(), "", M);
  NewKernel->takeName(F);

  ValueToValueMapTy VV;
  Function::arg_iterator j = NewKernel->arg_begin();
  for (Function::const_arg_iterator i = F->arg_begin(), e = F->arg_end();
       i != e; ++i) {
    j->setName(i->getName());
    VV[&*i] = &*j;
    ++j;
  }

  for (int i = 0; j != NewKernel->arg_end(); ++i, ++j) {
    j->setName("_local" + Twine(i));
    VV[Locals[i]] = &*j;
  }

  SmallVector<ReturnInst *, 1> RI;

  // As of LLVM 5.0 we need to let CFI to make module level changes,
  // otherwise there will be an assertion. The changes are likely
  // additional debug info nodes added when cloning the function into
  // the other.  For some reason it doesn't want to reuse the old ones.
  CloneFunctionIntoAbs(NewKernel, F, VV, RI);

  for (size_t i = 0; i < Locals.size(); ++i) {
    setFuncArgAddressSpaceMD(NewKernel, F->arg_size() + i,
                             SPIR_ADDRESS_SPACE_LOCAL);
  }
  return NewKernel;
}

static bool automaticLocals(Module &M, FunctionVec &OldKernels) {
  unsigned long ModStrategy = 0;
  getModuleIntMetadata(M, "device_autolocals_to_args", ModStrategy);

  if (ModStrategy == POCL_AUTOLOCALS_TO_ARGS_NEVER) {
    return false;
  }
  bool changed = false;

  // store the new and old kernel pairs in order to regenerate
  // all the metadata that used to point to the unmodified
  // kernels
  FunctionMapping kernels;

  std::string ErrorInfo;
  FunctionVec NewFuncs;
  for (Module::iterator mi = M.begin(), me = M.end(); mi != me; ++mi) {
    // This is to prevent recursion with llvm 3.9. The new kernels are
    // recognized as kernelsToProcess.
    if (find(NewFuncs.begin(), NewFuncs.end(), &*mi) != NewFuncs.end())
      continue;
    if (!isKernelToProcess(*mi))
      continue;

    Function *F = &*mi;

    Function *new_kernel = processAutomaticLocals(F, ModStrategy);
    if (new_kernel != F)
      changed = true;
    kernels[F] = new_kernel;
    NewFuncs.push_back(new_kernel);
  }

  if (changed) {
    regenerate_kernel_metadata(M, kernels);
    /* Delete the old kernels. */
    for (FunctionMapping::const_iterator i = kernels.begin(), e = kernels.end();
         i != e; ++i) {
      Function *old_kernel = (*i).first;
      Function *new_kernel = (*i).second;
      if (old_kernel == new_kernel)
        continue;
      OldKernels.push_back(old_kernel);
    }
  }
  return changed;
}

#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
char AutomaticLocals::ID = 0;

bool AutomaticLocals::runOnModule(Module &M) {
  FunctionVec OldKernels;
  bool Ret = automaticLocals(M, OldKernels);
  for (auto K : OldKernels)
    K->eraseFromParent();
  return Ret;
}

void AutomaticLocals::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.setPreservesCFG();
}

REGISTER_OLD_MPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

#else

// One thing to note when accessing inner level IR analyses is cached results
// for deleted IR. If a function is deleted in a module pass, its address is
// still used as the key for cached analyses. Take care in the pass to either
// clear the results for that function or not use inner analyses at all.
llvm::PreservedAnalyses AutomaticLocals::run(llvm::Module &M,
                                             llvm::ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  FunctionVec OldKernels;
  bool Ret = automaticLocals(M, OldKernels);
  for (auto K : OldKernels) {
    FAM.clear(*K, "parallel.bc");
    K->eraseFromParent();
  }

  return Ret ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

REGISTER_NEW_MPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

#endif

} // namespace pocl
