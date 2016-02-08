// LLVM module pass to process automatic locals.
// 
// Copyright (c) 2011 Universidad Rey Juan Carlos
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
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "pocl.h"

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

#include "LLVMUtils.h"
#include "Workgroup.h"

POP_COMPILER_DIAGS

using namespace std;
using namespace llvm;
using namespace pocl;

namespace {
  class AutomaticLocals : public ModulePass {
  
  public:
    static char ID;
    AutomaticLocals() : ModulePass(ID) {}
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual bool runOnModule(Module &M);

  private:
    Function *ProcessAutomaticLocals(Function *F);
  };
}

char AutomaticLocals::ID = 0;
static RegisterPass<AutomaticLocals> X("automatic-locals",
				      "Processes automatic locals");

void
AutomaticLocals::getAnalysisUsage(AnalysisUsage &AU) const {
#if (LLVM_OLDER_THAN_3_7)
  AU.addRequired<DataLayoutPass>();
#endif
}

bool
AutomaticLocals::runOnModule(Module &M)
{
  bool changed = false;

  // store the new and old kernel pairs in order to regenerate
  // all the metadata that used to point to the unmodified
  // kernels
  FunctionMapping kernels;

  string ErrorInfo;

  for (Module::iterator mi = M.begin(), me = M.end(); mi != me; ++mi) {
    if (!Workgroup::isKernelToProcess(*mi))
      continue;
  
    Function *F = &*mi;

    Function *new_kernel = ProcessAutomaticLocals(F);
    if (new_kernel != F)
      changed = true;
    kernels[F] = new_kernel;
  }

  if (changed)
    {
      regenerate_kernel_metadata(M, kernels);

      /* Delete the old kernels. */
      for (FunctionMapping::const_iterator i = kernels.begin(),
             e = kernels.end(); i != e; ++i) 
        {
          Function *old_kernel = (*i).first;
          Function *new_kernel = (*i).second;
          if (old_kernel == new_kernel) continue;
          old_kernel->eraseFromParent();
        }
    }
  return changed;
}

Function *
AutomaticLocals::ProcessAutomaticLocals(Function *F)
{
  Module *M = F->getParent();
  
  SmallVector<GlobalVariable *, 8> locals;

  SmallVector<Type *, 8> parameters;
  for (Function::const_arg_iterator i = F->arg_begin(),
         e = F->arg_end();
       i != e; ++i)
    parameters.push_back(i->getType());
    
  for (Module::global_iterator i = M->global_begin(),
         e = M->global_end();
       i != e; ++i) {
    std::string funcName = "";
    funcName = F->getName().str();
    if (is_automatic_local(funcName, *i)) {
      locals.push_back(&*i);
      // Add the parameters to the end of the function parameter list.
      parameters.push_back(i->getType());
    }
  }

  if (locals.empty()) {
    // This kernel fingerprint has not changed.
    return F;
  }
  
  // Create the new function.
  FunctionType *ft = FunctionType::get(F->getReturnType(),
                                       parameters,
                                       F->isVarArg());
  Function *new_kernel = Function::Create(ft,
                                          F->getLinkage(),
                                          "",
                                          M);
  new_kernel->takeName(F);
  
  ValueToValueMapTy vv;
  Function::arg_iterator j = new_kernel->arg_begin();
  for (Function::const_arg_iterator i = F->arg_begin(),
         e = F->arg_end();
       i != e; ++i) {
    j->setName(i->getName());
    vv[&*i] = &*j;
    ++j;
  }
  
  for (int i = 0; j != new_kernel->arg_end(); ++i, ++j) {
    j->setName("_local" + Twine(i));
    vv[locals[i]] = &*j;
  }
                                 
  SmallVector<ReturnInst *, 1> ri;
  CloneFunctionInto(new_kernel, F, vv, false, ri);

  return new_kernel;
}

