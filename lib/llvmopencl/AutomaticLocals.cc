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
    Function *processAutomaticLocals(Function *F);
  };
}

char AutomaticLocals::ID = 0;
static RegisterPass<AutomaticLocals> X("automatic-locals",
				      "Processes automatic locals");

#if (LLVM_OLDER_THAN_3_7)
void
AutomaticLocals::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DataLayoutPass>();
}
#else
void
AutomaticLocals::getAnalysisUsage(AnalysisUsage &) const {
}
#endif

bool
AutomaticLocals::runOnModule(Module &M)
{
  bool changed = false;

  // store the new and old kernel pairs in order to regenerate
  // all the metadata that used to point to the unmodified
  // kernels
  FunctionMapping kernels;

  string ErrorInfo;
  std::vector<Function*> NewFuncs;
  for (Module::iterator mi = M.begin(), me = M.end(); mi != me; ++mi) {
    // This is to prevent recursion with llvm 3.9. The new kernels are
    // recognized as kernelsToProcess.
    if (find(NewFuncs.begin(), NewFuncs.end(), &*mi) != NewFuncs.end())
      continue;
    if (!Workgroup::isKernelToProcess(*mi))
      continue;

    Function *F = &*mi;

    Function *new_kernel = processAutomaticLocals(F);
    if (new_kernel != F)
      changed = true;
    kernels[F] = new_kernel;
    NewFuncs.push_back(new_kernel);
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

// Recursively descend a Value's users and convert any constant expressions into
// regular instructions.
static void breakConstantExpressions(llvm::Value *Val, llvm::Function *Func) {
  std::vector<llvm::Value *> Users(Val->user_begin(), Val->user_end());
  for (auto *U : Users) {
    if (auto *CE = llvm::dyn_cast<llvm::ConstantExpr>(U)) {
      // First, make sure no users of this constant expression are themselves
      // constant expressions.
      breakConstantExpressions(U, Func);

      // Convert this constant expression to an instruction.
      llvm::Instruction *I = CE->getAsInstruction();
      I->insertBefore(&*Func->begin()->begin());
      CE->replaceAllUsesWith(I);
      CE->destroyConstant();
    }
  }
}

Function *
AutomaticLocals::processAutomaticLocals(Function *F) {

  Module *M = F->getParent();

  SmallVector<GlobalVariable *, 8> Locals;

  SmallVector<Type *, 8> Parameters;
  for (Function::const_arg_iterator i = F->arg_begin(),
         e = F->arg_end(); i != e; ++i)
    Parameters.push_back(i->getType());

  for (Module::global_iterator i = M->global_begin(),
         e = M->global_end(); i != e; ++i) {
    std::string FuncName = "";
    FuncName = F->getName().str();
    if (isAutomaticLocal(FuncName, *i)) {
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
  CloneFunctionInto(NewKernel, F, VV, true, RI);

  return NewKernel;
}

