// LLVM module pass to get information from kernel functions.
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

#include "locl.h"
#include "llvm/Argument.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/DerivedTypes.h"

using namespace std;
using namespace llvm;

cl::opt<string> Kernel("kernel",
		       cl::desc("Kernel function name"),
		       cl::value_desc("kernel"));

namespace {
  class Trampoline : public ModulePass {
  
  public:
    
    static char ID;
    Trampoline() : ModulePass(ID) {}
    
    virtual bool runOnModule(Module &M);
  };
  
  char Trampoline::ID = 0;
  INITIALIZE_PASS(Trampoline, "trampoline", "Trampoline identification pass", false, false);
}

bool
Trampoline::runOnModule(Module &M)
{
  int num_args;

  Function **functions_to_delete = new Function *[M.size()];
  memset(functions_to_delete, M.size() * sizeof(Function *), 0);

  unsigned index = 0;

  for (Module::iterator i = M.begin(), e = M.end(); i != e; ++i) {
    Function &F = *i;
    if (F.getName() != Kernel) {
      assert(index < M.size());

      functions_to_delete[index] = &F;
      ++index;
    } else {
      F.deleteBody();

      num_args = 0;
      for (Function::const_arg_iterator ii = F.arg_begin(),
	     ee = F.arg_end();
	   ii != ee; ++ii)
	++num_args;

      if (num_args == 0)
	continue;

      Constant **is_pointer = new Constant *[num_args];
      Constant **is_local = new Constant *[num_args];
      
      int i = 0;
      for (Function::const_arg_iterator ii = F.arg_begin(),
	     ee = F.arg_end();
	   ii != ee; ++ii) {
	const Type *t = ii->getType();

	new GlobalVariable(M, t, false, GlobalVariable::ExternalLinkage,
			   UndefValue::get(t), "_arg" + Twine(i));
	new GlobalVariable(M, IntegerType::get(M.getContext(), 32), false,
			   GlobalVariable::ExternalLinkage,
			   UndefValue::get(IntegerType::get(M.getContext(), 32)),
			   "_size" + Twine(i));

	const PointerType *p = dyn_cast<PointerType> (ii->getType());
	
	if (p == NULL) {
	  is_pointer[i] = ConstantInt::get(IntegerType::get(M.getContext(), 32), 0);
	  is_local[i] = ConstantInt::get(IntegerType::get(M.getContext(), 32), 0);
	  ++i;
	  continue;
	}
	
	is_pointer[i] = ConstantInt::get(IntegerType::get(M.getContext(), 32), 1);

	switch (p->getAddressSpace()) {
	case LOCL_ADDRESS_SPACE_GLOBAL:
	case LOCL_ADDRESS_SPACE_CONSTANT:
	  is_local[i] = ConstantInt::get(IntegerType::get(M.getContext(), 32), 0);
	  break;
	case LOCL_ADDRESS_SPACE_LOCAL:
	  is_local[i] = ConstantInt::get(IntegerType::get(M.getContext(), 32), 1);
	  break;
	default:
	  llvm_unreachable("Invalid address space qualifier on kernel argument!");
	}

	++i;
      }

      new GlobalVariable(M, VectorType::get(IntegerType::get(M.getContext(), 32), num_args),
			 true,
			 GlobalVariable::ExternalLinkage,
			 ConstantVector::get(is_pointer, num_args),
			 "_is_pointer");

      new GlobalVariable(M, VectorType::get(IntegerType::get(M.getContext(), 32), num_args),
			 true,
			 GlobalVariable::ExternalLinkage,
			 ConstantVector::get(is_local, num_args),
			 "_is_local");

      delete[] is_pointer;
      delete[] is_local;
    }
  }
  
  for (unsigned i = 0; i < M.size(); ++i) {
    if (functions_to_delete[i] != NULL)
      functions_to_delete[i]->eraseFromParent();
  }

  new GlobalVariable(M, IntegerType::get(M.getContext(), 32), true,
		     GlobalVariable::ExternalLinkage,
		     ConstantInt::get(IntegerType::get(M.getContext(), 32),
				      num_args),
		     "_num_args");

  return true;
}
