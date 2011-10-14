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

#include "pocl.h"
#include "llvm/Argument.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/DerivedTypes.h"

using namespace std;
using namespace llvm;

cl::opt<string>
Kernel("kernel",
       cl::desc("Kernel function name"),
       cl::value_desc("kernel"));

cl::opt<string>
Header("header",
       cl::desc("Output header file with kernel description macros"),
       cl::value_desc("header"));

namespace {
  class GenerateHeader : public FunctionPass {
  
  public:
    
    static char ID;
    GenerateHeader() : FunctionPass(ID) {}
    
    virtual bool runOnFunction(Function &F);
  };
}

char GenerateHeader::ID = 0;
static RegisterPass<GenerateHeader> X("generate-header",
				      "Kernel information header creation pass");

bool
GenerateHeader::runOnFunction(Function &F)
{
  if (F.getName() != Kernel)
    return false;

  string ErrorInfo;
  raw_fd_ostream out(Header.c_str(), ErrorInfo);

  int num_args = F.getFunctionType()->getNumParams();

  out << "#define _NUM_ARGS " << num_args << '\n';

  bool is_pointer[num_args];
  bool is_local[num_args];

  int i = 0;
  for (Function::const_arg_iterator ii = F.arg_begin(),
	 ee = F.arg_end();
       ii != ee; ++ii) {
    Type *t = ii->getType();

    const PointerType *p = dyn_cast<PointerType> (t);
	
    if (p == NULL) {
      is_pointer[i] = false;
      is_local[i] = false;
      ++i;
      continue;
    }
	
    is_pointer[i] = true;

    switch (p->getAddressSpace()) {
    case POCL_ADDRESS_SPACE_GLOBAL:
    case POCL_ADDRESS_SPACE_CONSTANT:
      is_local[i] = false;
      break;
    case POCL_ADDRESS_SPACE_LOCAL:
      is_local[i] = true;
      break;
    default:
      llvm_unreachable("Invalid address space qualifier on kernel argument!");
    }

    ++i;
  }

  out << "#define _ARG_IS_POINTER {";
  if (num_args != 0) {
    out << is_pointer[0];
    for (i = 1; i < num_args; ++i)
      out << ", " << is_pointer[i];
  }
  out << "}\n";

  out << "#define _ARG_IS_LOCAL {";
  if (num_args != 0) {
    out << is_local[0];
    for (i = 1; i < num_args; ++i)
      out << ", " << is_local[i];
  }
  out << "}\n";
    
  return false;
}
