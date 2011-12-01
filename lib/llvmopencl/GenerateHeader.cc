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
#include "Workgroup.h"
#include "llvm/Argument.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetData.h"

using namespace std;
using namespace llvm;
using namespace pocl;

cl::opt<string>
Header("header",
       cl::desc("Output header file with kernel description macros"),
       cl::value_desc("header"));

namespace {
  class GenerateHeader : public FunctionPass {
  
  public:
    
    static char ID;
    GenerateHeader() : FunctionPass(ID) {}
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const;
    virtual bool runOnFunction(Function &F);
  };
}

char GenerateHeader::ID = 0;
static RegisterPass<GenerateHeader> X("generate-header",
				      "Kernel information header creation pass");

void
GenerateHeader::getAnalysisUsage(AnalysisUsage &AU) const
{
  AU.addRequired<TargetData>();
}

bool
GenerateHeader::runOnFunction(Function &F)
{
  if (!Workgroup::isKernelToProcess(F))
    return false;

  string ErrorInfo;
  raw_fd_ostream out(Header.c_str(), ErrorInfo, raw_fd_ostream::F_Append);
      
  int num_args = F.getFunctionType()->getNumParams();
      
  out << "#define _" << F.getName() << "_NUM_ARGS " << num_args << '\n';
      
  bool is_pointer[num_args];
  bool is_local[num_args];
  
  int i = 0;
  for (Function::const_arg_iterator ii = F.arg_begin(),
         ee = F.arg_end();
       ii != ee; ++ii) {
    Type *t = ii->getType();
    
    if (isa<PointerType> (t)) {
      is_pointer[i] = true;
      // index 0 is for function attributes, parameters start at 1.
      if (F.paramHasAttr(i + 1, Attribute::NoCapture))
        is_local[i] = false;
      else
        is_local[i] = true;
    } else {
      is_pointer[i] = false;
      is_local[i] = false;
    }

    ++i;
  }
      
  out << "#define _" << F.getName() << "_ARG_IS_POINTER {";
  if (num_args != 0) {
    out << is_pointer[0];
    for (i = 1; i < num_args; ++i)
      out << ", " << is_pointer[i];
  }
  out << "}\n";
  
  out << "#define _" << F.getName() << "_ARG_IS_LOCAL {";
  if (num_args != 0) {
    out << is_local[0];
    for (i = 1; i < num_args; ++i)
      out << ", " << is_local[i];
  }
  out << "}\n";

  // TargetData &TD = getAnalysis<TargetData>();

  // SmallVector<unsigned, 8> locals;
  // for (Function::const_iterator ii = F.begin(), ee = F.end();
  //      ii != ee; ++ii) {
  //   if (const AllocaInst *alloca = dyn_cast<AllocaInst>(ii)) {
  //     if (alloca->getType()->getAddressSpace() == POCL_ADDRESS_SPACE_LOCAL) {
  //       Type *t = alloca->getAllocatedType();
  //       const ConstantInt *c = cast<ConstantInt>(alloca->getArraySize());
  //       unsigned size = TD.getTypeAllocSize(t) * c->getZExtValue();
  //       locals.push_back(size);
  //     }
  //   }
  // }
        
  // out << "#define _" << F.getName() << "_NUM_LOCALS "<< locals.size() << "\n";
  // out << "#define _" << F.getName() << "_LOCAL_SIZE {";
  // if (!locals.empty()) {
  //   out << locals[0];
  //   for (i = 1; i < locals.size(); ++i)
  //     out << ", " << locals[i];
  // }
  // out << "}\n";

  out << "#define _" << F.getName() << "_NUM_LOCALS 0\n";
  out << "#define _" << F.getName() << "_LOCAL_SIZE {}\n";

  return false;
}
