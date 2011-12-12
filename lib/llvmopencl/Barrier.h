// Class for barrier instructions, modelled as a CallInstr.
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

#define BARRIER_FUNCTION_NAME "pocl.barrier"

#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/Support/Casting.h"

namespace pocl {
  
  class Barrier : public llvm::CallInst {

  public:
    static Barrier *Create(llvm::Instruction *InsertBefore) {
      llvm::Module *M = InsertBefore->getParent()->getParent()->getParent();
      llvm::Function *F = llvm::cast<llvm::Function>
        (M->getOrInsertFunction(BARRIER_FUNCTION_NAME,
                                llvm::Type::getVoidTy(M->getContext()),
                                NULL));
      return llvm::cast<pocl::Barrier>
        (llvm::CallInst::Create(F, "", InsertBefore));
    }
    static bool classof(const Barrier *) { return true; };
    static bool classof(const llvm::CallInst *C) {
      return C->getCalledFunction()->getName() == BARRIER_FUNCTION_NAME;
    }
    static bool classof(const Instruction *I) {
      return (llvm::isa<llvm::CallInst>(I) &&
              classof(llvm::cast<llvm::CallInst>(I)));
    }
  };

}

