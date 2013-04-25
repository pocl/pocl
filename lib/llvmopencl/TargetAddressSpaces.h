// Header for TargetAddressSpaces, an LLVM pass that converts the
// generic address space ids to the target specific ones.
// 
// Copyright (c) 2013 Pekka Jääskeläinen / TUT
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

#ifndef _POCL_TARGET_ADDRESS_SPACES_H
#define _POCL_TARGET_ADDRESS_SPACES_H

#include "config.h"
#if (defined LLVM_3_1 or defined LLVM_3_2)
#include "llvm/Function.h"
#else
#include "llvm/IR/Function.h"
#endif

#include "llvm/Pass.h"

namespace pocl {
  /* pocl uses the fixed address space ids forced by the clang's
     -ffake-address-space-map internally until the end to be able to
     detect the different OpenCL address spaces ambiguously, regardless
     of the target. This pass converts the fake address space ids to
     the target-specific ones, if required by the code generator of that
     target. */       
  class TargetAddressSpaces : public llvm::ModulePass {
  public:
    static char ID;

    TargetAddressSpaces();
    virtual ~TargetAddressSpaces() {};

    virtual bool runOnModule(llvm::Module &M);    
  };
}

#endif
