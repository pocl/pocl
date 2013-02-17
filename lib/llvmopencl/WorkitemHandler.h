// Header for WorkitemHandler, a parent class for all implementations of
// work item handling.
// 
// Copyright (c) 2012 Pekka Jääskeläinen / TUT
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

#ifndef _POCL_WORKITEM_HANDLER_H
#define _POCL_WORKITEM_HANDLER_H

#include "config.h"
#if (defined LLVM_3_1 or defined LLVM_3_2)
#include "llvm/Function.h"
#else
#include "llvm/IR/Function.h"
#endif

#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {
  class DominatorTree;
}

namespace pocl {
  class Workgroup;
  class Kernel;

  class WorkitemHandler : public llvm::FunctionPass {
  public:

    WorkitemHandler(char& ID);

    virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const = 0;
    virtual bool runOnFunction(llvm::Function &F);

    virtual void Initialize(pocl::Kernel *K);

  protected:
    
    void movePhiNodes(llvm::BasicBlock* src, llvm::BasicBlock* dst);
    bool fixUndominatedVariableUses(llvm::DominatorTree *DT, llvm::Function &F);
    bool dominatesUse(llvm::DominatorTree *DT, llvm::Instruction &I, unsigned i);

    int LocalSizeX, LocalSizeY, LocalSizeZ;

    unsigned size_t_width;

    /* The global variables that store the current local id. */
    llvm::Value *localIdZ, *localIdY, *localIdX;

  };

  extern llvm::cl::opt<bool> AddWIMetadata;
}

#endif
