// Header for WorkitemHandler, a parent class for all implementations of
// work item handling.
//
// Copyright (c) 2012-2019 Pekka Jääskeläinen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#ifndef _POCL_WORKITEM_HANDLER_H
#define _POCL_WORKITEM_HANDLER_H

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "config.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

POP_COMPILER_DIAGS

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
    bool fixUndominatedVariableUses(llvm::DominatorTreeWrapperPass *DT, llvm::Function &F);
    bool dominatesUse(llvm::DominatorTreeWrapperPass *DT, llvm::Instruction &I, unsigned i);

    // The type of size_t for the current target.
    llvm::Type *SizeT;
    // The width of size_t for the current target.
    int SizeTWidth;

    // The Module global variables that hold the place of the current local
    // id until privatized.
    llvm::Value *LocalIdZGlobal, *LocalIdYGlobal, *LocalIdXGlobal;
  };

  extern llvm::cl::opt<bool> AddWIMetadata;
  extern llvm::cl::opt<int> LockStepSIMDWidth;

  // Used to communicate the work-group dimensions of the currently
  // compiled kernel command to the workitem loop.
  // TODO: Something cleaner than a global value.
  extern size_t WGLocalSizeX;
  extern size_t WGLocalSizeY;
  extern size_t WGLocalSizeZ;
  // Set to true to generate a global offset 0 specialized WG function.
  extern bool WGAssumeZeroGlobalOffset;
  extern bool WGDynamicLocalSize;
}

#endif
