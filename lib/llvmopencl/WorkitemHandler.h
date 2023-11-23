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

#ifndef POCL_WORKITEM_HANDLER_H
#define POCL_WORKITEM_HANDLER_H

#include "config.h"

#include "Kernel.h"

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

#include "llvm/Support/CommandLine.h"

namespace pocl {

class WorkitemHandler {
public:
  virtual void Initialize (pocl::Kernel *K);

protected:
  void movePhiNodes(llvm::BasicBlock *Src, llvm::BasicBlock *Dst);
  bool fixUndominatedVariableUses (llvm::DominatorTree &DT, llvm::Function &F);
  bool dominatesUse (llvm::DominatorTree &DT, llvm::Instruction &Inst,
                     unsigned OpNum);

  // The type of size_t for the current target.
  llvm::Type *SizeT;
  // The width of size_t for the current target.
  int SizeTWidth;

  // The Module global variables that hold the place of the current local
  // id until privatized.
  llvm::Value *LocalIdZGlobal, *LocalIdYGlobal, *LocalIdXGlobal;

  // Copies of compilation parameters
  std::string KernelName;
  unsigned long AddressBits;
  bool WGAssumeZeroGlobalOffset;
  bool WGDynamicLocalSize;
  bool DeviceUsingArgBufferLauncher;
  bool DeviceIsSPMD;
  unsigned long WGLocalSizeX;
  unsigned long WGLocalSizeY;
  unsigned long WGLocalSizeZ;
  unsigned long WGMaxGridDimWidth;
};

  extern llvm::cl::opt<bool> AddWIMetadata;
  extern llvm::cl::opt<int> LockStepSIMDWidth;
}

#endif
