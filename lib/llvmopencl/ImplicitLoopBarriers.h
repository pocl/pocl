// Header for ImplicitLoopBarriers loop pass.
// 
// Copyright (c) 2012-2013 Pekka Jääskeläinen / TUT
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

#ifndef POCL_IMPLICIT_LOOP_BARRIERS_H
#define POCL_IMPLICIT_LOOP_BARRIERS_H

#include "config.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Pass.h>
#include <llvm/Passes/PassBuilder.h>

#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
#include <llvm/Analysis/LoopPass.h>
#else
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Transforms/Scalar/LoopPassManager.h>
#endif

namespace pocl {

#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER

class ImplicitLoopBarriers : public llvm::LoopPass
{
public:
  static char ID;
  ImplicitLoopBarriers() : LoopPass(ID){};
  virtual ~ImplicitLoopBarriers (){};

  virtual bool runOnLoop(llvm::Loop *L, llvm::LPPassManager &LPM) override;
  virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};

#else

class ImplicitLoopBarriers : public llvm::PassInfoMixin<ImplicitLoopBarriers> {
public:
  static void registerWithPB(llvm::PassBuilder &B);
  llvm::PreservedAnalyses run(llvm::Loop &L, llvm::LoopAnalysisManager &AM,
                              llvm::LoopStandardAnalysisResults &AR,
                              llvm::LPMUpdater &U);
  static bool isRequired() { return true; }
};

#endif

} // namespace pocl

#endif
