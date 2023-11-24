// Header for ImplicitConditionalBarriers pass.
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

#ifndef POCL_IMPLICIT_CONDITIONAL_BARRIERS_H
#define POCL_IMPLICIT_CONDITIONAL_BARRIERS_H

#include "config.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Pass.h>
#include <llvm/Passes/PassBuilder.h>

/**
 * Adds implicit barriers to branches to minimize the adverse effects from
 * "peeling" to the symmetry of the work-item loops created for conditional
 * barrier cases.
 *
 * In essence, the pass converts the following cases:

    .[P].
    |   |
    a   b
   [B]

   to

    .[P].
   [B] [B]
    |   |
    a   b
   [B]

   Legend: [P] is a BB with the predicate.
           [B] is a barrier.
           a and b are regular basic blocks.

   We can inject the barrier legally due to the barrier semantics:
   'a' or 'b' are entered by all or none of the work-items. Thus,
   the additional barrier is legal there as well.

   This creates a nice static trip count work-item parallel loop around 'a' and
   'b' instead of needing to peel the first iteration that creates
   an asymmetric loop with dynamic iteration count.
 */

namespace pocl {

#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER

class ImplicitConditionalBarriers : public llvm::FunctionPass
{
public:
  static char ID;
  ImplicitConditionalBarriers() : FunctionPass(ID){};
  virtual ~ImplicitConditionalBarriers (){};

  virtual bool runOnFunction(llvm::Function &F) override;
  virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};

#else

class ImplicitConditionalBarriers
    : public llvm::PassInfoMixin<ImplicitConditionalBarriers> {
public:
  static void registerWithPB(llvm::PassBuilder &B);
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

#endif
}

#endif
