// Header for VariableUniformityAnalysis function pass.
// 
// Copyright (c) 2013 Pekka Jääskeläinen / Tampere University of Technology
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

#ifndef POCL_VARIABLE_UNIFORMITY_ANALYSIS_H
#define POCL_VARIABLE_UNIFORMITY_ANALYSIS_H

#include "config.h"
#if (defined LLVM_3_1 or defined LLVM_3_2)
#include "llvm/Function.h"
#else
#include "llvm/IR/Function.h"
#endif

#include "llvm/Pass.h"

namespace pocl {
  /**
   * Analyses the variables in the function to figure out if a variable
   * value is
   *
   * a) 'uniform', i.e., always same for all work-items in the *same work-group*
   * b) 'varying', i.e., somehow dependent on the work-item id 
   * 
   * For safety, 'variable' is assumed, unless certain of a).
   */
  class VariableUniformityAnalysis : public llvm::FunctionPass {
  public:
    static char ID;

    VariableUniformityAnalysis();

    virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const;
    virtual bool runOnFunction(llvm::Function &F);
    virtual bool isUniform(llvm::Function *f, llvm::Value* v);
    virtual void setUniform(llvm::Function *f, llvm::Value *v, bool isUniform=true);
    virtual void analyzeBBDivergence(llvm::Function *f, 
                                     llvm::BasicBlock *bb, 
                                     llvm::BasicBlock *previousUniformBB);

  private:

    bool isUniformityAnalyzed(llvm::Function *f, llvm::Value *val) const;

    typedef std::map<llvm::Value*, bool> UniformityIndex;
    typedef std::map<llvm::Function *, UniformityIndex> UniformityCache;
    mutable UniformityCache uniformityCache_;

  };
}

#endif
