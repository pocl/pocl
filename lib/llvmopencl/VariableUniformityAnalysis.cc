// Implementation for VariableUniformityAnalysis function pass.
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

#include "config.h"
#include <sstream>
#include <iostream>

#if (defined LLVM_3_1 or defined LLVM_3_2)
#include "llvm/Metadata.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/ValueSymbolTable.h"
#else
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ValueSymbolTable.h"
#endif
#include "llvm/Support/CommandLine.h"
#include "WorkitemHandler.h"
#include "Kernel.h"
#include "VariableUniformityAnalysis.h"

//#define DEBUG_REFERENCE_FIXING

namespace pocl {

char VariableUniformityAnalysis::ID = 0;

using namespace llvm;

static
RegisterPass<VariableUniformityAnalysis> X(
    "uniformity", 
    "Analyses the variables of the function for uniformity (same value across WIs).",
    false, false);

VariableUniformityAnalysis::VariableUniformityAnalysis() : FunctionPass(ID) {
}


void
VariableUniformityAnalysis::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
}

bool
VariableUniformityAnalysis::runOnFunction(Function &F) {
  /* Do the actual analysis on-demand. */
  uniformityCache_[&F].clear();
  return false;
}

/**
 * Simple uniformity analysis that recursively analyses all the
 * operands affecting the value.
 *
 * Known uniform Values:
 * a) kernel arguments
 * b) constants
 * 
 */
bool 
VariableUniformityAnalysis::isUniform(llvm::Function *f, llvm::Value* v) {

  UniformityIndex &cache = uniformityCache_[f];
  UniformityIndex::const_iterator i = cache.find(v);
  if (i != cache.end()) {
    return (*i).second;
  }

  if (isa<llvm::Argument>(v) ||
      isa<llvm::Constant>(v)) {
    setUniform(f, v, true);
    return true;
  }

  if (isa<llvm::PHINode>(v)) {
    /* TODO: PHINodes need control flow analysis:
       even if the values are uniform, the selected
       value depends on the preceeding basic block which
       might depend on the ID. Assume they are not uniform
       for now in general and treat the loop iteration 
       variable as a special case (set externally from a LoopPass). 

       TODO: PHINodes can depend (indirectly or directly) on itself in loops 
       so this needs infinite recursion checking.
    */
    setUniform(f, v, false);
    return false;
  }

  llvm::Instruction *instr = dyn_cast<llvm::Instruction>(v);
  if (instr == NULL) {
    setUniform(f, v, false);
    return false;
  }
  // not computed previously, scan all operands of the instruction
  // and figure out their uniformity recursively
  bool uniform = true;
  for (unsigned opr = 0; opr < instr->getNumOperands(); ++opr) {    
    llvm::Value *operand = instr->getOperand(opr);
    if (!isUniform(f, operand)) {
      setUniform(f, v, false);
      return false;
    }
  }
  setUniform(f, v, true);
  return true;
}
  
void
VariableUniformityAnalysis::setUniform(llvm::Function *f, 
                                       llvm::Value *v, 
                                       bool isUniform) {

  UniformityIndex &cache = uniformityCache_[f];
  cache[v] = isUniform;
}

}
