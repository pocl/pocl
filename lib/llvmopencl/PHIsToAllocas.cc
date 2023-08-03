// LLVM function pass to convert all PHIs to allocas.
//
// Copyright (c) 2012-2019 Pekka Jääskeläinen
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

#include <iostream>

#include "config.h"

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include <llvm/IR/IRBuilder.h>

#include "PHIsToAllocas.h"
#include "Workgroup.h"
#include "WorkitemHandlerChooser.h"
#include "WorkitemLoops.h"
#include "VariableUniformityAnalysis.h"

namespace {
  static
  llvm::RegisterPass<pocl::PHIsToAllocas> X(
      "phistoallocas", "Convert all PHI nodes to allocas");
}

namespace pocl {

char PHIsToAllocas::ID = 0;

using namespace llvm;

void
PHIsToAllocas::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<pocl::WorkitemHandlerChooser>();
  AU.addPreserved<pocl::WorkitemHandlerChooser>();

  AU.addRequired<pocl::VariableUniformityAnalysis>();
  AU.addPreserved<pocl::VariableUniformityAnalysis>();
}

bool
PHIsToAllocas::runOnFunction(Function &F) {
  if (!isKernelToProcess(F))
    return false;

#ifdef CBS_NO_PHIS_IN_SPLIT
  bool RunWithCBS = true;
#else
  bool RunWithCBS = false;
#endif

  /* Skip PHIsToAllocas when we are not creating the work item loops,
     as it leads to worse code without benefits for the full replication method.
  */
  if (getAnalysis<pocl::WorkitemHandlerChooser>().chosenHandler() !=
          pocl::WorkitemHandlerChooser::POCL_WIH_LOOPS &&
      !(RunWithCBS &&
        getAnalysis<pocl::WorkitemHandlerChooser>().chosenHandler() ==
            pocl::WorkitemHandlerChooser::POCL_WIH_CBS))
    return false;

  typedef std::vector<llvm::Instruction* > InstructionVec;

  InstructionVec PHIs;

  for (Function::iterator bb = F.begin(); bb != F.end(); ++bb) {
    for (BasicBlock::iterator p = bb->begin(); 
         p != bb->end(); ++p) {
        Instruction* instr = &*p;
        if (isa<PHINode>(instr)) {
            PHIs.push_back(instr);
        }
    }
  }

  bool changed = false;
  for (InstructionVec::iterator i = PHIs.begin(); i != PHIs.end();
       ++i) {
      Instruction *instr = *i;
      BreakPHIToAllocas(dyn_cast<PHINode>(instr));
      changed = true;
  }  
  return changed;
}

/**
 * Convert a PHI to a read from a stack value and all the sources to
 * writes to the same stack value.
 *
 * Used to fix context save/restore issues with regions with PHI nodes in the
 * entry node (usually due to the use of work group scope variables such as
 * B-loop iteration variables). In case of PHI nodes at region entries, we cannot 
 * just insert the context restore code because it is assumed there are no
 * non-phi Instructions before PHIs which the context restore
 * code constitutes to. Secondly, in case the PHINode is at a
 * region entry (e.g. a B-Loop) adding new basic blocks before it would 
 * break the assumption of single entry regions.
 */
llvm::Instruction *
PHIsToAllocas::BreakPHIToAllocas(PHINode* phi) {

  // Loop iteration variables can be detected only when they are
  // implemented using PHI nodes. Maintain information of the
  // split PHI nodes in the VUA by first analyzing the function
  // with the PHIs intact and propagating the uniformity info
  // of the PHI nodes.
  VariableUniformityAnalysis &VUA = 
      getAnalysis<VariableUniformityAnalysis>();

  std::string allocaName = std::string(phi->getName().str()) + ".ex_phi";

  llvm::Function *function = phi->getParent()->getParent();

  const bool OriginalPHIWasUniform = VUA.isUniform(function, phi);

  IRBuilder<> builder(&*(function->getEntryBlock().getFirstInsertionPt()));

  llvm::Instruction *alloca = 
    builder.CreateAlloca(phi->getType(), 0, allocaName);

  for (unsigned incoming = 0; incoming < phi->getNumIncomingValues(); 
       ++incoming) {
      Value *val = phi->getIncomingValue(incoming);
      BasicBlock *incomingBB = phi->getIncomingBlock(incoming);
      builder.SetInsertPoint(incomingBB->getTerminator());
      llvm::Instruction *store = builder.CreateStore(val, alloca);
      if (OriginalPHIWasUniform)
          VUA.setUniform(function, store);
  }
  builder.SetInsertPoint(phi);

  llvm::Instruction *loadedValue = builder.CreateLoad(phi->getType(), alloca);
  phi->replaceAllUsesWith(loadedValue);

  if (OriginalPHIWasUniform) {
#ifdef DEBUG_PHIS_TO_ALLOCAS
      std::cout << "PHIsToAllocas: Original PHI was uniform" << std::endl
                << "original:";
      phi->dump();
      std::cout << "alloca:";
      alloca->dump();
      std::cout << "loadedValue:";
      loadedValue->dump();
#endif
      VUA.setUniform(function, alloca);
      VUA.setUniform(function, loadedValue);
  }
  phi->eraseFromParent();

  return loadedValue;
}


}
