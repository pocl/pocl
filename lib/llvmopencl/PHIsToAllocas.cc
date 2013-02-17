// LLVM function pass to convert all PHIs to allocas.
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

#include "PHIsToAllocas.h"
#include "Workgroup.h"
#include "WorkitemHandlerChooser.h"
#include "WorkitemLoops.h"

#include "config.h"

#ifdef LLVM_3_1
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/TypeBuilder.h"
#elif defined LLVM_3_2
#include "llvm/IRBuilder.h"
#include "llvm/TypeBuilder.h"
#else
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/TypeBuilder.h"
#endif

namespace {
  static
  llvm::RegisterPass<pocl::PHIsToAllocas> X(
      "phistoallocas", "Convert all PHI nodes to allocas");
}

namespace pocl {

char PHIsToAllocas::ID = 0;

using namespace llvm;

void
PHIsToAllocas::getAnalysisUsage(AnalysisUsage &AU) const
{
  AU.addRequired<pocl::WorkitemHandlerChooser>();
  AU.addPreserved<pocl::WorkitemHandlerChooser>();
}

bool
PHIsToAllocas::runOnFunction(Function &F)
{
  if (!Workgroup::isKernelToProcess(F))
    return false;

  /* Skip PHIsToAllocas when we are not creating the work item loops,
     as leads to worse code without benefits for the full replication method.
  */
  if (getAnalysis<pocl::WorkitemHandlerChooser>().chosenHandler() != 
      pocl::WorkitemHandlerChooser::POCL_WIH_LOOPS)
    return false;

  typedef std::vector<llvm::Instruction* > InstructionVec;

  InstructionVec PHIs;

  for (Function::iterator bb = F.begin(); bb != F.end(); ++bb) {
    for (BasicBlock::iterator p = bb->begin(); 
         p != bb->end(); ++p)
      {
        Instruction* instr = p;
        if (isa<PHINode>(instr))
          {
            PHIs.push_back(instr);
          }
      }

  }

  bool changed = false;
  for (InstructionVec::iterator i = PHIs.begin(); i != PHIs.end();
       ++i) 
    {
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
PHIsToAllocas::BreakPHIToAllocas(PHINode* phi) 
{
  std::string allocaName = std::string(phi->getName().str()) + ".ex_phi";

  llvm::Function *function = phi->getParent()->getParent();
  IRBuilder<> builder(function->getEntryBlock().getFirstInsertionPt());

  llvm::Instruction *alloca = 
    builder.CreateAlloca(phi->getType(), 0, allocaName);

  for (unsigned incoming = 0; incoming < phi->getNumIncomingValues(); 
       ++incoming)
    {
      Value *val = phi->getIncomingValue(incoming);
      BasicBlock *incomingBB = phi->getIncomingBlock(incoming);
      builder.SetInsertPoint(incomingBB->getTerminator());
      builder.CreateStore(val, alloca);
    }

  builder.SetInsertPoint(phi);

  llvm::Instruction *loadedValue = builder.CreateLoad(alloca);
  phi->replaceAllUsesWith(loadedValue);
  phi->eraseFromParent();
  return loadedValue;
}


}
