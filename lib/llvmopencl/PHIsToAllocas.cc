// LLVM function pass to convert all PHIs to allocas.
//
// Copyright (c) 2012-2019 Pekka Jääskeläinen
//               2024 Pekka Jääskeläinen / Intel Finland Oy
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

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/IR/IRBuilder.h>

#include "Barrier.h"
#include "LLVMUtils.h"
#include "PHIsToAllocas.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "Workgroup.h"
#include "WorkitemHandlerChooser.h"
#include "WorkitemLoops.h"
POP_COMPILER_DIAGS

#define PASS_NAME "phistoallocas"
#define PASS_CLASS pocl::PHIsToAllocas
#define PASS_DESC "Convert all PHI nodes to allocas"

// #define DEBUG_PHIS_TO_ALLOCAS

// Skip PHIsToAllocas when we are not creating the work item loops,
// as it leads to worse code without benefits for the full replication method.
// Note: re-enabling this causes workgroup/cond_barriers_in_for_cbs to fail
//#define CBS_NO_PHIS_IN_SPLIT
#include <iostream>

namespace pocl {

using namespace llvm;

static llvm::Instruction *
breakPHIToAllocas(PHINode *Phi, VariableUniformityAnalysisResult &VUA);

static bool needsPHIsToAllocas(Function &F, WorkitemHandlerType WIH) {
#ifdef CBS_NO_PHIS_IN_SPLIT
  bool RunWithCBS = true;
#else
  bool RunWithCBS = false;
#endif
  if (!isKernelToProcess(F))
    return false;

  if (WIH != WorkitemHandlerType::LOOPS &&
      !(RunWithCBS && WIH == WorkitemHandlerType::CBS))
    return false;

  return true;
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
static llvm::Instruction *
breakPHIToAllocas(PHINode *Phi, VariableUniformityAnalysisResult &VUA) {

  // Loop iteration variables can be detected only when they are
  // implemented using PHI nodes. Maintain information of the
  // split PHI nodes in the VUA by first analyzing the function
  // with the PHIs intact and propagating the uniformity info
  // of the PHI nodes.
  std::string AllocaName = std::string(Phi->getName().str()) + ".ex_phi";

  llvm::Function *Function = Phi->getParent()->getParent();

  const bool OriginalPHIWasUniform = VUA.isUniform(Function, Phi);
  IRBuilder<> Builder(&*(Function->getEntryBlock().getFirstInsertionPt()));

  llvm::Instruction *AllocaI =
    Builder.CreateAlloca(Phi->getType(), 0, AllocaName);

  for (unsigned Incoming = 0; Incoming < Phi->getNumIncomingValues();
       ++Incoming) {
      Value *Val = Phi->getIncomingValue(Incoming);
      BasicBlock *IncomingBB = Phi->getIncomingBlock(Incoming);
      Builder.SetInsertPoint(IncomingBB->getTerminator());
      llvm::Instruction *Store = Builder.CreateStore(Val, AllocaI);
      if (OriginalPHIWasUniform)
          VUA.setUniform(Function, Store);
  }
  Builder.SetInsertPoint(Phi);

  llvm::Instruction *LoadedValue = Builder.CreateLoad(Phi->getType(), AllocaI);
  Phi->replaceAllUsesWith(LoadedValue);

  if (OriginalPHIWasUniform) {
#ifdef DEBUG_PHIS_TO_ALLOCAS
      std::cout << "PHIsToAllocas: Original PHI was uniform" << std::endl
                << "original:";
      Phi->dump();
      std::cout << "alloca:";
      AllocaI->dump();
      std::cout << "loadedValue:";
      LoadedValue->dump();
#endif
      VUA.setUniform(Function, AllocaI);
      VUA.setUniform(Function, LoadedValue);
  }
  Phi->eraseFromParent();

  return LoadedValue;
}


llvm::PreservedAnalyses PHIsToAllocas::run(llvm::Function &F,
                                           llvm::FunctionAnalysisManager &AM) {

  WorkitemHandlerType WIH = AM.getResult<WorkitemHandlerChooser>(F).WIH;
  if (!needsPHIsToAllocas(F, WIH))
    return PreservedAnalyses::all();

  VariableUniformityAnalysisResult VUA =
      AM.getResult<VariableUniformityAnalysis>(F);

  llvm::LoopInfo &LI = AM.getResult<LoopAnalysis>(F);

  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<VariableUniformityAnalysis>();
  PAChanged.preserve<WorkitemHandlerChooser>();

  typedef std::vector<llvm::Instruction* > InstructionVec;

  InstructionVec PHIs;

  for (Function::iterator bb = F.begin(); bb != F.end(); ++bb) {
    for (BasicBlock::iterator p = bb->begin(); p != bb->end(); ++p) {
      Instruction *I = &*p;
      if (!isa<PHINode>(I))
        continue;

      // If this is a PHINode in a non-barrier loop header, we should not
      // convert it to allocas to enable easier loop analysis for loopvec and
      // to avoid storing the induction variable in the WI context. Repl
      // relies on all PHIs to be converted to allocas.
      llvm::Loop *L = LI.getLoopFor(I->getParent());
      if (L != nullptr && !Barrier::IsLoopWithBarrier(*L))
        continue;
      PHIs.push_back(I);
    }
  }

  bool Changed = false;
  for (InstructionVec::iterator i = PHIs.begin(); i != PHIs.end();
       ++i) {
      Instruction *instr = *i;
      if (breakPHIToAllocas(dyn_cast<PHINode>(instr), VUA) != nullptr)
        Changed = true;
  }

  return Changed ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
