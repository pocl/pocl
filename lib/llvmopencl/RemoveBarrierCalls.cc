// LLVM function pass to remove all barrier calls.
//
// Copyright (c) 2016 Pekka Jääskeläinen / TUT
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
#include "Barrier.h"
#include "LLVMUtils.h"
#include "RemoveBarrierCalls.h"
#include "VariableUniformityAnalysis.h"
#include "Workgroup.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#include <set>

#define PASS_NAME "remove-barriers"
#define PASS_CLASS pocl::RemoveBarrierCalls
#define PASS_DESC "Removes all barrier calls."

namespace pocl {

using namespace llvm;

static bool removeBarrierCalls(Function &F) {

  // Collect the barrier calls to be removed first, not remove them
  // instantly as it'd invalidate the iterators.
  std::set<Instruction*> BarriersToRemove;

  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE; ++BI) {
      Instruction *Instr = dyn_cast<Instruction>(BI);
      if (llvm::isa<Barrier>(Instr)) {
        BarriersToRemove.insert(Instr);
      }
    }
  }

  bool Changed = !BarriersToRemove.empty();
  for (auto B : BarriersToRemove) {
    B->eraseFromParent();
  }

  return Changed;
}

llvm::PreservedAnalyses
RemoveBarrierCalls::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<VariableUniformityAnalysis>();
  PAChanged.preserve<WorkitemHandlerChooser>();
  return removeBarrierCalls(F) ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
