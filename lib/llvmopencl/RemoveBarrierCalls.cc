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

#include <set>

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "pocl.h"
#include "Barrier.h"
#include "RemoveBarrierCalls.h"
#include "Workgroup.h"
#include "VariableUniformityAnalysis.h"

POP_COMPILER_DIAGS

using namespace llvm;

namespace {
  static
  RegisterPass<pocl::RemoveBarrierCalls> X("remove-barriers",
                                           "Removes all barrier calls.");
}

namespace pocl {

char RemoveBarrierCalls::ID = 0;

RemoveBarrierCalls::RemoveBarrierCalls() : FunctionPass(ID) {
}

bool
RemoveBarrierCalls::runOnFunction(Function &F) {

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

  for (auto B : BarriersToRemove) {
    B->eraseFromParent();
  }

  return false;
}

void
RemoveBarrierCalls::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addPreserved<pocl::VariableUniformityAnalysis>();
}

}



