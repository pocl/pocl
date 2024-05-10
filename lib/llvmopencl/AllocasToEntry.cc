// AllocasToEntry, an LLVM pass to move allocas to the function entry node.
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

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>

#include "AllocasToEntry.h"
#include "LLVMUtils.h"
POP_COMPILER_DIAGS

#include <iostream>
#include <sstream>

#define PASS_NAME "allocastoentry"
#define PASS_CLASS pocl::AllocasToEntry
#define PASS_DESC "Move allocas to the function entry node."

namespace pocl {

using namespace llvm;

static bool allocasToEntry(Function &F) {
  // This solves problem with dynamic stack objects that are
  // not supported by some targets (TCE).
  Function::iterator I = F.begin();
  Instruction *firstInsertionPt = &*(I++)->getFirstInsertionPt();

  bool Changed = false;
  for (Function::iterator E = F.end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE;) {
      AllocaInst *AllocaI = dyn_cast<AllocaInst>(BI++);
      if (AllocaI && isa<ConstantInt>(AllocaI->getArraySize())) {
        AllocaI->moveBefore(firstInsertionPt);
        Changed = true;
      }
    }
  }
  return Changed;
}

llvm::PreservedAnalyses AllocasToEntry::run(llvm::Function &F,
                                            llvm::FunctionAnalysisManager &AM) {
  allocasToEntry(F);
  return PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl

