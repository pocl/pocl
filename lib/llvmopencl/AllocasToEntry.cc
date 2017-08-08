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

#include <sstream>
#include <iostream>

#include "config.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>

#include "AllocasToEntry.h"

namespace pocl {

using namespace llvm;

namespace {
  static
  RegisterPass<pocl::AllocasToEntry> X("allocastoentry", 
                                       "Move allocas to the function entry node.");
}

char AllocasToEntry::ID = 0;


AllocasToEntry::AllocasToEntry() : FunctionPass(ID)
{
}

bool
AllocasToEntry::runOnFunction(Function &F)
{
  // This solves problem with dynamic stack objects that are 
  // not supported by some targets (TCE).
  Function::iterator I = F.begin();
  Instruction *firstInsertionPt = &*(I++)->getFirstInsertionPt();
    
  bool changed = false;
  for (Function::iterator E = F.end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE;) {
      AllocaInst *allocaInst = dyn_cast<AllocaInst>(BI++);
      if (allocaInst && isa<ConstantInt>(allocaInst->getArraySize())) {
        allocaInst->moveBefore(firstInsertionPt);
        changed = true;
      }
    }
  }
  return changed;
}

}
