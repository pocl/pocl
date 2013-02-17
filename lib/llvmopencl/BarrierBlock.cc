// Class for a basic block that just contains a barrier.
// 
// Copyright (c) 2011 Universidad Rey Juan Carlos
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

#include "BarrierBlock.h"
#include "Barrier.h"
#include "config.h"
#if (defined LLVM_3_1 or defined LLVM_3_2)
#include "llvm/Instructions.h"
#else
#include "llvm/IR/Instructions.h"
#endif
#include <cassert>

using namespace llvm;
using namespace pocl;

static bool
verify(const BasicBlock *B);

bool
BarrierBlock::classof(const BasicBlock *B)
{
  if ((B->size() == 2) &&
      isa<Barrier> (&B->front())) {
    assert(verify(B));
    return true;
  }

  return false;
}

static bool
verify(const BasicBlock *B)
{
  assert((B->size() == 2) && "Barriers blocks should have no functionality!");
  // const Instruction *barrier = B->getFirstNonPHI();
  // assert(isa<Barrier>(barrier) && "Barriers blocks should have no functionality!");
  // assert(B->getTerminator()->getPrevNode() == barrier &&
  //        "Barriers blocks should have no functionality!");
#if 1 // We want to allow barriers with more than one predecessors (?)
      // (for loop header barriers).
  assert(((B->getSinglePredecessor() != NULL) ||
          (B == &(B->getParent()->front()))) &&
         "Barrier blocks should have exactly one predecessor (except entry barrier)!");
#endif
#if 0  // We want to allow barriers with more than one successor (for latch barriers).
  assert((B->getTerminator()->getNumSuccessors() <= 1) &&
         "Barrier blocks should have one successor, or zero for exit barriers!");
#endif
  assert(isa<Barrier>(B->front()));

  return true;
}

