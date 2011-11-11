// LLVM function pass to canonicalize barriers.
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

#include "CanonicalizeBarriers.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Instructions.h"

using namespace llvm;
using namespace pocl;

#define BARRIER_FUNCTION_NAME "barrier"

namespace {
  static
  RegisterPass<CanonicalizeBarriers> X("barriers",
                                       "Barrier canizalization pass");
}

char CanonicalizeBarriers::ID = 0;

void
CanonicalizeBarriers::getAnalysisUsage(AnalysisUsage &AU) const
{
  AU.addPreserved<DominatorTree>();
  AU.addRequired<LoopInfo>();
  AU.addPreserved<LoopInfo>();
}

bool
CanonicalizeBarriers::runOnFunction(Function &F)
{
  DT = getAnalysisIfAvailable<DominatorTree>();
  LI = &getAnalysis<LoopInfo>();

  return ProcessFunction(F);
}

// Canonicalize barriers: ensure all barriers are in a separate BB
// containint only the barrier and the terminator, with just one
// predecessor and one successor. This allows us to use
// those BBs as markers only, they will not be replicated.
bool
CanonicalizeBarriers::ProcessFunction(Function &F)
{
  bool changed = false;

  InstructionSet PreSplitPoints;
  InstructionSet PostSplitPoints;
  InstructionSet BarriersToAdd;

  CallInst *barrier = NULL;

  for (Function::iterator i = F.begin(), e = F.end();
       i != e; ++i) {
    BasicBlock *b = i;
    for (BasicBlock::iterator i = b->begin(), e = b->end();
	 i != e; ++i) {
      if (CallInst *c = dyn_cast<CallInst>(i)) {
	if (Function *f = c->getCalledFunction()) {
	  if (f->getName().equals(BARRIER_FUNCTION_NAME)) {
            barrier = c;
            
            // We found a barrier, add the split points.
	    PreSplitPoints.insert(i);
	    PostSplitPoints.insert(i);
            
            // Is this barrier inside of a loop?
            Loop *loop = LI->getLoopFor(b);
            if (loop != NULL) {
              // We need loops to be canonicalized.  If the barrier
              // is in a loop, add a barrier in the preheader.
              BasicBlock *preheader = loop->getLoopPreheader();
              assert(preheader != NULL);
              Instruction *new_barrier = barrier->clone();
              new_barrier->insertBefore(preheader->getTerminator());
              changed = true;
              // No split point after preheader barriers, so we ensure
              // WI 0,0,0 starts at the loop header.  But still we need
              // a split before.
              PreSplitPoints.insert(new_barrier);

              // Add barriers before any loop backedge.  This
              // is to ensure all workitems run to the end of the loop
              // (because otherwise first WI will jump back to the
              // header and other WIs will skip portion of the
              // loop body).
              // We cannot add the barriers directly here to avoid
              // processing them when going on trough the loop, schedule
              // them to be added later. 
              BasicBlock *latch = loop->getLoopLatch();
              assert(latch != NULL);
              // If this barrier happens to be before the latch terminator,
              // there is no need to add an additional barrier.
              if (latch->getTerminator()->getPrevNode() != c)
                BarriersToAdd.insert(latch->getTerminator());
            }
          }
	}
      }
    }
  }

  // Add scheduled barriers.
  for (InstructionSet::iterator i = BarriersToAdd.begin(), e = BarriersToAdd.end();
       i != e; ++i) {
    assert(barrier != NULL);
    Instruction *new_barrier = barrier->clone();
    new_barrier->insertBefore(*i);
    changed = true;
    PreSplitPoints.insert(new_barrier);
    PostSplitPoints.insert(new_barrier);
  }

  // Finally add all the split points, now that we are done with the
  // iterators.
  for (InstructionSet::iterator i = PreSplitPoints.begin(), e = PreSplitPoints.end();
       i != e; ++i) {
    BasicBlock *b = (*i)->getParent();
    BasicBlock *new_b = b->splitBasicBlock(*i);
    new_b->takeName(b);
    b->setName(new_b->getName() + ".prebarrier");

    // Update analysis
    if (DT)
      DT->runOnFunction(F);
    Loop *l = LI->getLoopFor(b);
    if (l)
      l->addBasicBlockToLoop(new_b, LI->getBase());

    changed = true;
  }
  for (InstructionSet::iterator i = PostSplitPoints.begin(), e = PostSplitPoints.end();
       i != e; ++i) {
    BasicBlock *b = (*i)->getParent();
    BasicBlock *new_b = b->splitBasicBlock((*i)->getNextNode(), b->getName() + ".postbarrier");

    // Update analysis
    if (DT)
      DT->runOnFunction(F);
    Loop *l = LI->getLoopFor(b);
    if (l)
      l->addBasicBlockToLoop(new_b, LI->getBase());

    changed = true;
  }

  if (DT)
    DT->verifyAnalysis();
  LI->verifyAnalysis();
  
  return changed;
}
