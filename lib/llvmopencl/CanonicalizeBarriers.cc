// LLVM function pass to canonicalize barriers.
// 
// Copyright (c) 2011 Universidad Rey Juan Carlos
//               2012 Pekka Jääskeläinen / Tampere University of Technology
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
#include "CanonicalizeBarriers.h"
#include "BarrierBlock.h"
#include "Barrier.h"
#include "Workgroup.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <iostream>

#if (defined LLVM_3_1 or defined LLVM_3_2)
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#else
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#endif

using namespace llvm;
using namespace pocl;

namespace {
  static
  RegisterPass<CanonicalizeBarriers> X("barriers",
                                       "Barrier canonicalization pass");
}

char CanonicalizeBarriers::ID = 0;

void
CanonicalizeBarriers::getAnalysisUsage(AnalysisUsage &AU) const
{
}

bool
CanonicalizeBarriers::runOnFunction(Function &F)
{
  if (!Workgroup::isKernelToProcess(F))
    return false;

  BasicBlock *entry = &F.getEntryBlock();
  if (!isa<BarrierBlock>(entry)) {
    BasicBlock *effective_entry = SplitBlock(entry, 
                                             &(entry->front()),
                                             this);
    effective_entry->takeName(entry);
    entry->setName("entry.barrier");
    Barrier::Create(entry->getTerminator());
  }

  for (Function::iterator i = F.begin(), e = F.end(); i != e; ++i) {
    BasicBlock *b = i;
    TerminatorInst *t = b->getTerminator();
    if ((t->getNumSuccessors() == 0) && (!isa<BarrierBlock>(b))) {
      /* In case the bb is already terminated with a barrier,
         split before the barrier so we dot create an empty
         parallel region.
         
         This is because the assumptions of the other passes in the 
         compilation that are 
         a) exit node is a barrier block 
         b) there are no empty parallel regions (which would be formed 
         between the explicit barrier and the added one). */
      BasicBlock *exit; 
      if (Barrier::endsWithBarrier(b))
        exit = SplitBlock(b, t->getPrevNode(), this);
      else
        exit = SplitBlock(b, t, this);
      exit->setName("exit.barrier");
      Barrier::Create(t);
    }
  }

  DT = getAnalysisIfAvailable<DominatorTree>();
  LI = getAnalysisIfAvailable<LoopInfo>();

  bool changed = ProcessFunction(F);

  if (DT)
    DT->verifyAnalysis();
  if (LI)
    LI->verifyAnalysis();

  return changed;
}


// Canonicalize barriers: ensure all barriers are in a separate BB
// containing only the barrier and the terminator, with just one
// predecessor and one successor. This allows us to use
// those BBs as markers only, they will not be replicated.
bool
CanonicalizeBarriers::ProcessFunction(Function &F)
{
  bool changed = false;

  InstructionSet Barriers;

  for (Function::iterator i = F.begin(), e = F.end();
       i != e; ++i) 
    {
      BasicBlock *b = i;
      for (BasicBlock::iterator i = b->begin(), e = b->end();
           i != e; ++i) 
        {
          if (isa<Barrier>(i))
            {
              Barriers.insert(i);
            }
        }
    }
  
  // Finally add all the split points, now that we are done with the
  // iterators.
  for (InstructionSet::iterator i = Barriers.begin(), e = Barriers.end();
       i != e; ++i) {
    BasicBlock *b = (*i)->getParent();

    // Split post barrier first cause it does not make the barrier
    // to belong to another basic block.
    TerminatorInst  *t = b->getTerminator();
    // if ((t->getNumSuccessors() > 1) ||
    //     (t->getPrevNode() != *i)) {
    // Change: barriers with several successors are all right
    // they just start several parallel regions. Simplifies
    // loop handling.

    const bool HAS_NON_BRANCH_INSTRUCTIONS_AFTER_BARRIER = 
      t->getPrevNode() != *i;

    if (HAS_NON_BRANCH_INSTRUCTIONS_AFTER_BARRIER) {
      BasicBlock *new_b = SplitBlock(b, (*i)->getNextNode(), this);
      new_b->setName(b->getName() + ".postbarrier");
      changed = true;
    }

    BasicBlock *predecessor = b->getSinglePredecessor();
    if (predecessor != NULL) {
      TerminatorInst *pt = predecessor->getTerminator();
      if ((pt->getNumSuccessors() == 1) &&
          (&b->front() == (*i))) {
        // Barrier is at the beginning of the BB,
        // which has a single predecessor with just
        // one successor (the barrier itself), thus
        // no need to split before barrier.
        continue;
      }
    }
    if ((b == &(b->getParent()->getEntryBlock())) &&
        (&b->front() == (*i)))
      continue;
    
    // If no instructions before barrier, do not split
    // (allow multiple predecessors, eases loop handling).
    // if (&b->front() == (*i))
    //   continue;
    BasicBlock *new_b = SplitBlock(b, *i, this);
    new_b->takeName(b);
    b->setName(new_b->getName() + ".prebarrier");
    changed = true;
  }

  /* Prune empty regions. That is, if there are two successive
     barriers, remove the other one. */
  bool emptyRegionDeleted = false;  
  do {
    emptyRegionDeleted = false;
    for (Function::iterator i = F.begin(), e = F.end();
         i != e; ++i) 
      {
        BasicBlock *b = i;
        llvm::TerminatorInst *t = b->getTerminator();
        if (!Barrier::endsWithBarrier(b) || t->getNumSuccessors() != 1) continue;

        BasicBlock *successor = t->getSuccessor(0);

        if (Barrier::hasOnlyBarrier(successor) && 
            successor->getSinglePredecessor() == b &&
            successor->getTerminator()->getNumSuccessors() == 1)
          {
            b->getTerminator()->setSuccessor(0, successor->getTerminator()->getSuccessor(0));
            successor->replaceAllUsesWith(b);
            successor->eraseFromParent();
            emptyRegionDeleted = true;
            changed = true;
            break;
          }
      }
  } while (emptyRegionDeleted);
  

  return changed;
}
