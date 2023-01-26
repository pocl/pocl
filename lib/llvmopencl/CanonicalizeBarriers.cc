// LLVM function pass to canonicalize barriers.
// 
// Copyright (c) 2011 Universidad Rey Juan Carlos
//               2012-2014 Pekka Jääskeläinen / Tampere University of Technology
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

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "pocl.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Dominators.h"

#include "CanonicalizeBarriers.h"
#include "Barrier.h"
#include "Workgroup.h"
#include "VariableUniformityAnalysis.h"
#include "WorkitemHandlerChooser.h"

POP_COMPILER_DIAGS

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
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addPreserved<VariableUniformityAnalysis>();
  AU.addRequired<WorkitemHandlerChooser>();
  AU.addPreserved<WorkitemHandlerChooser>();
}

bool
CanonicalizeBarriers::runOnFunction(Function &F)
{
  if (!Workgroup::isKernelToProcess(F))
    return false;
  bool changed = false;

  BasicBlock *entry = &F.getEntryBlock();
  if (!Barrier::hasOnlyBarrier(entry)) {
    BasicBlock *effective_entry = SplitBlock(entry, 
                                             &(entry->front()));

    effective_entry->takeName(entry);
    entry->setName("entry.barrier");
    Barrier::Create(entry->getTerminator());
    changed |= true;
  }

  for (Function::iterator i = F.begin(), e = F.end(); i != e; ++i) {

    BasicBlock *b = &*i;
    auto t = b->getTerminator();

    const bool isExitNode =
      (t->getNumSuccessors() == 0) && (!Barrier::hasOnlyBarrier(b));

    // The function exits should have barriers.
    if (isExitNode && !Barrier::hasOnlyBarrier(b)) {
      /* In case the bb is already terminated with a barrier,
         split before the barrier so we don't create an empty
         parallel region.

         This is because the assumptions of the other passes in the
         compilation that are 
         a) exit node is a barrier block 
         b) there are no empty parallel regions (which would be formed 
         between the explicit barrier and the added one). */
      BasicBlock *exit; 
      if (Barrier::endsWithBarrier(b))
        exit = SplitBlock(b, t->getPrevNode());
      else
        exit = SplitBlock(b, t);
      exit->setName("exit.barrier");
      Barrier::Create(t);
      changed |= true;
    }
  }

  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  return ProcessFunction(F) || changed;
}

// Canonicalize barriers: ensure all barriers are in a separate BB
// containing only the barrier and the terminator, with just one
// predecessor. This allows us to use those BBs as markers only, 
// they will not be replicated.
bool
CanonicalizeBarriers::ProcessFunction(Function &F) {

  auto WIH = getAnalysis<WorkitemHandlerChooser>().chosenHandler();

  bool changed = false;

  InstructionSet Barriers;

  for (Function::iterator i = F.begin(), e = F.end();
       i != e; ++i) {
    BasicBlock *b = &*i;
    for (BasicBlock::iterator i = b->begin(), e = b->end();
         i != e; ++i) {
      if (isa<Barrier>(i)) {
        Barriers.insert(&*i);
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
    Instruction *t = b->getTerminator();
    // if ((t->getNumSuccessors() > 1) ||
    //     (t->getPrevNode() != *i)) {
    // Change: barriers with several successors are all right
    // they just start several parallel regions. Simplifies
    // loop handling.

    const bool HasNonBranchInstructionsAfterBarrier =
        t->getPrevNode() != *i ||
        (WIH == WorkitemHandlerChooser::POCL_WIH_CBS &&
         t->getNumSuccessors() > 1);

    if (HasNonBranchInstructionsAfterBarrier) {
      BasicBlock *new_b = SplitBlock(b, (*i)->getNextNode());
      new_b->setName(b->getName() + ".postbarrier");
      changed = true;
    }

    BasicBlock *predecessor = b->getSinglePredecessor();
    if (predecessor != NULL) {
      auto pt = predecessor->getTerminator();
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
    BasicBlock *new_b = SplitBlock(b, *i);
    new_b->takeName(b);
    b->setName(new_b->getName() + ".prebarrier");
    changed = true;
  }

  // Prune empty regions. That is, if there are two successive
  // pure barrier blocks without side branches, remove the other one.
  bool emptyRegionDeleted = false;
  do {
    emptyRegionDeleted = false;
    for (Function::iterator i = F.begin(), e = F.end();
         i != e; ++i) {
        BasicBlock *b = &*i;
        auto t = b->getTerminator();
        if (!Barrier::endsWithBarrier(b) || t->getNumSuccessors() != 1)
          continue;

        BasicBlock *successor = t->getSuccessor(0);

        if (Barrier::hasOnlyBarrier(successor) && 
            successor->getSinglePredecessor() == b) {
            b->replaceAllUsesWith(successor);
            b->eraseFromParent();
            emptyRegionDeleted = true;
            changed = true;
            break;
          }
      }
  } while (emptyRegionDeleted);
  

  return changed;
}
