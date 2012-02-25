// LLVM function pass to replicate barrier tails (successors to barriers).
// 
// Copyright (c) 2011 Universidad Rey Juan Carlos and
//               2012 Pekka Jääskeläinen / TUT
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

#include "BarrierTailReplication.h"
#include "Barrier.h"
#include "Workgroup.h"
#include "llvm/InstrTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <iostream>

using namespace llvm;
using namespace pocl;

//#define DEBUG_BARRIER_REPL

static bool block_has_barrier(const BasicBlock *bb);
  
namespace {
  static
  RegisterPass<BarrierTailReplication> X("barriertails",
					 "Barrier tail replication pass");
}

char BarrierTailReplication::ID = 0;

void
BarrierTailReplication::getAnalysisUsage(AnalysisUsage &AU) const
{
  AU.addRequired<DominatorTree>();
  AU.addPreserved<DominatorTree>();
  AU.addRequired<LoopInfo>();
  AU.addPreserved<LoopInfo>();
}

bool
BarrierTailReplication::runOnFunction(Function &F)
{
  if (!Workgroup::isKernelToProcess(F))
    return false;
  
  DT = &getAnalysis<DominatorTree>();
  LI = &getAnalysis<LoopInfo>();
  
  bool changed = ProcessFunction(F);

  DT->verifyAnalysis();
  LI->verifyAnalysis();

  /* The created tails might contain PHI nodes with
     operands referring to the non-predecessor (split point)
     BB. These must be cleaned to avoid breakage later on.
   */
  for (Function::iterator i = F.begin(), e = F.end();
       i != e; ++i)
    {
      llvm::BasicBlock *bb = i;
      changed |= CleanupPHIs(bb);
    }      

  return changed;
}

bool
BarrierTailReplication::ProcessFunction(Function &F)
{
  BasicBlockSet processed_bbs;

  return FindBarriersDFS(&F.getEntryBlock(), processed_bbs);
}  


// Recursively (depht-first) look for barriers in all possible
// execution paths starting on entry, replicating the barrier
// successors to ensure there is a separate function exit BB
// for each combination of traversed barriers. The set
// processed_bbs stores the 
bool
BarrierTailReplication::FindBarriersDFS(BasicBlock *bb,
                                        BasicBlockSet &processed_bbs)
{
  bool changed = false;

  // Check if we already visited this BB (to avoid
  // infinite recursion in case of unbarriered loops).
  if (processed_bbs.count(bb) != 0)
    return changed;

  processed_bbs.insert(bb);

  TerminatorInst *t = bb->getTerminator();

  if (block_has_barrier(bb)) {
    changed = ReplicateJoinedSubgraphs(bb, bb);
  }

  // Find barriers in the successors (depth first).
  for (unsigned i = 0, e = t->getNumSuccessors(); i != e; ++i)
    changed |= FindBarriersDFS(t->getSuccessor(i), processed_bbs);

  return changed;
}


// Only replicate those parts of the subgraph that are not
// dominated by a (barrier) basic block, to avoid excesive
// (and confusing) code replication.
bool
BarrierTailReplication::ReplicateJoinedSubgraphs(BasicBlock *dominator,
                                                 BasicBlock *subgraph_entry)
{
  bool changed = false;

  assert(DT->dominates(dominator, subgraph_entry));

  Function *f = dominator->getParent();

  TerminatorInst *t = subgraph_entry->getTerminator();
  Loop *l = LI->getLoopFor(subgraph_entry);
  for (int i = 0, e = t->getNumSuccessors(); i != e; ++i) {
    BasicBlock *b = t->getSuccessor(i);
    if (l != NULL && l->getHeader() == b) {
      // This is a loop backedge. Do not find subgraphs across
      // those.
      continue;
    }
    if (DT->dominates(dominator, b))
      changed |= ReplicateJoinedSubgraphs(dominator, b);
    else {
      BasicBlock *replicated_subgraph_entry =
        ReplicateSubgraph(b, f);
      /* Ensure there's a barrier just before the terminator
         branch of the split point BB. Otherwise the PR gets
         replicated multiple times and there will be illegal
         references in the another branch. Without this
         the case with an early return before a barrier fails.

         Basically it's the case:

         something1;
         if (data_dependent_condition) 
            return;
         else
            barrier();
         something2;

         Here something1 -> ddc -> return and something1 -> ddcs -> barrier
         got replicated separately and the variable references were broken.
         This inserts a barrier just before the branch in the condition
         check of the if block.

         TODO: check from Carlos how the "tail replication" is supposed to
         work without the barrier. It's not entirely clear to me (Pekka).
      */
      t->setSuccessor(i, replicated_subgraph_entry);
      changed = true;
      // We have modified the function. Possibly created new loops.
      // Update analysis passes.
      DT->runOnFunction(*f);
      LI->getBase().Calculate(DT->getBase());

    }
  }

  return changed;
}

// Removes phi elements for which there is no successors (anymore).
bool
BarrierTailReplication::CleanupPHIs(llvm::BasicBlock *BB)
{

  bool changed = false;
#ifdef DEBUG_BARRIER_REPL
  std::cerr << "### CleanupPHIs for BB:" << std::endl;
  BB->dump();
#endif

  for (BasicBlock::iterator BI = BB->begin(), BE = BB->end(); BI != BE; )
    {
      PHINode *PN = dyn_cast<PHINode>(BI);
      if (PN == NULL) break;

      bool PHIRemoved = false;
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i < e; ++i)
        {
          bool isSuccessor = false;
          // find if the predecessor branches to this one (anymore)
          for (unsigned s = 0, 
                 se = PN->getIncomingBlock(i)->getTerminator()->getNumSuccessors();
               s < se; ++s) {
            if (PN->getIncomingBlock(i)->getTerminator()->getSuccessor(s) == BB) 
              {
                isSuccessor = true;
                break;
              }
          }
          if (!isSuccessor)
            {
#ifdef DEBUG_BARRIER_REPL
              std::cerr << "removing incoming value " << i << " from PHINode:" << std::endl;
              PN->dump();            
#endif
              PN->removeIncomingValue(i, true);
              changed = true;
              e--;
              if (e == 0)
                {
                  PHIRemoved = true;
                  break;
                }
              i = 0; 
              continue;
            }
        }
      if (PHIRemoved)
        BI = BB->begin();
      else
        BI++;
    }
  return changed;
}

BasicBlock *
BarrierTailReplication::ReplicateSubgraph(BasicBlock *entry,
                                          Function *f)
{
  // Find all basic blocks to replicate.
  BasicBlockVector subgraph;
  FindSubgraph(subgraph, entry);

  // Replicate subgraph maintaining control flow.
  BasicBlockVector v;
  ValueValueMap m;
  ReplicateBasicBlocks(v, m, subgraph, f);
  UpdateReferences(v, m);

  // Return entry block of replicated subgraph.
  return cast<BasicBlock>(m[entry]);
}


void
BarrierTailReplication::FindSubgraph(BasicBlockVector &subgraph,
                                     BasicBlock *entry)
{
  subgraph.push_back(entry);

  const TerminatorInst *t = entry->getTerminator();
  Loop *l = LI->getLoopFor(entry);
  for (unsigned i = 0, e = t->getNumSuccessors(); i != e; ++i) {
    BasicBlock *successor = t->getSuccessor(i);
    if ((l != NULL)  && (l->getHeader() == successor)) {
      // This is a loop backedge. Do not find subgraphs across
      // those.
      continue;
    }
    FindSubgraph(subgraph, successor);
  }
}


void
BarrierTailReplication::ReplicateBasicBlocks(BasicBlockVector &new_graph,
                                             ValueValueMap &reference_map,
                                             BasicBlockVector &graph,
                                             Function *f)
{
  for (BasicBlockVector::const_iterator i = graph.begin(),
	 e = graph.end();
       i != e; ++i) {
    BasicBlock *b = *i;
    BasicBlock *new_b = BasicBlock::Create(b->getContext(),
					   b->getName() + ".btr",
					   f);
    reference_map.insert(std::make_pair(b, new_b));
    new_graph.push_back(new_b);

    for (BasicBlock::iterator i2 = b->begin(), e2 = b->end();
	 i2 != e2; ++i2) {
      Instruction *i = i2->clone();
      reference_map.insert(std::make_pair(i2, i));
      new_b->getInstList().push_back(i);
    }
  }
}


void
BarrierTailReplication::UpdateReferences(const BasicBlockVector &graph,
                                         const ValueValueMap &reference_map)
{
  for (BasicBlockVector::const_iterator i = graph.begin(),
	 e = graph.end();
       i != e; ++i) {
    BasicBlock *b = *i;
    for (BasicBlock::iterator i2 = b->begin(), e2 = b->end();
	 i2 != e2; ++i2) {
      Instruction *i = i2;
      for (ValueValueMap::const_iterator i3 =
             reference_map.begin(), e3 = reference_map.end();
           i3 != e3; ++i3) {
        i->replaceUsesOfWith(i3->first, i3->second);
      }
    }
  }

}


static bool
block_has_barrier(const BasicBlock *bb)
{
  for (BasicBlock::const_iterator i = bb->begin(), e = bb->end();
       i != e; ++i) {
    if (isa<Barrier>(i))
      return true;
  }

  return false;
}
