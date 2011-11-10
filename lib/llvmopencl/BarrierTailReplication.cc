// LLVM function pass to replicate barrier tails (successors to barriers).
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

#include "BarrierTailReplication.h"
#include "llvm/InstrTypes.h"
#include "llvm/Instructions.h"

using namespace llvm;
using namespace pocl;

#define BARRIER_FUNCTION_NAME "barrier"

static bool block_has_barrier(const BasicBlock *bb);
  
namespace {
  static
  RegisterPass<BarrierTailReplication> X("barriertails",
					 "Barrier tail replication pass",
					 false, false);
}

char BarrierTailReplication::ID = 0;

void
BarrierTailReplication::getAnalysisUsage(AnalysisUsage &AU) const
{
  AU.addRequired<DominatorTree>();
  AU.addRequired<LoopInfo>();
}

bool
BarrierTailReplication::runOnFunction(Function &F)
{
  DT = &getAnalysis<DominatorTree>();
  LI = &getAnalysis<LoopInfo>();

  return ProcessFunction(F);
}

bool
BarrierTailReplication::ProcessFunction(Function &F)
{
  BasicBlockSet processed_bbs;

  FindBarriersDFS(&F.getEntryBlock(), processed_bbs);

  return true;
}  


// Recursively (depht-first) look for barriers in all possible
// execution paths starting on entry, replicating the barrier
// successors to ensure there is a separate function exit BB
// for each combination of traversed barriers. The set
// processed_bbs stores the 
void
BarrierTailReplication::FindBarriersDFS(BasicBlock *bb,
                                        BasicBlockSet &processed_bbs)
{
  // Check if we already visited this BB (to avoid
  // infinite recursion in case of unbarriered loops).
  if (processed_bbs.count(bb) != 0)
    return;

  processed_bbs.insert(bb);

  TerminatorInst *t = bb->getTerminator();
  Function *f = bb->getParent();

  if (block_has_barrier(bb)) {
    // This block has a barrier, replicate all successors.
    // Even the path starting in an unique successor is replicated,
    // as it the path might be joined by another path in a
    // sucessor BB (see ifbarrier4.ll in tests).
    Loop *l = LI->getLoopFor(bb);
    for (unsigned i = 0, e = t->getNumSuccessors(); i != e; ++i) {
      BasicBlock *subgraph_entry = t->getSuccessor(i);
      if ((l != NULL)  && (l->getHeader() == subgraph_entry)) {
        // Do not replicate the path leading to the loop header,
        // as would lead to infinite unrolling.
        continue;
      }
      BasicBlock *replicated_subgraph_entry =
	ReplicateSubgraph(subgraph_entry, f);
      t->setSuccessor(i, replicated_subgraph_entry);
    }
  }

  // Find barriers in the successors (depth first).
  for (unsigned i = 0, e = t->getNumSuccessors(); i != e; ++i)
    FindBarriersDFS(t->getSuccessor(i), processed_bbs);
}


BasicBlock *
BarrierTailReplication::ReplicateSubgraph(BasicBlock *entry,
                                          Function *f)
{
  // Find all basic blocks to replicate.
  std::set<BasicBlock *> subgraph;
  FindSubgraph(subgraph, entry);

  // Replicate subgraph maintaining control flow.
  std::set<BasicBlock *> v;
  std::map<Value *, Value *> m;
  ReplicateBasicBlocks(v, m, subgraph, f);
  UpdateReferences(v, m);

  // We have modified the function. Possibly created new loops.
  // Update analysis passes.
  DT->runOnFunction(*f);
  LI->releaseMemory();
  LI->getBase().Calculate(DT->getBase());

  // Return entry block of replicated subgraph.
  return cast<BasicBlock>(m[entry]);
}


void
BarrierTailReplication::FindSubgraph(BasicBlockSet &subgraph,
                                     BasicBlock *entry)
{
  // This check is not enough when we have loops inside barriers.
  // Use LoopInfo.
  // if (subgraph.count(entry) != 0)
  //   return;

  subgraph.insert(entry);

  const TerminatorInst *t = entry->getTerminator();
  Loop *l = LI->getLoopFor(entry);
  for (unsigned i = 0, e = t->getNumSuccessors(); i != e; ++i) {
    BasicBlock *successor = t->getSuccessor(i);
    if ((l != NULL)  && (l->getHeader() == successor))
      continue;
    FindSubgraph(subgraph, successor);
  }
}


void
BarrierTailReplication::ReplicateBasicBlocks(BasicBlockSet &new_graph,
                                             ValueValueMap &reference_map,
                                             BasicBlockSet &graph,
                                             Function *f)
{
  for (std::set<BasicBlock *>::const_iterator i = graph.begin(),
	 e = graph.end();
       i != e; ++i) {
    BasicBlock *b = *i;
    BasicBlock *new_b = BasicBlock::Create(b->getContext(),
					   b->getName(),
					   f);
    reference_map.insert(std::make_pair(b, new_b));
    new_graph.insert(new_b);

    for (BasicBlock::iterator i2 = b->begin(), e2 = b->end();
	 i2 != e2; ++i2) {
      Instruction *i = i2->clone();
      reference_map.insert(std::make_pair(i2, i));
      new_b->getInstList().push_back(i);
    }
  }
}


void
BarrierTailReplication::UpdateReferences(const BasicBlockSet &graph,
                                         const ValueValueMap &reference_map)
{
  for (std::set<BasicBlock *>::const_iterator i = graph.begin(),
	 e = graph.end();
       i != e; ++i) {
    BasicBlock *b = *i;
    for (BasicBlock::iterator i2 = b->begin(), e2 = b->end();
	 i2 != e2; ++i2) {
      Instruction *i = i2;
      for (std::map<Value *, Value *>::const_iterator i3 =
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
    if (const CallInst *c = dyn_cast<CallInst>(i)) {
      const Value *v = c->getCalledValue();
      if (v->getName().equals(BARRIER_FUNCTION_NAME))
	return true;
    }
  }

  return false;
}




