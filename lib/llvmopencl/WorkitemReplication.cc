// LLVM function pass to replicate kernel body for several workitems.
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

#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
using namespace llvm;

#define BARRIER_FUNCTION_NAME "barrier"

static BasicBlock *find_barriers_dfs(BasicBlock *bb,
				     BasicBlock *entry,
				     std::set<BasicBlock *> &processed_barriers,
				     std::set<BasicBlock *> &bbs_to_replicate);
static bool block_has_barrier(const BasicBlock *bb);
static bool find_subgraph(std::set<BasicBlock *> &subgraph,
			  BasicBlock *entry,
			  BasicBlock *exit);
static void replicate_basicblocks(std::set<BasicBlock *> &new_graph,
				  std::map<Value *, Value *> &reference_map,
				  const std::set<BasicBlock *> &graph);
static void update_references(const std::set<BasicBlock *>&graph,
			      const std::map<Value *, Value *> &reference_map);

namespace {
  class WorkitemReplication : public FunctionPass {

  public:
    static char ID;
    WorkitemReplication(): FunctionPass(ID) {}

    virtual bool runOnFunction(Function &F);
  };

  char WorkitemReplication::ID = 0;
  static
  RegisterPass<WorkitemReplication> X("workitems",
				      "Workitem tail replication pass",
				      false, false);
}

bool
WorkitemReplication::runOnFunction(Function &F)
{
  std::set<BasicBlock *> processed_barriers;
  std::set<BasicBlock *> bbs_to_replicate;

  BasicBlock *exit = find_barriers_dfs(&(F.getEntryBlock()),
				       &(F.getEntryBlock()),
				       processed_barriers,
				       bbs_to_replicate);

  std::set<BasicBlock *> v;
  std::map<Value *, Value *> m;
  replicate_basicblocks(v, m, bbs_to_replicate);
  update_references(v, m);
  if (exit != NULL) {
    BranchInst::Create(cast<BasicBlock>(m[&(F.getEntryBlock())]),
		       exit->getTerminator());
    exit->getTerminator()->eraseFromParent();
  }

  return true;
}

static BasicBlock*
find_barriers_dfs(BasicBlock *bb,
		  BasicBlock *entry,
		  std::set<BasicBlock *> &processed_barriers,
		  std::set<BasicBlock *> &bbs_to_replicate)
{
  TerminatorInst *t = bb->getTerminator();
  
  if (block_has_barrier(bb) &&
      (processed_barriers.count(bb) == 0))
    {      
      // Replicate subgraph from entry to the barrier for
      // all workitems.
      std::set<BasicBlock *> subgraph;
      find_subgraph(subgraph, entry, bb);
      for (std::set<BasicBlock *>::const_iterator i = subgraph.begin(),
	     e = subgraph.end();
	   i != e; ++i) {
	if (block_has_barrier(*i) &&
	    processed_barriers.count(*i) == 0) {
	  // Subgraph to this barriers has still unprocessed barriers. Do
	  // not process this barrier yet (it will be done when coming
	  // from the path through the previous barrier).
	  return NULL;
	}
      }

      // Mark this barrier as processed.
      processed_barriers.insert(bb);
      
      // Replicate subgraph from entry to the barrier for
      // all workitems.
      std::set<BasicBlock *> v;
      std::map<Value *, Value *> m;
      replicate_basicblocks(v, m, subgraph);
      update_references(v, m);

      // Make original subgraph branch to replicated one.
      BasicBlock *b = cast<BasicBlock>(m[entry]);
      m.clear();
      m.insert(std::make_pair(bb, b));
      update_references(subgraph, m);

      // Continue processing after the barrier.
      bb = t->getSuccessor(0);
      subgraph.clear();
      v.clear();
      m.clear();
      BasicBlock *exit = find_barriers_dfs(bb, bb, processed_barriers, subgraph);
      replicate_basicblocks(v, m, subgraph);
      update_references(v, m);
      if (exit != NULL) {
	BranchInst::Create(cast<BasicBlock>(m[bb]), exit->getTerminator());
	exit->getTerminator()->eraseFromParent();
      }

      return NULL;
    }

  bbs_to_replicate.insert(bb);

  if (t->getNumSuccessors() == 0)
    return t->getParent();

  // Find barriers in the successors (depth first).
  BasicBlock *r = NULL;
  BasicBlock *s;
  for (unsigned i = 0, e = t->getNumSuccessors(); i != e; ++i) {
    s = find_barriers_dfs(t->getSuccessor(i), entry, processed_barriers, bbs_to_replicate);
    if (s != NULL) {
      assert((r == NULL || r == s) &&
	     "More than one function tail within same barrier path!\n");
      r = s;
    }
  }

  return r;
}

static bool
block_has_barrier(const BasicBlock *bb)
{
  for (BasicBlock::const_iterator i = bb->begin(), e = bb->end();
       i != e; ++i) {
    if (const CallInst *c = dyn_cast<CallInst>(i)) {
      const Value *v = c->getCalledValue();
      if (v->getName().equals(BARRIER_FUNCTION_NAME)) {
	assert((bb->size() == 2) &&
	       (bb->getTerminator()->getNumSuccessors() == 1) &&
	       ("Invalid barrier basicblock found!\n"));
	return true;
      }
    }
  }

  return false;
}

static bool
find_subgraph(std::set<BasicBlock *> &subgraph,
	      BasicBlock *entry,
	      BasicBlock *exit)
{
  if (entry == exit)
    return true;

  bool found = false;
  const TerminatorInst *t = entry->getTerminator();
  for (unsigned i = 0, e = t->getNumSuccessors(); i != e; ++i)
    found |= find_subgraph(subgraph, t->getSuccessor(i), exit);

  if (found)
    subgraph.insert(entry);

  return found;
}

static void
replicate_basicblocks(std::set<BasicBlock *> &new_graph,
		      std::map<Value *, Value *> &reference_map,
		      const std::set<BasicBlock *> &graph)
{
  for (std::set<BasicBlock *>::const_iterator i = graph.begin(),
	 e = graph.end();
       i != e; ++i) {
    BasicBlock *b = *i;
    BasicBlock *new_b = BasicBlock::Create(b->getContext(),
					   b->getName(),
					   b->getParent());
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

static void
update_references(const std::set<BasicBlock *>&graph,
		  const std::map<Value *, Value *> &reference_map)
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
