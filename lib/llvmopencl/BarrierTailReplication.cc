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
#include "llvm/Function.h"
#include "llvm/InstrTypes.h"
#include "llvm/Instructions.h"
#include <map>
#include <set>

using namespace llvm;
using namespace locl;

#define BARRIER_FUNCTION_NAME "barrier"

static void find_barriers_dfs(BasicBlock *bb, std::set<BasicBlock *> &processed_bbs);
static bool block_has_barrier(const BasicBlock *bb);
static BasicBlock *replicate_subgraph(BasicBlock *entry,
				      Function *f);
static void find_subgraph(std::set<BasicBlock *> &subgraph,
			  BasicBlock *entry);
static void replicate_basicblocks(std::set<BasicBlock *> &new_graph,
				  std::map<Value *, Value *> &reference_map,
				  const std::set<BasicBlock *> &graph,
				  Function *f);
static void update_references(const std::set<BasicBlock *>&graph,
			      const std::map<Value *, Value *> &reference_map);
  
namespace {
  static
  RegisterPass<BarrierTailReplication> X("barriertails",
					 "Barrier tail replication pass",
					 false, false);
}

char BarrierTailReplication::ID = 0;

bool
BarrierTailReplication::runOnFunction(Function &F)
{
  std::set<BasicBlock *> processed_bbs;

  find_barriers_dfs(&(F.getEntryBlock()), processed_bbs);

  return true;
}

static void
find_barriers_dfs(BasicBlock *bb, std::set<BasicBlock *> &processed_bbs)
{
  if (processed_bbs.count(bb) != 0)
    return;

  processed_bbs.insert(bb);

  Function *f = bb->getParent();
  TerminatorInst *t = bb->getTerminator();

  if (block_has_barrier(bb)) {
    // Replicate all successors.
    for (unsigned i = 0, e = t->getNumSuccessors(); i != e; ++i) {
      BasicBlock *subgraph_entry = t->getSuccessor(i);
      BasicBlock *replicated_subgraph_entry =
	replicate_subgraph(subgraph_entry, f);
      t->setSuccessor(i, replicated_subgraph_entry);
    }
  }

  // Find barriers in the successors (depth first).
  for (unsigned i = 0, e = t->getNumSuccessors(); i != e; ++i)
    find_barriers_dfs(t->getSuccessor(i), processed_bbs);
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

static BasicBlock *
replicate_subgraph(BasicBlock *entry, Function *f)
{
  // Find all basic blocks to replicate.
  std::set<BasicBlock *> subgraph;
  find_subgraph(subgraph, entry);

  // Replicate subgraph maintaining control flow.
  std::set<BasicBlock *> v;
  std::map<Value *, Value *> m;
  replicate_basicblocks(v, m, subgraph, f);
  update_references(v, m);

  // Return entry block of replicated subgraph.
  return cast<BasicBlock>(m[entry]);
}

static void
find_subgraph(std::set<BasicBlock *> &subgraph,
	      BasicBlock *entry)
{
  if (subgraph.count(entry) != 0)
    return;

  subgraph.insert(entry);

  const TerminatorInst *t = entry->getTerminator();
  for (unsigned i = 0, e = t->getNumSuccessors(); i != e; ++i)
    find_subgraph(subgraph, t->getSuccessor(i));
}

static void
replicate_basicblocks(std::set<BasicBlock *> &new_graph,
		      std::map<Value *, Value *> &reference_map,
		      const std::set<BasicBlock *> &graph,
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

