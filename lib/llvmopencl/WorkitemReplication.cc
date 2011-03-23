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

#include "WorkitemReplication.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/IRBuilder.h"

using namespace llvm;
using namespace locl;

#define BARRIER_FUNCTION_NAME "barrier"

static bool block_has_barrier(const BasicBlock *bb);
static bool find_subgraph(std::set<BasicBlock *> &subgraph,
			  BasicBlock *entry,
			  BasicBlock *exit);
static void purge_subgraph(std::set<BasicBlock *> &new_subgraph,
			   const std::set<BasicBlock *> &original_subgraph,
			   const BasicBlock *exit);
static void split_barriers(Function &F);

cl::list<int>
LocalSize("local-size",
	  cl::desc("Local size (x y z)"),
	  cl::multi_val(3));

namespace {
  static
  RegisterPass<WorkitemReplication> X("workitems",
				      "Workitem replication pass",
				      false, false);
}

char WorkitemReplication::ID = 0;

bool
WorkitemReplication::doInitialization(Module &M)
{
  // Allocate space for workitem reference maps. Workitem 0 does
  // not need it.
  int i = LocalSize[2] * LocalSize[1] * LocalSize[0] - 1;
  ReferenceMap = new ValueValueMap[i];

  LocalX = M.getGlobalVariable("_local_x");
  LocalY = M.getGlobalVariable("_local_y");
  LocalZ = M.getGlobalVariable("_local_z");

  return false;
}

bool
WorkitemReplication::runOnFunction(Function &F)
{
  BasicBlockSet subgraph;

  // Split basicblock at barriers. Barrier blocks must have a single
  // predecessor, single sucessor, and just the barrier call and
  // the terminator as body.
  split_barriers(F);

  BasicBlock *exit = findBarriersDFS(&(F.getEntryBlock()),
				     &(F.getEntryBlock()),
				     subgraph);

  if (exit != NULL)
    replicateWorkitemSubgraph(subgraph, &(F.getEntryBlock()), exit);

  return true;
}

bool
WorkitemReplication::doFinalization(Module &M)
{
  delete []ReferenceMap;

  return false;
}

// Perform a deep first seach on the subgraph starting on bb. On the
// outermost recursive call, entry = bb. Barriers are handled by
// recursive calls, so on return only subgraph needs still to be
// replicated. Return value is last basicblock (exit) of original graph.
BasicBlock*
WorkitemReplication::findBarriersDFS(BasicBlock *bb,
				     BasicBlock *entry,
				     BasicBlockSet &subgraph)
{
  TerminatorInst *t = bb->getTerminator();
  
  if (block_has_barrier(bb) &&
      (ProcessedBarriers.count(bb) == 0))
    {      
      BasicBlockSet pre_subgraph;
      find_subgraph(pre_subgraph, entry, bb);
      pre_subgraph.erase(bb); // Remove barrier basicblock from subgraph.
      for (std::set<BasicBlock *>::const_iterator i = pre_subgraph.begin(),
	     e = pre_subgraph.end();
	   i != e; ++i) {
	if (block_has_barrier(*i) &&
	    ProcessedBarriers.count(*i) == 0) {
	  // Subgraph to this barriers has still unprocessed barriers. Do
	  // not process this barrier yet (it will be done when coming
	  // from the path through the previous barrier).
	  return NULL;
	}
      }

      // Mark this barrier as processed.
      ProcessedBarriers.insert(bb);
      
      // Replicate subgraph from entry to the barrier for
      // all workitems.
      replicateWorkitemSubgraph(pre_subgraph, entry, bb->getSinglePredecessor());

      // Continue processing after the barrier.
      BasicBlockSet post_subgraph;
      bb = t->getSuccessor(0);
      BasicBlock *exit = findBarriersDFS(bb, bb, post_subgraph);
      replicateWorkitemSubgraph(post_subgraph, bb, exit);

      return NULL;
    }

  subgraph.insert(bb);

  if (t->getNumSuccessors() == 0)
    return t->getParent();

  // Find barriers in the successors (depth first).
  BasicBlock *r = NULL;
  BasicBlock *s;
  for (unsigned i = 0, e = t->getNumSuccessors(); i != e; ++i) {
    s = findBarriersDFS(t->getSuccessor(i), entry, subgraph);
    if (s != NULL) {
      assert((r == NULL || r == s) &&
	     "More than one function tail within same barrier path!\n");
      r = s;
    }
  }

  return r;
}

void
WorkitemReplication::replicateWorkitemSubgraph(BasicBlockSet subgraph,
					       BasicBlock *entry,
					       BasicBlock *exit)
{
  BasicBlockSet s;

  assert (entry != NULL && exit != NULL);

  IRBuilder<> builder(entry->getContext());

  for (int z = 0; z < LocalSize[2]; ++z) {
    for (int y = 0; y < LocalSize[1]; ++y) {
      for (int x = 0; x < LocalSize[0]; ++x) {
	
	if (x == (LocalSize[0] - 1) &&
	    y == (LocalSize[1] - 1) &&
	    z == (LocalSize[2] - 1)) {
	  builder.SetInsertPoint(entry, entry->front());
	  if (LocalX != NULL) {
	    builder.CreateStore(ConstantInt::get(IntegerType::
						 get(entry->getContext(),
						     32), x), LocalX);
	    
	  }
	  return;
	}

	int i = (z + 1) * (y + 1) * (x + 1) - 1;

	replicateBasicblocks(s, ReferenceMap[i], subgraph);
	purge_subgraph(s, subgraph,
		       cast<BasicBlock> (ReferenceMap[i][exit]));
	updateReferences(s, ReferenceMap[i]);
	
	ReferenceMap[i].erase(exit->getTerminator());
	BranchInst::Create(cast<BasicBlock>(ReferenceMap[i][entry]),
			   exit->getTerminator());
	exit->getTerminator()->eraseFromParent();

	builder.SetInsertPoint(entry, entry->front());
	if (LocalX != NULL) {
	  builder.CreateStore(ConstantInt::get(IntegerType::
					       get(entry->getContext(),
						   32), x), LocalX);
	  
	}
	if (x == 0) {
	  if (LocalY != NULL) {
	    builder.CreateStore(ConstantInt::get(IntegerType::
						 get(entry->getContext(),
						     32), y), LocalY);
	    
	  }
	  if (y == 0) {
	    if (LocalZ != NULL) {
	      builder.CreateStore(ConstantInt::get(IntegerType::
						   get(entry->getContext(),
						       32), z), LocalZ);
	    }
	  }
	}

	subgraph = s;
	entry = cast<BasicBlock>(ReferenceMap[i][entry]);
	exit = cast<BasicBlock>(ReferenceMap[i][exit]);

	s.clear();
      }
    }
  }
}

static void
purge_subgraph(std::set<BasicBlock *> &new_subgraph,
	       const std::set<BasicBlock *> &original_subgraph,
	       const BasicBlock *exit)
{
  for (std::set<BasicBlock *>::iterator i = new_subgraph.begin(),
	 e = new_subgraph.end();
       i != e; ++i) {
    if (*i == exit)
      continue;
    // Remove CFG edges going to basic blocks which
    // were not contained in original graph (impossible
    // branches).
    TerminatorInst *t = (*i)->getTerminator();
    for (unsigned u = 0; u < t->getNumSuccessors(); ++u) {
      if (original_subgraph.count(t->getSuccessor(u)) == 0) {
	BasicBlock *unreachable = BasicBlock::Create((*i)->getContext(),
						     "unreachable",
						     (*i)->getParent());
	new UnreachableInst((*i)->getContext(), unreachable);
	t->setSuccessor(u, unreachable);
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
      if (v->getName().equals(BARRIER_FUNCTION_NAME)) {
	assert((bb->size() == 2) &&
	       (bb->getSinglePredecessor() != NULL) &&
	       (bb->getTerminator()->getNumSuccessors() == 1) &&
	       ("Invalid barrier basicblock found!\n"));
	return true;
      }
    }
  }

  return false;
}

// Find subgraph between entry and exit basicblocks.
static bool
find_subgraph(BasicBlockSet &subgraph,
	      BasicBlock *entry,
	      BasicBlock *exit)
{
  if (entry == exit) {
    subgraph.insert(entry);
    return true;
  }

  bool found = false;
  const TerminatorInst *t = entry->getTerminator();
  for (unsigned i = 0, e = t->getNumSuccessors(); i != e; ++i)
    found |= find_subgraph(subgraph, t->getSuccessor(i), exit);

  if (found)
    subgraph.insert(entry);

  return found;
}

void
WorkitemReplication::replicateBasicblocks(BasicBlockSet &new_graph,
					  ValueValueMap &reference_map,
					  const BasicBlockSet &graph)
{
  for (std::set<BasicBlock *>::const_iterator i = graph.begin(),
	 e = graph.end();
       i != e; ++i) {
    BasicBlock *b = *i;
    BasicBlock *new_b = BasicBlock::Create(b->getContext(),
					   b->getName(),
					   b->getParent());
    
    reference_map[b] = new_b;
    new_graph.insert(new_b);

    for (BasicBlock::iterator i2 = b->begin(), e2 = b->end();
	 i2 != e2; ++i2) {
      if (isReplicable(i2)) {
	Instruction *i = i2->clone();
	reference_map.insert(std::make_pair(i2, i));
	new_b->getInstList().push_back(i);
      }
    }
  }
}

void
WorkitemReplication::updateReferences(const BasicBlockSet &graph,
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

bool
WorkitemReplication::isReplicable(const Instruction *i)
{
  if (const StoreInst *s = dyn_cast<StoreInst>(i)) {
    const Value *v = s->getPointerOperand();
    const GlobalVariable *gv = dyn_cast<GlobalVariable>(v);
    if (gv == LocalX || gv == LocalY || gv == LocalZ)
      return false;
  }

  return true;
}

static void
split_barriers(Function &F)
{
  std::set<Instruction *> SplitPoints;

  for (Function::iterator i = F.begin(), e = F.end();
       i != e; ++i) {
    BasicBlock *b = i;
    for (BasicBlock::iterator i = b->begin(), e = b->end();
	 i != e; ++i) {
      if (CallInst *c = dyn_cast<CallInst>(i)) {
	if (Function *f = c->getCalledFunction()) {
	  if (f->getName().equals(BARRIER_FUNCTION_NAME)) {
	    BasicBlock::iterator j = i;
	    SplitPoints.insert(j);
	    SplitPoints.insert(++j);
	  }
	}
      }
    }
  }

  for (std::set<Instruction *>::iterator i = SplitPoints.begin(),
	 e = SplitPoints.end();
       i != e; ++i) {
    BasicBlock *b = (*i)->getParent();
    b->splitBasicBlock(*i);
  }
}
