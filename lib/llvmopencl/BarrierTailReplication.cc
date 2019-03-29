// LLVM function pass to replicate barrier tails (successors to barriers).
//
// Copyright (c) 2011 Universidad Rey Juan Carlos and
//               2012-2019 Pekka Jääskeläinen
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
#include <algorithm>

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "pocl.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"

#include "BarrierTailReplication.h"
#include "Barrier.h"
#include "Workgroup.h"
#include "VariableUniformityAnalysis.h"

POP_COMPILER_DIAGS

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
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addPreserved<DominatorTreeWrapperPass>();
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addPreserved<LoopInfoWrapperPass>();

  AU.addPreserved<VariableUniformityAnalysis>();
}

bool
BarrierTailReplication::runOnFunction(Function &F)
{
  if (!Workgroup::isKernelToProcess(F))
    return false;
  
#ifdef DEBUG_BARRIER_REPL
  std::cerr << "### BTR on " << F.getName().str() << std::endl;
#endif

  DTP = &getAnalysis<DominatorTreeWrapperPass>();
  DT = &DTP->getDomTree();

  LI = &getAnalysis<LoopInfoWrapperPass>();

  bool changed = ProcessFunction(F);


  // In LLVM 7+, it is replaced by some assert(verify()) in LLVM itself
#ifdef LLVM_OLDER_THAN_7_0
  DT->verifyDomTree();
#endif

  LI->verifyAnalysis();
  /* The created tails might contain PHI nodes with operands 
     referring to the non-predecessor (split point) BB. 
     These must be cleaned to avoid breakage later on.
   */
  for (Function::iterator i = F.begin(), e = F.end();
       i != e; ++i)
    {
      llvm::BasicBlock *bb = &*i;
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

  if (block_has_barrier(bb)) {
#ifdef DEBUG_BARRIER_REPL
    std::cerr << "### block " << bb->getName().str() << " has barrier, RJS" << std::endl;
#endif
    BasicBlockSet processed_bbs_rjs;
    changed = ReplicateJoinedSubgraphs(bb, bb, processed_bbs_rjs);
  }

  auto t = bb->getTerminator();

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
                                                 BasicBlock *subgraph_entry,
                                                 BasicBlockSet &processed_bbs)
{
  bool changed = false;

  assert(DT->dominates(dominator, subgraph_entry));

  Function *f = dominator->getParent();

  auto t = subgraph_entry->getTerminator();
  for (int i = 0, e = t->getNumSuccessors(); i != e; ++i) {
    BasicBlock *b = t->getSuccessor(i);
#ifdef DEBUG_BARRIER_REPL
    std::cerr << "### traversing from " << subgraph_entry->getName().str() 
              << " to " << b->getName().str() << std::endl;
#endif

    // Check if we already handled this BB and all its branches.
    if (processed_bbs.count(b) != 0) 
      {
#ifdef DEBUG_BARRIER_REPL
        std::cerr << "### already processed " << std::endl;
#endif
        continue;
      }

    const bool isBackedge = DT->dominates(b, subgraph_entry);
    if (isBackedge) {
      // This is a loop backedge. Do not find subgraphs across
      // those.
#ifdef DEBUG_BARRIER_REPL
      std::cerr << "### a loop backedge, skipping" << std::endl;
#endif
      continue;
    }
    if (DT->dominates(dominator, b))
      {
#ifdef DEBUG_BARRIER_REPL
        std::cerr << "### " << dominator->getName().str() << " dominates "
                  << b->getName().str() << std::endl;
#endif
        changed |= ReplicateJoinedSubgraphs(dominator, b, processed_bbs);
      } 
    else           
      {
#ifdef DEBUG_BARRIER_REPL
        std::cerr << "### " << dominator->getName().str() << " does not dominate "
                  << b->getName().str() << " replicating " << std::endl;
#endif
        BasicBlock *replicated_subgraph_entry =
          ReplicateSubgraph(b, f);
        t->setSuccessor(i, replicated_subgraph_entry);
        changed = true;
      }

    if (changed)
      {
        // We have modified the function. Possibly created new loops.
        // Update analysis passes.
        DTP->runOnFunction(*f);

        LI->runOnFunction(*f);
      }
  }
  processed_bbs.insert(subgraph_entry);
  return changed;
}

// Removes phi elements for which there are no successors (anymore).
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
#ifdef DEBUG_BARRIER_REPL
              std::cerr << "now:" << std::endl;
              PN->dump();
#endif
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

  ValueToValueMapTy m;
  ReplicateBasicBlocks(v, m, subgraph, f);
  UpdateReferences(v, m);

  // Return entry block of replicated subgraph.
  return cast<BasicBlock>(m[entry]);
}


void
BarrierTailReplication::FindSubgraph(BasicBlockVector &subgraph,
                                     BasicBlock *entry)
{
  // The subgraph can have internal branches (join points)
  // avoid replicating these parts multiple times within the
  // same tail.
  if (std::count(subgraph.begin(), subgraph.end(), entry) > 0)
    return;

  subgraph.push_back(entry);

  auto t = entry->getTerminator();
  for (unsigned i = 0, e = t->getNumSuccessors(); i != e; ++i) {
    BasicBlock *successor = t->getSuccessor(i);
    const bool isBackedge = DT->dominates(successor, entry);
    if (isBackedge) continue;
    FindSubgraph(subgraph, successor);
  }
}


void
BarrierTailReplication::ReplicateBasicBlocks(BasicBlockVector &new_graph,
                                             ValueToValueMapTy &reference_map,
                                             BasicBlockVector &graph,
                                             Function *f)
{
#ifdef DEBUG_BARRIER_REPL
  std::cerr << "### ReplicateBasicBlocks: " << std::endl;
#endif
  for (BasicBlockVector::const_iterator i = graph.begin(),
         e = graph.end();
       i != e; ++i) {
    BasicBlock *b = *i;
    BasicBlock *new_b = BasicBlock::Create(b->getContext(),
					   b->getName() + ".btr",
					   f);
    reference_map.insert(std::make_pair(b, new_b));
    new_graph.push_back(new_b);

#ifdef DEBUG_BARRIER_REPL
    std::cerr << "Replicated BB: " << new_b->getName().str() << std::endl;
#endif

    for (BasicBlock::iterator i2 = b->begin(), e2 = b->end();
         i2 != e2; ++i2) {
      Instruction *i = i2->clone();
      reference_map.insert(std::make_pair(&*i2, i));
      new_b->getInstList().push_back(i);
    }

    // Add predicates to PHINodes of basic blocks the replicated
    // block jumps to (backedges).
    auto t = new_b->getTerminator();
    for (unsigned i = 0, e = t->getNumSuccessors(); i != e; ++i) {
      BasicBlock *successor = t->getSuccessor(i);
      if (std::count(graph.begin(), graph.end(), successor) == 0) {
        // Successor is not in the graph, possible backedge.
        for (BasicBlock::iterator i  = successor->begin(), e = successor->end();
             i != e; ++i) {
          PHINode *phi = dyn_cast<PHINode>(i);
          if (phi == NULL)
            break; // All PHINodes already checked.

          // Get value for original incoming edge and add new predicate.
          Value *v = phi->getIncomingValueForBlock(b);
          Value *new_v = reference_map.find(v) == reference_map.end() ?
            NULL : reference_map[v];

          if (new_v == NULL) {
            /* This case can happen at least when replicating a latch 
               block in a b-loop. The value produced might be from a common
               path before the replicated part. Then just use the original value.*/
            new_v = v;
#if 0
            std::cerr << "### could not find a replacement block for phi node ("
                      << b->getName().str() << ")" << std::endl;
            phi->dump();
            v->dump();
            f->viewCFG();
            assert (0);
#endif
          }
          phi->addIncoming(new_v, new_b);
        }
      }
    }
  }

#ifdef DEBUG_BARRIER_REPL
  std::cerr << std::endl;
#endif
}


void
BarrierTailReplication::UpdateReferences(const BasicBlockVector &graph,
                                         ValueToValueMapTy &reference_map)
{
  for (BasicBlockVector::const_iterator i = graph.begin(),
	 e = graph.end();
       i != e; ++i) {
    BasicBlock *b = *i;
    for (BasicBlock::iterator i2 = b->begin(), e2 = b->end();
         i2 != e2; ++i2) {
      Instruction *i = &*i2;
      RemapInstruction(i, reference_map,
                       RF_IgnoreMissingLocals | RF_NoModuleLevelChanges);
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
