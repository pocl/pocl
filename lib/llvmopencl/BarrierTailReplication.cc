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

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/Transforms/Utils/LoopSimplify.h>

#include "Barrier.h"
#include "BarrierTailReplication.h"
#include "LLVMUtils.h"
#include "VariableUniformityAnalysis.h"
#include "Workgroup.h"
#include "WorkitemHandlerChooser.h"

POP_COMPILER_DIAGS

//#define DEBUG_BARRIER_REPL

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <vector>

#define PASS_NAME "barriertails"
#define PASS_CLASS pocl::BarrierTailReplication
#define PASS_DESC "Barrier tail replication pass"

namespace pocl {

using namespace llvm;

class BarrierTailReplicationImpl {

public:
  bool runOnFunction(llvm::Function &F);
  BarrierTailReplicationImpl(llvm::DominatorTree &DT, llvm::LoopInfo &LI)
      : DT(DT), LI(LI){};

private:
  typedef std::set<llvm::BasicBlock *> BasicBlockSet;
  typedef std::vector<llvm::BasicBlock *> BasicBlockVector;
  typedef std::map<llvm::Value *, llvm::Value *> ValueValueMap;

  llvm::DominatorTree &DT;
  llvm::LoopInfo &LI;

  bool ProcessFunction(llvm::Function &F);
  bool FindBarriersDFS(llvm::BasicBlock *BB, BasicBlockSet &ProcessedBBs);
  bool ReplicateJoinedSubgraphs(llvm::BasicBlock *Dominator,
                                llvm::BasicBlock *SubgraphEntry,
                                BasicBlockSet &ProcessedBBs);

  llvm::BasicBlock *ReplicateSubgraph(llvm::BasicBlock *Entry,
                                      llvm::Function *F);
  void FindSubgraph(BasicBlockVector &Subgraph, llvm::BasicBlock *Entry);
  void ReplicateBasicBlocks(BasicBlockVector &NewGraph,
                            llvm::ValueToValueMapTy &ReferenceMap,
                            BasicBlockVector &Graph, llvm::Function *F);
  void UpdateReferences(const BasicBlockVector &Graph,
                        llvm::ValueToValueMapTy &ReferenceMap);

  bool CleanupPHIs(llvm::BasicBlock *BB);
};

bool BarrierTailReplicationImpl::runOnFunction(Function &F) {
#ifdef DEBUG_BARRIER_REPL
  std::cerr << "### BTR on " << F.getName().str() << std::endl;
#endif

  bool changed = ProcessFunction(F);

  LI.verify(DT);
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

bool BarrierTailReplicationImpl::ProcessFunction(Function &F) {
  BasicBlockSet processed_bbs;

  return FindBarriersDFS(&F.getEntryBlock(), processed_bbs);
}

static bool blockHasBarrier(const BasicBlock *BB) {
  for (BasicBlock::const_iterator i = BB->begin(), e = BB->end(); i != e; ++i) {
    if (isa<Barrier>(i))
      return true;
  }

  return false;
}

// Recursively (depht-first) look for barriers in all possible
// execution paths starting on entry, replicating the barrier
// successors to ensure there is a separate function exit BB
// for each combination of traversed barriers. The set
// processed_bbs stores the
bool BarrierTailReplicationImpl::FindBarriersDFS(BasicBlock *BB,
                                                 BasicBlockSet &ProcessedBBs) {
  bool changed = false;

  // Check if we already visited this BB (to avoid
  // infinite recursion in case of unbarriered loops).
  if (ProcessedBBs.count(BB) != 0)
    return changed;

  ProcessedBBs.insert(BB);

  if (blockHasBarrier(BB)) {
#ifdef DEBUG_BARRIER_REPL
    std::cerr << "### block " << bb->getName().str() << " has barrier, RJS" << std::endl;
#endif
    BasicBlockSet processed_bbs_rjs;
    changed = ReplicateJoinedSubgraphs(BB, BB, processed_bbs_rjs);
  }

  auto t = BB->getTerminator();

  // Find barriers in the successors (depth first).
  for (unsigned i = 0, e = t->getNumSuccessors(); i != e; ++i)
    changed |= FindBarriersDFS(t->getSuccessor(i), ProcessedBBs);

  return changed;
}

// Only replicate those parts of the subgraph that are not
// dominated by a (barrier) basic block, to avoid excesive
// (and confusing) code replication.
bool BarrierTailReplicationImpl::ReplicateJoinedSubgraphs(BasicBlock *Dominator, BasicBlock *SubgraphEntry,
    BasicBlockSet &ProcessedBBs) {
  bool changed = false;

  assert(DT.dominates(Dominator, SubgraphEntry));

  Function *f = Dominator->getParent();

  auto t = SubgraphEntry->getTerminator();
  for (int i = 0, e = t->getNumSuccessors(); i != e; ++i) {
    BasicBlock *b = t->getSuccessor(i);
#ifdef DEBUG_BARRIER_REPL
    std::cerr << "### traversing from " << subgraph_entry->getName().str() 
              << " to " << b->getName().str() << std::endl;
#endif

    // Check if we already handled this BB and all its branches.
    if (ProcessedBBs.count(b) != 0)
      {
#ifdef DEBUG_BARRIER_REPL
        std::cerr << "### already processed " << std::endl;
#endif
        continue;
      }

      const bool isBackedge = DT.dominates(b, SubgraphEntry);
      if (isBackedge) {
        // This is a loop backedge. Do not find subgraphs across
        // those.
#ifdef DEBUG_BARRIER_REPL
      std::cerr << "### a loop backedge, skipping" << std::endl;
#endif
      continue;
    }
    if (DT.dominates(Dominator, b)) {
#ifdef DEBUG_BARRIER_REPL
        std::cerr << "### " << dominator->getName().str() << " dominates "
                  << b->getName().str() << std::endl;
#endif
        changed |= ReplicateJoinedSubgraphs(Dominator, b, ProcessedBBs);
    } else {
#ifdef DEBUG_BARRIER_REPL
        std::cerr << "### " << dominator->getName().str() << " does not dominate "
                  << b->getName().str() << " replicating " << std::endl;
#endif
        BasicBlock *replicated_subgraph_entry =
          ReplicateSubgraph(b, f);
        t->setSuccessor(i, replicated_subgraph_entry);
        changed = true;
    }

    if (changed) {
      // We have modified the function. Possibly created new loops.
      // Update analysis passes.
#if LLVM_VERSION_MAJOR < 11
      DT.releaseMemory();
#else
      DT.reset();
#endif
      DT.recalculate(*f);
      LI.releaseMemory();
      LI.analyze(DT);
    }
  }
  ProcessedBBs.insert(SubgraphEntry);
  return changed;
}

// Removes phi elements for which there are no successors (anymore).
bool BarrierTailReplicationImpl::CleanupPHIs(llvm::BasicBlock *BB) {

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

BasicBlock *BarrierTailReplicationImpl::ReplicateSubgraph(BasicBlock *Entry,
                                                          Function *F) {
  // Find all basic blocks to replicate.
  BasicBlockVector Subgraph;
  FindSubgraph(Subgraph, Entry);

  // Replicate subgraph maintaining control flow.
  BasicBlockVector V;

  ValueToValueMapTy VVM;
  ReplicateBasicBlocks(V, VVM, Subgraph, F);
  UpdateReferences(V, VVM);

  // Return entry block of replicated subgraph.
  return cast<BasicBlock>(VVM[Entry]);
}

void BarrierTailReplicationImpl::FindSubgraph(BasicBlockVector &Subgraph,
                                              BasicBlock *Entry) {
  // The subgraph can have internal branches (join points)
  // avoid replicating these parts multiple times within the
  // same tail.
  if (std::count(Subgraph.begin(), Subgraph.end(), Entry) > 0)
    return;

  Subgraph.push_back(Entry);

  auto Tntor = Entry->getTerminator();
  for (unsigned I = 0, E = Tntor->getNumSuccessors(); I != E; ++I) {
    BasicBlock *successor = Tntor->getSuccessor(I);
    const bool isBackedge = DT.dominates(successor, Entry);
    if (isBackedge) continue;
    FindSubgraph(Subgraph, successor);
  }
}

void BarrierTailReplicationImpl::ReplicateBasicBlocks(BasicBlockVector &NewGraph, ValueToValueMapTy &ReferenceMap,
    BasicBlockVector &Graph, Function *F) {
#ifdef DEBUG_BARRIER_REPL
  std::cerr << "### ReplicateBasicBlocks: " << std::endl;
#endif
  for (BasicBlockVector::const_iterator I = Graph.begin(),
         E = Graph.end();
       I != E; ++I) {
    BasicBlock *BB = *I;
    BasicBlock *NewBB = BasicBlock::Create(BB->getContext(),
             BB->getName() + ".btr",
             F);
    ReferenceMap.insert(std::make_pair(BB, NewBB));
    NewGraph.push_back(NewBB);

#ifdef DEBUG_BARRIER_REPL
    std::cerr << "Replicated BB: " << new_b->getName().str() << std::endl;
#endif

    for (BasicBlock::iterator I2 = BB->begin(), E2 = BB->end();
         I2 != E2; ++I2) {
      Instruction *Inst = I2->clone();
      ReferenceMap.insert(std::make_pair(&*I2, Inst));
#if LLVM_MAJOR < 16
      NewBB->getInstList().push_back(Inst);
#else
      Inst->insertInto(NewBB, NewBB->end());
#endif
    }

    // Add predicates to PHINodes of basic blocks the replicated
    // block jumps to (backedges).
    auto Tntor = NewBB->getTerminator();
    for (unsigned I = 0, E = Tntor->getNumSuccessors(); I != E; ++I) {
      BasicBlock *successor = Tntor->getSuccessor(I);
      if (std::count(Graph.begin(), Graph.end(), successor) == 0) {
        // Successor is not in the graph, possible backedge.
        for (BasicBlock::iterator BBI  = successor->begin(), BBE = successor->end();
             BBI != BBE; ++BBI) {
          PHINode *Phi = dyn_cast<PHINode>(BBI);
          if (Phi == NULL)
            break; // All PHINodes already checked.

          // Get value for original incoming edge and add new predicate.
          Value *OldV = Phi->getIncomingValueForBlock(BB);
          Value *NewV = ReferenceMap.find(OldV) == ReferenceMap.end() ?
            NULL : ReferenceMap[OldV];

          if (NewV == NULL) {
            /* This case can happen at least when replicating a latch 
               block in a b-loop. The value produced might be from a common
               path before the replicated part. Then just use the original value.*/
            NewV = OldV;
#if 0
            std::cerr << "### could not find a replacement block for phi node ("
                      << BB->getName().str() << ")" << std::endl;
            Phi->dump();
            OldV->dump();
            F->viewCFG();
            assert (0);
#endif
          }
          Phi->addIncoming(NewV, NewBB);
        }
      }
    }
  }

#ifdef DEBUG_BARRIER_REPL
  std::cerr << std::endl;
#endif
}

void BarrierTailReplicationImpl::UpdateReferences(
    const BasicBlockVector &Graph, ValueToValueMapTy &ReferenceMap) {
  for (BasicBlockVector::const_iterator BBVI = Graph.begin(),
   BBVE = Graph.end();
       BBVI != BBVE; ++BBVI) {
    BasicBlock *BB = *BBVI;
    for (BasicBlock::iterator BBI = BB->begin(), BBE = BB->end();
         BBI != BBE; ++BBI) {
      Instruction *Inst = &*BBI;
      RemapInstruction(Inst, ReferenceMap,
                       RF_IgnoreMissingLocals | RF_NoModuleLevelChanges);
    }
  }
}

#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
char BarrierTailReplication::ID = 0;

bool BarrierTailReplication::runOnFunction(Function &F) {
  if (!isKernelToProcess(F))
    return false;
  auto WIH = getAnalysis<WorkitemHandlerChooser>().chosenHandler();
  if (WIH == WorkitemHandlerType::CBS)
    return false;

  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

  BarrierTailReplicationImpl BTRI(DT, LI);

  return BTRI.runOnFunction(F);
}

void BarrierTailReplication::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addPreserved<DominatorTreeWrapperPass>();
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addPreserved<LoopInfoWrapperPass>();

  AU.addPreserved<VariableUniformityAnalysis>();
  AU.addRequired<WorkitemHandlerChooser>();
  AU.addPreserved<WorkitemHandlerChooser>();
}

} // namespace pocl

REGISTER_OLD_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

#else

llvm::PreservedAnalyses
BarrierTailReplication::run(llvm::Function &F,
                            llvm::FunctionAnalysisManager &FAM) {
  if (!isKernelToProcess(F))
    return PreservedAnalyses::all();

  WorkitemHandlerType WIH = FAM.getResult<WorkitemHandlerChooser>(F).WIH;
  if (WIH == WorkitemHandlerType::CBS)
    return PreservedAnalyses::all();

  llvm::DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  llvm::LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);

  BarrierTailReplicationImpl BTRI(DT, LI);

  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<VariableUniformityAnalysis>();
  PAChanged.preserve<WorkitemHandlerChooser>();
  PAChanged.preserve<LoopAnalysis>();
  PAChanged.preserve<DominatorTreeAnalysis>();

  return BTRI.runOnFunction(F) ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl

#endif
