// Implementation for VariableUniformityAnalysis function pass.
//
// Copyright (c) 2013-2019 Pekka Jääskeläinen
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
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Support/CommandLine.h"
POP_COMPILER_DIAGS

#include "Barrier.h"
#include "Kernel.h"
#include "LLVMUtils.h"
#include "VariableUniformityAnalysis.h"
#include "VariableUniformityAnalysisResult.hh"
#include "Workgroup.h"
#include "WorkitemHandler.h"

#include <iostream>
#include <map>
#include <sstream>

#include "pocl_llvm_api.h"

// #define DEBUG_UNIFORMITY_ANALYSIS

#ifdef DEBUG_UNIFORMITY_ANALYSIS
#include "DebugHelpers.h"
#endif

#define PASS_NAME "pocl-vua"
#define PASS_CLASS pocl::VariableUniformityAnalysis
#define PASS_DESC                                                              \
  "Analyses the variables of the function for uniformity (same value across "  \
  "WIs)."

namespace pocl {

using namespace llvm;

// Recursively mark the canonical induction variable PHI as uniform.
// If there's a canonical induction variable in loops, the variable
// update for each iteration should be uniform. Note: this does not yet
// imply all the work-items execute the loop same number of times!
void VariableUniformityAnalysisResult::markInductionVariables(Function &F,
                                                              llvm::Loop &L) {

  if (llvm::PHINode *inductionVar = L.getCanonicalInductionVariable()) {
#ifdef DEBUG_UNIFORMITY_ANALYSIS
    std::cerr << "### canonical induction variable, assuming uniform:";
    inductionVar->dump();
#endif
    setUniform(&F, inductionVar);
  }
  for (llvm::Loop *Subloop : L.getSubLoops()) {
    markInductionVariables(F, *Subloop);
  }
}

bool VariableUniformityAnalysisResult::runOnFunction(
    Function &F, llvm::LoopInfo &LI, llvm::PostDominatorTree &PDT) {

  if (!isKernelToProcess(F))
    return false;

#ifdef DEBUG_UNIFORMITY_ANALYSIS
  std::cerr << "### refreshing VUA" << std::endl;
  dumpCFG(F, F.getName().str() + ".vua.dot");
  F.dump();
#endif

  /* Do the actual analysis on-demand except for the basic block
     divergence analysis. */
  uniformityCache_[&F].clear();

  for (llvm::LoopInfo::iterator i = LI.begin(), e = LI.end(); i != e; ++i) {
    llvm::Loop *L = *i;
    markInductionVariables(F, *L);
  }

  setUniform(&F, &F.getEntryBlock());
  analyzeBBDivergence(&F, &F.getEntryBlock(), &F.getEntryBlock(), PDT);
  return false;
}

/**
 * Returns true in case the value should be privatized, e.g., a copy
 * should be created for each parallel work-item.
 *
 * This is not the same as !isUniform() because of some of the allocas.
 * Specifically, the loop iteration variables are sometimes uniform, 
 * that is, each work item sees the same induction variable value at every iteration, 
 * but the variables should be still replicated to avoid multiple increments
 * of the same induction variable by each work-item.
 */
bool VariableUniformityAnalysisResult::shouldBePrivatized(llvm::Function *F,
                                                          llvm::Value *Val) {
  if (!isUniform(F, Val)) return true;
  
  /* Check if the value is stored in stack (is an alloca or writes to an alloca). */
  /* It should be enough to context save the initial alloca and the stores to
     make sure each work-item gets their own stack slot and they are updated.
     How the value (based on which of those allocas) is computed does not matter as
     we are deadling with uniform computation. */

  if (isa<AllocaInst>(Val)) return true;

  if (isa<StoreInst>(Val) &&
      isa<AllocaInst>(dyn_cast<StoreInst>(Val)->getPointerOperand())) return true;
  return false;
}

/**  
 * BB divergence analysis.
 *
 * Define:
 * Uniform BB. A basic block which is known to be executed by all or none
 * of the work-items, that is, a BB where it's known safe to add a barrier.
 *
 * Divergent/varying BB. A basic block where work-items *might* diverge.
 * That is, it cannot be proven that all work-items execute the BB.
 *
 * Propagate the information from the entry downwards (breadth first). 
 * This avoids infinite recursion with loop back edges and enables
 * to keep book of the "last seen" uniform BB.
 *
 * The conditions to mark a BB 'uniform':
 *
 * a) the function entry, or
 * b) BBs that post-dominate at least one uniform BB (try the previously 
 *    found one), or
 * c) BBs that are branched to directly from a uniform BB using a uniform branch.
 *    Note: This assumes the CFG is well-formed in a way that there cannot be a divergent
 *    branch to the same BB in that case.
 *
 * Otherwise, assume divergent (might not be *proven* to be one!).
 *
 */
void VariableUniformityAnalysisResult::analyzeBBDivergence(
    llvm::Function *F, llvm::BasicBlock *BB,
    llvm::BasicBlock *PreviousUniformBB, llvm::PostDominatorTree &PDT) {

#ifdef DEBUG_UNIFORMITY_ANALYSIS
  std::cerr << "### Analyzing BB divergence (bb=" << bb->getName().str()
            << ", prevUniform=" << previousUniformBB->getName().str() << ")"
            << std::endl;
#endif

  auto Term = PreviousUniformBB->getTerminator();
  if (Term == NULL) {
    // this is most likely a function with a single basic block, the entry
    // node, which ends with a ret
    return;
  }

  llvm::BranchInst *BrInst = dyn_cast<llvm::BranchInst>(Term);
  llvm::SwitchInst *SwInst = dyn_cast<llvm::SwitchInst>(Term);

  if (BrInst == nullptr && SwInst == nullptr) {
    // Can only handle branches and switches for now.
    return;
  }

  // The BBs that were found uniform.
  std::vector<llvm::BasicBlock *> FoundUniforms;

  // Condition c)
  if ((BrInst && (!BrInst->isConditional() ||
                  isUniform(F, BrInst->getCondition()))) ||
      (SwInst && isUniform(F, SwInst->getCondition()))) {
    // This is a branch with a uniform condition, propagate the uniformity
    // to the BB of interest.
    for (unsigned suc = 0, end = Term->getNumSuccessors(); suc < end; ++suc) {
      llvm::BasicBlock *Successor = Term->getSuccessor(suc);
      // TODO: should we check that there are no divergent entries to this
      // BB even though if the currently checked condition is uniform?
      setUniform(F, Successor, true);
      FoundUniforms.push_back(Successor);
    }
  }

  // Condition b)
  if (FoundUniforms.size() == 0) {
    if (PDT.dominates(BB, PreviousUniformBB)) {
      setUniform(F, BB, true);
      FoundUniforms.push_back(BB);
    }
  }

  /* Assume diverging. */
  if (!isUniformityAnalyzed(F, BB))
    setUniform(F, BB, false);

  for (auto UniformBB : FoundUniforms) {

    // Propagate the Uniform BB data downwards.
    auto NextTerm = UniformBB->getTerminator();

    for (unsigned Suc = 0, End = NextTerm->getNumSuccessors(); Suc < End;
         ++Suc) {
      llvm::BasicBlock *NextBB = NextTerm->getSuccessor(Suc);
      if (!isUniformityAnalyzed(F, NextBB)) {
        analyzeBBDivergence(F, NextBB, UniformBB, PDT);
      }
    }
  }
}

bool VariableUniformityAnalysisResult::isUniformityAnalyzed(
    llvm::Function *F, llvm::Value *V) const {
  UniformityIndex &Cache = uniformityCache_[F];
  UniformityIndex::const_iterator I = Cache.find(V);
  if (I != Cache.end()) {
    return true;
  }
  return false;
}

/**
 * Simple uniformity analysis that recursively analyses all the
 * operands affecting the value.
 *
 * Known uniform Values that act as "leafs" in the recursive uniformity
 * check logic:
 * a) kernel arguments
 * b) constants
 * c) OpenCL C identifiers that are constant for all work-items in a work-group
 * 
 */
bool VariableUniformityAnalysisResult::isUniform(llvm::Function *F,
                                                 llvm::Value *V) {

  UniformityIndex &Cache = uniformityCache_[F];
  UniformityIndex::const_iterator I = Cache.find(V);
  if (I != Cache.end()) {
    return (*I).second;
  }

  if (llvm::BasicBlock *BB = dyn_cast<llvm::BasicBlock>(V)) {
    if (BB == &F->getEntryBlock()) {
      setUniform(F, V, true);
      return true;
    }
  }

  if (isa<llvm::Argument>(V)) {
    setUniform(F, V, true);
    return true;
  }

  if (isa<llvm::ConstantInt>(V)) {
    setUniform(F, V, true);
    return true;
  }

  if (isa<llvm::AllocaInst>(V)) {
    /* Allocas might or might not be divergent. These are produced 
       from work-item private arrays or the PHIsToAllocas. It depends
       what is written to them whether they are really divergent. 
       
       We need to figure out if any of the stores to the alloca contain 
       work-item id dependent data. Take a white listing approach that
       detects the ex-phi allocas of loop iteration variables of non-diverging
       loops. 

       Currently the following case is white listed:
       a) are scalars, and
       b) are accessed only with load and stores (e.g. address not taken) from
          uniform basic blocks, and
       c) the stored data is uniform

       Because alloca data can be modified in loops and thus be dependent on
       itself, we need a bit involved mechanism to handle it. First create 
       a copy of the uniformity cache, then assume the alloca itself is uniform, 
       then check if all the stores to the alloca contain uniform data. If
       our initial assumption was wrong, restore the cache from the backup.
    */
    UniformityCache backupCache(uniformityCache_);
    setUniform(F, V);

    bool isUniformAlloca = true;
    llvm::Instruction *instruction = dyn_cast<llvm::AllocaInst>(V);
    for (Instruction::use_iterator ui = instruction->use_begin(),
           ue = instruction->use_end();
         ui != ue; ++ui) {
      llvm::Instruction *user = cast<Instruction>(ui->getUser());
      if (user == NULL) continue;
      
      llvm::StoreInst *store = dyn_cast<llvm::StoreInst>(user);
      if (store) {
        if (!isUniform(F, store->getValueOperand()) ||
            !isUniform(F, store->getParent())) {
          if (!isUniform(F, store->getParent())) {
#ifdef DEBUG_UNIFORMITY_ANALYSIS
            std::cerr << "### alloca was written in a non-uniform BB" << std::endl;
            store->getParent()->dump();
            /* TODO: This is a problematic chicken-egg situation because the 
               BB uniformity check ends up analyzing allocas in phi-removed code:
               the loop constructs refer to these allocas and at that point we
               do not yet know if the BB itself is uniform. This leads to not
               being able to detect loop iteration variables as uniform. */
#endif          
          }
          isUniformAlloca = false;
          break;
        }
      } else if (isa<llvm::LoadInst>(user) || isa<llvm::BitCastInst>(user)) {
      } else if (isa<llvm::CallInst>(user)) {
        CallInst *CallInstr = dyn_cast<CallInst>(user);
        Function *Callee = CallInstr->getCalledFunction();
        if (Callee != nullptr &&
            (Callee->getName().startswith("llvm.lifetime.end") ||
             Callee->getName().startswith("llvm.lifetime.start"))) {
#ifdef DEBUG_UNIFORMITY_ANALYSIS
          std::cerr << "### alloca is used by llvm.lifetime" << std::endl;
          user->dump();
#endif
        } else {
#ifdef DEBUG_UNIFORMITY_ANALYSIS
          std::cerr << "### alloca has a suspicious user" << std::endl;
          user->dump();
#endif
          isUniformAlloca = false;
          break;
        }
      } else {
#ifdef DEBUG_UNIFORMITY_ANALYSIS
        std::cerr << "### alloca has a suspicious user" << std::endl;
        user->dump();
#endif
        isUniformAlloca = false;
        break;
      }
    }

    if (!isUniformAlloca) {
      // restore the old uniform data as our guess was wrong
      uniformityCache_ = backupCache;
    }
    setUniform(F, V, isUniformAlloca);
    
    return isUniformAlloca;
  }

  /* TODO: global memory loads are uniform in case they are accessing
     the higher scope ids (group_id_?). */
  if (isa<llvm::LoadInst>(V)) {
    llvm::LoadInst *load = dyn_cast<llvm::LoadInst>(V);
    llvm::Value *pointer = load->getPointerOperand();
    llvm::Module *M = load->getParent()->getParent()->getParent();

    if (pointer == M->getGlobalVariable("_group_id_x") ||
        pointer == M->getGlobalVariable("_group_id_y") ||
        pointer == M->getGlobalVariable("_group_id_z") ||
        pointer == M->getGlobalVariable("_work_dim") ||
        pointer == M->getGlobalVariable("_num_groups_x") ||
        pointer == M->getGlobalVariable("_num_groups_y") ||
        pointer == M->getGlobalVariable("_num_groups_z") ||
        pointer == M->getGlobalVariable("_global_offset_x") ||
        pointer == M->getGlobalVariable("_global_offset_y") ||
        pointer == M->getGlobalVariable("_global_offset_z") ||
        pointer == M->getGlobalVariable("_local_size_x") ||
        pointer == M->getGlobalVariable("_local_size_y") ||
        pointer == M->getGlobalVariable("_local_size_z") ||
        pointer == M->getGlobalVariable(PoclGVarBufferName)) {

      setUniform(F, V, true);
      return true;
    }
  }

  if (isa<llvm::PHINode>(V)) {
    /* TODO: PHINodes need control flow analysis:
       even if the values are uniform, the selected
       value depends on the preceeding basic block which
       might depend on the ID. Assume they are not uniform
       for now in general and treat the loop iteration 
       variable as a special case (set externally from a LoopPass). 

       TODO: PHINodes can depend (indirectly or directly) on itself in loops 
       so it would need infinite recursion checking.
    */
    setUniform(F, V, false);
    return false;
  }

  llvm::Instruction *instr = dyn_cast<llvm::Instruction>(V);
  if (instr == NULL) {
    setUniform(F, V, false);
    return false;
  }

  // Atomic operations might look like uniform if only considering the operands
  // (access a global memory location of which ordering by default is not
  // constrained), but their semantics have ordering: Each work-item should get
  // their own value from that memory location.
  if (instr->isAtomic()) {
      setUniform(F, V, false);
      return false;
  }

  // not computed previously, scan all operands of the instruction
  // and figure out their uniformity recursively
  for (unsigned opr = 0; opr < instr->getNumOperands(); ++opr) {    
    llvm::Value *operand = instr->getOperand(opr);
    if (!isUniform(F, operand)) {
      setUniform(F, V, false);
      return false;
    }
  }
  setUniform(F, V, true);
  return true;
}

void VariableUniformityAnalysisResult::setUniform(llvm::Function *F,
                                                  llvm::Value *V,
                                                  bool isUniform) {

  UniformityIndex &Cache = uniformityCache_[F];
  Cache[V] = isUniform;

#ifdef DEBUG_UNIFORMITY_ANALYSIS
  std::cerr << "### ";
  if (isUniform) 
    std::cerr << "uniform ";
  else
    std::cerr << "varying ";

  if (isa<llvm::BasicBlock>(V)) {
    std::cerr << "BB: " << V->getName().str() << std::endl;
  } else {
    V->dump();
  }
#endif
}

bool VariableUniformityAnalysisResult::doFinalization(llvm::Module & /*M*/) {
  uniformityCache_.clear();
  return true;
}

#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
char VariableUniformityAnalysis::ID = 0;

bool VariableUniformityAnalysis::runOnFunction(Function &F) {
  pImpl = new VariableUniformityAnalysisResult;
  llvm::LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  llvm::PostDominatorTree &PDT =
      getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();
  return pImpl->runOnFunction(F, LI, PDT);
}

void VariableUniformityAnalysis::getAnalysisUsage(
    llvm::AnalysisUsage &AU) const {
  AU.addRequired<PostDominatorTreeWrapperPass>();
  AU.addPreserved<PostDominatorTreeWrapperPass>();

  AU.addRequired<LoopInfoWrapperPass>();
  AU.addPreserved<LoopInfoWrapperPass>();
  // required by LoopInfo:
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addPreserved<DominatorTreeWrapperPass>();
}

VariableUniformityAnalysis::~VariableUniformityAnalysis() {
  delete pImpl;
  pImpl = nullptr;
}

REGISTER_OLD_FANALYSIS(PASS_NAME, PASS_CLASS, PASS_DESC);

#else

llvm::AnalysisKey VariableUniformityAnalysis::Key;

VariableUniformityAnalysis::Result
VariableUniformityAnalysis::run(llvm::Function &F,
                                llvm::FunctionAnalysisManager &AM) {
  llvm::LoopInfo &LI = AM.getResult<llvm::LoopAnalysis>(F);
  llvm::PostDominatorTree &PDT =
      AM.getResult<llvm::PostDominatorTreeAnalysis>(F);

  VariableUniformityAnalysisResult Res;
  Res.runOnFunction(F, LI, PDT);
  return Res;
}

bool VariableUniformityAnalysisResult::invalidate(
    llvm::Function &F, const llvm::PreservedAnalyses PA,
    llvm::AnalysisManager<llvm::Function>::Invalidator &Inv) {
  // TODO: this is required by the LoopPasses that use this analysis; however,
  // it's most likely incorrect. We should convert LoopPasses to FunctionPasses
  // and properly invalidate VUA
  return false;
#if 0
  auto PAC = PA.getChecker<VariableUniformityAnalysis>();
  bool Preserved = (PAC.preserved() ||
    PAC.preservedSet<AllAnalysesOn<Function>>());
  if (!Preserved) {
    uniformityCache_.erase(&F);
  }
  return !Preserved;
#endif
}

REGISTER_NEW_FANALYSIS(PASS_NAME, PASS_CLASS, PASS_DESC);

#endif

} // namespace pocl
