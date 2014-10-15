// Implementation for VariableUniformityAnalysis function pass.
// 
// Copyright (c) 2013-2014 Pekka Jääskeläinen / Tampere University of Technology
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
#include <sstream>
#include <iostream>

#ifdef LLVM_3_2
#include "llvm/Metadata.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/ValueSymbolTable.h"
#include "llvm/DataLayout.h"
#else
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/IR/DataLayout.h"
#endif
#include "llvm/Support/CommandLine.h"
#include "llvm/Analysis/PostDominators.h"

#include "WorkitemHandler.h"
#include "Kernel.h"
#include "VariableUniformityAnalysis.h"
#include "Barrier.h"

//#define DEBUG_UNIFORMITY_ANALYSIS

namespace pocl {

char VariableUniformityAnalysis::ID = 0;

using namespace llvm;

static
RegisterPass<VariableUniformityAnalysis> X(
    "uniformity", 
    "Analyses the variables of the function for uniformity (same value across WIs).",
    false, false);

VariableUniformityAnalysis::VariableUniformityAnalysis() : FunctionPass(ID) {
}


void
VariableUniformityAnalysis::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<PostDominatorTree>();
  AU.addPreserved<PostDominatorTree>();
  AU.addRequired<LoopInfo>();
  AU.addPreserved<LoopInfo>();
  // required by LoopInfo:
#if (defined LLVM_3_2 or defined LLVM_3_3 or defined LLVM_3_4)
  AU.addRequired<DominatorTree>();
  AU.addPreserved<DominatorTree>();
#else
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addPreserved<DominatorTreeWrapperPass>();
#endif

#ifdef LLVM_3_1
  AU.addRequired<TargetData>();
  AU.addPreserved<TargetData>();
#elif (defined LLVM_3_2 or defined LLVM_3_3 or defined LLVM_3_4)
  AU.addRequired<DataLayout>();
  AU.addPreserved<DataLayout>();
#else
  AU.addRequired<DataLayoutPass>();
  AU.addPreserved<DataLayoutPass>();
#endif
}

bool
VariableUniformityAnalysis::runOnFunction(Function &F) {

#ifdef DEBUG_UNIFORMITY_ANALYSIS
  std::cerr << "### refreshing VUA" << std::endl;
#endif  

  /* Do the actual analysis on-demand except for the basic block 
     divergence analysis. */
  uniformityCache_[&F].clear();  

  /* Mark the canonical induction variable PHI as uniform. 
     If there's a canonical induction variable in loops, the variable
     update for each iteration should be uniform. Note: this does not yet imply
     all the work-items execute the loop same number of times! */
  llvm::LoopInfo &LI = getAnalysis<LoopInfo>();
  for (llvm::LoopInfo::iterator i = LI.begin(), e = LI.end(); i != e; ++i) {
    llvm::Loop *L = *i;
    if (llvm::PHINode *inductionVar = L->getCanonicalInductionVariable()) {
#ifdef DEBUG_UNIFORMITY_ANALYSIS
      std::cerr << "### canonical induction variable, assuming uniform:";
      inductionVar->dump();
#endif
      setUniform(&F, inductionVar);
    }    
  }

  setUniform(&F, &F.getEntryBlock());
  analyzeBBDivergence(&F, &F.getEntryBlock(), &F.getEntryBlock());
  //  F.viewCFG();
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
bool
VariableUniformityAnalysis::shouldBePrivatized
(llvm::Function *f, llvm::Value *val) {
  if (!isUniform(f, val)) return true;
  
  /* Check if the value is stored in stack (is an alloca or writes to an alloca). */
  /* It should be enough to context save the initial alloca and the stores to
     make sure each work-item gets their own stack slot and they are updated.
     How the value (based on which of those allocas) is computed does not matter as
     we are deadling with uniform computation. */

  if (isa<AllocaInst>(val)) return true;

  if (isa<StoreInst>(val) && 
      isa<AllocaInst>(dyn_cast<StoreInst>(val)->getPointerOperand())) return true;
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
 *
 * Otherwise, assume divergent (might not be *proven* to be one!).
 * 
 */
void
VariableUniformityAnalysis::analyzeBBDivergence
(llvm::Function *f, llvm::BasicBlock *bb, llvm::BasicBlock *previousUniformBB) {

#ifdef DEBUG_UNIFORMITY_ANALYSIS
  std::cerr << "### Analyzing BB divergence (bb=" << bb->getName().str() 
            << ", prevUniform=" << previousUniformBB->getName().str() << ")" 
            << std::endl;
#endif
 

  llvm::BasicBlock *newPreviousUniformBB = previousUniformBB;

  llvm::BranchInst *br = 
    dyn_cast<llvm::BranchInst>(previousUniformBB->getTerminator());  

  if (br == NULL) {
    // this is most likely a function with a single basic block, the entry node, which
    // ends with a ret
    return;
  }

  // Condition c)
  if ((!br->isConditional() || isUniform(f, br->getCondition()))) {
    for (unsigned suc = 0, end = br->getNumSuccessors(); suc < end; ++suc) {
      if (br->getSuccessor(suc) == bb) {
        setUniform(f, bb, true);
        newPreviousUniformBB = bb;
        break;
      }
    }
  } 

  // Condition b)
  if (newPreviousUniformBB != bb) {
    llvm::PostDominatorTree *PDT = &getAnalysis<PostDominatorTree>();
    if (PDT->dominates(bb, previousUniformBB)) {
      setUniform(f, bb, true);
      newPreviousUniformBB = bb;
    }
  } 

  /* Assume diverging. */
  if (!isUniformityAnalyzed(f, bb))
    setUniform(f, bb, false);

  llvm::BranchInst *nextbr = dyn_cast<llvm::BranchInst>(bb->getTerminator());  

  if (nextbr == NULL) return; /* ret */

  /* Propagate the data downward. */
  for (unsigned suc = 0, end = nextbr->getNumSuccessors(); suc < end; ++suc) {
    llvm::BasicBlock *nextbb = nextbr->getSuccessor(suc);
    if (!isUniformityAnalyzed(f, nextbb)) {
      analyzeBBDivergence(f, nextbb, newPreviousUniformBB);
    }
  }
}

bool
VariableUniformityAnalysis::isUniformityAnalyzed(llvm::Function *f, llvm::Value *v) const {
  UniformityIndex &cache = uniformityCache_[f];
  UniformityIndex::const_iterator i = cache.find(v);
  if (i != cache.end()) {
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
bool 
VariableUniformityAnalysis::isUniform(llvm::Function *f, llvm::Value* v) {

  UniformityIndex &cache = uniformityCache_[f];
  UniformityIndex::const_iterator i = cache.find(v);
  if (i != cache.end()) {
    return (*i).second;
  }

  if (llvm::BasicBlock *bb = dyn_cast<llvm::BasicBlock>(v)) {
    if (bb == &f->getEntryBlock()) {
      setUniform(f, v, true);
      return true;
    }
  }

  if (isa<llvm::Argument>(v)) {
    setUniform(f, v, true);
    return true;
  }

  if (isa<llvm::ConstantInt>(v)) {
    setUniform(f, v, true);
    return true;
  }

  if (isa<llvm::AllocaInst>(v)) {
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
    setUniform(f, v);

    bool isUniformAlloca = true;
    llvm::Instruction *instruction = dyn_cast<llvm::AllocaInst>(v);
    for (Instruction::use_iterator ui = instruction->use_begin(),
           ue = instruction->use_end();
         ui != ue; ++ui) {
#if defined LLVM_3_2 || defined LLVM_3_3 || defined LLVM_3_4
      llvm::Instruction *user = cast<Instruction>(*ui);
#else
      llvm::Instruction *user = cast<Instruction>(ui->getUser());
#endif
      if (user == NULL) continue;
      
      llvm::StoreInst *store = dyn_cast<llvm::StoreInst>(user);
      if (store) {
        if (!isUniform(f, store->getValueOperand()) || 
            !isUniform(f, store->getParent())) {
          if (!isUniform(f, store->getParent())) {
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
      } else if (dyn_cast<llvm::LoadInst>(user) != NULL) {
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
    setUniform(f, v, isUniformAlloca);
    
    return isUniformAlloca;
  }

  /* TODO: global memory loads are uniform in case they are accessing
     the higher scope ids (group_id_?). */
  if (isa<llvm::LoadInst>(v)) {
    llvm::LoadInst *load = dyn_cast<llvm::LoadInst>(v);
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
        pointer == M->getGlobalVariable("_local_size_z")) {

      setUniform(f, v, true);
      return true;
    } 
  }

  if (isa<llvm::PHINode>(v)) {
    /* TODO: PHINodes need control flow analysis:
       even if the values are uniform, the selected
       value depends on the preceeding basic block which
       might depend on the ID. Assume they are not uniform
       for now in general and treat the loop iteration 
       variable as a special case (set externally from a LoopPass). 

       TODO: PHINodes can depend (indirectly or directly) on itself in loops 
       so it would need infinite recursion checking.
    */
    setUniform(f, v, false);
    return false;
  }

  llvm::Instruction *instr = dyn_cast<llvm::Instruction>(v);
  if (instr == NULL) {
    setUniform(f, v, false);
    return false;
  }

  // Atomic operations might look like uniform if only considering the operands
  // (access a global memory location of which ordering by default is not
  // constrained), but their semantics have ordering: Each work-item should get
  // their own value from that memory location. isAtomic() check was introduced
  // only in LLVM 3.6 so we have to use a more general mayWriteToMemory check
  // with earlier versions. 
#if defined(LLVM_3_2) || defined(LLVM_3_3) || defined(LLVM_3_4) || defined(LLVM_3_5)
  // Volatile loads are set to mayWriteToMemory (perhaps because they might read
  // an I/O register which might update at read) which are treated as a special case 
  // here as we know this is not the case in OpenCL C memory accesses.
  if (instr->mayWriteToMemory() && !isa<llvm::LoadInst>(instr)) {
      setUniform(f, v, false);
      return false;
  }
#else
  if (instr->isAtomic()) {
      setUniform(f, v, false);
      return false;
  }
#endif

  // not computed previously, scan all operands of the instruction
  // and figure out their uniformity recursively
  for (unsigned opr = 0; opr < instr->getNumOperands(); ++opr) {    
    llvm::Value *operand = instr->getOperand(opr);
    if (!isUniform(f, operand)) {
      setUniform(f, v, false);
      return false;
    }
  }
  setUniform(f, v, true);
  return true;
}
  
void
VariableUniformityAnalysis::setUniform(llvm::Function *f, 
                                       llvm::Value *v, 
                                       bool isUniform) {

  UniformityIndex &cache = uniformityCache_[f];
  cache[v] = isUniform;

#ifdef DEBUG_UNIFORMITY_ANALYSIS
  std::cerr << "### ";
  if (isUniform) 
    std::cerr << "uniform ";
  else
    std::cerr << "varying ";

  if (isa<llvm::BasicBlock>(v)) {
    std::cerr << "BB: " << v->getName().str() << std::endl;
  } else {
    v->dump();
  }
#endif
}

bool
VariableUniformityAnalysis::doFinalization(llvm::Module& /*M*/) {
  uniformityCache_.clear();
  return true;
}

}
