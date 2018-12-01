// LLVM function pass to replicate the kernel body for all work items
// in a work group.
//
// Copyright (c) 2011-2012 Carlos Sánchez de La Lama / URJC and
//               2012-2018 Pekka Jääskeläinen / TUT
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

#include <sstream>
#include <iostream>

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "pocl.h"

#include "llvm/IR/Metadata.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Support/CommandLine.h"

#include "WorkitemHandler.h"
#include "Kernel.h"
#include "DebugHelpers.h"

POP_COMPILER_DIAGS

//#define DEBUG_REFERENCE_FIXING

namespace pocl {

using namespace llvm;

/* This is used to communicate the work-group dimensions of the currently
   compiled kernel command to the workitem loop.

   TODO: Something cleaner than a global value. */
size_t WGLocalSizeX = 1;
size_t WGLocalSizeY = 1;
size_t WGLocalSizeZ = 1;
bool WGDynamicLocalSize = false;

cl::opt<bool>
AddWIMetadata("add-wi-metadata", cl::init(false), cl::Hidden,
  cl::desc("Adds a work item identifier to each of the instruction in work items."));


WorkitemHandler::WorkitemHandler(char& ID) : FunctionPass(ID) {
}

bool
WorkitemHandler::runOnFunction(Function &) {
  return false;
}

void
WorkitemHandler::Initialize(Kernel *K) {

  llvm::Module *M = K->getParent();

  llvm::Type *localIdType;
  size_t_width = 0;
#ifdef LLVM_OLDER_THAN_3_7
  if (M->getDataLayout()->getPointerSize(0) == 8)
    size_t_width = 64;
  else if (M->getDataLayout()->getPointerSize(0) == 4)
    size_t_width = 32;
  else
    assert (false && "Only 32 and 64 bit size_t widths supported.");
#else
  if (M->getDataLayout().getPointerSize(0) == 8)
    size_t_width = 64;
  else if (M->getDataLayout().getPointerSize(0) == 4)
    size_t_width = 32;
  else
    assert (false && "Only 32 and 64 bit size_t widths supported.");
#endif

  localIdType = IntegerType::get(K->getContext(), size_t_width);

  localIdZ = M->getOrInsertGlobal(POCL_LOCAL_ID_Z_GLOBAL, localIdType);
  localIdY = M->getOrInsertGlobal(POCL_LOCAL_ID_Y_GLOBAL, localIdType);
  localIdX = M->getOrInsertGlobal(POCL_LOCAL_ID_X_GLOBAL, localIdType);
}


bool
WorkitemHandler::dominatesUse
(llvm::DominatorTreeWrapperPass *DTP, Instruction &I, unsigned i) {
  DominatorTree *DT = &DTP->getDomTree();
  Instruction *Op = cast<Instruction>(I.getOperand(i));
  BasicBlock *OpBlock = Op->getParent();
  PHINode *PN = dyn_cast<PHINode>(&I);

  // DT can handle non phi instructions for us.
  if (!PN) 
    {
      // Definition must dominate use unless use is unreachable!
      return Op->getParent() == I.getParent() ||
        DT->dominates(Op, &I);
    }

  // PHI nodes are more difficult than other nodes because they actually
  // "use" the value in the predecessor basic blocks they correspond to.
  unsigned j = PHINode::getIncomingValueNumForOperand(i);
  BasicBlock *PredBB = PN->getIncomingBlock(j);
  return (PredBB && DT->dominates(OpBlock, PredBB));
}

/* Fixes the undominated variable uses.

   These appear when a conditional barrier kernel is replicated to
   form a copy of the *same basic block* in the alternative 
   "barrier path".

   E.g., from

   A -> [exit], A -> B -> [exit]

   a replicated CFG as follows, is created:

   A1 -> (T) A2 -> [exit1],  A1 -> (F) A2' -> B1, B2 -> [exit2] 

   The regions are correct because of the barrier semantics
   of "all or none". In case any barrier enters the [exit1]
   from A1, all must (because there's a barrier in the else
   branch).

   Here at A2 and A2' one creates the same variables. 
   However, B2 does not know which copy
   to refer to, the ones created in A2 or ones in A2' (correct).
   The mapping data contains only one possibility, the
   one that was placed there last. Thus, the instructions in B2 
   might end up referring to the variables defined in A2 
   which do not nominate them.

   The variable references are fixed by exploiting the knowledge
   of the naming convention of the cloned variables. 

   One potential alternative way would be to collect the refmaps per BB,
   not globally. Then as a final phase traverse through the 
   basic blocks starting from the beginning and propagating the
   reference data downwards, the data from the new BB overwriting
   the old one. This should ensure the reachability without 
   the costly dominance analysis.
*/
bool
WorkitemHandler::fixUndominatedVariableUses(llvm::DominatorTreeWrapperPass *DT,
                                            llvm::Function &F)
{
  bool changed = false;
  DT->runOnFunction(F);

  for (Function::iterator i = F.begin(), e = F.end(); i != e; ++i) 
    {
      llvm::BasicBlock *bb = &*i;
      for (llvm::BasicBlock::iterator ins = bb->begin(), inse = bb->end();
           ins != inse; ++ins)
        {
          for (unsigned opr = 0; opr < ins->getNumOperands(); ++opr)
            {
              if (!isa<Instruction>(ins->getOperand(opr))) continue;
              Instruction *operand = cast<Instruction>(ins->getOperand(opr));
              if (dominatesUse(DT, *ins, opr)) 
                  continue;
#ifdef DEBUG_REFERENCE_FIXING
              std::cout << "### dominance error!" << std::endl;
              operand->dump();
              std::cout << "### does not dominate:" << std::endl;
              ins->dump();
#endif
              StringRef baseName;
              std::pair< StringRef, StringRef > pieces = 
                operand->getName().rsplit('.');
              if (pieces.second.startswith("pocl_"))
                baseName = pieces.first;
              else
                baseName = operand->getName();
              
              Value *alternative = NULL;

              unsigned int copy_i = 0;
              do {
                std::ostringstream alternativeName;
                alternativeName << baseName.str();
                if (copy_i > 0)
                  alternativeName << ".pocl_" << copy_i;

#ifdef LLVM_OLDER_THAN_4_0
                alternative = 
                  F.getValueSymbolTable().lookup(alternativeName.str());
#else
                alternative = 
                  F.getValueSymbolTable()->lookup(alternativeName.str());
#endif

                if (alternative != NULL)
                  {
                    ins->setOperand(opr, alternative);
                    if (dominatesUse(DT, *ins, opr))
                      break;
                  }
                     
                if (copy_i > 10000 && alternative == NULL)
                  break; /* ran out of possibilities */
                ++copy_i;
              } while (true);

              if (alternative != NULL) {
#ifdef DEBUG_REFERENCE_FIXING
                  std::cout << "### found the alternative:" << std::endl;
                  alternative->dump();
#endif                      
                  changed |= true;
                } else {
#ifdef DEBUG_REFERENCE_FIXING
                  std::cout << "### didn't find an alternative for" << std::endl;
                  operand->dump();
                  std::cerr << "### BB:" << std::endl;
                  operand->getParent()->dump();
                  std::cerr << "### the user BB:" << std::endl;
                  ins->getParent()->dump();
#endif
                  std::cerr << "Could not find a dominating alternative variable." << std::endl;
                  dumpCFG(F, "broken.dot");
                  abort();
              }
            }
        }
    }
  return changed;
}

/**
 * Moves the phi nodes in the beginning of the src to the beginning of
 * the dst. 
 *
 * MergeBlockIntoPredecessor function from llvm discards the phi nodes
 * of the replicated BB because it has only one entry.
 */
void
WorkitemHandler::movePhiNodes(llvm::BasicBlock* src, llvm::BasicBlock* dst) {
  while (PHINode *PN = dyn_cast<PHINode>(src->begin())) 
    PN->moveBefore(dst->getFirstNonPHI());
}


} // namespace pocl
