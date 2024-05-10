//===- BreakConstantGEPs.cpp - Change constant GEPs into GEP instructions - --//
// 
// pocl note: This pass is taken from The SAFECode project with trivial modifications.
//            Automatic locals might cause constant GEPs which cause problems during 
//            converting the locals to kernel function arguments for thread safety.
//
//                          The SAFECode Compiler 
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This pass changes all GEP constant expressions into GEP instructions.  This
// permits the rest of SAFECode to put run-time checks on them if necessary.
//
//===----------------------------------------------------------------------===//

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/ADT/Statistic.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>

#include "BreakConstantGEPs.h"
#include "LLVMUtils.h"
#include "Workgroup.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#include <iostream>
#include <map>
#include <utility>

#define DEBUG_TYPE "break-constgeps"

#define PASS_NAME "break-constgeps"
#define PASS_CLASS pocl::BreakConstantGEPs
#define PASS_DESC "Remove GEP Constant Expressions"

// Statistics
STATISTIC (GEPChanges,   "Number of Converted GEP Constant Expressions");
STATISTIC (TotalChanges, "Number of Converted Constant Expressions");

namespace pocl {

using namespace llvm;

static bool breakConstantGEPs(Function &F);

//
// Function: hasConstantGEP()
//
// Description:
//  This function determines whether the given value is a constant expression
//  that has a constant GEP expression embedded within it.
//
// Inputs:
//  V - The value to check.
//
// Return value:
//  NULL  - This value is not a constant expression with a constant expression
//          GEP within it.
//  ~NULL - A pointer to the value casted into a ConstantExpr is returned.
//
static ConstantExpr *
hasConstantGEP (Value * V) {
  if (ConstantExpr * CE = dyn_cast<ConstantExpr>(V)) {
    bool isGEPOrCast =
        CE->getOpcode() == Instruction::GetElementPtr ||
        CE->getOpcode() == Instruction::BitCast;

    isGEPOrCast |= CE->getOpcode() == Instruction::AddrSpaceCast;
    if (isGEPOrCast) {
      return CE;
    } else {
      for (unsigned index = 0; index < CE->getNumOperands(); ++index) {
        if (hasConstantGEP (CE->getOperand(index)))
          return CE;
      }
    }
  }

  return 0;
}

//
// Function: convertGEP()
//
// Description:
//  Convert a GEP constant expression into a GEP instruction.
//
// Inputs:
//  CE       - The GEP constant expression.
//  InsertPt - The instruction before which to insert the new GEP instruction.
//
// Return value:
//  A pointer to the new GEP instruction is returned.
//
static Instruction *
convertGEP (ConstantExpr * CE, Instruction * InsertPt) {
  //
  // Create iterators to the indices of the constant expression.
  //
  std::vector<Value *> Indices;
  for (unsigned index = 1; index < CE->getNumOperands(); ++index) {
    Indices.push_back (CE->getOperand (index));
  }

  //
  // Update the statistics.
  //
  ++GEPChanges;

  //
  // Make the new GEP instruction.
  //
  /* The first NULL is the Type. It is not used at all, just asserted 
   * against. And it asserts, no matter what is passed. Except NULL. 
   * Seems this API is still "fluctuation in progress"*/
  return (GetElementPtrInst::Create (NULL,
                                     CE->getOperand(0),
                                     Indices,
                                     CE->getName(),
                                     InsertPt));
}

//
// Function: convertExpression()
//
// Description:
//  Convert a constant expression into an instruction.  This routine does *not*
//  perform any recursion, so the resulting instruction may have constant
//  expression operands.
//
static Instruction *
convertExpression (ConstantExpr * CE, Instruction * InsertPt) {
  //
  // Convert this constant expression into a regular instruction.
  //
  Instruction * NewInst = 0;
  switch (CE->getOpcode()) {
    case Instruction::GetElementPtr: {
      NewInst = convertGEP (CE, InsertPt);
      break;
    }

    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Mul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor: {
      Instruction::BinaryOps Op = (Instruction::BinaryOps)(CE->getOpcode());
      NewInst = BinaryOperator::Create (Op,
                                        CE->getOperand(0),
                                        CE->getOperand(1),
                                        CE->getName(),
                                        InsertPt);
      break;
    }

    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::AddrSpaceCast:
    case Instruction::BitCast: {
      Instruction::CastOps Op = (Instruction::CastOps)(CE->getOpcode());
      NewInst = CastInst::Create (Op,
                                  CE->getOperand(0),
                                  CE->getType(),
                                  CE->getName(),
                                  InsertPt);
      break;
    }

    case Instruction:: FCmp:
    case Instruction:: ICmp: {
      Instruction::OtherOps Op = (Instruction::OtherOps)(CE->getOpcode());
      NewInst = CmpInst::Create (Op,
                                 (llvm::CmpInst::Predicate)CE->getPredicate(),
                                 CE->getOperand(0),
                                 CE->getOperand(1),
                                 CE->getName(),
                                 InsertPt);
      break;
    }

    case Instruction:: Select:
      NewInst = SelectInst::Create (CE->getOperand(0),
                                    CE->getOperand(1),
                                    CE->getOperand(2),
                                    CE->getName(),
                                    InsertPt);
      break;

    case Instruction:: ExtractElement:
    case Instruction:: InsertElement:
    case Instruction:: ShuffleVector:
    case Instruction:: InsertValue:
    default:
      assert (0 && "Unhandled constant expression!\n");
      break;
  }

  //
  // Update the statistics.
  //
  ++TotalChanges;

  return NewInst;
}

//
// Method: runOnFunction()
//
// Description:
//  Entry point for this LLVM pass.
//
// Return value:
//  true  - The function was modified.
//  false - The function was not modified.
//
static bool breakConstantGEPs(Function &F) {

  if (!pocl::isKernelToProcess(F))
    return false;

  bool modified = false;

  // Worklist of values to check for constant GEP expressions
  std::vector<Instruction *> Worklist;

  //
  // Initialize the worklist by finding all instructions that have one or more
  // operands containing a constant GEP expression.
  //
  for (Function::iterator BB = F.begin(); BB != F.end(); ++BB) {
    for (BasicBlock::iterator i = BB->begin(); i != BB->end(); ++i) {
      //
      // Scan through the operands of this instruction.  If it is a constant
      // expression GEP, insert an instruction GEP before the instruction.
      //
      Instruction * I = &*i;
      for (unsigned index = 0; index < I->getNumOperands(); ++index) {
        if (hasConstantGEP (I->getOperand(index))) {
          Worklist.push_back (I);
        }
      }
    }
  }

  //
  // Determine whether we will modify anything.
  //
  if (Worklist.size()) modified = true;

  //
  // While the worklist is not empty, take an item from it, convert the
  // operands into instructions if necessary, and determine if the newly
  // added instructions need to be processed as well.
  //
  while (Worklist.size()) {
    Instruction * I = Worklist.back();
    Worklist.pop_back();

    //
    // Scan through the operands of this instruction and convert each into an
    // instruction.  Note that this works a little differently for phi
    // instructions because the new instruction must be added to the
    // appropriate predecessor block.
    //
    if (PHINode * PHI = dyn_cast<PHINode>(I)) {
      for (unsigned index = 0; index < PHI->getNumIncomingValues(); ++index) {
        //
        // For PHI Nodes, if an operand is a constant expression with a GEP, we
        // want to insert the new instructions in the predecessor basic block.
        //
        // Note: It seems that it's possible for a phi to have the same
        // incoming basic block listed multiple times; this seems okay as long
        // the same value is listed for the incoming block.
        //
        Instruction * InsertPt = PHI->getIncomingBlock(index)->getTerminator();
        if (ConstantExpr * CE = hasConstantGEP (PHI->getIncomingValue(index))) {
          Instruction * NewInst = convertExpression (CE, InsertPt);
          for (unsigned i2 = index; i2 < PHI->getNumIncomingValues(); ++i2) {
            if ((PHI->getIncomingBlock (i2)) == PHI->getIncomingBlock (index))
              PHI->setIncomingValue (i2, NewInst);
          }
          Worklist.push_back (NewInst);
        }
      }
    } else {
      for (unsigned index = 0; index < I->getNumOperands(); ++index) {
        //
        // For other instructions, we want to insert instructions replacing
        // constant expressions immediently before the instruction using the
        // constant expression.
        //
        if (ConstantExpr * CE = hasConstantGEP (I->getOperand(index))) {
          Instruction * NewInst = convertExpression (CE, I);
          I->replaceUsesOfWith (CE, NewInst);
          Worklist.push_back (NewInst);
        }
      }
    }
  }

  return modified;
}

llvm::PreservedAnalyses
BreakConstantGEPs::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();
  PAChanged.preserveSet<CFGAnalyses>();
  return breakConstantGEPs(F) ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC)

} // namespace pocl
