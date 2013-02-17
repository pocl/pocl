//===- BreakConstantGEPs.h - Change constant GEPs into GEP instructions --- --//
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

#ifndef BREAKCONSTANTGEPS_H
#define BREAKCONSTANTGEPS_H

#include "config.h"
#if (defined LLVM_3_1 or defined LLVM_3_2)
#include "llvm/Module.h"
#else
#include "llvm/IR/Module.h"
#endif
#include "llvm/Analysis/Dominators.h"
#include "llvm/Pass.h"

using namespace llvm;

//
// Pass: BreakConstantGEPs
//
// Description:
//  This pass modifies a function so that it uses GEP instructions instead of
//  GEP constant expressions.
//
struct BreakConstantGEPs : public FunctionPass {
  private:
    // Private methods

    // Private variables

  public:
    static char ID;
    BreakConstantGEPs() : FunctionPass(ID) {}
    const char *getPassName() const {return "Remove Constant GEP Expressions";}
    virtual bool runOnFunction (Function & F);
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      // This pass does not modify the control-flow graph of the function
      AU.setPreservesCFG();
    }
};

#endif
