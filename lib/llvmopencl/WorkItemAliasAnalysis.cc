/*
    Copyright (c) 2012-2017 Tampere University of Technology.

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
 */
/**
 * @file WorkItemAliasAnalysis.cc
 *
 * Definition of WorkItemAliasAnalysis class.
 *
 * @author Vladimír Guzma 2012,
 * @author Pekka Jääskeläinen 2015-2017
 */

#include <iostream>

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "pocl.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Pass.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

POP_COMPILER_DIAGS

using namespace llvm;

typedef llvm::MemoryLocation Location;
typedef llvm::AliasResult AliasResult;

/// WorkItemAliasAnalysis - This is a simple alias analysis
/// implementation that uses pocl metadata to make sure memory accesses from
/// different work items are not aliasing.
///

// LLVM 3.8+

class WorkItemAAResult : public AAResultBase<WorkItemAAResult> {
    friend AAResultBase<WorkItemAAResult>;

public:
    static char ID;

    WorkItemAAResult(const TargetLibraryInfo &)
        : AAResultBase() {}
    WorkItemAAResult(const WorkItemAAResult &)
        : AAResultBase() {}
    WorkItemAAResult(WorkItemAAResult &&)
        : AAResultBase() {}

    AliasResult alias(const MemoryLocation &LocA, const MemoryLocation &LocB);
};

class WorkItemAA {
public:
    typedef WorkItemAAResult Result;

    static AnalysisKey *ID() { return &PassID; }

    WorkItemAAResult run(Function &F, AnalysisManager<Function> *AM);

    /// \brief Provide access to a name for this pass for debugging purposes.
    static StringRef name() { return "WorkItemAliasAnalysis"; }

private:
    static AnalysisKey PassID;
};

/// Legacy wrapper pass to provide the (WorkItemAAWrapperPass) object.
class WorkItemAliasAnalysis : public FunctionPass {
    std::unique_ptr<WorkItemAAResult> Result;

    virtual void anchor();

public:
    static char ID;

    WorkItemAliasAnalysis() : FunctionPass(ID) {};

    WorkItemAAResult &getResult() { return *Result; }
    const WorkItemAAResult &getResult() const { return *Result; }

    bool runOnFunction(Function &F) override;
    void getAnalysisUsage(AnalysisUsage &AU) const override;
};

AnalysisKey WorkItemAA::PassID;

char WorkItemAAResult::ID = 0;
void WorkItemAliasAnalysis::anchor() {}

WorkItemAAResult WorkItemAA::run(Function &F, AnalysisManager<Function> *AM) {
    return WorkItemAAResult(AM->getResult<WorkItemAA>(F));
}

bool WorkItemAliasAnalysis::runOnFunction(
#ifdef LLVM_OLDER_THAN_10_0
    IGNORE_UNUSED
#endif
    llvm::Function &f) {
  auto &TLIWP = getAnalysis<TargetLibraryInfoWrapperPass>();
#ifndef LLVM_OLDER_THAN_10_0
  auto tli = TLIWP.getTLI(f);
#else
  auto tli = TLIWP.getTLI();
#endif
  Result.reset(new WorkItemAAResult(tli));
  return false;
}

// Register this pass...
char WorkItemAliasAnalysis::ID = 0;
RegisterPass<WorkItemAliasAnalysis>
    X("wi-aa", "Work item alias analysis.", false, false);
// Register it also to pass group
RegisterAnalysisGroup<WorkItemAAResult> Y(X);

void
WorkItemAliasAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
}

/**
 * Test if memory locations are from different work items from same region.
 * Then they can not alias.
 */

#ifdef LLVM_OLDER_THAN_13_0
#define NO_ALIAS NoAlias
#else
#define NO_ALIAS AliasResult::Kind::NoAlias
#endif

AliasResult
WorkItemAAResult::alias(const Location &LocA, const Location &LocB) {
    // If either of the memory references is empty, it doesn't matter what the
    // pointer values are. This allows the code below to ignore this special
    // case.
    if (LocA.Size == 0 || LocB.Size == 0)
      return NO_ALIAS;

    // Pointers from different address spaces do not alias
    if (cast<PointerType>(LocA.Ptr->getType())->getAddressSpace() != 
        cast<PointerType>(LocB.Ptr->getType())->getAddressSpace()) {
      return NO_ALIAS;
    }
    // In case code is created by pocl, we can also use metadata.
    if (isa<Instruction>(LocA.Ptr) && isa<Instruction>(LocB.Ptr)) {
        const Instruction* valA = dyn_cast<Instruction>(LocA.Ptr);
        const Instruction* valB = dyn_cast<Instruction>(LocB.Ptr);
        if (valA->getMetadata("wi") && valB->getMetadata("wi")) {
            const MDNode* mdA = valA->getMetadata("wi");
            const MDNode* mdB = valB->getMetadata("wi");
            // Compare region ID. If they are same, different work items
            // imply no aliasing. If regions are different or work items
            // are same anything can happen.
            // Fall back to other AAs.
            const MDNode* mdRegionA = dyn_cast<MDNode>(mdA->getOperand(1));
            const MDNode* mdRegionB = dyn_cast<MDNode>(mdB->getOperand(1)); 
            ConstantInt* C1 = dyn_cast<ConstantInt>(
              dyn_cast<ConstantAsMetadata>(mdRegionA->getOperand(1))->getValue());
            ConstantInt* C2 = dyn_cast<ConstantInt>(
              dyn_cast<ConstantAsMetadata>(mdRegionB->getOperand(1))->getValue());
            if (C1->getValue() == C2->getValue()) {
                // Now we have both locations from same region. Check for different
                // work items.
                MDNode* iXYZ= dyn_cast<MDNode>(mdA->getOperand(2));
                MDNode* jXYZ= dyn_cast<MDNode>(mdB->getOperand(2));
                assert(iXYZ->getNumOperands() == 4);
                assert(jXYZ->getNumOperands() == 4);

                ConstantInt *CIX = 
                  dyn_cast<ConstantInt>(
                      dyn_cast<ConstantAsMetadata>(
                        iXYZ->getOperand(1))->getValue());
                ConstantInt *CJX = 
                  dyn_cast<ConstantInt>(
                    dyn_cast<ConstantAsMetadata>(
                      jXYZ->getOperand(1))->getValue());

                ConstantInt *CIY = 
                  dyn_cast<ConstantInt>(
                    dyn_cast<ConstantAsMetadata>(
                      iXYZ->getOperand(2))->getValue());
                ConstantInt *CJY = 
                  dyn_cast<ConstantInt>(
                    dyn_cast<ConstantAsMetadata>(
                      jXYZ->getOperand(2))->getValue());
                
                ConstantInt *CIZ = 
                  dyn_cast<ConstantInt>(
                    dyn_cast<ConstantAsMetadata>(
                      iXYZ->getOperand(3))->getValue());
                ConstantInt *CJZ = 
                  dyn_cast<ConstantInt>(
                    dyn_cast<ConstantAsMetadata>(
                      jXYZ->getOperand(3))->getValue());
                
                if ( !(CIX->getValue() == CJX->getValue()
                    && CIY->getValue() == CJY->getValue()
                    && CIZ->getValue() == CJZ->getValue())) {
                  return NO_ALIAS;
                }                
            }        
        }
    }
  
    return WorkItemAAResult::alias(LocA, LocB);
}
