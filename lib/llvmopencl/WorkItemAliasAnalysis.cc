/*
    Copyright (c) 2012 Tampere University of Technology.

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
 * @author Vladimír Guzma 2012
 */
#include <iostream>

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Pass.h"
#include "llvm/Metadata.h"
#include "llvm/Constants.h"


using namespace llvm;

namespace {
/// WorkItemAliasAnalysis - This is a simple alias analysis
/// implementation that uses pocl metadata to make sure memory accesses from
/// different work items are not aliasing.
class WorkItemAliasAnalysis : public FunctionPass, public AliasAnalysis {
public:
    static char ID; 
    WorkItemAliasAnalysis() : FunctionPass(ID) {}

    /// getAdjustedAnalysisPointer - This method is used when a pass implements
    /// an analysis interface through multiple inheritance.  If needed, it
    /// should override this to adjust the this pointer as needed for the
    /// specified pass info.
    virtual void *getAdjustedAnalysisPointer(AnalysisID PI) {
        if (PI == &AliasAnalysis::ID)
            return (AliasAnalysis*)this;
        return this;
    }
    private:
        virtual void getAnalysisUsage(AnalysisUsage &AU) const;
        virtual bool runOnFunction(Function &F);
        virtual AliasResult alias(const Location &LocA, const Location &LocB);

    };
}

// Register this pass...
char WorkItemAliasAnalysis::ID = 0;
RegisterPass<WorkItemAliasAnalysis>
    X("wi-aa", "Work item alias analysis.", false, false);
// Register it also to pass group
RegisterAnalysisGroup<AliasAnalysis> Y(X);  

FunctionPass *createWorkItemAliasAnalysisPass() {
    return new WorkItemAliasAnalysis();
}

extern "C" {                                
    FunctionPass*
    create_workitem_aa_plugin() {
        return new WorkItemAliasAnalysis();
    }
}

void
WorkItemAliasAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
    AliasAnalysis::getAnalysisUsage(AU);
}

bool
WorkItemAliasAnalysis::runOnFunction(Function &F) {
    InitializeAliasAnalysis(this);
    return false;
}

/**
 * Test if memory locations are from different work items from same region.
 * Then they can not alias.
 */
AliasAnalysis::AliasResult
WorkItemAliasAnalysis::alias(const Location &LocA,
                                    const Location &LocB) {
    // If either of the memory references is empty, it doesn't matter what the
    // pointer values are. This allows the code below to ignore this special
    // case.
    if (LocA.Size == 0 || LocB.Size == 0)
        return NoAlias;
    
    // Pointers from different address spaces do not alias
    if (cast<PointerType>(LocA.Ptr->getType())->getAddressSpace() != 
        cast<PointerType>(LocB.Ptr->getType())->getAddressSpace()) {
        return NoAlias;
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
            ConstantInt* C1 = dyn_cast<ConstantInt>(mdA->getOperand(1));
            ConstantInt* C2 = dyn_cast<ConstantInt>(mdB->getOperand(1));
            if (C1->getValue() == C2->getValue()) {
                // Now we have both locations from same region. Check for different
                // work items.
                int differs = 0;
                for (unsigned int i = 2; i < mdA->getNumOperands() -1; i++) {
                    // Last operand stores the line number in original bitcode
                    // which is of no use to us here.
                    ConstantInt *CA = dyn_cast<ConstantInt>(mdA->getOperand(i));
                    ConstantInt *CB = dyn_cast<ConstantInt>(mdB->getOperand(i));
                    if (CA->getValue() != CB->getValue()) {
                        differs ++;
                    }
                }
                if (differs != 0) {
                    return NoAlias;
                }
            }
        
        }
    }
  
    // Forward the query to the next analysis.
    return AliasAnalysis::alias(LocA, LocB);
}
