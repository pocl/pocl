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
#ifndef LLVM_OLDER_THAN_3_7
# include "llvm/Analysis/TargetLibraryInfo.h"
#endif
#include "llvm/Analysis/Passes.h"
#include "llvm/Pass.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

POP_COMPILER_DIAGS

using namespace llvm;

#ifdef LLVM_OLDER_THAN_3_7
typedef AliasAnalysis::AliasResult AliasResult;
#else
typedef llvm::MemoryLocation Location;
typedef llvm::AliasResult AliasResult;
#endif

/// WorkItemAliasAnalysis - This is a simple alias analysis
/// implementation that uses pocl metadata to make sure memory accesses from
/// different work items are not aliasing.
///
#ifdef LLVM_OLDER_THAN_3_8
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

#ifdef LLVM_OLDER_THAN_3_7
    virtual void initializePass() {
        InitializeAliasAnalysis(this);
    }
    virtual bool runOnFunction(llvm::Function &) {
        InitializeAliasAnalysis(this);
        return false;
    }
#else
    virtual bool runOnFunction(llvm::Function &F) {
        InitializeAliasAnalysis(this, &F.getParent()->getDataLayout());
        return false;
    }
#endif

    private:
        virtual void getAnalysisUsage(AnalysisUsage &AU) const;
        virtual AliasResult alias(const Location &LocA, const Location &LocB);
};

#else

// LLVM 3.8+

class WorkItemAAResult : public AAResultBase<WorkItemAAResult> {
    friend AAResultBase<WorkItemAAResult>;

public:
    static char ID;

#ifdef LLVM_OLDER_THAN_3_9
    WorkItemAAResult(const TargetLibraryInfo &TLI)
        : AAResultBase(TLI) {}
    WorkItemAAResult(const WorkItemAAResult &Arg)
        : AAResultBase(Arg.TLI) {}
    WorkItemAAResult(WorkItemAAResult &&Arg)
        : AAResultBase(Arg.TLI) {}
#else
    WorkItemAAResult(const TargetLibraryInfo &)
        : AAResultBase() {}
    WorkItemAAResult(const WorkItemAAResult &)
        : AAResultBase() {}
    WorkItemAAResult(WorkItemAAResult &&)
        : AAResultBase() {}
#endif

    AliasResult alias(const MemoryLocation &LocA, const MemoryLocation &LocB);
};

class WorkItemAA {
public:
    typedef WorkItemAAResult Result;

#ifdef LLVM_OLDER_THAN_4_0
    /// \brief Opaque, unique identifier for this analysis pass.
    static void *ID() { return (void *)&PassID; }
#else
    static AnalysisKey *ID() { return &PassID; }
#endif

    WorkItemAAResult run(Function &F, AnalysisManager<Function> *AM);

    /// \brief Provide access to a name for this pass for debugging purposes.
    static StringRef name() { return "WorkItemAliasAnalysis"; }

private:
#ifdef LLVM_OLDER_THAN_4_0
    static char PassID;
#else
    static AnalysisKey PassID;
#endif
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

#ifdef LLVM_OLDER_THAN_4_0
char WorkItemAA::PassID;
#else
AnalysisKey WorkItemAA::PassID;
#endif
char WorkItemAAResult::ID = 0;
void WorkItemAliasAnalysis::anchor() {}

WorkItemAAResult WorkItemAA::run(Function &F, AnalysisManager<Function> *AM) {
    return WorkItemAAResult(AM->getResult<WorkItemAA>(F));
}

bool WorkItemAliasAnalysis::runOnFunction(llvm::Function &) {
    auto &TLIWP = getAnalysis<TargetLibraryInfoWrapperPass>();
    Result.reset(new WorkItemAAResult(TLIWP.getTLI()));
    return false;
}

#endif // LLVM 3.8+

// Register this pass...
char WorkItemAliasAnalysis::ID = 0;
RegisterPass<WorkItemAliasAnalysis>
    X("wi-aa", "Work item alias analysis.", false, false);
// Register it also to pass group
#ifdef LLVM_OLDER_THAN_3_8
RegisterAnalysisGroup<AliasAnalysis> Y(X);
#else
RegisterAnalysisGroup<WorkItemAAResult> Y(X);
#endif

void
WorkItemAliasAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesAll();
#ifdef LLVM_OLDER_THAN_3_8
    AliasAnalysis::getAnalysisUsage(AU);
#else
    AU.addRequired<TargetLibraryInfoWrapperPass>();
#endif
}

/**
 * Test if memory locations are from different work items from same region.
 * Then they can not alias.
 */

AliasResult
#ifdef LLVM_OLDER_THAN_3_8
WorkItemAliasAnalysis::alias(const Location &LocA, const Location &LocB) {
#else
WorkItemAAResult::alias(const Location &LocA, const Location &LocB) {
#endif
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
                    return NoAlias;
                }                
            }        
        }
    }
  
#ifdef LLVM_OLDER_THAN_3_8
    // Forward the query to the next analysis.
    return AliasAnalysis::alias(LocA, LocB);
#else
    return WorkItemAAResult::alias(LocA, LocB);
#endif
}
