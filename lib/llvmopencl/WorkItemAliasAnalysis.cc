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


#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Pass.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "LLVMUtils.h"
#include "WorkItemAliasAnalysis.h"
POP_COMPILER_DIAGS

#include <iostream>

#define PASS_NAME "wi-aa"
#define PASS_CLASS pocl::WorkItemAliasAnalysis
#define PASS_DESC "Work item alias analysis."

namespace pocl {

using namespace llvm;

typedef llvm::MemoryLocation Location;
typedef llvm::AliasResult AliasResult;

#if LLVM_MAJOR < 16
#define AAResultB AAResultBase<WorkItemAAResult>
#else
#define AAResultB AAResultBase
#endif

/// WorkItemAliasAnalysis - This is a simple alias analysis
/// implementation that uses pocl metadata to make sure memory accesses from
/// different work items are not aliasing.
///

class WorkItemAAResult : public AAResultB {
    friend AAResultB;

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
char WorkItemAAResult::ID = 0;

/**
 * Test if memory locations are from different work items from same region.
 * Then they can not alias.
 */

#define NO_ALIAS AliasResult::Kind::NoAlias

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
            const MDNode *mdRegionB = dyn_cast<MDNode>(mdB->getOperand(1));
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

                ConstantInt *CIX = dyn_cast<ConstantInt>(
                    dyn_cast<ConstantAsMetadata>(iXYZ->getOperand(1))
                        ->getValue());
                ConstantInt *CJX = dyn_cast<ConstantInt>(
                    dyn_cast<ConstantAsMetadata>(jXYZ->getOperand(1))
                        ->getValue());

                ConstantInt *CIY = dyn_cast<ConstantInt>(
                    dyn_cast<ConstantAsMetadata>(iXYZ->getOperand(2))
                        ->getValue());
                ConstantInt *CJY = dyn_cast<ConstantInt>(
                    dyn_cast<ConstantAsMetadata>(jXYZ->getOperand(2))
                        ->getValue());

                ConstantInt *CIZ = dyn_cast<ConstantInt>(
                    dyn_cast<ConstantAsMetadata>(iXYZ->getOperand(3))
                        ->getValue());
                ConstantInt *CJZ = dyn_cast<ConstantInt>(
                    dyn_cast<ConstantAsMetadata>(jXYZ->getOperand(3))
                        ->getValue());

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

llvm::AnalysisKey WorkItemAliasAnalysis::Key;

WorkItemAliasAnalysis::Result
WorkItemAliasAnalysis::run(llvm::Function &F,
                           llvm::AnalysisManager<Function> &AM) {
  TargetLibraryInfo &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  return WorkItemAAResult(TLI);
}

REGISTER_NEW_FANALYSIS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
