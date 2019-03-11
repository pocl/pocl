// Header for WorkitemLoops function pass.
// 
// Copyright (c) 2012 Pekka Jääskeläinen / TUT
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

#ifndef _POCL_WORKITEM_LOOPS_H
#define _POCL_WORKITEM_LOOPS_H

#include <map>
#include <vector>

#include "pocl.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/IR/IRBuilder.h"

#include "WorkitemHandler.h"
#include "ParallelRegion.h"

namespace llvm {
#ifdef LLVM_OLDER_THAN_3_9
  struct PostDominatorTree;
#else
  struct PostDominatorTreeWrapperPass;
#endif
}

namespace pocl {
  class Workgroup;

  class WorkitemLoops : public pocl::WorkitemHandler {

  public:
    static char ID;

  WorkitemLoops() : pocl::WorkitemHandler(ID),
                    original_parallel_regions(nullptr) {}

    virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const;
    virtual bool runOnFunction(llvm::Function &F);

  private:

    typedef std::vector<llvm::BasicBlock *> BasicBlockVector;
    typedef std::set<llvm::Instruction* > InstructionIndex;
    typedef std::vector<llvm::Instruction* > InstructionVec;
    typedef std::map<std::string, llvm::Instruction*> StrInstructionMap;

    llvm::DominatorTree *DT;
#ifdef LLVM_OLDER_THAN_3_7
    llvm::LoopInfo *LI;
#else
    llvm::LoopInfoWrapperPass *LI;
#endif

#ifdef LLVM_OLDER_THAN_3_9
    llvm::PostDominatorTree *PDT;
#else
    llvm::PostDominatorTreeWrapperPass *PDT;
#endif

    llvm::DominatorTreeWrapperPass *DTP;

    ParallelRegion::ParallelRegionVector *original_parallel_regions;

    StrInstructionMap contextArrays;

    virtual bool ProcessFunction(llvm::Function &F);

    void FixMultiRegionVariables(ParallelRegion *region);
    void AddContextSaveRestore(llvm::Instruction *instruction);
    void releaseParallelRegions();

    llvm::Value *GetLinearWiIndex(llvm::IRBuilder<> &builder, llvm::Module *M,
                                  ParallelRegion *region);
    llvm::Instruction *AddContextSave(llvm::Instruction *instruction,
                                      llvm::Instruction *alloca);
    llvm::Instruction *AddContextRestore
        (llvm::Value *val, llvm::Instruction *alloca, 
         llvm::Instruction *before=NULL, 
         bool isAlloca=false);
    llvm::Instruction *GetContextArray(llvm::Instruction *val);

    std::pair<llvm::BasicBlock *, llvm::BasicBlock *>
    CreateLoopAround
        (ParallelRegion &region, llvm::BasicBlock *entryBB, llvm::BasicBlock *exitBB, 
         bool peeledFirst, llvm::Value *localIdVar, size_t LocalSizeForDim,
         bool addIncBlock=true, llvm::Value *DynamicLocalSize=NULL);

    llvm::BasicBlock *
      AppendIncBlock
      (llvm::BasicBlock* after, 
       llvm::Value *localIdVar);

    ParallelRegion* RegionOfBlock(llvm::BasicBlock *bb);

    bool ShouldNotBeContextSaved(llvm::Instruction *instr);

    llvm::Type *RecursivelyAlignArrayType(llvm::Type *ArrayType,
                                          llvm::Type *ElementType,
                                          size_t Alignment,
                                          const llvm::DataLayout &Layout);

    std::map<llvm::Instruction*, unsigned> tempInstructionIds;
    size_t tempInstructionIndex;
    // An alloca in the kernel which stores the first iteration to execute
    // in the inner (dimension 0) loop. This is set to 1 in an peeled iteration
    // to skip the 0, 0, 0 iteration in the loops.
    llvm::Value *localIdXFirstVar;
  };
}

#endif
