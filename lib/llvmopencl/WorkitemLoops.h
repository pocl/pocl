// Header for WorkitemLoops function pass.
//
// Copyright (c) 2012 Pekka Jääskeläinen / TUT
//               2022-2023 Pekka Jääskeläinen / Intel Finland Oy
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#ifndef _POCL_WORKITEM_LOOPS_H
#define _POCL_WORKITEM_LOOPS_H

#include <map>
#include <vector>

#include "pocl.h"

#include <llvm/ADT/Twine.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Transforms/Utils/ValueMapper.h>
#include <llvm/IR/IRBuilder.h>

#include "WorkitemHandler.h"
#include "ParallelRegion.h"

namespace llvm {
  struct PostDominatorTreeWrapperPass;
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
    typedef std::map<std::string, llvm::AllocaInst *> StrInstructionMap;

    llvm::DominatorTree *DT;
    llvm::LoopInfoWrapperPass *LI;

    llvm::PostDominatorTreeWrapperPass *PDT;

    llvm::DominatorTreeWrapperPass *DTP;

    ParallelRegion::ParallelRegionVector *original_parallel_regions;

    StrInstructionMap contextArrays;

    // Points to the __pocl_local_mem_alloca pseudo function declaration, if
    // it's been referred to in the processed module.
    llvm::Function *LocalMemAllocaFuncDecl;

    // Points to the __pocl_work_group_alloca pseudo function declaration, if
    // it's been referred to in the processed module.
    llvm::Function *WorkGroupAllocaFuncDecl;

    // Points to the work-group size computation instruction in the entry
    // block of the currently handled function.
    llvm::Instruction *WGSizeInstr;

    virtual bool ProcessFunction(llvm::Function &F);

    void fixMultiRegionVariables(ParallelRegion *region);
#if LLVM_MAJOR > 13
    bool handleLocalMemAllocas(Kernel &K);
#endif
    void addContextSaveRestore(llvm::Instruction *instruction);
    void releaseParallelRegions();

    // Returns an instruction in the entry block which computes the
    // total size of work-items in the work-group. If it doesn't
    // exist, creates it to the end of the entry block.
    llvm::Instruction *getWorkGroupSizeInstr(llvm::Function &F);

    llvm::Value *GetLinearWiIndex(llvm::IRBuilder<> &builder, llvm::Module *M,
                                  ParallelRegion *region);
    llvm::Instruction *AddContextSave (llvm::Instruction *instruction,
                                       llvm::AllocaInst *alloca);
    llvm::Instruction *AddContextRestore (llvm::Value *val,
                                          llvm::AllocaInst *alloca,
                                          llvm::Type *InstType,
                                          bool PoclWrapperStructAdded,
                                          llvm::Instruction *before = NULL,
                                          bool isAlloca = false);
    llvm::AllocaInst *getContextArray(llvm::Instruction *val,
                                      bool &PoclWrapperStructAdded);

    std::pair<llvm::BasicBlock *, llvm::BasicBlock *>
    CreateLoopAround(ParallelRegion &region, llvm::BasicBlock *entryBB,
                     llvm::BasicBlock *exitBB, bool peeledFirst,
                     llvm::Value *localIdVar, size_t LocalSizeForDim,
                     bool addIncBlock = true,
                     llvm::Value *DynamicLocalSize = NULL);

    llvm::BasicBlock *
      AppendIncBlock
      (llvm::BasicBlock* after, 
       llvm::Value *localIdVar);

    ParallelRegion* RegionOfBlock(llvm::BasicBlock *bb);

    bool shouldNotBeContextSaved(llvm::Instruction *instr);

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
