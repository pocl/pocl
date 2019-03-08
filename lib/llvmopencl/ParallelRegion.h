// Class definition for parallel regions, a group of BasicBlocks that
// each kernel should run in parallel.
// 
// Copyright (c) 2011 Universidad Rey Juan Carlos
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

#ifndef _POCL_PARALLEL_REGION_H
#define _POCL_PARALLEL_REGION_H

#include <vector>
#include <sstream>

#include "config.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/CFG.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace pocl {

#define POCL_LOCAL_ID_X_GLOBAL "_local_id_x"
#define POCL_LOCAL_ID_Y_GLOBAL "_local_id_y"
#define POCL_LOCAL_ID_Z_GLOBAL "_local_id_z"

class Kernel;

  // TODO Cleanup: this should not inherit vector but contain it.
  // It now exposes too much to the clients and leads to hard
  // to track errors when the API is changed.
  class ParallelRegion : public std::vector<llvm::BasicBlock *> {    
  public:
    typedef llvm::SmallVector<ParallelRegion *, 8> ParallelRegionVector;

    ParallelRegion(int forcedRegionId=-1);

    /* BarrierBlock *getEntryBarrier(); */
    ParallelRegion *replicate(llvm::ValueToValueMapTy &map,
                              const llvm::Twine &suffix);
    void remap(llvm::ValueToValueMapTy &map);
    void purge();
    void chainAfter(ParallelRegion *region);
    void insertPrologue(unsigned x, unsigned y, unsigned z);
    static void insertLocalIdInit(llvm::BasicBlock* entry,
                                  unsigned x,
                                  unsigned y,
                                  unsigned z);
    void dump();
    void dumpNames();
    void setEntryBBIndex(std::size_t index) { entryIndex_ = index; }
    void setExitBBIndex(std::size_t index) { exitIndex_ = index; }
    void SetExitBB(llvm::BasicBlock *block);
    void AddBlockBefore(llvm::BasicBlock *block, llvm::BasicBlock *before);
    void AddBlockAfter(llvm::BasicBlock *block, llvm::BasicBlock *after);

    llvm::BasicBlock* exitBB() { return at(exitIndex_); }
    llvm::BasicBlock* entryBB() { return at(entryIndex_); }
    void AddIDMetadata(llvm::LLVMContext& context, 
                       std::size_t x = 0, 
                       std::size_t y = 0, 
                       std::size_t z = 0);

    void AddParallelLoopMetadata(llvm::MDNode *Identifier);

    bool HasBlock(llvm::BasicBlock *bb);

    void InjectRegionPrintF();
    void InjectVariablePrintouts();

    void InjectPrintF
        (llvm::Instruction *before, std::string formatStr,
         std::vector<llvm::Value*>& params);

    static ParallelRegion *
      Create(const llvm::SmallPtrSet<llvm::BasicBlock *, 8>& bbs, 
             llvm::BasicBlock *entry, llvm::BasicBlock *exit);

    static void GenerateTempNames(llvm::BasicBlock *bb);

    llvm::Instruction* LocalIDXLoad();
    llvm::Instruction* LocalIDYLoad();
    llvm::Instruction* LocalIDZLoad();

    void LocalizeIDLoads();

    int GetID() const { return pRegionId; }

  private:
    llvm::Instruction* LocalIDXLoadInstr;
    llvm::Instruction* LocalIDYLoadInstr;
    llvm::Instruction* LocalIDZLoadInstr;

    bool Verify();
    /// The indices of entry and exit, not pointers, for finding the BBs in the
    /// replicated PRs too.
    std::size_t exitIndex_;
    std::size_t entryIndex_;

    /// Identifier for the parallel region.
    int pRegionId;
    static int idGen;

  };
    
}
                              
#endif
