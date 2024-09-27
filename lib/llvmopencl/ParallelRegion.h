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

#ifndef POCL_PARALLEL_REGION_H
#define POCL_PARALLEL_REGION_H

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/CFG.h>
#include <llvm/Transforms/Utils/ValueMapper.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
POP_COMPILER_DIAGS

#include <functional>
#include <map>
#include <sstream>
#include <vector>

#include "config.h"

namespace pocl {

class Kernel;

  class ParallelRegion {
  private:
    using BBContainer = std::vector<llvm::BasicBlock *>;

  public:
    typedef llvm::SmallVector<ParallelRegion *, 8> ParallelRegionVector;

    using iterator = BBContainer::iterator;
    using const_iterator = BBContainer::const_iterator;

    ParallelRegion(int forcedRegionId=-1);

    iterator begin() { return BBs_.begin(); }
    iterator end() { return BBs_.end(); }

    const_iterator begin() const { return BBs_.begin(); }
    const_iterator end() const { return BBs_.end(); }

    llvm::BasicBlock *front() { return BBs_.front(); }
    llvm::BasicBlock *back() { return BBs_.back(); }

    llvm::BasicBlock *at(size_t index) { return BBs_.at(index); }
    llvm::BasicBlock *operator[](size_t index) { return BBs_[index]; }

    std::size_t size() const { return BBs_.size(); }
    bool empty() const { return BBs_.empty(); }

    void push_back(llvm::BasicBlock *BB) { BBs_.push_back(BB); }

    void insert(const_iterator pos, llvm::BasicBlock *BB) {
      BBs_.insert(pos, BB);
    }

    template <typename InIteratorT>
    void insert(const_iterator pos, InIteratorT first, InIteratorT last) {
      BBs_.insert(pos, first, last);
    }

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

    void addParallelLoopMetadata(
        llvm::MDNode *Identifier,
        std::function<bool(llvm::Instruction *)> IsLoadUnconditionallySafe);

    bool hasBlock(llvm::BasicBlock *Block);

    void InjectRegionPrintF();
    void InjectVariablePrintouts();

    void InjectPrintF
        (llvm::Instruction *before, std::string formatStr,
         std::vector<llvm::Value*>& params);

    static ParallelRegion *
      Create(const llvm::SmallPtrSet<llvm::BasicBlock *, 8>& bbs,
             llvm::BasicBlock *entry, llvm::BasicBlock *exit);

    static void GenerateTempNames(llvm::BasicBlock *bb);

    llvm::Instruction *getOrCreateIDLoad(std::string IDGlobalName,
                                         llvm::Instruction *Before = nullptr);

    void LocalizeIDLoads();

    int getID() const { return pRegionId; }

    static int getNextID() { return idGen; }

  private:
    BBContainer BBs_;

    /// Cache for the instructions in the entry block that load the various WI
    /// ids.
    std::map<std::string, llvm::Instruction *> IDLoadInstrs;

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
