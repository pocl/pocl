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

#include "BarrierBlock.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/BasicBlock.h"
#include "llvm/Support/CFG.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/LLVMContext.h"
#include <vector>

namespace pocl {

  // TODO Cleanup: this should not inherit vector but contain it.
  // It now exposes too much to the clients and leads to hard
  // to track erros when API is changed.
  class ParallelRegion : public std::vector<llvm::BasicBlock *> {
    
  public:    
    /* BarrierBlock *getEntryBarrier(); */
    ParallelRegion *replicate(llvm::ValueToValueMapTy &map,
                              const llvm::Twine &suffix);
    void remap(llvm::ValueToValueMapTy &map);
    void purge();
    void chainAfter(ParallelRegion *region);
    void insertPrologue(unsigned x, unsigned y, unsigned z);
    void dump();
    void dumpNames();
    void setEntryBBIndex(std::size_t index) { entryIndex_ = index; }
    void setExitBBIndex(std::size_t index) { exitIndex_ = index; }
    llvm::BasicBlock* exitBB() { return at(exitIndex_); }
    llvm::BasicBlock* entryBB() { return at(entryIndex_); }
    void setID(
	llvm::LLVMContext& context, 
	std::size_t x = 0, 
	std::size_t y = 0, 
	std::size_t z = 0,
        std::size_t regionID = 0);

    static ParallelRegion *
      Create(const llvm::SmallPtrSet<llvm::BasicBlock *, 8>& bbs, 
             llvm::BasicBlock *entry, llvm::BasicBlock *exit);


    static void GenerateTempNames(llvm::BasicBlock *bb);

  private:
    bool Verify();
    /// The indices of entry and exit, not pointers, for finding the BBs in the
    /// replicated PRs too.
    std::size_t exitIndex_;
    std::size_t entryIndex_;
  };
    
}
                              
