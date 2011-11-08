// Header for BarrierTailReplication.cc function pass.
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

#include "llvm/Analysis/Dominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Function.h"
#include "llvm/Pass.h"
#include <map>
#include <set>

namespace pocl {
  class Workgroup;

  class BarrierTailReplication : public llvm::FunctionPass {

  public:
    static char ID;

    BarrierTailReplication(): FunctionPass(ID) {}

    virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const;
    virtual bool runOnFunction(llvm::Function &F);
    
  private:
    typedef std::set<llvm::BasicBlock *> BasicBlockSet;
    typedef std::map<llvm::Value *, llvm::Value *> ValueValueMap;

    llvm::DominatorTree *DT;
    llvm::LoopInfo *LI;

    bool ProcessFunction(llvm::Function &F);
    void FindBarriersDFS(llvm::BasicBlock *bb,
                         BasicBlockSet &processed_bbs);
    llvm::BasicBlock* ReplicateSubgraph(llvm::BasicBlock *entry,
                                        llvm::Function *f);
    void FindSubgraph(BasicBlockSet &subgraph,
                      llvm::BasicBlock *entry);
    void ReplicateBasicBlocks(BasicBlockSet &new_graph,
                              ValueValueMap &reference_map,
                              BasicBlockSet &graph,
                              llvm::Function *f);
    void UpdateReferences(const BasicBlockSet &graph,
                          const ValueValueMap &reference_map);

    friend class pocl::Workgroup;
  };
}
