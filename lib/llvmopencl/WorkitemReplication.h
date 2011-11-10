// Header for WorkitemReplication.cc function pass.
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

  class WorkitemReplication : public llvm::FunctionPass {

  public:
    static char ID;

  WorkitemReplication(): FunctionPass(ID) {}

    virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const;
    virtual bool runOnFunction(llvm::Function &F);

  private:
    typedef std::set<llvm::BasicBlock *> BasicBlockSet;
    typedef std::map<llvm::Value *, llvm::Value *> ValueValueMap;

    llvm::DominatorTree *DT;
    llvm::LoopInfo *LI;

    BasicBlockSet ProcessedBarriers;

    // This stores the reference substitution map for each workitem.
    // Even when some basic blocks are replicated more than once (due
    // to barriers creating several execution paths), blocks are
    // always processed from dominator to function end, making last
    // added reference the correct one, thus no need for more
    // complex bookeeping.
    ValueValueMap *ReferenceMap;

    llvm::GlobalVariable *LocalX, *LocalY, *LocalZ;

    bool ProcessFunction(llvm::Function &F);
    llvm::BasicBlock *FindBarriersDFS(llvm::BasicBlock *bb,
				      llvm::BasicBlock *entry,
				      BasicBlockSet &bbs_to_replicate);
    bool FindSubgraph(BasicBlockSet &subgraph,
                      llvm::BasicBlock *entry,
                      llvm::BasicBlock *exit);
    void replicateWorkitemSubgraph(BasicBlockSet subgraph,
				   llvm::BasicBlock *entry,
				   llvm::BasicBlock *exit);
    void replicateBasicblocks(BasicBlockSet &new_graph,
			      ValueValueMap &reference_map,
			      const BasicBlockSet &graph);
    void updateReferences(const BasicBlockSet &graph,
			  const ValueValueMap &reference_map);
    bool isReplicable(const llvm::Instruction *i);

    friend class pocl::Workgroup;
  };
}
