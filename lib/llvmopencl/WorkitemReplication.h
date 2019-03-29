// Header for WorkitemReplication function pass.
// 
// Copyright (c) 2011 Universidad Rey Juan Carlos and
//               2012-2015 Pekka Jääskeläinen / TUT
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

#ifndef _POCL_WORKITEM_REPLICATION_H
#define _POCL_WORKITEM_REPLICATION_H

#include <map>
#include <vector>

#include "pocl.h"

#include "llvm/IR/Dominators.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include "WorkitemHandler.h"

namespace pocl {
  class Workgroup;

  class WorkitemReplication : public pocl::WorkitemHandler {

  public:
    static char ID;

  WorkitemReplication() : pocl::WorkitemHandler(ID) {}

    virtual void getAnalysisUsage(llvm::AnalysisUsage &AU) const;
    virtual bool runOnFunction(llvm::Function &F);


  private:

    llvm::DominatorTree *DT;
    llvm::DominatorTreeWrapperPass *DTP;
    llvm::LoopInfoWrapperPass *LI;

    typedef std::set<llvm::BasicBlock *> BasicBlockSet;
    typedef std::vector<llvm::BasicBlock *> BasicBlockVector;
    typedef std::map<llvm::Value *, llvm::Value *> ValueValueMap;

    virtual bool ProcessFunction(llvm::Function &F);
  };
}

#endif
