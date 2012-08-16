// LLVM function pass to create a loop that runs all the work items 
// in a work group.
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

#define DEBUG_TYPE "workitem-loops"

#include "WorkitemLoops.h"
#include "Workgroup.h"
#include "Barrier.h"
#include "Kernel.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/ValueSymbolTable.h"
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#ifdef DUMP_RESULT_CFG
#include "llvm/Analysis/CFGPrinter.h"
#endif

using namespace llvm;
using namespace pocl;

namespace {
  static
  RegisterPass<WorkitemLoops> X("workitemloops", "Workitem loop generation pass");
}

char WorkitemLoops::ID = 0;

void
WorkitemLoops::getAnalysisUsage(AnalysisUsage &AU) const
{
  AU.addRequired<DominatorTree>();
  AU.addRequired<LoopInfo>();
  AU.addRequired<TargetData>();
}

bool
WorkitemLoops::runOnFunction(Function &F)
{
  if (!Workgroup::isKernelToProcess(F))
    return false;

  DT = &getAnalysis<DominatorTree>();
  LI = &getAnalysis<LoopInfo>();

  bool changed = ProcessFunction(F);
#ifdef DUMP_RESULT_CFG
  FunctionPass* cfgPrinter = createCFGPrinterPass();
  cfgPrinter->runOnFunction(F);
#endif

  changed |= fixUndominatedVariableUses(DT, F);
  return changed;
}

void
WorkitemLoops::CreateLoopAround
(ParallelRegion *region, llvm::Value *localIdVar, size_t LocalSizeForDim) 
{
#if 0
  std::cerr << "### creating loop around PR:" << std::endl;
  region->dump();    
#endif
  assert (region != NULL);
  assert (localIdVar != NULL);

  /* TODO: Process the variables: in case the variable is 
     1) created before the PR, load its value from the WI context
     data array
     2) live only inside the PR: do nothing
     3) created inside the PR and used outside, save its value
     at the end of the loop to the WI context data array */

  /* TODO: The Conditional barrier case needs the first iteration
     peeled. Check the case how it's best done. */

  /*

    Generate a structure like this for each loop level (x,y,z):

    for.preheader: 
    ; initialize the dimension id variable that is used as the iterator
    store i32 0, i32* %local_id_x, align 4
    br label %for.cond:

    for.cond:
    ; loop header, compare the id to the local size
    %0 = load i32* %_local_id_x, align 4
    %cmp = icmp ult i32 %0, i32 123
    br i1 %cmp, label %for.body, label %for.end

    for.body: 

    ; the parallel region code here

    br label %for.inc
        
    for.inc:
    %2 = load i32* %_local_id_x, align 4
    %inc = add nsw i32 %2, 1
    store i32 %inc, i32* %_local_id_x, align 4
    br label %for.cond

    for.end:

    TODO: Use a separate iteration variable across all the loops to iterate the context 
    data arrays to avoid needing multiplications to find the correct location, and to 
    enable easy vectorization of loading the context data when there are parallel iterations.
  */     

  llvm::BasicBlock *preLoopBB = region->entryBB()->getSinglePredecessor();
  llvm::BasicBlock *loopBodyEntryBB = region->entryBB();
  llvm::LLVMContext &C = loopBodyEntryBB->getContext();
  loopBodyEntryBB->setName("pregion.for.body");

  assert (region->exitBB()->getTerminator()->getNumSuccessors() == 1);

  llvm::BasicBlock *oldExit = region->exitBB()->getTerminator()->getSuccessor(0);

  llvm::BasicBlock *preheaderBB =
    BasicBlock::Create
    (C, "pregion.for.preheader",
     loopBodyEntryBB->getParent(), loopBodyEntryBB);

  llvm::BasicBlock *loopEndBB = 
    BasicBlock::Create
    (C, "pregion.for.end",
     region->exitBB()->getParent(), region->exitBB());

  llvm::BasicBlock *forCondBB = 
    BasicBlock::Create
    (C, "pregion.for.cond",
     region->exitBB()->getParent(), loopBodyEntryBB);

  llvm::BasicBlock *forIncBB = 
    BasicBlock::Create
    (C, "pregion.for.inc",
     region->exitBB()->getParent(), loopEndBB);

  preLoopBB->getTerminator()->replaceUsesOfWith(loopBodyEntryBB, preheaderBB);

  IRBuilder<> builder(preheaderBB);
  builder.CreateBr(forCondBB);

  /* chain region body and loop exit */
  region->exitBB()->getTerminator()->replaceUsesOfWith(oldExit, forIncBB);

  builder.SetInsertPoint(forIncBB);
  builder.CreateBr(forCondBB);

  builder.SetInsertPoint(loopEndBB);
  builder.CreateBr(oldExit);

  builder.SetInsertPoint(preheaderBB->getTerminator());

  llvm::Module& M = *preheaderBB->getParent()->getParent();

  int size_t_width;

  if (M.getPointerSize() == llvm::Module::Pointer64)
    {
      size_t_width = 64;
    }
  else if (M.getPointerSize() == llvm::Module::Pointer32) 
    {
      size_t_width = 32;
    }
  else 
    {
      assert (false && "Target has an unsupported pointer width.");
    }       

  builder.CreateStore
    (ConstantInt::get(IntegerType::get(C, size_t_width), 0), 
     localIdVar);

  builder.SetInsertPoint(forCondBB);
  llvm::Value *cmpResult = 
    builder.CreateICmpULT
    (builder.CreateLoad(localIdVar),
     (ConstantInt::get
      (IntegerType::get(C, size_t_width), 
       LocalSizeForDim)));
      
  builder.CreateCondBr(cmpResult, loopBodyEntryBB, loopEndBB);

  builder.SetInsertPoint(forIncBB->getTerminator());

  /* Create the iteration variable increment */
  builder.CreateStore
    (builder.CreateAdd
     (builder.CreateLoad(localIdVar),
      ConstantInt::get(IntegerType::get(C, size_t_width), 1)),
     localIdVar);
      
      
}

bool
WorkitemLoops::ProcessFunction(Function &F)
{
  Kernel *K = cast<Kernel> (&F);
  CheckLocalSize(K);

  unsigned workitem_count = LocalSizeZ * LocalSizeY * LocalSizeX;

  BasicBlockVector original_bbs;
  for (Function::iterator i = F.begin(), e = F.end(); i != e; ++i) {
      if (!Barrier::hasBarrier(i))
        original_bbs.push_back(i);
  }

  ParallelRegion::ParallelRegionVector* original_parallel_regions =
    K->getParallelRegions(LI);

  std::vector<SmallVector<ParallelRegion *, 8> > parallel_regions(workitem_count);

  for (ParallelRegion::ParallelRegionVector::iterator
           i = original_parallel_regions->begin(), 
           e = original_parallel_regions->end();
       i != e; ++i) 
  {
      ParallelRegion *region = (*i);
      llvm::Module *M = F.getParent();
      CreateLoopAround(region, M->getGlobalVariable("_local_id_z"), LocalSizeZ);
      CreateLoopAround(region, M->getGlobalVariable("_local_id_y"), LocalSizeY);
      CreateLoopAround(region, M->getGlobalVariable("_local_id_x"), LocalSizeX);
  }

  //F.viewCFG();

  return true;
}

