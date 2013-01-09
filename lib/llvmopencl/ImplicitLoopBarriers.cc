// LLVM function pass that adds implicit barriers to loops.
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

#include "config.h"
#include "ImplicitLoopBarriers.h"
#include "Barrier.h"
#include "Workgroup.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#if (defined LLVM_3_1 or defined LLVM_3_2)
#include "llvm/Constants.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#else
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#endif

using namespace llvm;
using namespace pocl;

namespace {
  static
  RegisterPass<ImplicitLoopBarriers> X("implicit-loop-barriers",
                                       "Adds implicit barriers to loops");
}

char ImplicitLoopBarriers::ID = 0;

void
ImplicitLoopBarriers::getAnalysisUsage(AnalysisUsage &AU) const
{
  AU.addRequired<DominatorTree>();
  AU.addPreserved<DominatorTree>();
}

bool
ImplicitLoopBarriers::runOnLoop(Loop *L, LPPassManager &LPM)
{
  if (!Workgroup::isKernelToProcess(*L->getHeader()->getParent()))
    return false;

  DT = &getAnalysis<DominatorTree>();

  bool changed = ProcessLoop(L, LPM);

  DT->verifyAnalysis();

  return changed;
}


/**
 * Adds a barrier to the first BB of each loop.
 *
 * Note: it's not safe to do this in case the loop is not executed
 * by all work items. Therefore this is not enabled by default.
 */
bool
ImplicitLoopBarriers::ProcessLoop(Loop *L, LPPassManager &LPM)
{
  assert (L->isLoopSimplifyForm());

  llvm::BasicBlock* firstBB = *L->block_begin();

  assert (firstBB != NULL);

  Barrier::Create(firstBB->getFirstNonPHI());
  return true;
}
