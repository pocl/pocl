// Header for DebugHelpers, tools for debugging the kernel compiler.
// 
// Copyright (c) 2013 Pekka Jääskeläinen / TUT
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

#ifndef _POCL_DEBUG_HELPERS_H
#define _POCL_DEBUG_HELPERS_H

#include <string>

#include "ParallelRegion.h"

#include "config.h"
#if (defined LLVM_3_1 || defined LLVM_3_2)
#include "llvm/Function.h"
#else
#include "llvm/IR/Function.h"
#endif
#include "llvm/Pass.h"

#if _MSC_VER
#  include <set>
#endif

namespace pocl {
  // View CFG with visual aids to debug kernel compiler problems.
  void dumpCFG(llvm::Function& F, std::string fname="", 
               ParallelRegion::ParallelRegionVector* regions=NULL,
               std::set<llvm::BasicBlock*> *highlights=NULL);

  // Split large basic blocks to smaller one so dot doesn't crash when
  // calling viewCFG on it. This should be fixed in LLVM upstream.
  //
  // @return True in case the function was changed.
  bool chopBBs(llvm::Function& F, llvm::Pass &P);
};

#endif
