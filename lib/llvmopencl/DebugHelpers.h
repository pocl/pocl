// Header for DebugHelpers, tools for debugging the kernel compiler.
//
// Copyright (c) 2019 Pekka Jääskeläinen
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

#ifndef POCL_DEBUG_HELPERS_H
#define POCL_DEBUG_HELPERS_H

#include "config.h"

#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Pass.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/IR/Function.h>

namespace llvm {
class Region;
}

#include "ParallelRegion.h"

#include <set>
#include <string>

namespace pocl {
  // View CFG with visual aids to debug kernel compiler problems.
void dumpCFG(llvm::Function &F, std::string fname = "",
             const std::vector<llvm::Region *> *Regions = nullptr,
             const ParallelRegion::ParallelRegionVector *ParRegions = nullptr,
             const std::set<llvm::BasicBlock *> *highlights = nullptr);

// Split large basic blocks to smaller one so dot doesn't crash when
// calling viewCFG on it. This should be fixed in LLVM upstream.
//
// @return True in case the function was changed.
bool chopBBs (llvm::Function &F, llvm::Pass &P);

  class PoCLCFGPrinter : public llvm::PassInfoMixin<PoCLCFGPrinter> {
  public:
    explicit PoCLCFGPrinter(llvm::raw_ostream &OutS, llvm::StringRef Pref = "")
        : OS(OutS) {
      Prefix = Pref.str();
      Prefix += "_";
    }
    static void registerWithPB(llvm::PassBuilder &B);
    llvm::PreservedAnalyses run(llvm::Module &M,
                                llvm::ModuleAnalysisManager &AM);
    static bool isRequired() { return true; }

  private:
    std::string Prefix;
    llvm::raw_ostream &OS;
    void dumpModule(llvm::Module &M);
  };
};

// Controls the debug output from Kernel.cc parallel region generation:
//#define DEBUG_PR_CREATION

// Controls the debug output from ImplicitConditionalBarriers.cc:
//#define DEBUG_COND_BARRIERS

// Controls the debug output from PHIsToAllocas.cc
//#define DEBUG_PHIS_TO_ALLOCAS

#endif
