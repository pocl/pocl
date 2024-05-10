// Header for WorkitemHandlerChooser function pass.
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

#ifndef POCL_WORKITEM_HANDLER_CHOOSER_H
#define POCL_WORKITEM_HANDLER_CHOOSER_H

#include "config.h"

#include <llvm/Passes/PassBuilder.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Pass.h>

#include "WorkitemHandler.h"

namespace pocl {

enum class WorkitemHandlerType { FULL_REPLICATION, LOOPS, CBS, INVALID };
// this is required because we can only return class/struct from LLVM Analysis
struct WorkitemHandlerResult {
  WorkitemHandlerType WIH;
  WorkitemHandlerResult() : WIH(WorkitemHandlerType::LOOPS) {}
  WorkitemHandlerResult(WorkitemHandlerType A) : WIH(A) {}
  bool invalidate(llvm::Function &F, const llvm::PreservedAnalyses PA,
                  llvm::AnalysisManager<llvm::Function>::Invalidator &Inv);
};

class WorkitemHandlerChooser
    : public llvm::AnalysisInfoMixin<WorkitemHandlerChooser> {
public:
  static llvm::AnalysisKey Key;
  using Result = WorkitemHandlerResult;

  static void registerWithPB(llvm::PassBuilder &PB);
  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};
} // namespace pocl

#endif
