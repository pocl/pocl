// OptimizeWorkItemFuncCalls is an LLVM pass to optimize calls to work-item
// functions like get_local_size().
//
// Copyright (c) 2017-2019 Pekka Jääskeläinen
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

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>

#include "LLVMUtils.h"
#include "OptimizeWorkItemFuncCalls.h"
#include "VariableUniformityAnalysis.h"
#include "WorkitemHandlerChooser.h"
POP_COMPILER_DIAGS

#include <iostream>
#include <map>
#include <set>

#define PASS_NAME "optimize-wi-func-calls"
#define PASS_CLASS pocl::OptimizeWorkItemFuncCalls
#define PASS_DESC "Optimize work-item function calls."

namespace pocl {

using namespace llvm;

static bool optimizeWorkItemFuncCalls(Function &F) {

  // Let's avoid reoptimizing pocl_printf in the kernel compiler. It should
  // be optimized already in the bitcode library, and we do not want to
  // aggressively inline it to the kernel, causing compile time expansion.
  if (F.getName().starts_with("__pocl_print") &&
      !F.hasFnAttribute(Attribute::OptimizeNone)) {
    F.addFnAttr(Attribute::OptimizeNone);
    F.addFnAttr(Attribute::NoInline);
  }
  return false;
}

llvm::PreservedAnalyses
OptimizeWorkItemFuncCalls::run(llvm::Function &F,
                               llvm::FunctionAnalysisManager &AM) {
  PreservedAnalyses PAChanged = PreservedAnalyses::none();
  PAChanged.preserve<WorkitemHandlerChooser>();

  return optimizeWorkItemFuncCalls(F) ? PAChanged : PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
