// Removes optnone keyword from get_global_id().
//
// Copyright (c) 2017 Michal Babej / TUT
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
#include "LLVMUtils.h"
#include "RemoveOptnoneFromWIFunc.h"
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
POP_COMPILER_DIAGS

#include <iostream>

#define PASS_NAME "remove-optnone"
#define PASS_CLASS pocl::RemoveOptnoneFromWIFunc
#define PASS_DESC "Remove optnone keyword from workitem functions."

namespace pocl {

using namespace llvm;

static bool removeOptnoneFromWIFunc(Function &F) {
  /* Adding "optnone" to get_global_id() solves the problem
   * that some pass in opt introduces switch tables which the
   * variable uniformity analysis cannot analyze.
   *
   * However having optnone prevents some later optimizations
   * and creates problems in certain workitem tests.
   */
  const char *name = "_Z13get_global_idj";
  StringRef nameref(name);
  bool changed = false;

  if (F.getName().equals(nameref)) {
    F.removeFnAttr(Attribute::AttrKind::OptimizeNone);
    changed = true;
  }
  return changed;
}

#if LLVM_MAJOR < MIN_LLVM_NEW_PASSMANAGER
char RemoveOptnoneFromWIFunc::ID = 0;

bool RemoveOptnoneFromWIFunc::runOnFunction(Function &F) {
  return removeOptnoneFromWIFunc(F);
}

void RemoveOptnoneFromWIFunc::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

REGISTER_OLD_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

#else

llvm::PreservedAnalyses
RemoveOptnoneFromWIFunc::run(llvm::Function &F,
                             llvm::FunctionAnalysisManager &AM) {
  removeOptnoneFromWIFunc(F);
  return PreservedAnalyses::all();
}

REGISTER_NEW_FPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

#endif

} // namespace pocl
