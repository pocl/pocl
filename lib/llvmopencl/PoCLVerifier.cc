// LLVM module pass that checks invariants of the PoCL kernel compiler.
//
// Copyright (c) 2026 Tim Besard
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/Twine.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/Support/ErrorHandling.h>

#include "Barrier.h"
#include "LLVMUtils.h"
#include "PoCLVerifier.h"
POP_COMPILER_DIAGS

#define PASS_NAME "pocl-verify"
#define PASS_CLASS pocl::PoCLVerifier
#define PASS_DESC "Checks invariants of the PoCL kernel compiler."

namespace pocl {

using namespace llvm;

// The work-group passes only handle barriers in the kernel itself, so every
// function that reaches one must have been inlined by now. A barrier left in
// a called function would be dropped, and the kernel would run without it.
static const Function *findCalledFunctionWithBarrier(const Module &M) {
  const Function *BarrierFn = M.getFunction(BARRIER_FUNCTION_NAME);
  if (BarrierFn == nullptr)
    return nullptr;
  for (const User *B : BarrierFn->users()) {
    const auto *I = dyn_cast<Instruction>(B);
    if (I == nullptr)
      continue;
    const Function *F = I->getFunction();
    for (const User *U : F->users())
      if (isa<CallBase>(U))
        return F;
  }
  return nullptr;
}

llvm::PreservedAnalyses PoCLVerifier::run(llvm::Module &M,
                                          llvm::ModuleAnalysisManager &AM) {
  if (const Function *F = findCalledFunctionWithBarrier(M))
    report_fatal_error(Twine("pocl-verify: failed to inline ") + F->getName() +
                       ", which calls a work-group barrier");
  return PreservedAnalyses::all();
}

REGISTER_NEW_MPASS(PASS_NAME, PASS_CLASS, PASS_DESC);

} // namespace pocl
