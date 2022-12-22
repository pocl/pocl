// unify print function signature
//
// Copyright (c) 2022 Pekka Jääskeläinen
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

// Printf requires special treatment at bitcode link time for seamless SPIR-V
// import support: The printf we link in might be SPIR-V compliant with the
// format string address space in the constant space or the other way around.
// The calls to the printf, however, depend whether we are importing the kernel
// from SPIR or through OpenCL C native compilation path. In the former, the
// kernel calls refer to constant address space in the format string, and in the
// latter, when compiling natively to CPUs and other flat address space targets,
// the calls see an AS0 format string address space due to Clang's printf
// declaration adhering to target address spaces.
//
// In this function we fix calls to the printf to refer to one in the bitcode
// library's printf, with the correct AS for the format string. Other considered
// options include building two different bitcode libraries: One with SPIR-V
// address spaces, another with the target's (flat) AS. This would be
// problematic in other ways and redundant.

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wno-maybe-uninitialized"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#pragma GCC diagnostic pop

#include "UnifyPrintf.h"

using namespace llvm;

int unifyPrintfFingerPrint(Module *Program, const Module *Lib) {

  Function *CalledPrintf = Program->getFunction("printf");
  Function *LibPrintf = Lib->getFunction("printf");
  Function *NewPrintf = nullptr;

  assert(LibPrintf != nullptr);
  if (CalledPrintf != nullptr &&
      CalledPrintf->getArg(0)->getType() != LibPrintf->getArg(0)->getType()) {
    CalledPrintf->setName("_old_printf");
    // Create a declaration with a fingerprint with the correct format argument
    // type which we will import from the BC library.
    NewPrintf = Function::Create(LibPrintf->getFunctionType(),
                                 LibPrintf->getLinkage(), "printf", Program);
  } else {
    // No printf fingerprint mismatch detected in this module.
    return 0;
  }

  // Fix the printf calls to point to the library imported declaration.
  while (CalledPrintf->getNumUses() > 0) {
    auto U = CalledPrintf->user_begin();
    CallInst *Call = dyn_cast<CallInst>(*U);
    if (Call == nullptr)
      continue;
    auto Cast = CastInst::CreatePointerBitCastOrAddrSpaceCast(
        Call->getArgOperand(0), NewPrintf->getArg(0)->getType(), "fmt_str_cast",
        Call);
    Call->setCalledFunction(NewPrintf);
    Call->setArgOperand(0, Cast);
  }
  CalledPrintf->eraseFromParent();

  return 0;
}
