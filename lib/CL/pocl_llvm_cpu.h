// Check x86 CPU feature requirements.
//
// Copyright (c) 2026 PoCL developers
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

#ifndef POCL_LLVM_CPU_H
#define POCL_LLVM_CPU_H

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/TargetParser/X86TargetParser.h>

/* Check all CPU defaults, including features absent from the host map. */
inline bool
pocl_cpu_has_features(llvm::StringRef CPU,
                      const llvm::StringMap<bool> &Available) {
  if (llvm::X86::parseArchX86(CPU) == llvm::X86::CK_None)
    return false;
  llvm::SmallVector<llvm::StringRef, 64> Required;
  llvm::X86::getFeaturesForCPU(CPU, Required);
  for (llvm::StringRef F : Required) {
    /* LLVM omits x87 from host detection; it is baseline for host devices. */
    if (F != "x87" && !Available.lookup(F))
      return false;
  }
  return true;
}

#endif
