// Check CPU compatibility with masked host features.
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

#include "pocl_llvm_cpu.h"
#include "poclu.h"

#include <cstdlib>
#include <cstdio>

int main() {
  llvm::SmallVector<llvm::StringRef, 64> Features;
  llvm::X86::getFeaturesForCPU("znver4", Features);
  llvm::StringMap<bool> Host;
  for (auto F : Features)
    Host[F] = true;
  Host["cmov"] = true;
  Host.erase("x87");
  TEST_ASSERT(pocl_cpu_has_features("znver4", Host));
  TEST_ASSERT(pocl_cpu_has_features("skylake-avx512", Host));
  Host["avx512f"] = false;
  TEST_ASSERT(!pocl_cpu_has_features("znver4", Host));
  TEST_ASSERT(!pocl_cpu_has_features("skylake-avx512", Host));
  TEST_ASSERT(pocl_cpu_has_features("haswell", Host));
  Host.erase("avx");
  TEST_ASSERT(!pocl_cpu_has_features("haswell", Host));
  TEST_ASSERT(pocl_cpu_has_features("core2", Host));
  TEST_ASSERT(!pocl_cpu_has_features("not-a-real-cpu", Host));
  return EXIT_SUCCESS;
}
