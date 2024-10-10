/* OpenCL runtime library: Dynalib library utility functions implemented
   using the LLVM Support library.

   Copyright (c) 2024 Pekka Jääskeläinen / Intel Finland Oy

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include "pocl_dynlib.h"

#include <llvm/Support/DynamicLibrary.h>
#include <unordered_map>

using namespace llvm::sys;

// Needed book keeping data to implement the API.
struct DynlibData {
  DynamicLibrary DL;
  std::string Path;
};

std::unordered_map<void *, DynlibData> LoadedLibs;

void *pocl_dynlib_open(const char *Path, int, int) {
  std::string Err;
  DynamicLibrary DL = DynamicLibrary::getLibrary(Path, &Err);
  if (!DL.isValid()) {
    POCL_MSG_ERR("DynamicLibrary::getLibrary() failed: '%s'\n", Err.c_str());
    return NULL;
  }
  void *Handle = DL.getOSSpecificHandle();
  DynlibData D = {DL, Path};
  LoadedLibs[Handle] = D;
  return Handle;
}

int pocl_dynlib_close(void *Handle) {
  auto L = LoadedLibs.find(Handle);
  if (L == LoadedLibs.end())
    return 0;
  DynamicLibrary::closeLibrary((*L).second.DL);
  LoadedLibs.erase(L);
  return 1;
}

void *pocl_dynlib_symbol_address(void *, const char *SymbolName) {
  return DynamicLibrary::SearchForAddressOfSymbol(SymbolName);
}

const char *pocl_dynlib_pathname(void *Address) {
  POCL_ABORT_UNIMPLEMENTED("pocl_dynlib_pathname using LLVM");
}
