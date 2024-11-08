// Lightweight bitcode linker to replace llvm::Linker.
//
// Copyright (c) 2014 Kalle Raiskila
//               2016-2022 Pekka Jääskeläinen
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

#ifndef POCL_LINKER_H
#define POCL_LINKER_H

#include "llvm/IR/Module.h"

#include "pocl_cl.h" // cl_device_id

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

/**
 * Link in module lib to krn.
 * This function searches for each undefined symbol 
 * in krn from lib, cloning as needed. For big modules,
 * this is faster than calling llvm::Linker and then
 * running DCE.
 *
 * log is used to report errors if we run into undefined symbols
 */
int link(llvm::Module *Program, const llvm::Module *Lib, std::string &Log,
         cl_device_id ClDev, bool StripAllDebugInfo);

int copyKernelFromBitcode(const char* Name, llvm::Module *ParallelBC,
                          const llvm::Module *Program,
                          const char **DevAuxFuncs);

bool moveProgramScopeVarsOutOfProgramBc(llvm::LLVMContext *Context,
                                        llvm::Module *ProgramBC,
                                        llvm::Module *OutputBC,
                                        unsigned DeviceLocalAS);

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif
