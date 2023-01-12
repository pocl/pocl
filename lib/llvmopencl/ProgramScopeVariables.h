// Header for ProgramScopeVariables pass.
//
// Copyright (c) 2022 Michal Babej / Intel Finland Oy
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

#ifndef _POCL_PROGRAM_SCOPE_VARS_H
#define _POCL_PROGRAM_SCOPE_VARS_H

#include <llvm/IR/Module.h>

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

int runProgramScopeVariablesPass(llvm::Module *Program, unsigned DeviceGlobalAS,
                                 size_t &TotalGVarSize, std::string &Log);

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif
