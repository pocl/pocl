// Lightweight bitcode linker for MLIR
//
// Copyright (c) 2025 Topi Leppänen
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

#ifndef POCL_TRANSFORMS_LINKER_H
#define POCL_TRANSFORMS_LINKER_H

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

/**
 * Link in module lib to krn.
 * This function searches for each undefined symbol
 * in krn from lib, cloning as needed.
 *
 */
int mlirLink (mlir::OwningOpRef<mlir::ModuleOp> &Program,
              mlir::OwningOpRef<mlir::ModuleOp> &Lib,
              std::string &Log,
              const char **DevAuxFuncs,
              bool DeviceSidePrintf);

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif
