/* pocl_mlir_opt.cpp: Create pocl-mlir-opt binary for independently applying
   MLIR passes

   Copyright (c) 2025 Topi Leppänen / Tampere University

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include <mlir/Dialect/Func/Extensions/InlinerExtension.h>
#include <mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "pocl/Dialect/Dialect.hh"
#include "pocl/Transforms/Passes.hh"

int main(int argc, char **argv) {
  mlir::DialectRegistry Registry;

  mlir::registerAllDialects(Registry);
  mlir::registerAllPasses();
  mlir::registerAllExtensions(Registry);

  Registry.insert<mlir::pocl::PoclDialect>();
  mlir::pocl::registerPoclTransformsPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "PoCL MLIR Optimization Tool", Registry));
}
