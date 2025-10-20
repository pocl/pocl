/* pocl_mlir_file_util.cc: mlir file functions

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

#include "pocl_mlir_file_util.hh"

#include "pocl_cl.h"
#include "pocl_debug.h"

#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/FileUtilities.h>

int pocl::mlir::writeOutput(mlir::OwningOpRef<mlir::ModuleOp> &Module,
                            const char *OutputPath) {
  POCL_MSG_PRINT_GENERAL("Writing mlir module to %s\n", OutputPath);
  auto Output = mlir::openOutputFile(OutputPath);
  Module->print(Output->os());
  if (std::error_code Ec = Output->os().error()) {
    POCL_MSG_ERR("Failed to write to file %s: Error message %s\n", OutputPath,
                 Ec.message().c_str());
    return CL_FAILED;
  }
  Output->keep();
  return 0;
}

int pocl::mlir::openFile(const char *InputPath, mlir::MLIRContext *MlirContext,
                         mlir::OwningOpRef<mlir::ModuleOp> &Mod) {
  llvm::SourceMgr SourceMgr;
  auto InputFile = mlir::openInputFile(InputPath);
  if (!InputFile) {
    POCL_MSG_WARN("Failed to open file: %s\n", InputPath);
    return CL_FAILED;
  }
  SourceMgr.AddNewSourceBuffer(std::move(InputFile), mlir::SMLoc());
  Mod = mlir::parseSourceFile<mlir::ModuleOp>(SourceMgr, MlirContext);
  if (!Mod) {
    POCL_MSG_WARN("Can't parse mlir file: %s\n", InputPath);
    return CL_FAILED;
  }
  return 0;
}
