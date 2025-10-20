/* pocl_mlir_utils.cc: helpers for pocl MLIR API.

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

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/IndexToLLVM/IndexToLLVM.h>
#include <mlir/Conversion/MathToLLVM/MathToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/UBToLLVM/UBToLLVM.h>
#include <mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Func/Extensions/InlinerExtension.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Polygeist/Dialect/Dialect.h>

#include "pocl/Dialect/Dialect.hh"

#include "pocl_llvm.h"
#include "pocl_llvm_api.h"
#include "pocl_mlir.h"

void poclMlirRegisterDialects(PoclLLVMContextData *Data) {

  Data->MLIRContext = new mlir::MLIRContext();
  assert(Data->MLIRContext);
  mlir::DialectRegistry Registry;
  Registry.insert<mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                  mlir::affine::AffineDialect, mlir::math::MathDialect,
                  mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                  mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect,
                  mlir::index::IndexDialect, mlir::vector::VectorDialect,
                  mlir::DLTIDialect, cir::CIRDialect,
                  mlir::polygeist::PolygeistDialect, mlir::pocl::PoclDialect>();
  mlir::func::registerInlinerExtension(Registry);
  mlir::LLVM::registerInlinerInterface(Registry);
  // MLIR to LLVM conversion registrations:
  mlir::arith::registerConvertArithToLLVMInterface(Registry);
  mlir::ub::registerConvertUBToLLVMInterface(Registry);
  mlir::registerConvertFuncToLLVMInterface(Registry);
  mlir::registerConvertMathToLLVMInterface(Registry);
  mlir::index::registerConvertIndexToLLVMInterface(Registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(Registry);
  mlir::registerConvertMemRefToLLVMInterface(Registry);
  mlir::vector::registerConvertVectorToLLVMInterface(Registry);

  Data->MLIRContext->appendDialectRegistry(Registry);
}
