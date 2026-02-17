/* Passes.td

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

#ifndef POCL_MLIR_PASSES_TD
#define POCL_MLIR_PASSES_TD

include "mlir/Pass/PassBase.td"

def Workgroup : Pass<"pocl-workgroup", "mlir::ModuleOp"> {
	let summary = "DeSPMD conversion to add an affine.parallel operation around the"
		      " SPMD-program based on the gpu.workgroup_size attribute.";
	let constructor = "mlir::pocl::createWorkgroupPass()";
	let dependentDialects = [
	"mlir::affine::AffineDialect",
	"mlir::memref::MemRefDialect",
	"mlir::func::FuncDialect",
	"mlir::arith::ArithDialect",
	"mlir::gpu::GPUDialect",
	];
}

def ConvertAffineParallelToAffineFor : Pass<"pocl-affine-parallel-to-for", "mlir::ModuleOp"> {
	let summary = "Converts each dimension of parallel-op to for-operations";
	let constructor = "mlir::pocl::createConvertAffineParallelToAffineForPass()";
	let dependentDialects = [
	"mlir::affine::AffineDialect",
	];
}

def ConvertMemrefToLLVMKernelArgs : Pass<"pocl-convert-memref-to-llvm-kernel-args", "mlir::ModuleOp"> {
	let summary = "Wrapper to convert the lowered memref arguments to pocl's CPU device compatible function";
	let constructor = "mlir::pocl::createConvertMemrefToLLVMKernelArgsPass()";
	let dependentDialects = [
	"mlir::LLVM::LLVMDialect",
	"mlir::gpu::GPUDialect",
	];
}

def Linker : Pass<"pocl-linker", "mlir::ModuleOp"> {
	let summary = "Link in the builtin library."
	"NOTE: Not callable from CLI, as it requires external BuiltinLib argument";
	let constructor = "mlir::pocl::createLinkerPass(mlir::ModuleOp())";
	let dependentDialects = [
	"mlir::func::FuncDialect",
	];
}

def LowerOpenCLBuiltins : Pass<"pocl-lower-opencl-builtins", "mlir::ModuleOp"> {
	let summary = "Lower OpenCL builtin function calls to MLIR dialect-based implementations";
	let constructor = "mlir::pocl::createLowerOpenCLBuiltinsPass()";
	let dependentDialects = [
	"mlir::arith::ArithDialect",
	"mlir::func::FuncDialect",
	"mlir::gpu::GPUDialect",
	"mlir::scf::SCFDialect",
	"mlir::pocl::PoclDialect",
	];
}

def StripMemSpaces : Pass<"pocl-strip-memspaces", "mlir::ModuleOp"> {
	let summary = "TODO";
	let constructor = "mlir::pocl::createStripMemSpacesPass()";
	let dependentDialects = [
	"mlir::memref::MemRefDialect",
	];
}

#endif
