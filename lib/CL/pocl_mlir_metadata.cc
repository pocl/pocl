/* pocl_mlir_metadata.cc: part of pocl MLIR API dealing with kernel metadata.

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

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Casting.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/IR/BuiltinOps.h>

#include <string>

#include "pocl.h"
#include "pocl_cache.h"
#include "pocl_cl.h"
#include "pocl_file_util.h"
#include "pocl_llvm_api.h"
#include "pocl_mlir.h"
#include "pocl_mlir_file_util.hh"

int poclMlirGetKernelsMetadata(cl_program Program, unsigned DeviceI) {

  cl_context Ctx = Program->context;
  PoclLLVMContextData *LlvmCtx = (PoclLLVMContextData *)Ctx->llvm_context_data;
  PoclCompilerMutexGuard LockHolder(&LlvmCtx->Lock);

  char MlirPath[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_program_mlir_path(MlirPath, Program, DeviceI);
  if (!pocl_exists(MlirPath)) {
    POCL_MSG_ERR("Mlir IR %s does not exist\n", MlirPath);
    return CL_FAILED;
  }
  mlir::OwningOpRef<mlir::ModuleOp> MlirMod;
  if (pocl::mlir::openFile(MlirPath, LlvmCtx->MLIRContext, MlirMod)) {
    POCL_MSG_ERR("Failed to open mlir input file %s to parse metadata\n",
                 MlirPath);
    return CL_FAILED;
  }
  int KernelCount = 0;
  MlirMod->walk([&](mlir::func::FuncOp FuncOp) {
    if (FuncOp->hasAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName())) {
      pocl_kernel_metadata_t *Meta = &Program->kernel_meta[KernelCount];

      Meta->data = (void **)calloc(Program->num_devices, sizeof(void *));
      Meta->num_args = FuncOp.getFunctionBody().getNumArguments();
      Meta->name = strdup(FuncOp.getName().str().c_str());
      Meta->num_locals = 0;
      Meta->has_arg_metadata = 0;
      Meta->arg_info = (struct pocl_argument_info *)calloc(
          Meta->num_args, sizeof(struct pocl_argument_info));
      memset(Meta->arg_info, 0,
             sizeof(struct pocl_argument_info) * Meta->num_args);
      for (unsigned int I = 0; I < Meta->num_args; I++) {
        auto ArgInfo = FuncOp.getFunctionBody().getArgumentTypes()[I];
        std::string TypeName;
        llvm::raw_string_ostream Rso(TypeName);
        ArgInfo.print(Rso);
        Meta->arg_info[I].type_qualifier = 0;
        if (auto ArgInfoMemref = mlir::dyn_cast<mlir::MemRefType>(ArgInfo)) {
          Meta->arg_info[I].type_name = strdup(Rso.str().c_str());
          Meta->arg_info[I].type = POCL_ARG_TYPE_POINTER;
          Meta->arg_info[I].address_qualifier = CL_KERNEL_ARG_ADDRESS_GLOBAL;
          if (auto Ia = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
                  ArgInfoMemref.getMemorySpace()))
            if (Ia.getValue() == 5)
              Meta->arg_info[I].address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL;
          Meta->arg_info[I].access_qualifier = CL_KERNEL_ARG_ACCESS_READ_WRITE;
          Meta->arg_info[I].type_size = sizeof(void *);
        } else {
          Meta->arg_info[I].type = POCL_ARG_TYPE_NONE;
          auto SizeBits = ArgInfo.getIntOrFloatBitWidth();
          Meta->arg_info[I].type_size = SizeBits / 8;
          const int MaxTypeLength = 64;
          Meta->arg_info[I].type_name = (char *)calloc(1, MaxTypeLength);
          if (ArgInfo.isSignedInteger(32)) {
            snprintf(Meta->arg_info[I].type_name, MaxTypeLength, "int");
          } else if (ArgInfo.isInteger(32)) {
            snprintf(Meta->arg_info[I].type_name, MaxTypeLength, "unsigned");
          } else if (ArgInfo.isSignedInteger(64)) {
            snprintf(Meta->arg_info[I].type_name, MaxTypeLength, "long int");
          } else if (ArgInfo.isInteger(64)) {
            snprintf(Meta->arg_info[I].type_name, MaxTypeLength,
                     "unsigned long");
          } else if (ArgInfo.isF64()) {
            snprintf(Meta->arg_info[I].type_name, MaxTypeLength, "double");
          } else if (ArgInfo.isF32()) {
            snprintf(Meta->arg_info[I].type_name, MaxTypeLength, "float");
          } else if (ArgInfo.isF16()) {
            snprintf(Meta->arg_info[I].type_name, MaxTypeLength, "half");
          } else {
            POCL_MSG_WARN("!! Unknown MLIR argument type\n");
            snprintf(Meta->arg_info[I].type_name, MaxTypeLength,
                     "UNKNOWN_TYPE");
          }
        }
      }
      KernelCount++;
    }
  });
  return CL_SUCCESS;
}

unsigned poclMlirGetKernelCount(cl_program Program, unsigned DeviceI) {

  cl_context Ctx = Program->context;
  PoclLLVMContextData *LlvmCtx = (PoclLLVMContextData *)Ctx->llvm_context_data;
  PoclCompilerMutexGuard LockHolder(&LlvmCtx->Lock);

  char MlirPath[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_program_mlir_path(MlirPath, Program, DeviceI);
  if (!pocl_exists(MlirPath)) {
    POCL_MSG_ERR("Mlir IR %s does not exist\n", MlirPath);
    return 0;
  }
  mlir::OwningOpRef<mlir::ModuleOp> MlirMod;
  if (pocl::mlir::openFile(MlirPath, LlvmCtx->MLIRContext, MlirMod)) {
    POCL_MSG_ERR("Can't parse %s file in mlir_get_kernel_count\n", MlirPath);
    return 0;
  }
  int KernelCount = 0;
  MlirMod->walk([&](mlir::func::FuncOp FuncOp) {
    if (FuncOp->hasAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName())) {
      KernelCount++;
    }
  });
  POCL_MSG_PRINT_LLVM("Program has %d kernels\n", KernelCount);
  return KernelCount;
}
