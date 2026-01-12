/* ConvertMemrefToLLVMKernelArgs.cc - Wrap the memref-style function to
   pocl-compatible form

   Copyright (c) 2025 Topi Leppänen / Tampere University

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

#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include "pocl/Transforms/Passes.hh"

#include "pocl_util.h"

namespace {
#define GEN_PASS_DEF_CONVERTMEMREFTOLLVMKERNELARGS
#include "pocl/Transforms/Passes.h.inc"
} // namespace

namespace {
struct ConvertMemrefToLLVMKernelArgs
    : public impl::ConvertMemrefToLLVMKernelArgsBase<
          ConvertMemrefToLLVMKernelArgs> {

  void runOnOperation() override {
    mlir::ModuleOp Module = getOperation();
    mlir::MLIRContext *Context = Module.getContext();
    mlir::OpBuilder Builder(Context);
    Builder.setInsertionPointToEnd(Module.getBody());
    auto Loc = Builder.getUnknownLoc();

    // Define LLVM pointer and integer types
    auto PtrTy = mlir::LLVM::LLVMPointerType::get(Context);
    auto I64Ty = mlir::IntegerType::get(Context, 64);
    auto I32Ty = mlir::IntegerType::get(Context, 32);
    auto VoidTy = mlir::LLVM::LLVMVoidType::get(Context);

    mlir::SmallVector<mlir::LLVM::LLVMFuncOp> FuncOps;
    Module->walk([&](mlir::LLVM::LLVMFuncOp FuncOp) {
      if (FuncOp->hasAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName()))
        FuncOps.push_back(FuncOp);
    });
    mlir::LLVM::LLVMFuncOp KernFunc = *FuncOps.begin();

    // Define function type: (i8*, i8*, i64, i64, i64) -> void
    mlir::SmallVector<mlir::Type> ArgTypes = {
        PtrTy,              // args
        PtrTy,              // pocl_context_arg
        I64Ty, I64Ty, I64Ty // gid_x, gid_y, gid_z
    };

    auto Array3xI64Ty = mlir::LLVM::LLVMArrayType::get(Context, I64Ty, 3);

    auto PoclContextTy =
        mlir::LLVM::LLVMStructType::getLiteral(Context,
                                               {Array3xI64Ty, // NUM_GROUPS
                                                Array3xI64Ty, // GLOBAL_OFFSET
                                                Array3xI64Ty, // LOCAL_SIZE
                                                PtrTy,        // PRINTF_BUFFER
                                                PtrTy, // PRINTF_BUFFER_POSITION
                                                I32Ty, // PRINTF_BUFFER_CAPACITY
                                                PtrTy, // GLOBAL_VAR_BUFFER
                                                I32Ty, // WORK_DIM
                                                I32Ty}); // EXECUTION_FAILED
    auto FuncType = mlir::LLVM::LLVMFunctionType::get(VoidTy, ArgTypes);

    std::string KernelNameWithPrefix = std::string(KernFunc.getSymName());
    int PrefixLen = strlen("pocl_mlir_");
    std::string KernelName = KernelNameWithPrefix.substr(PrefixLen);
    std::string FuncName = "_pocl_kernel_" + KernelName + "_workgroup";
    auto Func =
        mlir::LLVM::LLVMFuncOp::create(Builder, Loc, FuncName, FuncType);
    auto &EntryBlock = *Func.addEntryBlock(Builder);
    Builder.setInsertionPointToStart(&EntryBlock);

    mlir::Value Args = EntryBlock.getArgument(0);
    mlir::Value ContextArg = EntryBlock.getArgument(1);
    mlir::Value GidX = EntryBlock.getArgument(2);
    mlir::Value GidY = EntryBlock.getArgument(3);
    mlir::Value GidZ = EntryBlock.getArgument(4);

    // Prepare arguments
    mlir::SmallVector<mlir::Value> KernArgs = {};
    mlir::Attribute Attr = KernFunc->getAttr("CL_arg_count");
    int64_t ClArgCount = 0;
    if (auto IntegerAttr = mlir::dyn_cast<mlir::IntegerAttr>(Attr)) {
      ClArgCount = IntegerAttr.getInt();
    } else {
      POCL_MSG_ERR(
          "Couldn't extract the CL_arg_count attribute from mlir bitcode\n");
      signalPassFailure();
    }

    int MLIRArgIndex = 0;
    for (int ClArgIndex = 0; ClArgIndex < ClArgCount; ClArgIndex++) {
      auto Arg = KernFunc.getArgument(MLIRArgIndex);
      if (mlir::isa<mlir::LLVM::LLVMPointerType>(Arg.getType())) {
        auto ZeroAttr = mlir::IntegerAttr::get(I64Ty, 0);
        auto IntZero =
            mlir::LLVM::ConstantOp::create(Builder, Loc, I64Ty, ZeroAttr);
        auto NullPtr = mlir::LLVM::ZeroOp::create(Builder, Loc, PtrTy);

        mlir::Value V1 = mlir::LLVM::LoadOp::create(
            Builder, Loc, PtrTy,
            mlir::LLVM::GEPOp::create(Builder, Loc, PtrTy, PtrTy, Args,
                                      mlir::ArrayRef<mlir::LLVM::GEPArg>{
                                          mlir::LLVM::GEPArg(ClArgIndex)}));

        mlir::Value V2 = mlir::LLVM::LoadOp::create(Builder, Loc, PtrTy, V1);

        KernArgs.push_back(NullPtr);
        KernArgs.push_back(V2);
        KernArgs.push_back(IntZero); // offset
        KernArgs.push_back(IntZero); // size
        KernArgs.push_back(IntZero); // stride

        MLIRArgIndex += 5;
      } else {
        auto ArgType = Arg.getType();
        mlir::Value V1 = mlir::LLVM::LoadOp::create(
            Builder, Loc, PtrTy,
            mlir::LLVM::GEPOp::create(Builder, Loc, PtrTy, PtrTy, Args,
                                      mlir::ArrayRef<mlir::LLVM::GEPArg>{
                                          mlir::LLVM::GEPArg(ClArgIndex)}));
        mlir::Value V2 = mlir::LLVM::LoadOp::create(Builder, Loc, ArgType, V1);
        KernArgs.push_back(V2);

        MLIRArgIndex++;
      }
    }

    int PcArgCount = PoclContextTy.getBody().size();
    int PcArgIndex = 0;
    for (; PcArgIndex < 3; PcArgIndex++) {
      auto ArrayGepOp = mlir::LLVM::GEPOp::create(
          Builder, Loc, PtrTy, PoclContextTy, ContextArg,
          mlir::ArrayRef<mlir::LLVM::GEPArg>{0, mlir::LLVM::GEPArg(PcArgIndex)},
          mlir::LLVM::GEPNoWrapFlags::inbounds);
      for (int TmpArrayIdx = 0; TmpArrayIdx < 3; TmpArrayIdx++) {
        auto GepOp = mlir::LLVM::GEPOp::create(
            Builder, Loc, PtrTy, Array3xI64Ty, ArrayGepOp,
            mlir::ArrayRef<mlir::LLVM::GEPArg>{0,
                                               mlir::LLVM::GEPArg(TmpArrayIdx)},
            mlir::LLVM::GEPNoWrapFlags::inbounds);
        auto ArgType = KernFunc.getArgumentTypes()[MLIRArgIndex];
        mlir::Value V1 =
            mlir::LLVM::LoadOp::create(Builder, Loc, ArgType, GepOp);
        KernArgs.push_back(V1);
        MLIRArgIndex++;
      }
    }
    for (; PcArgIndex < PcArgCount; PcArgIndex++) {
      auto GepOp = mlir::LLVM::GEPOp::create(
          Builder, Loc, PtrTy, PoclContextTy, ContextArg,
          mlir::ArrayRef<mlir::LLVM::GEPArg>{0, mlir::LLVM::GEPArg(PcArgIndex)},
          mlir::LLVM::GEPNoWrapFlags::inbounds);
      auto ArgType = KernFunc.getArgumentTypes()[MLIRArgIndex];
      mlir::Value V1 = mlir::LLVM::LoadOp::create(Builder, Loc, ArgType, GepOp);
      KernArgs.push_back(V1);
      MLIRArgIndex++;
    }

    KernArgs.push_back(GidX);
    KernArgs.push_back(GidY);
    KernArgs.push_back(GidZ);

    mlir::LLVM::CallOp::create(Builder, Loc, KernFunc, KernArgs);
    mlir::LLVM::ReturnOp::create(Builder, Loc, mlir::ValueRange{});
  }
};
} // namespace

std::unique_ptr<mlir::Pass>
mlir::pocl::createConvertMemrefToLLVMKernelArgsPass() {
  return std::make_unique<ConvertMemrefToLLVMKernelArgs>();
}
