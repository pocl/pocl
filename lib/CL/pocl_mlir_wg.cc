/* pocl_mlir_wg.cc: part of pocl MLIR API dealing with parallel.mlir,
   optimization passes and codegen.

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
#include "config.h"

#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/TargetParser/Host.h>

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMPass.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/LLVMCommon/LoweringOptions.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/VectorToSCF/VectorToSCF.h>
#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include "pocl/Transforms/Passes.hh"

#include "common.h"
#include "pocl.h"
#include "pocl_cache.h"
#include "pocl_debug.h"
#include "pocl_file_util.h"
#include "pocl_llvm_api.h"
#include "pocl_mlir.h"
#include "pocl_mlir_file_util.hh"
#include "pocl_mlir_passes.hh"

static void generateLlvmFunctionNowrite(mlir::OwningOpRef<mlir::ModuleOp> &Mod,
                                        mlir::MLIRContext *MLIRContext) {

  mlir::PassManager PMLower(MLIRContext);
  PMLower.addPass(mlir::pocl::createConvertAffineParallelToAffineForPass());
  if (mlir::failed(PMLower.run(*Mod))) {
    POCL_MSG_PRINT_LLVM("Failed lowering the affine parallel to affine for\n");
  }

  pocl::mlir::runAffinePasses(Mod, true);

  auto TargetTriple = llvm::sys::getDefaultTargetTriple();
  mlir::Operation *ModOp = Mod.get();
  ModOp->setAttr(mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
                 mlir::StringAttr::get(MLIRContext, TargetTriple));

  std::string Error;
  const llvm::Target *Target =
      llvm::TargetRegistry::lookupTarget(llvm::Triple(TargetTriple), Error);
  if (!Target) {
    POCL_ABORT("Failed getting the default target: %s\n", Error.c_str());
  }
  llvm::TargetOptions Opt;
  std::optional<llvm::Reloc::Model> RM = std::nullopt;
  llvm::TargetMachine *TargetMachine = Target->createTargetMachine(
      llvm::Triple(TargetTriple), "generic", "", Opt, RM);
  std::string DataLayout =
      TargetMachine->createDataLayout().getStringRepresentation();

  mlir::PassManager PMLLVM(MLIRContext);
  PMLLVM.addPass(mlir::createSetLLVMModuleDataLayoutPass({DataLayout}));
  PMLLVM.addPass(mlir::createLowerAffinePass());
  PMLLVM.addPass(mlir::createSCFToControlFlowPass());
  PMLLVM.addPass(mlir::pocl::createStripMemSpacesPass());
  PMLLVM.addPass(mlir::createConvertToLLVMPass());
  PMLLVM.addPass(mlir::createReconcileUnrealizedCastsPass());
  PMLLVM.addPass(mlir::pocl::createConvertMemrefToLLVMKernelArgsPass());

  if (mlir::failed(PMLLVM.run(*Mod))) {
    POCL_MSG_PRINT_LLVM("Failed running the MLIR-To-LLVM IR lowering passes\n");
  }
  return;
}

int pocl::mlir::runAffinePasses(mlir::OwningOpRef<mlir::ModuleOp> &Mod,
                                bool RaiseAffine) {
  mlir::GreedyRewriteConfig CanonicalizerConfig;
  CanonicalizerConfig.setMaxIterations(400);
  mlir::PassManager PMAffine(Mod->getContext());
  mlir::OpPassManager &OptPmAffine = PMAffine.nest<mlir::func::FuncOp>();
  PMAffine.addPass(mlir::createCanonicalizerPass());
  PMAffine.addPass(mlir::createCanonicalizerPass());
  PMAffine.addPass(mlir::createInlinerPass());
  PMAffine.addPass(mlir::createLoopInvariantCodeMotionPass());
  PMAffine.addNestedPass<mlir::func::FuncOp>(
      mlir::affine::createLoopCoalescingPass());
  PMAffine.addPass(mlir::affine::createLoopFusionPass());
  PMAffine.addPass(mlir::createCSEPass());
  PMAffine.addPass(mlir::createLoopInvariantCodeMotionPass());
  PMAffine.addPass(mlir::createMem2Reg());
  OptPmAffine.addPass(
      mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  PMAffine.addPass(mlir::createCanonicalizerPass());
  PMAffine.addPass(mlir::affine::createLoopFusionPass());
  PMAffine.addNestedPass<mlir::func::FuncOp>(
      mlir::affine::createLoopCoalescingPass());
  PMAffine.addPass(mlir::createCanonicalizerPass());
  if (mlir::failed(PMAffine.run(*Mod))) {
    Mod->dump();
    return CL_FAILED;
  }
  return CL_SUCCESS;
}

static int runPoclPasses(mlir::OwningOpRef<mlir::ModuleOp> &Mod,
                         mlir::MLIRContext *MLIRContext,
                         _cl_command_node *Command, int Specialize) {
  _cl_command_run *RunCommand = &Command->command.run;

  POCL_MEASURE_START(mlir_workgroup_ir_func_gen);

  // Set the specialization properties.
  if (Specialize) {
    assert(RunCommand);
    long WGLocalSizeX = RunCommand->pc.local_size[0];
    long WGLocalSizeY = RunCommand->pc.local_size[1];
    long WGLocalSizeZ = RunCommand->pc.local_size[2];
    mlir::DenseI64ArrayAttr LocalSize = mlir::DenseI64ArrayAttr::get(
        MLIRContext, {WGLocalSizeX, WGLocalSizeY, WGLocalSizeZ});
    (*Mod)->setAttr("gpu.workgroup_size", LocalSize);
  }

  mlir::GreedyRewriteConfig CanonicalizerConfig;
  CanonicalizerConfig.setMaxIterations(400);

  mlir::PassManager Pm(MLIRContext);
  mlir::OpPassManager &OptPm = Pm.nest<mlir::func::FuncOp>();
  Pm.addPass(mlir::createCanonicalizerPass());
  Pm.addPass(mlir::createInlinerPass());
  OptPm.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  OptPm.addPass(mlir::createCSEPass());
  OptPm.addPass(mlir::createMem2Reg());
  Pm.addPass(mlir::createCanonicalizerPass());

  Pm.addPass(mlir::pocl::createWorkgroupPass());

  Pm.addPass(mlir::createCanonicalizerPass());
  Pm.addPass(mlir::memref::createNormalizeMemRefsPass());
  Pm.addPass(mlir::createCanonicalizerPass());
  Pm.addPass(mlir::createSymbolDCEPass());
  OptPm.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  OptPm.addPass(mlir::createMem2Reg());
  OptPm.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  OptPm.addPass(mlir::createLoopInvariantCodeMotionPass());
  OptPm.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  OptPm.addPass(mlir::createCSEPass());
  OptPm.addPass(mlir::createMem2Reg());
  Pm.addPass(mlir::createSymbolDCEPass());
  Pm.addPass(mlir::createCanonicalizerPass());
  if (mlir::failed(Pm.run(*Mod))) {
    POCL_MSG_PRINT_LLVM("Failed running the MLIR compiler passes\n");
    return CL_FAILED;
  }
  if (mlir::failed(mlir::verify(*Mod))) {
    Mod->dump();
    return CL_FAILED;
  }

  if (pocl::mlir::runAffinePasses(Mod, false) == CL_FAILED) {
    POCL_MSG_PRINT_LLVM("Failed running the MLIR affine passes\n");
    return CL_FAILED;
  }

  // Remove all function atributes, TODO: Check if some would still be needed
  // At least the cir ones need to be removed, since hls tools may not know
  // about them
  for (auto Attr : (*Mod)->getAttrs()) {
    (*Mod)->removeAttr(Attr.getName());
  }

  POCL_MEASURE_FINISH(mlir_workgroup_ir_func_gen);

  return 0;
}

int poclMlirGenerateStandardWorkgroupFunctionNowrite(
    cl_kernel Kernel, _cl_command_node *Command,
    mlir::OwningOpRef<mlir::ModuleOp> &Module, int Specialize) {
  cl_context Ctx = Kernel->context;
  PoclLLVMContextData *PoCLLLVMContext =
      (PoclLLVMContextData *)Ctx->llvm_context_data;

  std::vector<mlir::func::FuncOp> FuncsToDelete;
  Module->walk([&](mlir::func::FuncOp Func) {
    auto IsKernel =
        Func->hasAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName());
    std::string FuncName = Func.getName().str();
    if (IsKernel && FuncName != std::string(Kernel->name)) {
      FuncsToDelete.push_back(Func);
    }
  });
  for (auto Func : FuncsToDelete) {
    Func.erase();
  }

  int Res =
      runPoclPasses(Module, PoCLLLVMContext->MLIRContext, Command, Specialize);
  return Res;
}

int poclMlirGenerateStandardWorkgroupFunction(
    unsigned DeviceI, cl_device_id Device, cl_kernel Kernel,
    _cl_command_node *Command, int Specialize, const char *Cachedir) {
  cl_context Ctx = Kernel->context;
  PoclLLVMContextData *PoCLLLVMContext =
      (PoclLLVMContextData *)Ctx->llvm_context_data;
  char ProgramMLIRPath[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_program_mlir_path(ProgramMLIRPath, Kernel->program, DeviceI);
  std::string FinalBinaryPath = Cachedir;
  FinalBinaryPath += POCL_PARALLEL_MLIR_FILENAME;

  if (pocl_exists(FinalBinaryPath.c_str()))
    return CL_SUCCESS;

  mlir::OwningOpRef<mlir::ModuleOp> InputModule;
  int Error = pocl::mlir::openFile(ProgramMLIRPath,
                                   PoCLLLVMContext->MLIRContext, InputModule);
  if (Error)
    return Error;

  POCL_MSG_PRINT_GENERAL("Calling generate standard function for kernel %s\n",
                         Kernel->name);

  Error = poclMlirGenerateStandardWorkgroupFunctionNowrite(
      Kernel, Command, InputModule, Specialize);
  if (Error)
    return Error;

  Error = pocl_mkdir_p(Cachedir);
  if (Error) {
    POCL_MSG_PRINT_GENERAL("Unable to create directory %s.\n", Cachedir);
    return Error;
  }
  return pocl::mlir::writeOutput(InputModule, FinalBinaryPath.c_str());
}

int poclMlirGenerateLlvmFunction(unsigned DeviceI, cl_device_id Device,
                                 cl_kernel Kernel, _cl_command_node *Command,
                                 int Specialize, const char *Cachedir) {
  cl_context Ctx = Kernel->context;
  PoclLLVMContextData *PoCLLLVMContext =
      (PoclLLVMContextData *)Ctx->llvm_context_data;
  std::string ParallelStdPath = Cachedir;
  ParallelStdPath += POCL_PARALLEL_MLIR_FILENAME;

  std::string FinalBinaryPath = Cachedir;
  FinalBinaryPath += POCL_PARALLEL_BC_FILENAME;

  if (pocl_exists(FinalBinaryPath.c_str()))
    return CL_SUCCESS;

  mlir::OwningOpRef<mlir::ModuleOp> InputModule;
  int Error = pocl::mlir::openFile(ParallelStdPath.c_str(),
                                   PoCLLLVMContext->MLIRContext, InputModule);
  if (Error)
    return Error;

  POCL_MSG_PRINT_GENERAL("Calling generate_llvm_function for kernel %s\n",
                         Kernel->name);

  generateLlvmFunctionNowrite(InputModule, PoCLLLVMContext->MLIRContext);

  std::string KernelLlvmMlirPath = Cachedir;
  KernelLlvmMlirPath += "/parallel_llvm.mlir";
  Error = pocl::mlir::writeOutput(InputModule, KernelLlvmMlirPath.c_str());
  if (Error)
    return Error;

  std::string KernelParallelLlPath = Cachedir;
  KernelParallelLlPath += POCL_PARALLEL_BC_FILENAME;

  std::string InvokeMlir = MLIRTRANSLATE_EXECUTABLE;
  InvokeMlir += " -o ";
  InvokeMlir += KernelParallelLlPath;
  InvokeMlir += " --mlir-to-llvmir ";
  InvokeMlir += KernelLlvmMlirPath;
  POCL_MSG_PRINT_LLVM("MLIR-Translate cmd: %s\n", InvokeMlir.c_str());
  system(InvokeMlir.c_str());

  return 0;
}
