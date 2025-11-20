/* pocl_mlir_build.cc: part of pocl's MLIR API which deals with
   producing program.mlir

   Copyright (c) 2025 Topi Leppanen

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

#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

#include "pocl.h"
#include "pocl_cache.h"
#include "pocl_cl.h"
#include "pocl_debug.h"
#include "pocl_file_util.h"
#include "pocl_llvm_api.h"
#include "pocl_mlir.h"
#include "pocl_mlir_file_util.hh"
#include "pocl_util.h"

#include "pocl/Transforms/Linker.h"

static int getMLIRKernelLibrary(cl_device_id Device,
                                PoclLLVMContextData *LlvmCtx,
                                mlir::OwningOpRef<mlir::ModuleOp> &Lib) {
  std::string KernellibCommon, Kernellib, KernellibFallback;

#ifdef ENABLE_POCL_BUILDING
  if (pocl_get_bool_option("POCL_BUILDING", 0)) {
    KernellibCommon = BUILDDIR;
    KernellibCommon += "/lib/kernel/mlir";
  } else // POCL_BUILDING == 0, use install dir
#endif
  {
    char Temp[POCL_MAX_PATHNAME_LENGTH];
    pocl_get_private_datadir(Temp);
    KernellibCommon = Temp;
  }

  KernellibCommon += "/";
  Kernellib = KernellibCommon + "kernel-mlirbc";
  Kernellib += ".mlir";

  if (Device->kernellib_fallback_name) {
    KernellibFallback = KernellibCommon + Device->kernellib_fallback_name;
    KernellibFallback += ".mlir";
  }

  int ReadStatus = 0;
  if (pocl_exists(Kernellib.c_str())) {
    POCL_MSG_PRINT_LLVM("Using %s as the built-in lib.\n", Kernellib.c_str());
    ReadStatus = poclMlirOpenFile(Kernellib.c_str(), LlvmCtx->MLIRContext, Lib);
  } else {
    if (Device->kernellib_fallback_name &&
        pocl_exists(KernellibFallback.c_str())) {
      POCL_MSG_WARN("Using fallback %s as the built-in lib.\n",
                    KernellibFallback.c_str());
      ReadStatus = poclMlirOpenFile(KernellibFallback.c_str(),
                                    LlvmCtx->MLIRContext, Lib);
    } else
      POCL_ABORT("Kernel library file %s doesn't exist.\n", Kernellib.c_str());
  }
  return ReadStatus;
}

static bool generateProgramMLIR(PoclLLVMContextData *Context,
                                mlir::OwningOpRef<mlir::ModuleOp> &Mod,
                                cl_device_id Device, std::string &Log) {

  mlir::OwningOpRef<mlir::ModuleOp> BuiltinLib;
  auto Error = getMLIRKernelLibrary(Device, Context, BuiltinLib);
  if (Error) {
    POCL_MSG_ERR("Failed retrieving kernel library with error code %d\n",
                 Error);
  }

  if (mlirLink(Mod, BuiltinLib, Log, Device->device_aux_functions,
               Device->device_side_printf != CL_FALSE))
    return true;

  return false;
}

POCL_EXPORT
int poclMlirBuildProgram(cl_program Program, unsigned DeviceI,
                         cl_uint NumInputHeaders,
                         const cl_program *InputHeaders,
                         const char **HeaderIncludeNames, int LinkingProgram) {
  char ProgramMlirPath[POCL_MAX_PATHNAME_LENGTH];
  char TempIncludeDir[POCL_MAX_PATHNAME_LENGTH];
  std::string UserOptions(Program->compiler_options ? Program->compiler_options
                                                    : "");

  size_t N = 0;
  int Error;
  cl_context Ctx = Program->context;
  PoclLLVMContextData *LlvmCtx = (PoclLLVMContextData *)Ctx->llvm_context_data;

  if (NumInputHeaders > 0) {
    Error = pocl_cache_create_tempdir(TempIncludeDir);
    if (Error) {
      POCL_MSG_ERR("pocl_cache_create_tempdir (%s)"
                   " failed with %i\n",
                   TempIncludeDir, Error);
      return Error;
    }
    std::string Tempdir(TempIncludeDir);

    for (N = 0; N < NumInputHeaders; N++) {
      char *InputHeader = InputHeaders[N]->source;
      size_t InputHeaderSize = strlen(InputHeader);
      const char *HeaderName = HeaderIncludeNames[N];
      std::string Header(HeaderName);
      /* TODO this path stuff should be in utils */
      std::string Path(Tempdir);
      Path.append("/");
      Path.append(HeaderName);
      size_t LastSlash = Header.rfind('/');
      if (LastSlash != std::string::npos) {
        std::string Dir(Path, 0, (Tempdir.size() + 1 + LastSlash));
        pocl_mkdir_p(Dir.c_str());
      }
      pocl_write_file(Path.c_str(), InputHeader, InputHeaderSize, 0);
    }
  }

  cl_device_id Device = Program->devices[DeviceI];

  pocl_cache_create_program_cachedir(Program, DeviceI, Program->source,
                                     strlen(Program->source), ProgramMlirPath);
  pocl_cache_program_mlir_path(ProgramMlirPath, Program, DeviceI);

  if (pocl_exists(ProgramMlirPath)) {
    return CL_SUCCESS;
  }

  char SourceFile[POCL_MAX_PATHNAME_LENGTH];
  POCL_RETURN_ERROR_ON(pocl_cache_write_program_source(SourceFile, Program),
                       CL_OUT_OF_HOST_MEMORY, "Could not write program source");

  std::string CgeistCommand = "";
#ifdef ENABLE_POLYGEIST
  CgeistCommand += POLYGEIST_EXECUTABLE;
  CgeistCommand += std::string(" -include ") + SRCDIR + "/include/_kernel.h ";
  CgeistCommand += "-DPOCL_DEVICE_ADDRESS_BITS=32 -D__OPENCL_C_VERSION__=120 ";
  CgeistCommand += std::string("-I") + LLVM_LIBDIR + "/clang/22/include/ ";
  CgeistCommand += "-I. ";
  CgeistCommand += UserOptions + " ";
  CgeistCommand += "-function=* ";
  CgeistCommand += "-S -O3 -scal-rep=false ";
  CgeistCommand += SourceFile;
  CgeistCommand += " > ";
  CgeistCommand += ProgramMlirPath;

#else
  CgeistCommand += CLANGCC;
  CgeistCommand += std::string(" -include ") + SRCDIR + "/include/_kernel.h ";
  CgeistCommand += UserOptions + " ";
  CgeistCommand += "-fclangir -emit-cir ";
  CgeistCommand += SourceFile;
  /*-cir-flatten-cfg*/
  CgeistCommand += " -o - |";
  CgeistCommand += CIROPT_EXECUTABLE;
  CgeistCommand += " -mem2reg -cir-mlir-scf-prepare -cir-to-mlir -mem2reg -cse "
                   "-canonicalize -o ";
  CgeistCommand += ProgramMlirPath;
#endif

  POCL_MSG_PRINT_LLVM("Mlir frontend compiler command:%s\n",
                      CgeistCommand.c_str());
  system(CgeistCommand.c_str());

  mlir::OwningOpRef<mlir::ModuleOp> MlirMod;
  auto ParsingStatus =
      poclMlirOpenFile(ProgramMlirPath, LlvmCtx->MLIRContext, MlirMod);
  if (ParsingStatus) {
    POCL_ABORT("Can't parse program.mlir file in build_program\n");
  }

  std::string Log("Error(s) while linking: \n");
  generateProgramMLIR(LlvmCtx, MlirMod, Device, Log);

  std::string CgeistOutputPath = ProgramMlirPath;
  CgeistOutputPath += ".cgeist";
  pocl_rename(ProgramMlirPath, CgeistOutputPath.c_str());
  poclMlirWriteOutput(MlirMod, ProgramMlirPath);

  cl_int Status = CL_SUCCESS;

  return Status;
}

int poclMlirLinkProgram(cl_program Program, unsigned DeviceI,
                        cl_uint NumInputPrograms,
                        unsigned char **CurDeviceBinaries,
                        size_t *CurDeviceBinarySizes, void **CurLlvmIrs,
                        int LinkDeviceBuiltinLibrary,
                        int LinkingIntoNewClProgram) {
  return CL_SUCCESS;
}

int poclMlirGenerateWorkgroupFunctionNowrite(unsigned DeviceI,
                                             cl_device_id Device,
                                             cl_kernel Kernel,
                                             _cl_command_node *Command,
                                             void **Output, int Specialize,
                                             cl_program Program) {
  cl_context Ctx = Program->context;
  PoclLLVMContextData *LLVMCtx = (PoclLLVMContextData *)Ctx->llvm_context_data;

  char Cachedir[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_kernel_cachedir_path(Cachedir, Program, DeviceI, Kernel, "",
                                  Command, Specialize);
  char KernelStandardMlirPath[POCL_MAX_PATHNAME_LENGTH];
  strncpy(KernelStandardMlirPath, Cachedir, POCL_MAX_PATHNAME_LENGTH);
  strcat(KernelStandardMlirPath, POCL_PARALLEL_MLIR_FILENAME);

  if (!pocl_exists(KernelStandardMlirPath)) {
    int Error = poclMlirGenerateStandardWorkgroupFunction(
        DeviceI, Device, Kernel, Command, Specialize, Cachedir);

    POCL_MSG_PRINT_LLVM("Generated %s\n", KernelStandardMlirPath);
    if (Error) {
      POCL_ABORT("MLIR: pocl_mlir_generate_standard_workgroup_function()"
                 " failed for kernel %s\n",
                 Kernel->name);
    }
  }

  char KernelParallelLlPath[POCL_MAX_PATHNAME_LENGTH];
  strncpy(KernelParallelLlPath, Cachedir, POCL_MAX_PATHNAME_LENGTH);
  strcat(KernelParallelLlPath, POCL_PARALLEL_BC_FILENAME);
  if (!pocl_exists(KernelParallelLlPath)) {
    int Error =
        poclMlirGenerateLlvmFunction(Command->program_device_i, Device, Kernel,
                                     Command, Specialize, Cachedir);

    POCL_MSG_PRINT_LLVM("Generated %s\n", KernelParallelLlPath);
    if (Error) {
      POCL_ABORT("MLIR: pocl_mlir_generate_llvm_workgroup_function()"
                 " failed for kernel %s\n",
                 Kernel->name);
    }
  }

  llvm::SMDiagnostic Err;
  std::unique_ptr<llvm::Module> FinalLlvmModule =
      llvm::parseIRFile(KernelParallelLlPath, Err, *LLVMCtx->Context);

  // Purposefully leak the unique_ptr, the module is freed manually later
  llvm::Module *Modp = FinalLlvmModule.release();
  if (!Modp) {
    Err.print(KernelParallelLlPath, llvm::errs());
  }

  *Output = Modp;

  return 0;
}

void poclDestroyMlirModule(void *Module) {
  llvm::Module *Modp = (llvm::Module *)Module;
  delete Modp;
}
