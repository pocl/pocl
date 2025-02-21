/* pocl_llvm_spirv.cc: implementation of SPIR-V related functions

   Copyright (c) 2025 Michal Babej / Intel Finland Oy

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

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wmaybe-uninitialized")
#include <llvm/ADT/StringRef.h>
POP_COMPILER_DIAGS
IGNORE_COMPILER_WARNING("-Wunused-parameter")
#include <llvm/Support/Casting.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassTimingInfo.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/SourceMgr.h>

#if LLVM_MAJOR >= 18
#include <llvm/Frontend/Driver/CodeGenOptions.h>
#endif
#include <llvm/Support/CommandLine.h>

#include "LLVMUtils.h"
POP_COMPILER_DIAGS

#include "common.h"
#include "pocl.h"
#include "pocl_cache.h"
#include "pocl_file_util.h"
#include "pocl_llvm.h"
#include "pocl_llvm_api.h"
#include "pocl_spir.h"
#include "pocl_util.h"

#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef HAVE_LLVM_SPIRV_LIB
#include <LLVMSPIRVLib/LLVMSPIRVLib.h>
#endif

#include "spirv_parser.hh"

// Specify list of allowed/disallowed extensions
#define LLVM17_INTEL_EXTS                                                      \
  "+SPV_INTEL_subgroups"                                                       \
  ",+SPV_INTEL_usm_storage_classes"                                            \
  ",+SPV_INTEL_arbitrary_precision_integers"                                   \
  ",+SPV_INTEL_arbitrary_precision_fixed_point"                                \
  ",+SPV_INTEL_arbitrary_precision_floating_point"                             \
  ",+SPV_INTEL_kernel_attributes"                                              \
  ",+SPV_KHR_no_integer_wrap_decoration"                                       \
  ",+SPV_EXT_shader_atomic_float_add"                                          \
  ",+SPV_EXT_shader_atomic_float_min_max"                                      \
  ",+SPV_INTEL_function_pointers"                                              \
  ",+SPV_KHR_integer_dot_product"

#if LLVM_MAJOR >= 18
#define ALLOW_INTEL_EXTS LLVM17_INTEL_EXTS ",+SPV_EXT_shader_atomic_float16_add"
#else
#define ALLOW_INTEL_EXTS LLVM17_INTEL_EXTS
#endif

  /*
  possibly useful:
    "+SPV_INTEL_unstructured_loop_controls,"
    "+SPV_INTEL_blocking_pipes,"
    "+SPV_INTEL_function_pointers,"
    "+SPV_INTEL_io_pipes,"
    "+SPV_INTEL_inline_assembly,"
    "+SPV_INTEL_optimization_hints,"
    "+SPV_INTEL_float_controls2,"
    "+SPV_INTEL_vector_compute,"
    "+SPV_INTEL_fast_composite,"
    "+SPV_INTEL_variable_length_array,"
    "+SPV_INTEL_fp_fast_math_mode,"
    "+SPV_INTEL_long_constant_composite,"
    "+SPV_INTEL_memory_access_aliasing,"
    "+SPV_INTEL_runtime_aligned,"
    "+SPV_INTEL_arithmetic_fence,"
    "+SPV_INTEL_bfloat16_conversion,"
    "+SPV_INTEL_global_variable_decorations,"
    "+SPV_INTEL_non_constant_addrspace_printf,"
    "+SPV_INTEL_hw_thread_queries,"
    "+SPV_INTEL_complex_float_mul_div,"
    "+SPV_INTEL_split_barrier,"
    "+SPV_INTEL_masked_gather_scatter"

  probably not useful:
    "+SPV_INTEL_media_block_io,+SPV_INTEL_device_side_avc_motion_estimation,"
    "+SPV_INTEL_fpga_loop_controls,+SPV_INTEL_fpga_memory_attributes,"
    "+SPV_INTEL_fpga_memory_accesses,"
    "+SPV_INTEL_fpga_reg,+SPV_INTEL_fpga_buffer_location,"
    "+SPV_INTEL_fpga_cluster_attributes,"
    "+SPV_INTEL_loop_fuse,"
    "+SPV_INTEL_optnone," // this one causes crash
    "+SPV_INTEL_fpga_dsp_control,"
    "+SPV_INTEL_fpga_invocation_pipelining_attributes,"
    "+SPV_INTEL_token_type,"
    "+SPV_INTEL_debug_module,"
    "+SPV_INTEL_joint_matrix,"
  */

#define ALLOW_EXTS "+SPV_KHR_no_integer_wrap_decoration"

#ifdef USE_LLVM_SPIRV_TARGET
int pocl_llvm_initialize_spirv_ext_option() {
  llvm::cl::Option *O = nullptr;
  llvm::StringMap<llvm::cl::Option *> &Opts = llvm::cl::getRegisteredOptions();
  O = Opts["spirv-ext"];
  if (O == nullptr) {
    POCL_MSG_WARN("Level0 : Note: spirv-ext LLVM option not found");
    return false;
  }
  return O->addOccurrence(1, llvm::StringRef("spirv-ext"),
                          llvm::StringRef(ALLOW_INTEL_EXTS));
}
#endif


static void handleInOutPathArgs(bool &keepPath, char *Path, char *HiddenPath,
                                bool Reverse, const void *Content) {
  keepPath = false;
  if (Path) {
    keepPath = true;
    if (Path[0]) {
      strncpy(HiddenPath, Path, POCL_MAX_PATHNAME_LENGTH);
    } else {
      pocl_cache_tempname(HiddenPath, (Reverse ? ".bc" : ".spv"), NULL);
      strncpy(Path, HiddenPath, POCL_MAX_PATHNAME_LENGTH);
    }
  } else {
    assert(Content);
    pocl_cache_tempname(HiddenPath, (Reverse ? ".bc" : ".spv"), NULL);
  }
}

#if defined(HAVE_LLVM_SPIRV) || defined(HAVE_LLVM_SPIRV_LIB)

// implement IR <-> SPIRV conversion using llvm-spirv or LLVMSPIRVLib
static int convertBCorSPV(char *InputPath,
                          const char *InputContent, // LLVM bitcode as string
                          uint64_t InputSize, std::string *BuildLog,
                          int useIntelExts,
                          int Reverse, // add "-r"
                          char *OutputPath, char **OutContent,
                          uint64_t *OutSize, pocl_version_t TargetVersion) {
  char HiddenOutputPath[POCL_MAX_PATHNAME_LENGTH];
  char HiddenInputPath[POCL_MAX_PATHNAME_LENGTH];
  std::vector<std::string> CompilationArgs;
  std::vector<const char *> CompilationArgs2;
  std::vector<uint8_t> FinalSpirv;
  llvm::Module *Mod = nullptr;
  bool keepOutputPath, keepInputPath;
  char *Content = nullptr;
  uint64_t ContentSize = 0;
  llvm::LLVMContext LLVMCtx;

  if (!Reverse && TargetVersion.major==0) {
    POCL_MSG_ERR("Invalid SPIR-V target version!");
    return -1;
  }

#ifdef HAVE_LLVM_SPIRV_LIB
  std::string Errors;
  SPIRV::TranslatorOpts::ExtensionsStatusMap EnabledExts;
  if (useIntelExts) {
    EnabledExts[SPIRV::ExtensionID::SPV_INTEL_subgroups] =
    EnabledExts[SPIRV::ExtensionID::SPV_INTEL_usm_storage_classes] =
    EnabledExts[SPIRV::ExtensionID::SPV_INTEL_arbitrary_precision_integers] =
    EnabledExts[SPIRV::ExtensionID::SPV_INTEL_arbitrary_precision_fixed_point] =
    EnabledExts[SPIRV::ExtensionID::SPV_INTEL_arbitrary_precision_floating_point] =
    EnabledExts[SPIRV::ExtensionID::SPV_INTEL_kernel_attributes] = true;
  }
  EnabledExts[SPIRV::ExtensionID::SPV_KHR_integer_dot_product] = true;
  EnabledExts[SPIRV::ExtensionID::SPV_KHR_no_integer_wrap_decoration] = true;
  EnabledExts[SPIRV::ExtensionID::SPV_EXT_shader_atomic_float_add] = true;
  EnabledExts[SPIRV::ExtensionID::SPV_EXT_shader_atomic_float_min_max] = true;
#if LLVM_MAJOR >= 18
  EnabledExts[SPIRV::ExtensionID::SPV_EXT_shader_atomic_float16_add] = true;
#endif

  SPIRV::VersionNumber TargetVersionEnum = SPIRV::VersionNumber::SPIRV_1_3;
  switch (TargetVersion.major * 100 + TargetVersion.minor) {
  default:
    if (!Reverse) {
      POCL_MSG_ERR("Unrecognized SPIR-V version: %u.%u\n",
                   static_cast<unsigned>(TargetVersion.major),
                   static_cast<unsigned>(TargetVersion.minor));
      return 1;
    }
    break;
  case 100:
    TargetVersionEnum = SPIRV::VersionNumber::SPIRV_1_0;
    break;
  case 101:
    TargetVersionEnum = SPIRV::VersionNumber::SPIRV_1_1;
    break;
  case 102:
    TargetVersionEnum = SPIRV::VersionNumber::SPIRV_1_2;
    break;
  case 103:
    TargetVersionEnum = SPIRV::VersionNumber::SPIRV_1_3;
    break;
  case 104:
    TargetVersionEnum = SPIRV::VersionNumber::SPIRV_1_4;
    break;
  case 105:
    TargetVersionEnum = SPIRV::VersionNumber::SPIRV_1_5;
    break;
  case 106:
    TargetVersionEnum = SPIRV::VersionNumber::SPIRV_1_6;
    break;
  }

  SPIRV::TranslatorOpts Opts(TargetVersionEnum, EnabledExts);
  Opts.setDesiredBIsRepresentation(SPIRV::BIsRepresentation::OpenCL20);
#endif
  int r = -1;

  handleInOutPathArgs(keepOutputPath, OutputPath, HiddenOutputPath, Reverse,
                      OutContent);

  handleInOutPathArgs(keepInputPath, InputPath, HiddenInputPath, Reverse,
                      &InputContent);

  if (InputContent && InputSize) {
    r = pocl_write_file(HiddenInputPath, InputContent, InputSize, 0);
    if (r != 0) {
      BuildLog->append("failed to write input file for llvm-spirv\n");
      goto FINISHED;
    }
  }

#ifdef HAVE_LLVM_SPIRV_LIB
  if (Reverse) {
    // SPIRV to BC
    std::string InputS;
    if (InputContent && InputSize) {
      InputS.append(InputContent, InputSize);
    } else {
      r = pocl_read_file(InputPath, &Content, &ContentSize);
      if (r != 0) {
        BuildLog->append("ConvertBC2SPIRV: failed to read input file:\n");
        BuildLog->append(InputPath);
        goto FINISHED;
      }
      InputS.append(Content, ContentSize);
      free(Content);
      Content = nullptr;
      ContentSize = 0;
    }

    std::stringstream InputSS(InputS);
    Mod = nullptr;

    // TODO maybe use context from program ?
    if (!readSpirv(LLVMCtx, Opts, InputSS, Mod, Errors)) {
      BuildLog->append("LLVMSPIRVLib: Write failed with errors:\n");
      BuildLog->append(Errors.c_str());
      goto FINISHED;
    }
    std::string OutputBC;
    writeModuleIRtoString(Mod, OutputBC);
    assert(OutputBC.size() > 20);
    Content = (char *)malloc(OutputBC.size());
    assert(Content);
    memcpy(Content, OutputBC.data(), OutputBC.size());
    ContentSize = OutputBC.size();

  } else {
    // BC to SPIRV
    std::stringstream SS;
    if (InputContent && InputSize) {
      Mod = parseModuleIRMem(InputContent, InputSize, &LLVMCtx);
    } else {
      assert(InputPath);
      Mod = parseModuleIR(InputPath, &LLVMCtx);
    }
    if (Mod == nullptr) {
      BuildLog->append("ConvertBC2SPIRV: failed to parse input module\n");
      goto FINISHED;
    }

    // TODO maybe use context from program ?
    if (!writeSpirv(Mod, Opts, SS, Errors)) {
      BuildLog->append("LLVMSPIRVLib: writeSPIRV failed with errors:\n");
      BuildLog->append(Errors.c_str());
      goto FINISHED;
    }
    SS.flush();
    std::string IntermediateSpirv = SS.str();
    assert(IntermediateSpirv.size() > 20);

    SPIRVParser::applyAtomicCmpXchgWorkaround(
        (const int32_t *)IntermediateSpirv.data(), IntermediateSpirv.size() / 4,
        FinalSpirv);

    Content = (char *)malloc(FinalSpirv.size());
    assert(Content);
    memcpy(Content, FinalSpirv.data(), FinalSpirv.size());
    ContentSize = IntermediateSpirv.size();
  }

#else

  // generate program.spv
  CompilationArgs.push_back(pocl_get_path("LLVM_SPIRV", LLVM_SPIRV));
#if (LLVM_MAJOR == 15) || (LLVM_MAJOR == 16)
#ifdef LLVM_OPAQUE_POINTERS
  CompilationArgs.push_back("--opaque-pointers");
#endif
#endif
  if (useIntelExts)
    CompilationArgs.push_back("--spirv-ext=" ALLOW_INTEL_EXTS);
  else
    CompilationArgs.push_back("--spirv-ext=" ALLOW_EXTS);

  if (!Reverse) {
    const auto TargetVersionOpt = std::string("--spirv-max-version=") +
                                  std::to_string(TargetVersion.major) + "." +
                                  std::to_string(TargetVersion.minor);
    CompilationArgs.push_back(TargetVersionOpt);
  }

  if (Reverse) {
    CompilationArgs.push_back("-r");
    CompilationArgs.push_back("--spirv-target-env=CL2.0");
  }
  CompilationArgs.push_back("-o");
  CompilationArgs.push_back(HiddenOutputPath);
  CompilationArgs.push_back(HiddenInputPath);
  CompilationArgs2.resize(CompilationArgs.size() + 1);
  for (unsigned i = 0; i < CompilationArgs.size(); ++i)
    CompilationArgs2[i] = (char *)CompilationArgs[i].data();
  CompilationArgs2[CompilationArgs.size()] = nullptr;

  // TODO removed output capture
  r = pocl_run_command(CompilationArgs2.data());
  if (r != 0) {
    BuildLog->append("llvm-spirv failed\n");
    goto FINISHED;
  }

  Content = nullptr;
  ContentSize = 0;
  r = pocl_read_file(HiddenOutputPath, &Content, &ContentSize);
  if (r != 0) {
    BuildLog->append("failed to read output file from llvm-spirv\n");
    goto FINISHED;
  }
  if (!Reverse) {
    size_t ContentWords = ContentSize / 4;
    SPIRVParser::applyAtomicCmpXchgWorkaroundInPlace((int32_t *)Content,
                                                     &ContentWords);
    ContentSize = ContentWords * 4;
  }
#endif

  if (keepOutputPath) {
    r = pocl_write_file(HiddenOutputPath, Content, ContentSize, 0);
    if (r != 0) {
      BuildLog->append("failed to write output file\n");
      goto FINISHED;
    }
  }

  if (OutContent && OutSize) {
    *OutContent = Content;
    *OutSize = ContentSize;
  } else {
    free(Content);
  }

  r = 0;

FINISHED:
  if (pocl_get_bool_option("POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES", 0) != 0) {
#if 0
    POCL_MSG_PRINT_LLVM("LLVM SPIR-V conversion tempfiles: %s -> %s",
                        HiddenInputPath, HiddenOutputPath);
#endif
  } else {
    if (!keepInputPath)
      pocl_remove(HiddenInputPath);
    if (!keepOutputPath)
      pocl_remove(HiddenOutputPath);
  }

  return r;
}


#elif defined(USE_LLVM_SPIRV_TARGET)

// SPIRV backend did not exist until LLVM 15
static_assert(LLVM_MAJOR > 14);

// implement IR -> SPIRV conversion using LLVM SPIRV backend
static int convertBCorSPV(char *InputPath,
                          const char *InputContent, // LLVM bitcode as string
                          uint64_t InputSize, std::string *BuildLog,
                          int useIntelExts,
                          int Reverse, // add "-r"
                          char *OutputPath, char **OutContent,
                          uint64_t *OutSize, pocl_version_t TargetVersion) {
  char HiddenOutputPath[POCL_MAX_PATHNAME_LENGTH];
  char HiddenInputPath[POCL_MAX_PATHNAME_LENGTH];
  std::vector<uint8_t> FinalSpirv;
  bool keepOutputPath, keepInputPath;
  llvm::Module *Mod = nullptr;
  char *Content = nullptr;
  uint64_t ContentSize = 0;
  size_t ContentWords = 0;
  llvm::LLVMContext LLVMCtx;
  std::string InputS;
  std::stringstream SS;
  std::string IntermediateSpirv;
  const char *Triple = "spirv64-unknown-unknown";
  int r = 0;

  if (Reverse) {
    POCL_MSG_ERR("Called convertBCorSPV(Reverse) with SPIRV backend\n");
    return -1;
  }

  handleInOutPathArgs(keepOutputPath, OutputPath, HiddenOutputPath, Reverse,
                      *OutContent);

  handleInOutPathArgs(keepInputPath, InputPath, HiddenInputPath, Reverse,
                      InputContent);

  if (InputContent && InputSize) {
    r = pocl_write_file(HiddenInputPath, InputContent, InputSize, 0);
    if (r != 0) {
      BuildLog->append("failed to write input file for SPIRV\n");
      goto FINISHED;
    }
    InputS.append(InputContent, InputSize);
  } else {
    assert(InputPath);
    r = pocl_read_file(InputPath, &Content, &ContentSize);
    if (r != 0) {
      BuildLog->append("ConvertBC2SPIRV: failed to read input file:\n");
      BuildLog->append(InputPath);
      goto FINISHED;
    }
    InputS.append(Content, ContentSize);
    free(Content);
    Content = nullptr;
    ContentSize = 0;
  }

  switch (TargetVersion.major * 100 + TargetVersion.minor) {
  case 100:
    Triple = "spirv64v1.0-unknown-unknown";
    break;
  case 101:
    Triple = "spirv64v1.1-unknown-unknown";
    break;
  default:
  case 102:
    Triple = "spirv64v1.2-unknown-unknown";
    break;
  case 103:
    Triple = "spirv64v1.3-unknown-unknown";
    break;
  case 104:
    Triple = "spirv64v1.4-unknown-unknown";
    break;
  case 105:
    Triple = "spirv64v1.5-unknown-unknown";
    break;
#if LLVM_MAJOR >= 18
    case 106:
    Triple = "spirv64v1.6-unknown-unknown";
#endif
    break;
  }

  Mod = parseModuleIRMem((char *)InputS.data(), InputS.size(), &LLVMCtx);
  if (Mod == nullptr) {
    BuildLog->append("failed to parse input LLVM IR\n");
    r = CL_BUILD_PROGRAM_FAILURE;
    goto FINISHED;
  }
  pocl_lock_t Lock;
  POCL_INIT_LOCK(Lock);
  if (pocl_llvm_codegen2(Triple, "", "", CL_DEVICE_TYPE_GPU, &Lock,
                         Mod, CL_FALSE, CL_TRUE, &Content, &ContentSize) != CL_SUCCESS) {
    BuildLog->append("failed to convert LLVM IR to SPIR-V "
                     "using LLVM SPIRV backend\n");
    POCL_DESTROY_LOCK(Lock);
    r = CL_BUILD_PROGRAM_FAILURE;
    goto FINISHED;
  }
  POCL_DESTROY_LOCK(Lock);

  ContentWords = ContentSize / 4;
  SPIRVParser::applyAtomicCmpXchgWorkaroundInPlace((int32_t *)Content,
                                                   &ContentWords);
  ContentSize = ContentWords * 4;

  if (keepOutputPath) {
    r = pocl_write_file(HiddenOutputPath, Content, ContentSize, 0);
    if (r != 0) {
      BuildLog->append("failed to write output file\n");
      goto FINISHED;
    }
  }

  if (OutContent && OutSize) {
    *OutContent = Content;
    *OutSize = ContentSize;
  } else {
    free(Content);
  }

  r = 0;

FINISHED:
  if (pocl_get_bool_option("POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES", 0) != 0) {
    POCL_MSG_PRINT_LLVM("SPIRV backend conversion tempfiles: %s -> %s",
                        HiddenInputPath, HiddenOutputPath);
  } else {
    if (!keepInputPath)
      pocl_remove(HiddenInputPath);
    if (!keepOutputPath)
      pocl_remove(HiddenOutputPath);
  }
  return r;
}

#else

// not implemented
static int convertBCorSPV(char *InputPath,
                          const char *InputContent, // LLVM bitcode as string
                          uint64_t InputSize, std::string *BuildLog,
                          int useIntelExts,
                          int Reverse, // add "-r"
                          char *OutputPath, char **OutContent,
                          uint64_t *OutSize, pocl_version_t TargetVersion) {
  POCL_MSG_ERR("No way to convert SPIR-V binaries to/from IR\n");
  return -1;
}
#endif

#if defined(HAVE_LLVM_SPIRV) || defined(HAVE_LLVM_SPIRV_LIB)

/* Note: this function exists only when building with LLVM-SPIRV / LLVMSPIRVLib.
 * The LLVM SPIRV backend can only convert in IR -> SPIRV direction. */
int pocl_convert_spirv_to_bitcode(char *TempSpirvPath, const char *SpirvContent,
                                  uint64_t SpirvSize, cl_program Program,
                                  cl_uint DeviceI, int UseIntelExts,
                                  char *TempBitcodePathOut,
                                  char **BitcodeContent,
                                  uint64_t *BitcodeSize) {

  std::string BuildLog;
  int R = convertBCorSPV(
      TempSpirvPath, SpirvContent, SpirvSize, &BuildLog, UseIntelExts,
      1, // = Reverse.
      TempBitcodePathOut, BitcodeContent, BitcodeSize,
      // Target version for SPIR-V emission. Ignored in reverse translation.
      pocl_version_t{});
  if (!BuildLog.empty())
    pocl_append_to_buildlog(Program, DeviceI, strdup(BuildLog.c_str()),
                            BuildLog.size());
  return R;
}

#else

int pocl_convert_spirv_to_bitcode(char *TempSpirvPath, const char *SpirvContent,
                                  uint64_t SpirvSize, cl_program Program,
                                  cl_uint DeviceI, int UseIntelExts,
                                  char *TempBitcodePathOut,
                                  char **BitcodeContent,
                                  uint64_t *BitcodeSize) {
  POCL_MSG_ERR("No way to convert SPIR-V binaries to IR\n");
  return -1;
}
#endif

int pocl_convert_bitcode_to_spirv(char *TempBitcodePath, const char *Bitcode,
                                  uint64_t BitcodeSize, cl_program Program,
                                  cl_uint DeviceI, int UseIntelExts,
                                  char *TempSpirvPathOut, char **SpirvContent,
                                  uint64_t *SpirvSize,
                                  pocl_version_t TargetVersion) {

  std::string BuildLog;
  int R = convertBCorSPV(
      TempBitcodePath, Bitcode, BitcodeSize, &BuildLog, UseIntelExts,
      0, // = Reverse
      TempSpirvPathOut, SpirvContent, SpirvSize, TargetVersion);

  if (!BuildLog.empty())
    pocl_append_to_buildlog(Program, DeviceI, strdup(BuildLog.c_str()),
                            BuildLog.size());
  return R;
}

int pocl_convert_bitcode_to_spirv2(char *TempBitcodePath, const char *Bitcode,
                                   uint64_t BitcodeSize, void *BuildLog,
                                   int UseIntelExts, char *TempSpirvPathOut,
                                   char **SpirvContent, uint64_t *SpirvSize,
                                   pocl_version_t TargetVersion) {

  return convertBCorSPV(TempBitcodePath, Bitcode, BitcodeSize,
                        (std::string *)BuildLog, UseIntelExts, 0, // = Reverse
                        TempSpirvPathOut, SpirvContent, SpirvSize,
                        TargetVersion);
}
