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
#include "pocl_compiler_macros.h"
#include "pocl_file_util.h"
#include "pocl_llvm.h"
#include "pocl_llvm_api.h"
#include "pocl_run_command.h"
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
#include <LLVMSPIRVLib.h>

#if LLVM_SPIRV_LIB_MAXVER >= 0x00010600
static const pocl_version_t MaxSPIRVLibSupportedVersion{1, 6};
#elif LLVM_SPIRV_LIB_MAXVER >= 0x00010500
static const pocl_version_t MaxSPIRVLibSupportedVersion{1, 5};
#elif LLVM_SPIRV_LIB_MAXVER >= 0x00010400
static const pocl_version_t MaxSPIRVLibSupportedVersion{1, 4};
#elif LLVM_SPIRV_LIB_MAXVER >= 0x00010300
static const pocl_version_t MaxSPIRVLibSupportedVersion{1, 3};
#elif LLVM_SPIRV_LIB_MAXVER >= 0x00010200
static const pocl_version_t MaxSPIRVLibSupportedVersion{1, 2};
#elif LLVM_SPIRV_LIB_MAXVER >= 0x00010100
static const pocl_version_t MaxSPIRVLibSupportedVersion{1, 1};
#else
static const pocl_version_t MaxSPIRVLibSupportedVersion{1, 0};
#endif

#else // HAVE_LLVM_SPIRV_LIB
static const pocl_version_t MaxSPIRVLibSupportedVersion{1, 5};
#endif

#include "spirv_parser.hh"

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
                          llvm::StringRef("all"));
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

extern "C" int pocl_reload_program_bc(char *program_bc_path, cl_program program,
                                      cl_uint device_i);

int pocl_preprocess_spirv_input(cl_program program) {
  int32_t *In = reinterpret_cast<int32_t *>(program->program_il);
  size_t NumW = program->program_il_size / sizeof(int32_t);
  if (SPIRVParser::applyAtomicCmpXchgWorkaroundInPlace(In, &NumW)) {
    program->program_il_size = NumW * sizeof(int32_t);
    return CL_SUCCESS;
  } else {
    return CL_INVALID_PROGRAM_EXECUTABLE;
  }
}

static bool getMaxSpirvVersion(pocl_version_t &MaxVersion,
                               size_t num_ils_with_version,
                               const cl_name_version *ils_with_version) {
  if (num_ils_with_version == 0) {
    MaxVersion.major = 1;
    MaxVersion.minor = 0;
    return false;
  }
  cl_version Max = CL_MAKE_VERSION(1, 0, 0);
  for (size_t i = 0; i < num_ils_with_version; ++i) {
    if (ils_with_version[i].version > Max)
      Max = ils_with_version[i].version;
  }
  MaxVersion.major = CL_VERSION_MAJOR(Max);
  MaxVersion.minor = CL_VERSION_MINOR(Max);
  return true;
}

#ifdef HAVE_LLVM_SPIRV_LIB

static std::map<std::string, SPIRV::ExtensionID> SPVExtMap = {
#define EXT(X) {#X, SPIRV::ExtensionID::X},
#include <LLVMSPIRVExtensions.inc>
#undef EXT
};

SPIRV::TranslatorOpts setupTranslOpts(const std::string &SupportedSPVExts,
                                      bool &UnrecognizedVersion,
                                      pocl_version_t TargetVersion) {
  SPIRV::TranslatorOpts::ExtensionsStatusMap EnabledExts;
  std::istringstream ISS(SupportedSPVExts);
  while (ISS) {
    std::string Token;
    std::getline(ISS, Token, ',');
    if (Token.empty())
      break;
    if (Token.front() == '+')
      Token.erase(0, 1);
    auto It = SPVExtMap.find(Token);
    if (It != SPVExtMap.end())
      EnabledExts[It->second] = true;
    else
      POCL_MSG_ERR("Unknown SPV extension: %s \n", Token.c_str());
  }

  // default to 1.2
  SPIRV::VersionNumber TargetVersionEnum = SPIRV::VersionNumber::SPIRV_1_2;
  switch (TargetVersion.major * 100 + TargetVersion.minor) {
  case 100:
    TargetVersionEnum = SPIRV::VersionNumber::SPIRV_1_0;
    break;
  case 101:
    TargetVersionEnum = SPIRV::VersionNumber::SPIRV_1_1;
    break;
  default:
    UnrecognizedVersion = true;
    POCL_FALLTHROUGH;
  case 102:
    TargetVersionEnum = SPIRV::VersionNumber::SPIRV_1_2;
    break;
#if LLVM_SPIRV_LIB_MAXVER >= 0x00010300
  case 103:
    TargetVersionEnum = SPIRV::VersionNumber::SPIRV_1_3;
    break;
#endif
#if LLVM_SPIRV_LIB_MAXVER >= 0x00010400
  case 104:
    TargetVersionEnum = SPIRV::VersionNumber::SPIRV_1_4;
    break;
#endif
#if LLVM_SPIRV_LIB_MAXVER >= 0x00010500
  case 105:
    TargetVersionEnum = SPIRV::VersionNumber::SPIRV_1_5;
    break;
#endif
#if LLVM_SPIRV_LIB_MAXVER >= 0x00010600
  case 106:
    TargetVersionEnum = SPIRV::VersionNumber::SPIRV_1_6;
    break;
#endif
  }

  SPIRV::TranslatorOpts Opts(TargetVersionEnum, EnabledExts);
  return Opts;
}
#endif

/* max number of lines in output of 'llvm-spirv --spec-const-info' */
#define MAX_SPEC_CONSTANT_LINES 4096
/* max bytes in output of 'llvm-spirv --spec-const-info' */
#define MAX_OUTPUT_BYTES 65536
#define MAX_SPEC_CONST_CMDLINE_LEN 8192
#define MAX_SPEC_CONST_OPT_LEN 256


#if defined(HAVE_LLVM_SPIRV_LIB)

/* if some SPIR-V spec constants were changed, use LLVMSPIRVLib
 * to generate new LLVM bitcode from SPIR-V with updated SpecConstants */
int pocl_regen_spirv_binary(cl_program Program, cl_uint DeviceI) {
  cl_device_id Device = Program->devices[DeviceI];

  bool UnrecognizedVersion = false;
  // Don't limit the Max SPIR-V version when doing reverse translation
  pocl_version_t MaxSupportedVersion = MaxSPIRVLibSupportedVersion;
  SPIRV::TranslatorOpts Opts =
      setupTranslOpts(Device->supported_spirv_extensions, UnrecognizedVersion,
                      MaxSupportedVersion);
  POCL_RETURN_ERROR_ON(UnrecognizedVersion, CL_INVALID_BINARY, "LLVM-SPIRV "
                       "Translator does not recognize the SPIR-V version\n");

  /* using --spirv-target-env=CL2.0 here enables llvm-spirv to produce proper
   * OpenCL 2.0 atomics, unfortunately it also enables generic ptrs, which not
   * all PoCL devices support, hence check the device */
  if (Device->generic_as_support)
    Opts.setDesiredBIsRepresentation(SPIRV::BIsRepresentation::OpenCL20);
  else
    Opts.setDesiredBIsRepresentation(SPIRV::BIsRepresentation::OpenCL12);

  for (unsigned I = 0; I < Program->num_spec_consts; ++I) {
    if (Program->spec_const_is_set[I]) {
      Opts.setSpecConst(Program->spec_const_ids[I],
                        Program->spec_const_values[I]);
    }
  }

  llvm::LLVMContext LLVMCtx;
  std::string Errors;
  std::string InputS((char *)Program->program_il, Program->program_il_size);
  std::stringstream InputSS(InputS);
  llvm::Module *Mod = nullptr;
  char *Content = nullptr;
  uint64_t ContentSize = 0;

  if (!readSpirv(LLVMCtx, Opts, InputSS, Mod, Errors)) {
    POCL_MSG_ERR("LLVMSPIRVLib failed to read SPIR-V with errors:\n%s\n",
                 Errors.c_str());
    return CL_INVALID_BINARY;
  }
  std::string OutputBC;
  writeModuleIRtoString(Mod, OutputBC);
  POCL_RETURN_ERROR_ON((OutputBC.size() < 20), CL_INVALID_BINARY,
                       "The result is not a valid SPIR-V\n");
  Content = (char *)malloc(OutputBC.size());
  POCL_RETURN_ERROR_COND((Content == nullptr), CL_OUT_OF_HOST_MEMORY);
  memcpy(Content, OutputBC.data(), OutputBC.size());
  ContentSize = OutputBC.size();
  delete Mod;

  if (Program->binaries[DeviceI])
    POCL_MEM_FREE(Program->binaries[DeviceI]);
  Program->binaries[DeviceI] = (unsigned char *)Content;
  Program->binary_sizes[DeviceI] = ContentSize;

  return CL_SUCCESS;
}

int pocl_get_program_spec_constants(cl_program program, char *spirv_path,
                                    const void *spirv_content,
                                    size_t spirv_len) {
  std::string InputS((const char *)spirv_content, spirv_len);
  std::stringstream InputSS(InputS);
  std::vector<llvm::SpecConstInfoTy> SpecConstInfoVec;
  if (!llvm::getSpecConstInfo(InputSS, SpecConstInfoVec))
    return CL_INVALID_BINARY;

  size_t NumConst = SpecConstInfoVec.size();
  program->num_spec_consts = NumConst;
  if (NumConst > 0) {
    program->spec_const_ids = (cl_uint *)calloc(NumConst, sizeof(cl_uint));
    program->spec_const_sizes = (cl_uint *)calloc(NumConst, sizeof(cl_uint));
    program->spec_const_values =
        (uint64_t *)calloc(NumConst, sizeof(uint64_t));
    program->spec_const_is_set = (char *)calloc(NumConst, sizeof(char));
    for (unsigned i = 0; i < program->num_spec_consts; ++i) {
      program->spec_const_ids[i] = SpecConstInfoVec[i].ID;
      program->spec_const_sizes[i] = SpecConstInfoVec[i].Size;
      program->spec_const_values[i] = 0;
      program->spec_const_is_set[i] = CL_FALSE;
    }
  }
  return CL_SUCCESS;
}

#elif defined(HAVE_LLVM_SPIRV)

/* if some SPIR-V spec constants were changed, use llvm-spirv --spec-const=...
 * to generate new LLVM bitcode from SPIR-V */
int pocl_regen_spirv_binary(cl_program program, cl_uint device_i) {
  int errcode = CL_SUCCESS;
  cl_device_id Device = program->devices[device_i];
  int spec_constants_changed = 0;
  char concated_spec_const_option[MAX_SPEC_CONST_CMDLINE_LEN] = {};
  concated_spec_const_option[0] = 0;
  std::string SpirvExts;
  std::string SpirvMaxVersion;
  char program_bc_spirv[POCL_MAX_PATHNAME_LENGTH] = {};
  char unlinked_program_bc_temp[POCL_MAX_PATHNAME_LENGTH] = {};
  program_bc_spirv[0] = 0;
  unlinked_program_bc_temp[0] = 0;

  // Don't limit the Max SPIR-V version when doing reverse translation

  /* using --spirv-target-env=CL2.0 here enables llvm-spirv to produce proper
   * OpenCL 2.0 atomics, unfortunately it also enables generic ptrs, which not
   * all PoCL devices support, hence check the device */
  const char *spirv_target_env = (Device->generic_as_support != CL_FALSE)
                                     ? "--spirv-target-env=CL2.0"
                                     : "--spirv-target-env=CL1.2";
  SpirvExts = "--spirv-ext=";
  if (Device->supported_spirv_extensions &&
      Device->supported_spirv_extensions[0]) {
    SpirvExts += Device->supported_spirv_extensions;
  } else {
    SpirvExts += "-all";
  }
  const char *args[] = {pocl_get_path("LLVM_SPIRV", LLVM_SPIRV),
                        concated_spec_const_option,
                        spirv_target_env,
                        SpirvExts.c_str(),
                        "-r",
                        "-o",
                        unlinked_program_bc_temp,
                        program_bc_spirv,
                        NULL};
  const char **final_args = args;

  errcode = pocl_cache_tempname(unlinked_program_bc_temp, ".bc", NULL);
  POCL_RETURN_ERROR_ON((errcode != 0), CL_BUILD_PROGRAM_FAILURE,
                       "failed to create tmpfile in pocl cache\n");

  errcode = pocl_cache_write_spirv(program_bc_spirv,
                                   (const char *)program->program_il,
                                   (uint64_t)program->program_il_size);
  POCL_RETURN_ERROR_ON((errcode != 0), CL_BUILD_PROGRAM_FAILURE,
                       "failed to write into pocl cache\n");

  for (unsigned i = 0; i < program->num_spec_consts; ++i)
    spec_constants_changed += program->spec_const_is_set[i];

  if (spec_constants_changed) {
    strcpy(concated_spec_const_option, "--spec-const=");
    for (unsigned i = 0; i < program->num_spec_consts; ++i) {
      if (program->spec_const_is_set[i]) {
        char opt[MAX_SPEC_CONST_OPT_LEN];
        snprintf(opt, MAX_SPEC_CONST_OPT_LEN, "%u:i%u:%zu ",
                 program->spec_const_ids[i], program->spec_const_sizes[i] * 8,
                 program->spec_const_values[i]);
        strcat(concated_spec_const_option, opt);
      }
    }
  } else {
    /* skip concated_spec_const_option */
    args[0] = NULL;
    args[1] = pocl_get_path("LLVM_SPIRV", LLVM_SPIRV);
    final_args = args + 1;
  }

  errcode = pocl_run_command(final_args);
  POCL_GOTO_ERROR_ON((errcode != 0), CL_INVALID_VALUE,
                     "External command (llvm-spirv translator) failed!\n");

  POCL_GOTO_ERROR_ON(
      (pocl_reload_program_bc(unlinked_program_bc_temp, program, device_i)),
      CL_INVALID_VALUE, "Can't read llvm-spirv converted bitcode file\n");

  errcode = CL_SUCCESS;

ERROR:
  if (pocl_get_bool_option("POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES", 0) == 0) {
    if (unlinked_program_bc_temp[0])
      pocl_remove(unlinked_program_bc_temp);
    if (program_bc_spirv[0])
      pocl_remove(program_bc_spirv);
  }
  return errcode;
}

int pocl_get_program_spec_constants(cl_program program, char *spirv_path,
                                    const void *spirv_content,
                                    size_t spirv_len) {
  const char *args[] = {pocl_get_path("LLVM_SPIRV", LLVM_SPIRV),
                        "--spec-const-info", spirv_path, NULL};
  char captured_output[MAX_OUTPUT_BYTES];
  size_t captured_bytes = MAX_OUTPUT_BYTES;
  int errcode = CL_SUCCESS;
  unsigned num_const = 0;

  errcode =
      pocl_run_command_capture_output(captured_output, &captured_bytes, args);
  POCL_RETURN_ERROR_ON((errcode != 0), CL_INVALID_BINARY,
                       "External command "
                       "(llvm-spirv --spec-const-info) failed!\n");

  captured_output[captured_bytes] = 0;
  char *lines[MAX_SPEC_CONSTANT_LINES];
  unsigned num_lines = 0;
  char delim[2] = {0x0A, 0x0};
  char *token = strtok(captured_output, delim);
  while (num_lines < MAX_SPEC_CONSTANT_LINES && token != NULL) {
    lines[num_lines++] = strdup(token);
    token = strtok(NULL, delim);
  }
  POCL_GOTO_ERROR_ON((num_lines == 0 || num_lines >= MAX_SPEC_CONSTANT_LINES),
                     CL_INVALID_BINARY, "Can't parse output from llvm-spirv\n");

  errcode = sscanf(
      lines[0], "Number of scalar specialization constants in the module = %u",
      &num_const);
  POCL_GOTO_ERROR_ON((errcode < 1 || num_const > num_lines), CL_INVALID_BINARY,
                     "Can't parse first line of output");

  program->num_spec_consts = num_const;
  if (num_const > 0) {
    program->spec_const_ids = (cl_uint *)calloc(num_const, sizeof(cl_uint));
    program->spec_const_sizes = (cl_uint *)calloc(num_const, sizeof(cl_uint));
    program->spec_const_values =
        (uint64_t *)calloc(num_const, sizeof(uint64_t));
    program->spec_const_is_set = (char *)calloc(num_const, sizeof(char));
    for (unsigned i = 0; i < program->num_spec_consts; ++i) {
      unsigned spec_id, spec_size;
      int r = sscanf(lines[i + 1], "Spec const id = %u, size in bytes = %u",
                     &spec_id, &spec_size);
      POCL_GOTO_ERROR_ON((r < 2), CL_INVALID_BINARY,
                         "Can't parse %u-th line of output:\n%s\n", i + 1,
                         lines[i + 1]);
      program->spec_const_ids[i] = spec_id;
      program->spec_const_sizes[i] = spec_size;
      program->spec_const_values[i] = 0;
      program->spec_const_is_set[i] = CL_FALSE;
    }
  }
  errcode = CL_SUCCESS;
ERROR:
  for (unsigned i = 0; i < num_lines; ++i)
    free(lines[i]);
  if (errcode != CL_SUCCESS) {
    program->num_spec_consts = 0;
  }
  return errcode;
}

#else

int pocl_get_program_spec_constants(cl_program program, char *spirv_path,
                                    const void *spirv_content,
                                    size_t spirv_len) {
  POCL_MSG_ERR("No way to parse spec constants from SPIRV\n");
  program->num_spec_consts = 0;
  return CL_INVALID_OPERATION;
}

int pocl_regen_spirv_binary(cl_program program, cl_uint device_i) {
  POCL_MSG_ERR("No way to regenerate SPIRV with new SpecConstants\n");
  return CL_INVALID_OPERATION;
}

#endif


#if defined(HAVE_LLVM_SPIRV) || defined(HAVE_LLVM_SPIRV_LIB)

// implement IR <-> SPIRV conversion using llvm-spirv or LLVMSPIRVLib
static int convertBCorSPV(char *InputPath,
                          const char *InputContent, // LLVM bitcode as string
                          uint64_t InputSize, std::string *BuildLog,
                          const char *SPVExtensions,
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
  std::string Errors;
  std::string SpirvExts("--spirv-ext=");

  if (!Reverse && TargetVersion.major==0) {
    POCL_MSG_ERR("Invalid SPIR-V target version!");
    return -1;
  }

  const auto TargetVersionOpt = std::string("--spirv-max-version=") +
                                std::to_string(TargetVersion.major) + "." +
                                std::to_string(TargetVersion.minor);

#ifdef HAVE_LLVM_SPIRV_LIB
  bool UnrecognizedVersion = false;
  SPIRV::TranslatorOpts Opts =
      setupTranslOpts(SPVExtensions, UnrecognizedVersion, TargetVersion);
  Opts.setDesiredBIsRepresentation(SPIRV::BIsRepresentation::OpenCL20);

  if (UnrecognizedVersion && !Reverse) {
    POCL_MSG_ERR("Unrecognized SPIR-V version: %u.%u\n",
                 static_cast<unsigned>(TargetVersion.major),
                 static_cast<unsigned>(TargetVersion.minor));
    return -1;
  }
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
    delete Mod;

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

    delete Mod;
  }

#else

  // generate program.spv
  CompilationArgs.push_back(pocl_get_path("LLVM_SPIRV", LLVM_SPIRV));
  SpirvExts.append(SPVExtensions);
  CompilationArgs.push_back(SpirvExts);
  CompilationArgs.push_back(TargetVersionOpt);

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
  assert(HiddenOutputPath[0]);
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
    assert(HiddenOutputPath[0]);
    r = pocl_write_file(HiddenOutputPath, Content, ContentSize, 0);
    if (r != 0) {
      free(Content);
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

// implement IR -> SPIRV conversion using LLVM SPIRV backend
static int convertBCorSPV(char *InputPath,
                          const char *InputContent, // LLVM bitcode as string
                          uint64_t InputSize, std::string *BuildLog,
                          const char *SPVExtensions,
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

  // TODO: we should set --spirv-ext to the Device supported extensions,
  // however it's a global option...
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
                          const char *SPVExtensions,
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
                                  cl_uint DeviceI, const char *SPVExtensions,
                                  char *TempBitcodePathOut,
                                  char **BitcodeContent,
                                  uint64_t *BitcodeSize) {

  std::string BuildLog;

  int R = convertBCorSPV(
      TempSpirvPath, SpirvContent, SpirvSize, &BuildLog, SPVExtensions,
      1, // = Reverse.
      TempBitcodePathOut, BitcodeContent, BitcodeSize,
      // Target version for SPIR-V emission. This is necessary to pass,
      // otherwise LLVM-SPIRV might return an error like:
      // Invalid SPIR-V module: incorrect SPIR-V version number 1.4 (66560) - it conflicts with maximum allowed version which is set to 1.2 (66304)
      // For Reverse translation, set this to the maximum supported by the library
      MaxSPIRVLibSupportedVersion);
  if (!BuildLog.empty())
    pocl_append_to_buildlog(Program, DeviceI, strdup(BuildLog.c_str()),
                            BuildLog.size());
  return R;
}

#else

int pocl_convert_spirv_to_bitcode(char *TempSpirvPath, const char *SpirvContent,
                                  uint64_t SpirvSize, cl_program Program,
                                  cl_uint DeviceI, const char *SPVExtensions,
                                  char *TempBitcodePathOut,
                                  char **BitcodeContent,
                                  uint64_t *BitcodeSize) {
  POCL_MSG_ERR("No way to convert SPIR-V binaries to IR\n");
  return -1;
}
#endif

int pocl_convert_bitcode_to_spirv(char *TempBitcodePath, const char *Bitcode,
                                  uint64_t BitcodeSize, cl_program Program,
                                  cl_uint DeviceI, const char *SPVExtensions,
                                  char *TempSpirvPathOut, char **SpirvContent,
                                  uint64_t *SpirvSize,
                                  pocl_version_t TargetVersion) {

  std::string BuildLog;
  int R = convertBCorSPV(
      TempBitcodePath, Bitcode, BitcodeSize, &BuildLog, SPVExtensions,
      0, // = Reverse
      TempSpirvPathOut, SpirvContent, SpirvSize, TargetVersion);

  if (!BuildLog.empty())
    pocl_append_to_buildlog(Program, DeviceI, strdup(BuildLog.c_str()),
                            BuildLog.size());
  return R;
}

int pocl_convert_bitcode_to_spirv2(char *TempBitcodePath, const char *Bitcode,
                                   uint64_t BitcodeSize, void *BuildLog,
                                   const char *SPVExtensions,
                                   char *TempSpirvPathOut, char **SpirvContent,
                                   uint64_t *SpirvSize,
                                   pocl_version_t TargetVersion) {

  return convertBCorSPV(TempBitcodePath, Bitcode, BitcodeSize,
                        (std::string *)BuildLog, SPVExtensions, 0, // = Reverse
                        TempSpirvPathOut, SpirvContent, SpirvSize,
                        TargetVersion);
}
