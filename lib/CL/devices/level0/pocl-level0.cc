/* pocl-level0.c - driver for LevelZero Compute API devices.

   Copyright (c) 2022-2023 Michal Babej / Intel Finland Oy

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


#include "common.h"
#include "common_driver.h"
#include "devices.h"
#include "pocl_cl.h"
#include "utlist.h"

#include "pocl-level0.h"
#include "pocl_cache.h"
#include "pocl_debug.h"
#include "pocl_file_util.h"
#include "pocl_hash.h"
#include "pocl_llvm.h"
#include "pocl_local_size.h"
#include "pocl_timing.h"
#include "pocl_util.h"

#include <ze_api.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "spirv_parser.hh"

using namespace SPIRVParser;

#include "level0-compilation.hh"
#include "level0-driver.hh"

using namespace pocl;

struct PoclL0EventData {
  pocl_cond_t Cond;
};

struct PoclL0QueueData {
  pocl_cond_t Cond;
  // list of event that have been submitted to the runtime, but not yet
  // to the device. They will be split into actual batches
  // in Flush/Notify callbacks
  BatchType UnsubmittedEventList;
};

static void pocl_level0_local_size_optimizer(cl_device_id Dev, cl_kernel Kernel,
                                             unsigned DeviceI, size_t GlobalX,
                                             size_t GlobalY, size_t GlobalZ,
                                             size_t *LocalX, size_t *LocalY,
                                             size_t *LocalZ) {
  assert(Kernel->data[DeviceI] != nullptr);
  Level0Kernel *L0Kernel = (Level0Kernel *)Kernel->data[DeviceI];
  ze_kernel_handle_t HKernel = L0Kernel->getAnyCreated();

  uint32_t SuggestedX = 0;
  uint32_t SuggestedY = 0;
  uint32_t SuggestedZ = 0;
  ze_result_t Res = ZE_RESULT_ERROR_DEVICE_LOST;
  if (HKernel != nullptr) {
    Res = zeKernelSuggestGroupSize(HKernel, GlobalX, GlobalY, GlobalZ,
                                   &SuggestedX, &SuggestedY, &SuggestedZ);
  }
  if (Res != ZE_RESULT_SUCCESS) {
    POCL_MSG_WARN("zeKernelSuggestGroupSize FAILED: %0x\n", (unsigned)Res);
    pocl_default_local_size_optimizer(Dev, Kernel, DeviceI, GlobalX, GlobalY,
                                      GlobalZ, LocalX, LocalY, LocalZ);
  } else {
    *LocalX = SuggestedX;
    *LocalY = SuggestedY;
    *LocalZ = SuggestedZ;
  }
}

void pocl_level0_init_device_ops(struct pocl_device_ops *Ops) {
  Ops->device_name = "level0";

  Ops->probe = pocl_level0_probe;
  Ops->init = pocl_level0_init;
  Ops->uninit = pocl_level0_uninit;
  Ops->reinit = pocl_level0_reinit;

  Ops->get_mapping_ptr = pocl_level0_get_mapping_ptr;
  Ops->free_mapping_ptr = pocl_level0_free_mapping_ptr;

  Ops->compute_local_size = pocl_level0_local_size_optimizer;

  Ops->alloc_mem_obj = pocl_level0_alloc_mem_obj;
  Ops->free = pocl_level0_free;
  Ops->svm_free = pocl_level0_svm_free;
  Ops->svm_alloc = pocl_level0_svm_alloc;
  Ops->usm_alloc = pocl_level0_usm_alloc;
  Ops->usm_free = pocl_level0_usm_free;
  Ops->usm_free_blocking = pocl_level0_usm_free_blocking;

  Ops->build_source = pocl_level0_build_source;
  Ops->build_binary = pocl_level0_build_binary;
  Ops->link_program = pocl_level0_link_program;
  Ops->free_program = pocl_level0_free_program;
  Ops->setup_metadata = pocl_level0_setup_metadata;
  Ops->supports_binary = pocl_level0_supports_binary;
  Ops->build_poclbinary = pocl_level0_build_poclbinary;
  Ops->compile_kernel = NULL;
  Ops->create_kernel = pocl_level0_create_kernel;
  Ops->free_kernel = pocl_level0_free_kernel;
  Ops->init_build = pocl_level0_init_build;

  Ops->join = pocl_level0_join;
  Ops->submit = pocl_level0_submit;
  Ops->broadcast = pocl_broadcast;
  Ops->notify = pocl_level0_notify;
  Ops->flush = pocl_level0_flush;
  Ops->build_hash = pocl_level0_build_hash;

  /* TODO get timing data from level0 API */
  /* ops->get_timer_value = pocl_level0_get_timer_value; */

  Ops->wait_event = pocl_level0_wait_event;
  Ops->notify_event_finished = pocl_level0_notify_event_finished;
  Ops->notify_cmdq_finished = pocl_level0_notify_cmdq_finished;
  Ops->free_event_data = pocl_level0_free_event_data;
  Ops->wait_event = pocl_level0_wait_event;
  Ops->update_event = pocl_level0_update_event;

  Ops->init_queue = pocl_level0_init_queue;
  Ops->free_queue = pocl_level0_free_queue;

  Ops->create_sampler = pocl_level0_create_sampler;
  Ops->free_sampler = pocl_level0_free_sampler;

  Ops->get_device_info_ext = pocl_level0_get_device_info_ext;
  Ops->set_kernel_exec_info_ext = pocl_level0_set_kernel_exec_info_ext;
  Ops->get_synchronized_timestamps = pocl_driver_get_synchronized_timestamps;
}


static int readProgramSpv(cl_program Program, cl_uint DeviceI,
                          const char *ProgramSpvPath) {
  /* Read binaries from program.spv to memory */
  if (Program->program_il_size == 0) {
    assert(ProgramSpvPath);
    assert(Program->program_il == nullptr);
    uint64_t Size = 0;
    char *Binary = nullptr;
    int Res = pocl_read_file(ProgramSpvPath, &Binary, &Size);
    POCL_RETURN_ERROR_ON((Res != 0), CL_BUILD_PROGRAM_FAILURE,
                         "Failed to read binaries from program.spv to "
                         "memory: %s\n",
                         ProgramSpvPath);
    Program->program_il = Binary;
    Program->program_il_size = Size;
  }
  return CL_SUCCESS;
}

static Level0Driver *DriverInstance = nullptr;

char *pocl_level0_build_hash(cl_device_id Device) {
  // TODO build hash
  char *Res = (char *)malloc(32);
  snprintf(Res, 32, "pocl-level0-spirv");
  return Res;
}

unsigned int pocl_level0_probe(struct pocl_device_ops *Ops) {
  int EnvCount = pocl_device_get_env_count(Ops->device_name);

  if (EnvCount <= 0) {
    return 0;
  }

  DriverInstance = new Level0Driver();

  POCL_MSG_PRINT_LEVEL0("Level Zero devices found: %u\n",
                        DriverInstance->getNumDevices());

  /* TODO: clamp device_count to env_count */

  return DriverInstance->getNumDevices();
}

cl_int pocl_level0_init(unsigned J, cl_device_id ClDevice,
                        const char *Parameters) {
  assert(J < DriverInstance->getNumDevices());
  POCL_MSG_PRINT_LEVEL0("Initializing device %u\n", J);

  Level0Device *Device = DriverInstance->createDevice(J, ClDevice, Parameters);

  if (Device == nullptr) {
    return CL_FAILED;
  }

  ClDevice->data = (void *)Device;

  return CL_SUCCESS;
}

cl_int pocl_level0_uninit(unsigned J, cl_device_id ClDevice) {
  Level0Device *Device = (Level0Device *)ClDevice->data;

  DriverInstance->releaseDevice(Device);
  /* TODO should this be done at all ? */
  if (DriverInstance->empty()) {
    delete DriverInstance;
    DriverInstance = nullptr;
  }

  return CL_SUCCESS;
}

cl_int pocl_level0_reinit(unsigned J, cl_device_id ClDevice, const char *parameters) {

  if (DriverInstance == nullptr) {
    DriverInstance = new Level0Driver();
  }

  assert(J < DriverInstance->getNumDevices());
  POCL_MSG_PRINT_LEVEL0("Initializing device %u\n", J);

  // TODO: parameters are not passed (this works ATM because they're ignored)
  Level0Device *Device = DriverInstance->createDevice(J, ClDevice, nullptr);

  if (Device == nullptr) {
    return CL_FAILED;
  }

  ClDevice->data = (void *)Device;

  return CL_SUCCESS;
}

static void convertProgramBcPathToSpv(char *ProgramBcPath,
                                      char *ProgramSpvPath) {
  strncpy(ProgramSpvPath, ProgramBcPath, POCL_MAX_PATHNAME_LENGTH);
  size_t Len = strlen(ProgramBcPath);
  assert(Len > 3);
  Len -= 2;
  ProgramSpvPath[Len] = 0;
  strncat(ProgramSpvPath, "spv", POCL_MAX_PATHNAME_LENGTH);
}

static constexpr unsigned DefaultCaptureSize = 128 * 1024;

static int runAndAppendOutputToBuildLog(cl_program Program, unsigned DeviceI,
                                        char *const *Args) {
  int Errcode = CL_SUCCESS;

  char *CapturedOutput = nullptr;
  size_t CaptureCapacity = 0;

  CapturedOutput = (char *)malloc(DefaultCaptureSize);
  POCL_RETURN_ERROR_ON((CapturedOutput == nullptr), CL_OUT_OF_HOST_MEMORY,
                       "Error while allocating temporary memory\n");
  CaptureCapacity = (DefaultCaptureSize) - 1;
  CapturedOutput[0] = 0;
  char *SavedCapturedOutput = CapturedOutput;

  std::string CommandLine;
  unsigned I = 0;
  while (Args[I] != nullptr) {
    CommandLine += " ";
    CommandLine += Args[I];
    ++I;
  }
  POCL_MSG_PRINT_LEVEL0("launching command: \n#### %s\n", CommandLine.c_str());

  std::string LaunchMsg;
  LaunchMsg.append("Output of ");
  LaunchMsg.append(Args[0]);
  LaunchMsg.append(":\n");
  if (LaunchMsg.size() < CaptureCapacity) {
    strncat(CapturedOutput, LaunchMsg.c_str(), CaptureCapacity);
    CapturedOutput += LaunchMsg.size();
    CaptureCapacity -= LaunchMsg.size();
  }

  Errcode =
      pocl_run_command_capture_output(CapturedOutput, &CaptureCapacity, Args);
  if (CaptureCapacity > 0) {
    CapturedOutput[CaptureCapacity] = 0;
  }

  pocl_append_to_buildlog(Program, DeviceI, SavedCapturedOutput,
                          strlen(SavedCapturedOutput));

  return Errcode;
}

// disabled for now, need to solve the problem of linking different version
#if 0
static int linkWithSpirvLink(cl_program Program, cl_uint DeviceI,
                             char ProgramSpvPathTemp[POCL_MAX_PATHNAME_LENGTH],
                             std::vector<std::string> &SpvBinaryPaths,
                             int CreateLibrary) {
  std::vector<std::string> CompilationArgs;
  std::vector<char *> CompilationArgs2;

  CompilationArgs.push_back(SPIRV_LINK);
  if (CreateLibrary != 0) {
    CompilationArgs.push_back("--create-library");
  }
  CompilationArgs.push_back("-o");
  CompilationArgs.push_back(ProgramSpvPathTemp);
  for (auto &Path : SpvBinaryPaths) {
    CompilationArgs.push_back(Path);
  }
  CompilationArgs2.reserve(CompilationArgs.size() + 1);
  for (unsigned i = 0; i < CompilationArgs.size(); ++i)
    CompilationArgs2[i] = (char *)CompilationArgs[i].data();
  CompilationArgs2[CompilationArgs.size()] = nullptr;

  int Err =
      runAndAppendOutputToBuildLog(Program, DeviceI, CompilationArgs2.data());
  POCL_RETURN_ERROR_ON((Err != 0), CL_BUILD_PROGRAM_FAILURE,
                       "spirv-link exited with nonzero code\n");
  POCL_RETURN_ERROR_ON(!pocl_exists(ProgramSpvPathTemp),
                       CL_LINK_PROGRAM_FAILURE, "spirv-link failed\n");
  return CL_SUCCESS;
}
#endif

static int linkWithLLVMLink(cl_program Program, cl_uint DeviceI,
                            char ProgramBcPathTemp[POCL_MAX_PATHNAME_LENGTH],
                            char ProgramSpvPathTemp[POCL_MAX_PATHNAME_LENGTH],
                            std::vector<std::string> &BcBinaryPaths,
                            int CreateLibrary) {
  std::vector<std::string> CompilationArgs;
  std::vector<char *> CompilationArgs2;

  CompilationArgs.push_back(LLVM_LINK);
//  if (CreateLibrary != 0) {
//    CompilationArgs.push_back("--create-library");
//  }
  CompilationArgs.push_back("-o");
  CompilationArgs.push_back(ProgramBcPathTemp);
  for (auto &Path : BcBinaryPaths) {
    CompilationArgs.push_back(Path);
  }
  CompilationArgs2.reserve(CompilationArgs.size() + 1);
  for (unsigned i = 0; i < CompilationArgs.size(); ++i)
    CompilationArgs2[i] = (char *)CompilationArgs[i].data();
  CompilationArgs2[CompilationArgs.size()] = nullptr;

  int Err =
      runAndAppendOutputToBuildLog(Program, DeviceI, CompilationArgs2.data());
  POCL_RETURN_ERROR_ON((Err != 0), CL_BUILD_PROGRAM_FAILURE,
                       "llvm-link exited with nonzero code\n");
  POCL_RETURN_ERROR_ON(!pocl_exists(ProgramBcPathTemp),
                       CL_LINK_PROGRAM_FAILURE, "llvm-link failed to "
                                                "produce output file\n");

  Err = pocl_convert_bitcode_to_spirv(ProgramBcPathTemp,
                                      nullptr, 0,
                                      Program, DeviceI,
                                      1, // useIntelExt
                                      ProgramSpvPathTemp,
                                      nullptr, nullptr);

  POCL_RETURN_ERROR_ON((Err != 0), CL_BUILD_PROGRAM_FAILURE,
                       "llvm-spirv exited with nonzero code\n");
  return CL_SUCCESS;
}

int pocl_level0_build_source(cl_program Program, cl_uint DeviceI,
                             cl_uint NumInputHeaders,
                             const cl_program *InputHeaders,
                             const char **HeaderIncludeNames, int LinkProgram) {
#ifdef ENABLE_LLVM
  int Err = CL_SUCCESS;
  POCL_MSG_PRINT_LLVM("building from sources for device %d\n", DeviceI);

  // last arg is 0 because we never link with Clang, let the spirv-link and
  // level0 do the linking
  int Errcode = pocl_llvm_build_program(Program, DeviceI, NumInputHeaders,
                                        InputHeaders, HeaderIncludeNames, 0);
  POCL_RETURN_ERROR_ON((Errcode != CL_SUCCESS), CL_BUILD_PROGRAM_FAILURE,
                       "Failed to build program from source\n");

  cl_device_id Dev = Program->devices[DeviceI];
  Level0Device *Device = (Level0Device *)Dev->data;

  char ProgramSpvPathTemp[POCL_MAX_PATHNAME_LENGTH] = {0};
  char ProgramBcPath[POCL_MAX_PATHNAME_LENGTH];
  char ProgramSpvPath[POCL_MAX_PATHNAME_LENGTH];

  pocl_cache_program_bc_path(ProgramBcPath, Program, DeviceI);
  pocl_cache_program_spv_path(ProgramSpvPath, Program, DeviceI);

  // result of pocl_llvm_build_program
  assert(pocl_exists(ProgramBcPath));
  // we don't need llvm::Module objects, only the bitcode
  pocl_llvm_free_llvm_irs(Program, DeviceI);

  if (pocl_exists(ProgramSpvPath) != 0) {
    POCL_MSG_PRINT_LEVEL0("Found compiled SPIR-V in cache\n");
    readProgramSpv(Program, DeviceI, ProgramSpvPath);
  } else {
    char *OutputBinary = nullptr;
    uint64_t OutputBinarySize = 0;
    assert(Program->binaries[DeviceI] != nullptr);
    assert(Program->binary_sizes[DeviceI] > 0);

    Err = pocl_convert_bitcode_to_spirv(nullptr,
                                        (char*)Program->binaries[DeviceI],
                                        Program->binary_sizes[DeviceI],
                                        Program, DeviceI,
                                        1, // useIntelExt
                                        ProgramSpvPathTemp,
                                        &OutputBinary,
                                        &OutputBinarySize);
    POCL_RETURN_ERROR_ON((Err != 0), CL_BUILD_PROGRAM_FAILURE,
                         "llvm-spirv exited with nonzero code\n");

    Program->program_il = OutputBinary;
    Program->program_il_size = OutputBinarySize;
    pocl_rename(ProgramSpvPathTemp, ProgramSpvPath);
    POCL_MSG_WARN("Final SPV written: %s\n", ProgramSpvPath);
  }

  assert(Program->program_il != nullptr);
  assert(Program->program_il_size > 0);
  assert(Program->binaries[DeviceI] != nullptr);
  assert(Program->binary_sizes[DeviceI] > 0);

  if (LinkProgram != 0) {
    return Device->createProgram(Program, DeviceI);
  } else {
    // only final (linked) programs have  ZE module
    assert(Program->data[DeviceI] == nullptr);
    return CL_SUCCESS;
  }
#else
  POCL_RETURN_ERROR_ON(1, CL_BUILD_PROGRAM_FAILURE,
                       "This device requires LLVM to build from sources\n");
#endif
}

int pocl_level0_supports_binary(cl_device_id Device, size_t Length,
                                const char *Binary) {
  if (pocl_bitcode_is_spirv_execmodel_kernel(Binary, Length) != 0) {
    return 1;
  }
  // TODO : possibly support native ZE binaries
  return 0;
}

char *pocl_level0_init_build(void *Data) {
  // the -O0 helps to avoid a bunch of issues created by Clang's optimization
  // (issues for llvm-spirv translator)
  // * the freeze instruction
  // * the vector instructions (llvm.vector.reduce.add.v4i32)
  // "InvalidBitWidth: Invalid bit width in input: 63" - happens with
  // test_convert_type_X
  return strdup("-O0");
}

int pocl_level0_build_binary(cl_program Program, cl_uint DeviceI,
                             int LinkProgram, int SpirBuild) {
  cl_device_id Dev = Program->devices[DeviceI];
  Level0Device *Device = (Level0Device *)Dev->data;

  char ProgramBcPath[POCL_MAX_PATHNAME_LENGTH];
  char ProgramSpvPath[POCL_MAX_PATHNAME_LENGTH];
  char ProgramSpvPathTemp[POCL_MAX_PATHNAME_LENGTH];
  ProgramSpvPathTemp[0] = 0;
  char ProgramBcPathTemp[POCL_MAX_PATHNAME_LENGTH];
  ProgramBcPathTemp[0] = 0;
  int Err = 0;

  if (Program->pocl_binaries[DeviceI] != nullptr) {
    /* we have pocl_binaries with BOTH SPIRV and IR Bitcode */

    pocl_cache_program_spv_path(ProgramSpvPath, Program, DeviceI);

    POCL_RETURN_ERROR_ON(
        (readProgramSpv(Program, DeviceI, ProgramSpvPath) != CL_SUCCESS),
        CL_BUILD_PROGRAM_FAILURE, "Could not read compiled program.spv at %s\n",
        ProgramSpvPath);

    // TODO is this really LLVM IR
    // program.bc should be read by clCreateProgramWithBinary
    assert(Program->binaries[DeviceI] != nullptr);
    assert(Program->binary_sizes[DeviceI] != 0);

  } else {

    char *OutputBinary = nullptr;
    uint64_t OutputBinarySize = 0;

    if (Program->pocl_binaries[DeviceI] == nullptr &&
        Program->binaries[DeviceI] == nullptr) {

      /* we have only program_il, which is SPIR-V*/
      assert(Program->program_il != nullptr);
      assert(Program->program_il_size > 0);
      Err = pocl_convert_spirv_to_bitcode(ProgramSpvPathTemp,
                                          Program->program_il,
                                          Program->program_il_size,
                                          Program, DeviceI,
                                          1, // useIntelExt
                                          ProgramBcPathTemp,
                                          &OutputBinary, &OutputBinarySize);
      POCL_RETURN_ERROR_ON((Err != 0), CL_BUILD_PROGRAM_FAILURE,
                           "failed to compile SPV -> BC\n");
      Program->binaries[DeviceI] = (unsigned char *)OutputBinary;
      Program->binary_sizes[DeviceI] = OutputBinarySize;
    } else {
      /* we have program->binaries[] which should be LLVM IR SPIR */
      assert(Program->binaries[DeviceI] != nullptr);
      assert(Program->binary_sizes[DeviceI] > 0);

      int Triple =
          pocl_bitcode_is_triple((char *)Program->binaries[DeviceI],
                            Program->binary_sizes[DeviceI], "spir-unknown");
      Triple +=
          pocl_bitcode_is_triple((char *)Program->binaries[DeviceI],
                            Program->binary_sizes[DeviceI], "spir64-unknown");
      POCL_RETURN_ERROR_ON((Triple == 0), CL_BUILD_PROGRAM_FAILURE,
                           "the binary supplied to level0 driver is "
                           "not a recognized binary type\n");

      Err = pocl_convert_bitcode_to_spirv(ProgramBcPathTemp,
                                          (char *)Program->binaries[DeviceI],
                                          Program->binary_sizes[DeviceI],
                                          Program, DeviceI,
                                          1, // useIntelExt
                                          ProgramSpvPathTemp,
                                          &OutputBinary, &OutputBinarySize);
      POCL_RETURN_ERROR_ON((Err != 0), CL_BUILD_PROGRAM_FAILURE,
                           "failed to compile BC -> SPV\n");
      Program->program_il = OutputBinary;
      Program->program_il_size = OutputBinarySize;
    }

    pocl_cache_create_program_cachedir(Program, DeviceI, Program->program_il,
                                       Program->program_il_size, ProgramBcPath);
    convertProgramBcPathToSpv(ProgramBcPath, ProgramSpvPath);
    pocl_rename(ProgramSpvPathTemp, ProgramSpvPath);
    pocl_rename(ProgramBcPathTemp, ProgramBcPath);
  }

  assert(Program->program_il != nullptr);
  assert(Program->program_il_size > 0);
  // TODO is this really LLVM IR
  assert(Program->binaries[DeviceI] != nullptr);
  assert(Program->binary_sizes[DeviceI] != 0);

  if (LinkProgram != 0) {
    return Device->createProgram(Program, DeviceI);
  } else {
    // only final (linked) programs have  ZE module
    assert(Program->data[DeviceI] == nullptr);
    return CL_SUCCESS;
  }
}

int pocl_level0_link_program(cl_program Program, cl_uint DeviceI,
                             cl_uint NumInputPrograms,
                             const cl_program *InputPrograms,
                             int CreateLibrary) {
  cl_device_id Dev = Program->devices[DeviceI];
  Level0Device *Device = (Level0Device *)Dev->data;
  char ProgramBcPath[POCL_MAX_PATHNAME_LENGTH];
  char ProgramSpvPath[POCL_MAX_PATHNAME_LENGTH];

  /* we have program->binaries[] which is SPIR-V */
  assert(Program->pocl_binaries[DeviceI] == nullptr);
  assert(Program->binaries[DeviceI] == nullptr);
  assert(Program->binary_sizes[DeviceI] == 0);

  std::vector<std::string> SpvBinaryPaths;
  std::vector<std::string> BcBinaryPaths;
  std::vector<char> SpvConcatBinary;

  cl_uint I;
  cl_uint ProgsHaveProgramBC = 0;
  for (I = 0; I < NumInputPrograms; I++) {
    assert(Dev == InputPrograms[I]->devices[DeviceI]);
    POCL_LOCK_OBJ(InputPrograms[I]);

    char *Spv = (char *)InputPrograms[I]->program_il;
    assert(Spv);
    size_t Size = InputPrograms[I]->program_il_size;
    assert(Size);
    SpvConcatBinary.insert(SpvConcatBinary.end(), Spv, Spv + Size);
    if (InputPrograms[I]->binary_sizes[DeviceI] > 0) {
      ++ProgsHaveProgramBC;
      pocl_cache_program_bc_path(ProgramBcPath, InputPrograms[I], DeviceI);
      assert(pocl_exists(ProgramBcPath));
      BcBinaryPaths.push_back(ProgramBcPath);
    }

    pocl_cache_program_spv_path(ProgramSpvPath, InputPrograms[I], DeviceI);
    assert(pocl_exists(ProgramSpvPath));
    SpvBinaryPaths.push_back(ProgramSpvPath);

    POCL_UNLOCK_OBJ(InputPrograms[I]);
  }
  if (ProgsHaveProgramBC != NumInputPrograms) {
    POCL_MSG_ERR("LevelZero: not all programs have program.bc\n");
    return CL_LINK_PROGRAM_FAILURE;
  }

  pocl_cache_create_program_cachedir(Program, DeviceI, SpvConcatBinary.data(),
                                     SpvConcatBinary.size(), ProgramBcPath);
  convertProgramBcPathToSpv(ProgramBcPath, ProgramSpvPath);

  if (pocl_exists(ProgramSpvPath) && pocl_exists(ProgramBcPath)) {
    POCL_MSG_PRINT_LEVEL0("Found linked SPIR-V in cache\n");
  } else {

    char ProgramSpvPathTemp[POCL_MAX_PATHNAME_LENGTH];
    pocl_cache_tempname(ProgramSpvPathTemp, ".spv", NULL);
    char ProgramBcPathTemp[POCL_MAX_PATHNAME_LENGTH];
    pocl_cache_tempname(ProgramBcPathTemp, ".bc", NULL);

//  if (linkWithSpirvLink(Program, DeviceI, ProgramSpvPathTemp,
//              SpvBinaryPaths, CreateLibrary) != CL_SUCCESS) {

//    POCL_MSG_WARN("LevelZero : failed to link using spirv-link, trying"
//                  "with llvm-link\n");
    if (linkWithLLVMLink(Program, DeviceI, ProgramBcPathTemp,
                         ProgramSpvPathTemp, BcBinaryPaths, 0) != CL_SUCCESS) {
      POCL_MSG_ERR("LevelZero: failed to link "
                   "with both spirv-link and llvm-link\n");
      return CL_LINK_PROGRAM_FAILURE;
    }
//  }

    pocl_rename(ProgramSpvPathTemp, ProgramSpvPath);
    pocl_rename(ProgramBcPathTemp, ProgramBcPath);
  }

  readProgramSpv(Program, DeviceI, ProgramSpvPath);
  assert(Program->program_il != nullptr);
  assert(Program->program_il_size > 0);
  if (Program->binary_sizes[DeviceI] == 0) {
    char *OutputBinary = nullptr;
    uint64_t OutputBinarySize = 0;
    int Err = pocl_read_file(ProgramBcPath, &OutputBinary, &OutputBinarySize);
    POCL_RETURN_ERROR_ON((Err != 0), CL_LINK_PROGRAM_FAILURE,
                         "failed to read BC file from cache\n");
    Program->binaries[DeviceI] = (unsigned char *)OutputBinary;
    Program->binary_sizes[DeviceI] = OutputBinarySize;
  }
  assert(Program->binaries[DeviceI] != nullptr);
  assert(Program->binary_sizes[DeviceI] > 0);

  if (CreateLibrary == 0) {
    return Device->createProgram(Program, DeviceI);
  } else {
    // only final (linked) programs have  ZE module
    assert(Program->data[DeviceI] == nullptr);
    return CL_SUCCESS;
  }
}

int pocl_level0_free_program(cl_device_id ClDevice, cl_program Program,
                             unsigned ProgramDeviceI) {
  Level0Device *Device = (Level0Device *)ClDevice->data;
#ifdef ENABLE_LLVM
  pocl_llvm_free_llvm_irs(Program, ProgramDeviceI);
#endif
  /* module can be NULL if compilation fails */
  Device->freeProgram(Program, ProgramDeviceI);
  return 0;
}

int pocl_level0_setup_metadata(cl_device_id Device, cl_program Program,
                               unsigned ProgramDeviceI) {
  assert(Program->data[ProgramDeviceI] != NULL);

  // TODO this is using program_il as source
  int32_t *Stream = (int32_t *)Program->program_il;
  size_t StreamSize = Program->program_il_size / 4;
  OpenCLFunctionInfoMap KernelInfoMap;
  if (!parseSPIRV(Stream, StreamSize, KernelInfoMap)) {
    POCL_MSG_ERR("Unable to parse SPIR-V module of the program\n");
    return 0;
  }

  Program->num_kernels = KernelInfoMap.size();
  if (Program->num_kernels == 0) {
    POCL_MSG_WARN("No kernels found in program.\n");
    return 1;
  }

  Program->kernel_meta = (pocl_kernel_metadata_t *)calloc(
      Program->num_kernels, sizeof(pocl_kernel_metadata_t));

  uint32_t Idx = 0;
  for (auto &I : KernelInfoMap) {
    std::string Name = I.first;
    OCLFuncInfo *FI = I.second.get();

    pocl_kernel_metadata_t *Meta = &Program->kernel_meta[Idx];
    Meta->data = (void **)calloc(Program->num_devices, sizeof(void *));
    Meta->num_args = FI->ArgTypeInfo.size();
    Meta->name = strdup(Name.c_str());

    // Level zero driver handles the static locals
    Meta->num_locals = 0;
    Meta->local_sizes = nullptr;

    Meta->max_subgroups =
        (size_t *)calloc(Program->num_devices, sizeof(size_t));
    Meta->compile_subgroups =
        (size_t *)calloc(Program->num_devices, sizeof(size_t));
    Meta->max_workgroup_size =
        (size_t *)calloc(Program->num_devices, sizeof(size_t));
    Meta->preferred_wg_multiple =
        (size_t *)calloc(Program->num_devices, sizeof(size_t));
    Meta->local_mem_size =
        (cl_ulong *)calloc(Program->num_devices, sizeof(cl_ulong));
    Meta->private_mem_size =
        (cl_ulong *)calloc(Program->num_devices, sizeof(cl_ulong));
    Meta->spill_mem_size =
        (cl_ulong *)calloc(Program->num_devices, sizeof(cl_ulong));

    // ZE kernel metadata; TODO with JIT, we don't have the ZE module
    // to extract the metadata - this needs to be extracted from SPIR-V
    // required workgroup size, attributes, subgroups, priv/local mem sizes
#if 0
    {
      ze_module_handle_t ModuleH = ProgramSPtr->get()->getAnyHandle();
      ze_kernel_handle_t HKernel = nullptr;
      ze_kernel_desc_t KernelDesc = {
          ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr,
          0, // flags | ZE_KERNEL_FLAG_FORCE_RESIDENCY
          Meta->name};
      LEVEL0_CHECK_RET(0, zeKernelCreate(ModuleH, &KernelDesc, &HKernel));

      ze_kernel_preferred_group_size_properties_t PrefGroupSize = {
          ZE_STRUCTURE_TYPE_KERNEL_PREFERRED_GROUP_SIZE_PROPERTIES, NULL, 0};
      ze_kernel_properties_t KernelProps{};
      KernelProps.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
      KernelProps.pNext = (void *)&PrefGroupSize;
      LEVEL0_CHECK_RET(0, zeKernelGetProperties(HKernel, &KernelProps));

      assert(Meta->num_args == KernelProps.numKernelArgs);
      Meta->reqd_wg_size[0] = KernelProps.requiredGroupSizeX;
      Meta->reqd_wg_size[1] = KernelProps.requiredGroupSizeY;
      Meta->reqd_wg_size[2] = KernelProps.requiredGroupSizeZ;
      // TODO: setup of these attributes is missing
      // meta->vec_type_hint
      // meta->wg_size_hint

      uint32_t AttrSize = 0;
      char *AttrString = nullptr;
      LEVEL0_CHECK_RET(
          0, zeKernelGetSourceAttributes(HKernel, &AttrSize, &AttrString));
      if (AttrSize > 0) {
        Meta->attributes = strdup(AttrString);
      }

      LEVEL0_CHECK_RET(0, zeKernelDestroy(HKernel));

      Meta->max_subgroups[ProgramDeviceI] = KernelProps.maxSubgroupSize;
      Meta->compile_subgroups[ProgramDeviceI] =
          KernelProps.requiredSubgroupSize;
      Meta->max_workgroup_size[ProgramDeviceI] = 0; // TODO
      Meta->preferred_wg_multiple[ProgramDeviceI] =
          PrefGroupSize.preferredMultiple;
      Meta->local_mem_size[ProgramDeviceI] = KernelProps.localMemSize;
      Meta->private_mem_size[ProgramDeviceI] = KernelProps.privateMemSize;
      Meta->spill_mem_size[ProgramDeviceI] = KernelProps.spillMemSize;
#if 0
      /// TODO:
      /// required number of subgroups per thread group,
      /// or zero if there is no required number of subgroups
      uint32_t requiredNumSubGroups;

      /// [out] required subgroup size,
      /// or zero if there is no required subgroup size
      uint32_t requiredSubgroupSize;
#endif
    }
#endif

    // ARGUMENTS
    if (Meta->num_args != 0u) {
      Meta->arg_info = (struct pocl_argument_info *)calloc(
          Meta->num_args, sizeof(struct pocl_argument_info));

      for (uint32_t J = 0; J < Meta->num_args; ++J) {
        cl_kernel_arg_address_qualifier Addr;
        cl_kernel_arg_access_qualifier Access;
        Addr = CL_KERNEL_ARG_ADDRESS_PRIVATE;
        Access = CL_KERNEL_ARG_ACCESS_NONE;
        Meta->arg_info[J].name = strdup(FI->ArgTypeInfo[J].Name.c_str());
        Meta->arg_info[J].type_name = nullptr;
        switch (FI->ArgTypeInfo[J].Type) {
        case OCLType::POD: {
          Meta->arg_info[J].type = POCL_ARG_TYPE_NONE;
          Meta->arg_info[J].type_size = FI->ArgTypeInfo[J].Size;
          break;
        }
        case OCLType::Pointer: {
          Meta->arg_info[J].type = POCL_ARG_TYPE_POINTER;
          Meta->arg_info[J].type_size = sizeof(cl_mem);
          switch (FI->ArgTypeInfo[J].Space) {
          case OCLSpace::Private:
            Addr = CL_KERNEL_ARG_ADDRESS_PRIVATE;
            break;
          case OCLSpace::Local:
            Addr = CL_KERNEL_ARG_ADDRESS_LOCAL;
            break;
          case OCLSpace::Global:
            Addr = CL_KERNEL_ARG_ADDRESS_GLOBAL;
            break;
          case OCLSpace::Constant:
            Addr = CL_KERNEL_ARG_ADDRESS_CONSTANT;
            break;
          case OCLSpace::Unknown:
            Addr = CL_KERNEL_ARG_ADDRESS_PRIVATE;
            break;
          }
          break;
        }
        case OCLType::Image: {
          Meta->arg_info[J].type = POCL_ARG_TYPE_IMAGE;
          Meta->arg_info[J].type_size = sizeof(cl_mem);
          Addr = CL_KERNEL_ARG_ADDRESS_GLOBAL;
          bool Readable = FI->ArgTypeInfo[J].Attrs.ReadableImg;
          bool Writable = FI->ArgTypeInfo[J].Attrs.WriteableImg;
          if (Readable && Writable) {
            Access = CL_KERNEL_ARG_ACCESS_READ_WRITE;
          }
          if (Readable && !Writable) {
            Access = CL_KERNEL_ARG_ACCESS_READ_ONLY;
          }
          if (!Readable && Writable) {
            Access = CL_KERNEL_ARG_ACCESS_WRITE_ONLY;
          }
          break;
        }
        case OCLType::Sampler: {
          Meta->arg_info[J].type = POCL_ARG_TYPE_SAMPLER;
          Meta->arg_info[J].type_size = sizeof(cl_mem);
          break;
        }
        case OCLType::Opaque: {
          POCL_MSG_ERR("Unknown OCL type OPaque\n");
          Meta->arg_info[J].type = POCL_ARG_TYPE_NONE;
          Meta->arg_info[J].type_size = FI->ArgTypeInfo[J].Size;
          break;
        }
        }
        Meta->arg_info[J].address_qualifier = Addr;
        Meta->arg_info[J].access_qualifier = Access;
        Meta->arg_info[J].type_qualifier = CL_KERNEL_ARG_TYPE_NONE;
        if (FI->ArgTypeInfo[J].Attrs.Constant) {
          Meta->arg_info[J].type_qualifier |= CL_KERNEL_ARG_TYPE_CONST;
        }
        if (FI->ArgTypeInfo[J].Attrs.Restrict) {
          Meta->arg_info[J].type_qualifier |= CL_KERNEL_ARG_TYPE_RESTRICT;
        }
        if (FI->ArgTypeInfo[J].Attrs.Volatile) {
          Meta->arg_info[J].type_qualifier |= CL_KERNEL_ARG_TYPE_VOLATILE;
        }
      }

      // TODO: POCL_HAS_KERNEL_ARG_TYPE_NAME missing
      Meta->has_arg_metadata = POCL_HAS_KERNEL_ARG_ACCESS_QUALIFIER |
                               POCL_HAS_KERNEL_ARG_ADDRESS_QUALIFIER |
                               POCL_HAS_KERNEL_ARG_TYPE_QUALIFIER |
                               POCL_HAS_KERNEL_ARG_NAME;
    }

    ++Idx;
  }

  return 1;
}

int pocl_level0_create_kernel(cl_device_id Device, cl_program Program,
                              cl_kernel Kernel, unsigned ProgramDeviceI) {
  assert(Program->data[ProgramDeviceI] != nullptr);
  Level0Program *L0Program = (Level0Program *)Program->data[ProgramDeviceI];
  Level0Kernel *Ker =
      DriverInstance->getJobSched().createKernel(L0Program, Kernel->name);
  Kernel->data[ProgramDeviceI] = Ker;
  return Ker ? CL_SUCCESS : CL_OUT_OF_RESOURCES;
}

int pocl_level0_free_kernel(cl_device_id Device, cl_program Program,
                            cl_kernel Kernel, unsigned ProgramDeviceI) {

  assert(Program->data[ProgramDeviceI] != nullptr);
  assert(Kernel->data[ProgramDeviceI] != nullptr);

  Level0Program *L0Program = (Level0Program *)Program->data[ProgramDeviceI];
  Level0Kernel *L0Kernel = (Level0Kernel *)Kernel->data[ProgramDeviceI];

  bool Res = DriverInstance->getJobSched().releaseKernel(L0Program, L0Kernel);
  assert(Res == true);

  return 0;
}

int pocl_level0_build_poclbinary(cl_program Program, cl_uint DeviceI) {

  assert(Program->build_status == CL_BUILD_SUCCESS);
  if (Program->num_kernels == 0) {
    return CL_SUCCESS;
  }

  /* For binaries of other than Executable type (libraries, compiled but
   * not linked programs, etc), do not attempt to compile the kernels. */
  if (Program->binary_type != CL_PROGRAM_BINARY_TYPE_EXECUTABLE) {
    return CL_SUCCESS;
  }

  assert(Program->binaries[DeviceI]);

  return CL_SUCCESS;
}

int pocl_level0_init_queue(cl_device_id Dev, cl_command_queue Queue) {
  PoclL0QueueData *QD = new PoclL0QueueData;
  Queue->data = QD;
  POCL_INIT_COND(QD->Cond);
  return CL_SUCCESS;
}

int pocl_level0_free_queue(cl_device_id Dev, cl_command_queue Queue) {
  PoclL0QueueData *QD = (PoclL0QueueData *)Queue->data;
  if (QD == nullptr)
    return CL_SUCCESS;

  POCL_DESTROY_COND(QD->Cond);
  delete QD;
  Queue->data = nullptr;
  return CL_SUCCESS;
}

void pocl_level0_notify_cmdq_finished(cl_command_queue Queue) {
  /* must be called with CQ already locked.
   * this must be a broadcast since there could be multiple
   * user threads waiting on the same command queue
   * in pthread_scheduler_wait_cq(). */
  PoclL0QueueData *QD = (PoclL0QueueData *)Queue->data;
  POCL_BROADCAST_COND(QD->Cond);
}

void pocl_level0_notify_event_finished(cl_event Event) {
  pocl_cond_t *EventCond = (pocl_cond_t *)Event->data;
  POCL_BROADCAST_COND(*EventCond);
}

void pocl_level0_free_event_data(cl_event Event) {
  if (Event->data == nullptr) {
    return;
  }
  pocl_cond_t *EventCond = (pocl_cond_t *)Event->data;
  POCL_DESTROY_COND(*EventCond);
  POCL_MEM_FREE(Event->data);
}

void pocl_level0_join(cl_device_id Device, cl_command_queue Queue) {
  POCL_LOCK_OBJ(Queue);
  PoclL0QueueData *QD = (PoclL0QueueData *)Queue->data;
  while (true) {
    if (Queue->command_count == 0) {
      POCL_UNLOCK_OBJ(Queue);
      return;
    } else {
      POCL_WAIT_COND(QD->Cond, Queue->pocl_lock);
    }
  }
}

static bool pocl_level0_queue_supports_batching(cl_command_queue CQ,
                                                Level0Device *Device) {
  if (!Device->supportsUniversalQueues())
    return false;
#ifdef LEVEL0_IMMEDIATE_CMDLIST
  // batching + Immediate CmdLists are currently not supported, TBD
  return false;
#else
  if (CQ->properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
    return false;
  if (CQ->properties & CL_QUEUE_PROFILING_ENABLE)
    return false;
  return true;
#endif
}

static bool pocl_level0_can_event_be_batched(cl_event Ev, cl_event LastEv) {
  assert(Ev != LastEv);

  // immediately ready event
  if (Ev->wait_list == nullptr)
    return true;
  // has one event only, and it's the previous in the queue
  if (Ev->wait_list->event == LastEv && Ev->wait_list->next == nullptr)
    return true;

  return false;
}

static void pocl_level0_batch_submitted_events(PoclL0QueueData *QD,
                                               Level0Device *Device,
                                               BatchType &SubmitBatch) {
  cl_event LastEv = nullptr;
  while (!QD->UnsubmittedEventList.empty()) {
    cl_event Ev = QD->UnsubmittedEventList.front();
    if (pocl_level0_can_event_be_batched(Ev, LastEv)) {
      SubmitBatch.push_back(Ev);
      QD->UnsubmittedEventList.pop_front();
      LastEv = Ev;
    } else {
      break;
    }
  }
  POCL_MSG_PRINT_LEVEL0("Processing Batch: Submitted %zu || New batch: %zu\n",
                        QD->UnsubmittedEventList.size(), SubmitBatch.size());
}

void pocl_level0_flush(cl_device_id ClDev, cl_command_queue Queue) {
  Level0Device *Device = (Level0Device *)ClDev->data;
  PoclL0QueueData *QD = (PoclL0QueueData *)Queue->data;

  if (!pocl_level0_queue_supports_batching(Queue, Device))
    return;

  BatchType SubmitBatch;
  POCL_LOCK_OBJ(Queue);
  if (!QD->UnsubmittedEventList.empty()) {
    pocl_level0_batch_submitted_events(QD, Device, SubmitBatch);
  }
  POCL_UNLOCK_OBJ(Queue);
  if (SubmitBatch.empty()) {
    POCL_MSG_PRINT_LEVEL0("FLUSH: SubmitBatch EMPTY\n");
  } else {
    POCL_MSG_PRINT_LEVEL0("FLUSH: SubmitBatch SIZE %zu\n", SubmitBatch.size());
    Device->pushCommandBatch(std::move(SubmitBatch));
    assert(SubmitBatch.empty());
  }
}

void pocl_level0_submit(_cl_command_node *Node, cl_command_queue Queue) {
  Level0Device *Device = (Level0Device *)Queue->device->data;
  PoclL0QueueData *QD = (PoclL0QueueData *)Queue->data;
  cl_event Ev = Node->sync.event.event;

  Node->ready = CL_TRUE;

  if (pocl_level0_queue_supports_batching(Queue, Device)) {
    POCL_LOCK_OBJ(Queue);
    QD->UnsubmittedEventList.push_back(Ev);
    // to limit latency of the first launch, limit the size of the batch
    if (QD->UnsubmittedEventList.size() >= BatchSizeLimit) {
      BatchType SubmitBatch;
      pocl_level0_batch_submitted_events(QD, Device, SubmitBatch);
      if (!SubmitBatch.empty())
        Device->pushCommandBatch(std::move(SubmitBatch));
    }
    POCL_UNLOCK_OBJ(Queue);
  } else {
    // fallback processing for all other events
    if (pocl_command_is_ready(Ev) != 0) {
      pocl_update_event_submitted(Ev);
      Device->pushCommand(Node);
    }
  }
  POCL_UNLOCK_OBJ(Ev);
}

void pocl_level0_notify(cl_device_id ClDev, cl_event Event, cl_event Finished) {
  _cl_command_node *Node = Event->command;
  Level0Device *Device = (Level0Device *)ClDev->data;

  if (Finished->status < CL_COMPLETE) {
    // remove the Event from unsubmitted list
    POCL_LOCK_OBJ(Event->queue);
    PoclL0QueueData *QD = (PoclL0QueueData *)Event->queue->data;
    auto It = std::find(QD->UnsubmittedEventList.begin(),
                        QD->UnsubmittedEventList.end(), Event);
    if (It != QD->UnsubmittedEventList.end())
      QD->UnsubmittedEventList.erase(It);
    POCL_UNLOCK_OBJ(Event->queue);
    pocl_update_event_failed(Event);
    return;
  }

  // node is ready to execute
  POCL_MSG_PRINT_LEVEL0("notify on event %zu | READY %i\n", Event->id,
                        Node->ready);

  assert(Event->queue);
  if (pocl_level0_queue_supports_batching(Event->queue, Device)) {
    BatchType SubmitBatch;
    POCL_LOCK_OBJ(Event->queue);
    PoclL0QueueData *QD = (PoclL0QueueData *)Event->queue->data;
    bool IsFirstEvent = !QD->UnsubmittedEventList.empty() &&
                        (Event == QD->UnsubmittedEventList.front());
    if (IsFirstEvent) {
      if (!QD->UnsubmittedEventList.empty()) {
        pocl_level0_batch_submitted_events(QD, Device, SubmitBatch);
      }
    }
    POCL_UNLOCK_OBJ(Event->queue);
    if (!SubmitBatch.empty()) {
      Device->pushCommandBatch(std::move(SubmitBatch));
    }
  } else {
    if (Node->ready == CL_TRUE && pocl_command_is_ready(Event) != 0) {
      pocl_update_event_submitted(Event);
      Device->pushCommand(Node);
    }
  }
}

void pocl_level0_update_event(cl_device_id ClDevice, cl_event Event) {
  if (Event->data == nullptr) {
    pocl_cond_t *EventCond = (pocl_cond_t *)malloc(sizeof(pocl_cond_t));
    assert(EventCond);
    POCL_INIT_COND(*EventCond);
    Event->data = (void *)EventCond;
  }
  if (Event->status == CL_QUEUED) {
    Event->time_queue = pocl_gettimemono_ns();
  }
  if (Event->status == CL_SUBMITTED) {
    Event->time_submit = pocl_gettimemono_ns();
  }
}

void pocl_level0_wait_event(cl_device_id ClDevice, cl_event Event) {
  POCL_MSG_PRINT_LEVEL0("device->wait_event on event %zu\n", Event->id);
  pocl_cond_t *EventCond = (pocl_cond_t *)Event->data;

  POCL_LOCK_OBJ(Event);
  while (Event->status > CL_COMPLETE) {
    POCL_WAIT_COND(*EventCond, Event->pocl_lock);
  }
  POCL_UNLOCK_OBJ(Event);
}

int pocl_level0_alloc_mem_obj(cl_device_id ClDevice, cl_mem Mem, void *HostPtr) {
  Level0Device *Device = (Level0Device *)ClDevice->data;
  pocl_mem_identifier *P = &Mem->device_ptrs[ClDevice->global_mem_id];

  assert(P->mem_ptr == NULL);

  P->extra = 0;
  /* for Images, ze_image_handler_t */
  P->extra_ptr = NULL;

  /* won't preallocate host-visible memory for images,
   * only for buffers */
  if (((Mem->flags & CL_MEM_ALLOC_HOST_PTR) != 0u) &&
      (Mem->mem_host_ptr == NULL) && (Mem->is_image != 0u)) {
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  // The cl_mems are allocated as shared USM by default to make clEnqueueMap()
  // implementation simpler. TODO: measure the perf. impact of this on some
  // relevant platform and use memcopies for map.

  void *Allocation = nullptr;
  // special handling for clCreateBuffer called on SVM or USM pointer
  if (((Mem->flags & CL_MEM_USE_HOST_PTR) != 0u) &&
      (Mem->mem_host_ptr_is_svm != 0)) {
    P->mem_ptr = Mem->mem_host_ptr;
    P->version = Mem->mem_host_ptr_version;
  } else if (Mem->flags & CL_MEM_DEVICE_ADDRESS_EXT) {
    // Treat cl_ext_buffer_device_address identically as USM Device.
    // If we passed an SVM/USM address, we can use it directly in the
    // previous branch. That should be at least a USM Device allocation.
    P->mem_ptr = Device->allocSharedMem(Mem->size, 0);
    P->version = 0;
  } else {
    bool Compress = false;
    if (pocl_get_bool_option("POCL_LEVEL0_COMPRESS", 0)) {
      Compress = (Mem->flags & CL_MEM_READ_ONLY) > 0;
    }
    Allocation = Device->allocSharedMem(Mem->size, Compress);
    if (Allocation == nullptr) {
      return CL_MEM_OBJECT_ALLOCATION_FAILURE;
    }
    P->mem_ptr = Allocation;
    P->version = 0;
  }

  if (Mem->is_image != 0u) {
    // image attributes must be already set up
    assert(Mem->image_channel_data_type != 0);
    assert(Mem->image_channel_order != 0);
    ze_image_handle_t Image = Device->allocImage(
        Mem->image_channel_data_type, Mem->image_channel_order, Mem->type,
        Mem->flags, Mem->image_width, Mem->image_height, Mem->image_depth,
        Mem->image_array_size);
    if (Image == nullptr) {
      if (Allocation != nullptr) {
        Device->freeMem(Allocation);
      }
      P->mem_ptr = nullptr;
      P->version = 0;
      return CL_MEM_OBJECT_ALLOCATION_FAILURE;
    } else {
      P->extra_ptr = (void *)Image;
    }
  }

  // since we allocate shared memory, use it for mem_host_ptr
  if (Mem->mem_host_ptr == nullptr) {
    assert((Mem->flags & CL_MEM_USE_HOST_PTR) == 0);
    Mem->mem_host_ptr = Allocation;
    Mem->mem_host_ptr_version = 0;
    ++Mem->mem_host_ptr_refcount;
  }

  POCL_MSG_PRINT_MEMORY("level0 ALLOCATED | MEM_HOST_PTR %p SIZE %zu | "
                        "level0 DEV BUF %p | STA BUF %p | EXTRA_PTR %p \n",
                        Mem->mem_host_ptr, Mem->size, P->mem_ptr,
                        (void *)P->extra, P->extra_ptr);

  return CL_SUCCESS;
}

void pocl_level0_free(cl_device_id ClDevice, cl_mem Mem) {
  Level0Device *Device = (Level0Device *)ClDevice->data;
  pocl_mem_identifier *P = &Mem->device_ptrs[ClDevice->global_mem_id];

  POCL_MSG_PRINT_MEMORY("level0 DEVICE FREE | PTR %p SIZE %zu \n", P->mem_ptr,
                        Mem->size);

  if (Mem->is_image != 0u) {
    assert(P->extra_ptr != nullptr);
    ze_image_handle_t Image = (ze_image_handle_t)P->extra_ptr;
    pocl::Level0Device::freeImage(Image);
  }

  // special handling for clCreateBuffer called on SVM pointer
  if (((Mem->flags & CL_MEM_USE_HOST_PTR) != 0u) &&
      (Mem->mem_host_ptr_is_svm != 0)) {
    P->mem_ptr = nullptr;
    P->version = 0;
  } else {
    Device->freeMem(P->mem_ptr);
  }

  if (Mem->mem_host_ptr != nullptr && Mem->mem_host_ptr == P->mem_ptr) {
    assert((Mem->flags & CL_MEM_USE_HOST_PTR) == 0);
    Mem->mem_host_ptr = nullptr;
    Mem->mem_host_ptr_version = 0;
    --Mem->mem_host_ptr_refcount;
    // TODO refcounting
    // assert(mem->mem_host_ptr_refcount == 0);
  }

  P->mem_ptr = nullptr;
  P->version = 0;
  P->extra_ptr = nullptr;
  P->extra = 0;
}

cl_int pocl_level0_get_mapping_ptr(void *Data, pocl_mem_identifier *MemId,
                                   cl_mem Mem, mem_mapping_t *Map) {
  /* assume buffer is allocated */
  assert(MemId->mem_ptr != NULL);

  if (Mem->is_image != 0u) {
    Map->host_ptr = (char *)MemId->mem_ptr + Map->offset;
  } else if ((Mem->flags & CL_MEM_USE_HOST_PTR) != 0u) {
    Map->host_ptr = (char *)Mem->mem_host_ptr + Map->offset;
  } else {
    Map->host_ptr = (char *)MemId->mem_ptr + Map->offset;
  }
  /* POCL_MSG_ERR ("map HOST_PTR: %p | SIZE %zu | OFFS %zu | DEV PTR: %p \n",
                  map->host_ptr, map->size, map->offset, mem_id->mem_ptr); */
  assert(Map->host_ptr);
  return CL_SUCCESS;
}

cl_int pocl_level0_free_mapping_ptr(void *Data, pocl_mem_identifier *MemId,
                                    cl_mem Mem, mem_mapping_t *Map) {
  Map->host_ptr = NULL;
  return CL_SUCCESS;
}

int pocl_level0_create_sampler(cl_device_id ClDevice, cl_sampler Samp,
                               unsigned ContextDeviceI) {
  Level0Device *Device = (Level0Device *)ClDevice->data;
  ze_sampler_handle_t HSampler = Device->allocSampler(
      Samp->addressing_mode, Samp->filter_mode, Samp->normalized_coords);
  if (HSampler == nullptr) {
    POCL_MSG_ERR("Failed to create sampler\n");
    return CL_FAILED;
  }
  Samp->device_data[ClDevice->dev_id] = HSampler;
  return CL_SUCCESS;
}

int pocl_level0_free_sampler(cl_device_id ClDevice, cl_sampler Samp,
                             unsigned ContextDeviceI) {

  ze_sampler_handle_t HSampler =
      (ze_sampler_handle_t)Samp->device_data[ClDevice->dev_id];
  if (HSampler != nullptr) {
    pocl::Level0Device::freeSampler(HSampler);
  }
  return CL_SUCCESS;
}

void *pocl_level0_svm_alloc(cl_device_id Dev, cl_svm_mem_flags Flags,
                            size_t Size) {
  Level0Device *Device = (Level0Device *)Dev->data;
  bool Compress = false;
  if (pocl_get_bool_option("POCL_LEVEL0_COMPRESS", 0)) {
    Compress = (Flags & CL_MEM_READ_ONLY) > 0;
  }
  return Device->allocSharedMem(Size, Compress);
}

void pocl_level0_svm_free(cl_device_id Dev, void *SvmPtr) {
  Level0Device *Device = (Level0Device *)Dev->data;
  Device->freeMem(SvmPtr);
}

void *pocl_level0_usm_alloc(cl_device_id Dev, unsigned AllocType,
                            cl_mem_alloc_flags_intel Flags, size_t Size,
                            cl_int *ErrCode) {
  Level0Device *Device = (Level0Device *)Dev->data;
  int errcode = CL_SUCCESS;
  void *Ptr = nullptr;
  ze_host_mem_alloc_flags_t HostZeFlags = 0;
  ze_device_mem_alloc_flags_t DevZeFlags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_CACHED;
  if (Flags & CL_MEM_ALLOC_WRITE_COMBINED_INTEL)
    HostZeFlags |= ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED;
  if (Flags & CL_MEM_ALLOC_INITIAL_PLACEMENT_DEVICE_INTEL)
    DevZeFlags |= ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT;
  if (Flags & CL_MEM_ALLOC_INITIAL_PLACEMENT_HOST_INTEL)
    HostZeFlags |= ZE_HOST_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT;

  switch (AllocType) {
  case CL_MEM_TYPE_HOST_INTEL:
    POCL_GOTO_ERROR_ON(!Device->supportsHostUSM(), CL_INVALID_OPERATION,
                       "Device does not support Host USM allocations\n");
    Ptr = Device->allocHostMem(Size, HostZeFlags);
    break;
  case CL_MEM_TYPE_DEVICE_INTEL:
    POCL_GOTO_ERROR_ON(!Device->supportsDeviceUSM(), CL_INVALID_OPERATION,
                       "Device does not support Device USM allocations\n");
    Ptr = Device->allocDeviceMem(Size, DevZeFlags);
    break;
  case CL_MEM_TYPE_SHARED_INTEL:
    POCL_GOTO_ERROR_ON(!Device->supportsSingleSharedUSM(), CL_INVALID_OPERATION,
                       "Device does not support Shared USM allocations\n");
    Ptr = Device->allocSharedMem(Size, false, DevZeFlags, HostZeFlags);
    break;
  default:
    POCL_MSG_ERR("Unknown USM AllocType requested\n");
    errcode = CL_INVALID_PROPERTY;
  }
ERROR:
  if (ErrCode)
    *ErrCode = errcode;
  return Ptr;
}

void pocl_level0_usm_free(cl_device_id Dev, void *SvmPtr) {
  Level0Device *Device = (Level0Device *)Dev->data;
  Device->freeMem(SvmPtr);
}

void pocl_level0_usm_free_blocking(cl_device_id Dev, void *SvmPtr) {
  Level0Device *Device = (Level0Device *)Dev->data;
  Device->freeMemBlocking(SvmPtr);
}

cl_int pocl_level0_get_device_info_ext(cl_device_id Dev,
                                       cl_device_info param_name,
                                       size_t param_value_size,
                                       void *param_value,
                                       size_t *param_value_size_ret) {
  Level0Device *Device = (Level0Device *)Dev->data;

  switch (param_name) {

  case CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL:
  case CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL:
  case CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL:
  case CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL:
  case CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL: {
    cl_bitfield Caps = Device->getMemCaps(param_name);
    POCL_RETURN_GETINFO(cl_bitfield, Caps);
  }

  case CL_DEVICE_SUB_GROUP_SIZES_INTEL: {
    const std::vector<size_t> &SupportedSGSizes =
        Device->getSupportedSubgroupSizes();
    if (!SupportedSGSizes.empty()) {
      POCL_RETURN_GETINFO_ARRAY(size_t, SupportedSGSizes.size(),
                                SupportedSGSizes.data());
    } else {
      POCL_RETURN_GETINFO(size_t, 0);
    }
  }

  default:
    return CL_INVALID_VALUE;
  }
}

/*

Enumeration type and values for the param_name parameter to
clGetMemAllocInfoINTEL to query information about a Unified Shared Memory
allocation. Optional allocation properties may also be queried using
clGetMemAllocInfoINTEL:

typedef cl_uint cl_mem_info_intel;

#define CL_MEM_ALLOC_TYPE_INTEL         0x419A
#define CL_MEM_ALLOC_BASE_PTR_INTEL     0x419B
#define CL_MEM_ALLOC_SIZE_INTEL         0x419C
#define CL_MEM_ALLOC_DEVICE_INTEL       0x419D
// CL_MEM_ALLOC_FLAGS_INTEL - defined above

Enumeration type and values describing the type of Unified Shared Memory
allocation. Returned by clGetMemAllocInfoINTEL when param_name is
CL_MEM_ALLOC_TYPE_INTEL:

typedef cl_uint cl_unified_shared_memory_type_intel;

#define CL_MEM_TYPE_UNKNOWN_INTEL       0x4196
#define CL_MEM_TYPE_HOST_INTEL          0x4197
#define CL_MEM_TYPE_DEVICE_INTEL        0x4198
#define CL_MEM_TYPE_SHARED_INTEL        0x4199

*/

/*

Accepted value for the param_name parameter to clSetKernelExecInfo to specify
that the kernel may indirectly access Unified Shared Memory allocations of the
specified type:

#define CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL      0x4200
#define CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL    0x4201
#define CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL    0x4202

Accepted value for the param_name parameter to clSetKernelExecInfo to specify a
set of Unified Shared Memory allocations that the kernel may indirectly access:

#define CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL                  0x4203

*/

cl_int pocl_level0_set_kernel_exec_info_ext(
    cl_device_id Dev, unsigned ProgramDeviceI, cl_kernel Kernel,
    cl_uint param_name, size_t param_value_size, const void *param_value) {

  assert(Kernel->data[ProgramDeviceI] != nullptr);
  Level0Kernel *L0Kernel = (Level0Kernel *)Kernel->data[ProgramDeviceI];
  assert(L0Kernel != nullptr);
  switch (param_name) {
  case CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM: {
    if (Dev->svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)
      return CL_SUCCESS;
    else {
      POCL_RETURN_ERROR_ON(
          1, CL_INVALID_OPERATION,
          "This device doesn't support fine-grain system allocations\n");
    }
  }
  case CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL:
  case CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL:
  case CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL: {
    cl_bool value;
    assert(param_value_size == sizeof(cl_bool));
    memcpy(&value, param_value, sizeof(cl_bool));
    ze_kernel_indirect_access_flag_t Flag;
    switch (param_name) {
    case CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL:
      Flag = ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST;
      break;
    case CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL:
      Flag = ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED;
      break;
    case CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL:
      Flag = ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE;
      break;
    }

    L0Kernel->setIndirectAccess(Flag, (value != CL_FALSE));
    return CL_SUCCESS;
  }

  case CL_KERNEL_EXEC_INFO_DEVICE_PTRS_EXT:
  case CL_KERNEL_EXEC_INFO_SVM_PTRS:
  case CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL: {
    std::map<void *, size_t> UsedPtrs;
    cl_uint NumElem = param_value_size / sizeof(void *);
    if (NumElem == 0)
      return CL_INVALID_ARG_VALUE;
    void **Elems = (void **)param_value;
    size_t AllocationSize;
    void *RawPtr;
    // find the allocation sizes for the pointers. Needed for L0 API
    for (cl_uint i = 0; i < NumElem; ++i) {
      AllocationSize = 0;
      RawPtr = nullptr;
      // TODO: DEVICE ptrs do not have vm_ptr set, the check will fail.

      if (param_name == CL_KERNEL_EXEC_INFO_DEVICE_PTRS_EXT) {
        pocl_raw_ptr *DevPtr =
            pocl_find_raw_ptr_with_dev_ptr(Kernel->context, Elems[i]);
        POCL_RETURN_ERROR_ON((DevPtr == nullptr), CL_INVALID_VALUE,
                             "Invalid pointer given to the call\n");
        AllocationSize = DevPtr->size;
        RawPtr = DevPtr->dev_ptr;
      } else {
        int err = pocl_svm_check_get_pointer(Kernel->context, Elems[i], 1,
                                             &AllocationSize, &RawPtr);
        POCL_RETURN_ERROR_ON((err != CL_SUCCESS), CL_INVALID_VALUE,
                             "Invalid pointer given to the call\n");
      }
      assert(AllocationSize > 0);
      UsedPtrs[RawPtr] = AllocationSize;
    }
    L0Kernel->setAccessedPointers(UsedPtrs);
    return CL_SUCCESS;
  }

  default:
    return CL_INVALID_VALUE;
  }
}
