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

#include "level0-compilation.hh"
#include "level0-driver.hh"

using namespace pocl;

static void pocl_level0_local_size_optimizer(cl_device_id Dev, cl_kernel Kernel,
                                             unsigned device_i, size_t global_x,
                                             size_t global_y, size_t global_z,
                                             size_t *local_x, size_t *local_y,
                                             size_t *local_z) {
  assert(Kernel->data[device_i] != nullptr);
  Level0Kernel *L0Kernel = (Level0Kernel *)Kernel->data[device_i];
  ze_kernel_handle_t hKernel = L0Kernel->getAnyCreated();

  uint32_t suggestedX = 0;
  uint32_t suggestedY = 0;
  uint32_t suggestedZ = 0;
  ze_result_t res = ZE_RESULT_ERROR_DEVICE_LOST;
  if (hKernel != nullptr) {
    res = zeKernelSuggestGroupSize(hKernel, global_x, global_y, global_z,
                                   &suggestedX, &suggestedY, &suggestedZ);
  }
  if (res != ZE_RESULT_SUCCESS) {
    POCL_MSG_WARN("zeKernelSuggestGroupSize FAILED\n");
    pocl_default_local_size_optimizer(Dev, Kernel, device_i, global_x, global_y,
                                      global_z, local_x, local_y, local_z);
  } else {
    *local_x = suggestedX;
    *local_y = suggestedY;
    *local_z = suggestedZ;
  }
}

void pocl_level0_init_device_ops(struct pocl_device_ops *ops) {
  ops->device_name = "level0";

  ops->probe = pocl_level0_probe;
  ops->init = pocl_level0_init;
  ops->uninit = pocl_level0_uninit;
  ops->reinit = pocl_level0_reinit;

  ops->get_mapping_ptr = pocl_level0_get_mapping_ptr;
  ops->free_mapping_ptr = pocl_level0_free_mapping_ptr;

  ops->compute_local_size = pocl_level0_local_size_optimizer;

  ops->alloc_mem_obj = pocl_level0_alloc_mem_obj;
  ops->free = pocl_level0_free;
  ops->svm_free = pocl_level0_svm_free;
  ops->svm_alloc = pocl_level0_svm_alloc;

  ops->build_source = pocl_level0_build_source;
  ops->build_binary = pocl_level0_build_binary;
  ops->link_program = pocl_level0_link_program;
  ops->free_program = pocl_level0_free_program;
  ops->setup_metadata = pocl_level0_setup_metadata;
  ops->supports_binary = pocl_level0_supports_binary;
  ops->build_poclbinary = pocl_level0_build_poclbinary;
  ops->compile_kernel = NULL;
  ops->create_kernel = pocl_level0_create_kernel;
  ops->free_kernel = pocl_level0_free_kernel;
  ops->init_build = pocl_level0_init_build;

  ops->join = pocl_level0_join;
  ops->submit = pocl_level0_submit;
  ops->broadcast = pocl_broadcast;
  ops->notify = pocl_level0_notify;
  ops->flush = pocl_level0_flush;
  ops->build_hash = pocl_level0_build_hash;

  /* TODO get timing data from level0 API */
  /* ops->get_timer_value = pocl_level0_get_timer_value; */

  ops->wait_event = pocl_level0_wait_event;
  ops->notify_event_finished = pocl_level0_notify_event_finished;
  ops->notify_cmdq_finished = pocl_level0_notify_cmdq_finished;
  ops->free_event_data = pocl_level0_free_event_data;
  ops->wait_event = pocl_level0_wait_event;
  ops->update_event = pocl_level0_update_event;

  ops->init_queue = pocl_level0_init_queue;
  ops->free_queue = pocl_level0_free_queue;

  ops->create_sampler = pocl_level0_create_sampler;
  ops->free_sampler = pocl_level0_free_sampler;

  ops->get_device_info_ext = pocl_level0_get_device_info_ext;
}


void appendToBuildLog(cl_program program, cl_uint device_i, char *Log,
                      size_t LogSize) {
  size_t ExistingLogSize = 0;
  if (LogSize == 0) {
    return;
  }

  if (program->build_log[device_i] != nullptr) {
    ExistingLogSize = strlen(program->build_log[device_i]);
    size_t TotalLogSize = LogSize + ExistingLogSize;
    char *NewLog = (char *)malloc(TotalLogSize);
    assert(NewLog);
    memcpy(NewLog, program->build_log[device_i], ExistingLogSize);
    memcpy(NewLog + ExistingLogSize, Log, LogSize);
    free(Log);
    free(program->build_log[device_i]);
    program->build_log[device_i] = NewLog;
  } else {
    program->build_log[device_i] = Log;
  }
}

static int readProgramSpv(cl_program program, cl_uint device_i,
                          const char *ProgramSpvPath) {
  /* Read binaries from program.spv to memory */
  if (program->program_il_size == 0) {
    assert(ProgramSpvPath);
    assert(program->program_il == nullptr);
    uint64_t Size = 0;
    char *Binary = nullptr;
    int Res = pocl_read_file(ProgramSpvPath, &Binary, &Size);
    POCL_RETURN_ERROR_ON((Res != 0), CL_BUILD_PROGRAM_FAILURE,
                         "Failed to read binaries from program.spv to "
                         "memory: %s\n",
                         ProgramSpvPath);
    program->program_il = Binary;
    program->program_il_size = Size;
  }
  return CL_SUCCESS;
}

static Level0Driver *DriverInstance = nullptr;

char *pocl_level0_build_hash(cl_device_id device) {
  // TODO build hash
  char *res = (char *)malloc(32);
  snprintf(res, 32, "pocl-level0-spirv");
  return res;
}

unsigned int pocl_level0_probe(struct pocl_device_ops *ops) {
  int env_count = pocl_device_get_env_count(ops->device_name);

  if (env_count <= 0) {
    return 0;
  }

  DriverInstance = new Level0Driver();

  POCL_MSG_PRINT_LEVEL0("Level Zero devices found: %u\n",
                        DriverInstance->getNumDevices());

  /* TODO: clamp device_count to env_count */

  return DriverInstance->getNumDevices();
}

cl_int pocl_level0_init(unsigned j, cl_device_id device,
                        const char *parameters) {
  assert(j < DriverInstance->getNumDevices());
  POCL_MSG_PRINT_LEVEL0("Initializing device %u\n", j);

  Level0Device *Device = DriverInstance->createDevice(j, device, parameters);

  if (Device == nullptr) {
    return CL_FAILED;
  }

  device->data = (void *)Device;

  return CL_SUCCESS;
}

cl_int pocl_level0_uninit(unsigned j, cl_device_id device) {
  Level0Device *Device = (Level0Device *)device->data;

  DriverInstance->releaseDevice(Device);
  /* TODO should this be done at all ? */
  if (DriverInstance->empty()) {
    delete DriverInstance;
    DriverInstance = nullptr;
  }

  return CL_SUCCESS;
}

cl_int pocl_level0_reinit(unsigned j, cl_device_id device) {

  if (DriverInstance == nullptr) {
    DriverInstance = new Level0Driver();
  }

  assert(j < DriverInstance->getNumDevices());
  POCL_MSG_PRINT_LEVEL0("Initializing device %u\n", j);

  // TODO: parameters are not passed (this works ATM because they're ignored)
  Level0Device *Device = DriverInstance->createDevice(j, device, nullptr);

  if (Device == nullptr) {
    return CL_FAILED;
  }

  device->data = (void *)Device;

  return CL_SUCCESS;
}

static void convertProgramBcToSpv(char *ProgramBcPath, char *ProgramSpvPath) {
  strncpy(ProgramSpvPath, ProgramBcPath, POCL_FILENAME_LENGTH);
  size_t len = strlen(ProgramBcPath);
  assert(len > 3);
  len -= 2;
  ProgramSpvPath[len] = 0;
  strncat(ProgramSpvPath, "spv", POCL_FILENAME_LENGTH);
}

static constexpr unsigned DefaultCaptureSize = 128 * 1024;

static int runAndAppendOutputToBuildLog(cl_program program,
                                              unsigned device_i,
                                              char *const *args) {
  int errcode = CL_SUCCESS;

  char *CapturedOutput = nullptr;
  size_t CaptureCapacity = 0;

  CapturedOutput = (char *)malloc(DefaultCaptureSize);
  POCL_RETURN_ERROR_ON((CapturedOutput == nullptr), CL_OUT_OF_HOST_MEMORY,
                       "Error while allocating temporary memory\n");
  CaptureCapacity = (DefaultCaptureSize) - 1;
  CapturedOutput[0] = 0;
  char *SavedCapturedOutput = CapturedOutput;

  std::string CommandLine;
  unsigned i = 0;
  while (args[i] != nullptr) {
    CommandLine += " ";
    CommandLine += args[i];
    ++i;
  }
  POCL_MSG_PRINT_LEVEL0("launching command: \n#### %s\n", CommandLine.c_str());

  std::string LaunchMsg;
  LaunchMsg.append("Output of ");
  LaunchMsg.append(args[0]);
  LaunchMsg.append(":\n");
  if (LaunchMsg.size() < CaptureCapacity) {
    strncat(CapturedOutput, LaunchMsg.c_str(), CaptureCapacity);
    CapturedOutput += LaunchMsg.size();
    CaptureCapacity -= LaunchMsg.size();
  }

  errcode =
      pocl_run_command_capture_output(CapturedOutput, &CaptureCapacity, args);
  if (CaptureCapacity > 0) {
    CapturedOutput[CaptureCapacity] = 0;
  }

  appendToBuildLog(program, device_i, SavedCapturedOutput,
                   strlen(SavedCapturedOutput));

  return errcode;
}

static int
compileProgramBcToSpv(cl_program program, cl_uint device_i,
                      const char ProgramBcPathTemp[POCL_FILENAME_LENGTH],
                      char ProgramSpvPathTemp[POCL_FILENAME_LENGTH]) {
  std::vector<std::string> CompilationArgs;
  std::vector<char *> CompilationArgs2;

  // generate program.spv
  CompilationArgs.clear();
  CompilationArgs.push_back(LLVM_SPIRV);
#ifdef LLVM_OPAQUE_POINTERS
  CompilationArgs.push_back("--opaque-pointers");
#endif
  CompilationArgs.push_back("-o");
  CompilationArgs.push_back(ProgramSpvPathTemp);
  CompilationArgs.push_back(ProgramBcPathTemp);

  CompilationArgs2.reserve(CompilationArgs.size() + 1);
  for (unsigned i = 0; i < CompilationArgs.size(); ++i)
    CompilationArgs2[i] = CompilationArgs[i].data();
  CompilationArgs2[CompilationArgs.size()] = nullptr;

  int err = runAndAppendOutputToBuildLog(program, device_i,
                                               CompilationArgs2.data());
  POCL_RETURN_ERROR_ON((err != CL_SUCCESS), CL_BUILD_PROGRAM_FAILURE,
                       "LLVM-SPIRV exited with nonzero code\n");
  POCL_RETURN_ERROR_ON(!pocl_exists(ProgramSpvPathTemp),
                       CL_BUILD_PROGRAM_FAILURE,
                       "LLVM-SPIRV produced no output\n");

  return err;
}

static int linkWithSpirvLink(cl_program program, cl_uint device_i,
                             char ProgramSpvPathTemp[POCL_FILENAME_LENGTH],
                             std::vector<std::string> &SpvBinaryPaths,
                             int create_library) {
  std::vector<std::string> CompilationArgs;
  std::vector<char *> CompilationArgs2;

  CompilationArgs.push_back(SPIRV_LINK);
  if (create_library != 0) {
    CompilationArgs.push_back("--create-library");
  }
  CompilationArgs.push_back("-o");
  CompilationArgs.push_back(ProgramSpvPathTemp);
  for (auto &Path : SpvBinaryPaths) {
    CompilationArgs.push_back(Path);
  }
  CompilationArgs2.reserve(CompilationArgs.size() + 1);
  for (unsigned i = 0; i < CompilationArgs.size(); ++i)
    CompilationArgs2[i] = CompilationArgs[i].data();
  CompilationArgs2[CompilationArgs.size()] = nullptr;

  int err = runAndAppendOutputToBuildLog(program, device_i,
                                               CompilationArgs2.data());
  POCL_RETURN_ERROR_ON((err != 0), CL_BUILD_PROGRAM_FAILURE,
                       "spirv-link exited with nonzero code\n");
  POCL_RETURN_ERROR_ON(!pocl_exists(ProgramSpvPathTemp),
                       CL_LINK_PROGRAM_FAILURE, "spirv-link failed\n");
  return CL_SUCCESS;
}

int pocl_level0_build_source(cl_program program, cl_uint device_i,
                             cl_uint num_input_headers,
                             const cl_program *input_headers,
                             const char **header_include_names,
                             int link_program) {
#ifdef ENABLE_LLVM
  int err = CL_SUCCESS;
  POCL_MSG_PRINT_LLVM("building from sources for device %d\n", device_i);

  // last arg is 0 because we never link with Clang, let the spirv-link and
  // level0 do the linking
  int errcode = pocl_llvm_build_program(program, device_i, num_input_headers,
                                        input_headers, header_include_names, 0);
  POCL_RETURN_ERROR_ON((errcode != CL_SUCCESS), CL_BUILD_PROGRAM_FAILURE,
                       "Failed to build program from source\n");

  cl_device_id dev = program->devices[device_i];
  Level0Device *Device = (Level0Device *)dev->data;

  char ProgramSpvPathTemp[POCL_FILENAME_LENGTH];
  char ProgramBcPathTemp[POCL_FILENAME_LENGTH];
  char ProgramBcPath[POCL_FILENAME_LENGTH];
  char ProgramSpvPath[POCL_FILENAME_LENGTH];

  pocl_cache_tempname(ProgramSpvPathTemp, ".spv", NULL);
  pocl_cache_tempname(ProgramBcPathTemp, ".bc", NULL);
  pocl_cache_program_bc_path(ProgramBcPath, program, device_i);
  pocl_cache_program_spv_path(ProgramSpvPath, program, device_i);

  // result of pocl_llvm_build_program
  assert(pocl_exists(ProgramBcPath));
  // we don't need llvm::Module objects, only the bitcode
  pocl_llvm_free_llvm_irs(program, device_i);
  pocl_rename(ProgramBcPath, ProgramBcPathTemp);

  if (pocl_exists(ProgramSpvPath) != 0) {
    goto CREATE_ZE_MODULE;
  }

  err = compileProgramBcToSpv(program, device_i, ProgramBcPathTemp,
                              ProgramSpvPathTemp);
  if (err != CL_SUCCESS) {
    return err;
  }
  pocl_rename(ProgramSpvPathTemp, ProgramSpvPath);
  POCL_MSG_WARN("Final SPV written: %s\n", ProgramSpvPath);

CREATE_ZE_MODULE:
  if (pocl_get_bool_option("POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES", 0) == 0) {
    pocl_remove(ProgramBcPathTemp);
  }

  readProgramSpv(program, device_i, ProgramSpvPath);
  assert(program->program_il != nullptr);
  assert(program->program_il_size > 0);

  if (link_program != 0) {
    return Device->createProgram(program, device_i);
  } else {
    // only final (linked) programs have  ZE module
    assert(program->data[device_i] == nullptr);
    return CL_SUCCESS;
  }
#else
  POCL_RETURN_ERROR_ON(1, CL_BUILD_PROGRAM_FAILURE,
                       "This device requires LLVM to build from sources\n");
#endif
}

int pocl_level0_supports_binary(cl_device_id device, size_t length,
                                const char *binary) {
  if (pocl_bitcode_is_spirv_execmodel_kernel(binary, length) != 0) {
    return 1;
  }
#ifdef ENABLE_SPIR
  if ((bitcode_is_triple(binary, length, "spir-unknown") != 0) ||
      (bitcode_is_triple(binary, length, "spir64-unknown") != 0)) {
    return 1;
  }
#endif
  // TODO : possibly support native ZE binaries
  return 0;
}

char *pocl_level0_init_build(void *data) {
  // the -O0 helps to avoid a bunch of issues created by Clang's optimization
  // (issues for llvm-spirv translator)
  // * the freeze instruction
  // * the vector instructions (llvm.vector.reduce.add.v4i32)
  // "InvalidBitWidth: Invalid bit width in input: 63" - happens with
  // test_convert_type_X
  return strdup("-O0");
}

int pocl_level0_build_binary(cl_program program, cl_uint device_i,
                             int link_program, int spir_build) {
  cl_device_id dev = program->devices[device_i];
  Level0Device *Device = (Level0Device *)dev->data;
  char ProgramBcPath[POCL_FILENAME_LENGTH];
  char ProgramSpvPath[POCL_FILENAME_LENGTH];

  if ((program->program_il != nullptr) && program->program_il_size > 0 &&
      program->pocl_binaries[device_i] == nullptr &&
      program->binaries[device_i] == nullptr) {
    char *Ptr = (char *)malloc(program->program_il_size);
    memcpy(Ptr, program->program_il, program->program_il_size);
    program->binaries[device_i] = (unsigned char *)Ptr;
    program->binary_sizes[device_i] = program->program_il_size;
  }

  if (program->pocl_binaries[device_i] != nullptr) {
    pocl_cache_program_spv_path(ProgramSpvPath, program, device_i);

    POCL_RETURN_ERROR_ON(
        (readProgramSpv(program, device_i, ProgramSpvPath) != CL_SUCCESS),
        CL_BUILD_PROGRAM_FAILURE,
        "PoCL binary doesn't contain program.spv at %s\n", ProgramSpvPath);

  } else {
    /* we have program->binaries[] which is SPIR-V,
     * or program->binaries[] which is LLVM IR SPIR,
     * or program_il which is SPIR-V*/
    char *Binary = nullptr;
    size_t BinarySize = 0;

    assert(program->binaries[device_i]);
    Binary = (char *)program->binaries[device_i];
    BinarySize = program->binary_sizes[device_i];

    int is_spirv = pocl_bitcode_is_spirv_execmodel_kernel(Binary, BinarySize);

    if (is_spirv != 0) {
      if (program->program_il == nullptr) {
        char *Ptr = (char *)malloc(BinarySize);
        memcpy(Ptr, Binary, BinarySize);
        program->program_il = Ptr;
        program->program_il_size = BinarySize;
      }

      pocl_cache_create_program_cachedir(program, device_i, Binary, BinarySize,
                                         ProgramBcPath);
      convertProgramBcToSpv(ProgramBcPath, ProgramSpvPath);

      if (pocl_exists(ProgramSpvPath) == 0) {
        char ProgramSpvPathTemp[POCL_FILENAME_LENGTH];
        pocl_cache_tempname(ProgramSpvPathTemp, ".spv", NULL);

        pocl_write_file(ProgramSpvPathTemp, Binary, BinarySize, 0, 0);
        POCL_RETURN_ERROR_ON(
            !pocl_exists(ProgramSpvPathTemp), CL_BUILD_PROGRAM_FAILURE,
            "failed to write SPIR-V file %s\n", ProgramSpvPathTemp);

        pocl_rename(ProgramSpvPathTemp, ProgramSpvPath);
      }
    } else {
#ifdef ENABLE_SPIR
      if ((bitcode_is_triple(Binary, BinarySize, "spir-unknown") != 0) ||
          (bitcode_is_triple(Binary, BinarySize, "spir64-unknown") != 0)) {
        char ProgramSpvPathTemp[POCL_FILENAME_LENGTH];
        char ProgramBcPathTemp[POCL_FILENAME_LENGTH];

        pocl_cache_tempname(ProgramSpvPathTemp, ".spv", NULL);
        pocl_cache_tempname(ProgramBcPathTemp, ".bc", NULL);
        int err = pocl_write_file(ProgramBcPathTemp, Binary, BinarySize, 0, 0);
        POCL_RETURN_ERROR_ON((err != 0), CL_BUILD_PROGRAM_FAILURE,
                             "failed to write BC file into cache\n");

        err = compileProgramBcToSpv(program, device_i, ProgramBcPathTemp,
                                    ProgramSpvPathTemp);
        if (err != CL_SUCCESS) {
          return err;
        }

        pocl_cache_create_program_cachedir(program, device_i, Binary,
                                           BinarySize, ProgramBcPath);
        convertProgramBcToSpv(ProgramBcPath, ProgramSpvPath);
        pocl_rename(ProgramSpvPathTemp, ProgramSpvPath);

        POCL_RETURN_ERROR_ON(
            (readProgramSpv(program, device_i, ProgramSpvPath) != CL_SUCCESS),
            CL_BUILD_PROGRAM_FAILURE,
            "Could not read compiled program.spv at %s\n", ProgramSpvPath);
      } else { // not SPIRV and not LLVM IR spir
#endif
        POCL_RETURN_ERROR_ON(
            1, CL_BUILD_PROGRAM_FAILURE,
            "the binary supplied to level0 driver is not SPIR-V, "
            "and it's not a recognized binary type\n");
      }
    }
  }

  assert(program->program_il != nullptr);
  assert(program->program_il_size > 0);

  if (link_program != 0) {
    return Device->createProgram(program, device_i);
  } else {
    // only final (linked) programs have  ZE module
    assert(program->data[device_i] == nullptr);
    return CL_SUCCESS;
  }
}

int pocl_level0_link_program(cl_program program, cl_uint device_i,
                             cl_uint num_input_programs,
                             const cl_program *input_programs,
                             int create_library) {
  cl_device_id dev = program->devices[device_i];
  Level0Device *Device = (Level0Device *)dev->data;
  char ProgramBcPath[POCL_FILENAME_LENGTH];
  char ProgramSpvPath[POCL_FILENAME_LENGTH];

  /* we have program->binaries[] which is SPIR-V */
  assert(program->pocl_binaries[device_i] == nullptr);
  assert(program->binaries[device_i] == nullptr);
  assert(program->binary_sizes[device_i] == 0);

  std::vector<std::string> SpvBinaryPaths;
  std::vector<char> SpvConcatBinary;

  cl_uint i;
  for (i = 0; i < num_input_programs; i++) {
    assert(dev == input_programs[i]->devices[device_i]);
    POCL_LOCK_OBJ(input_programs[i]);

    char *Spv = (char *)input_programs[i]->program_il;
    assert(Spv);
    size_t Size = input_programs[i]->program_il_size;
    assert(Size);
    SpvConcatBinary.insert(SpvConcatBinary.end(), Spv, Spv + Size);

    pocl_cache_program_spv_path(ProgramSpvPath, program, device_i);
    assert(pocl_exists(ProgramSpvPath));
    SpvBinaryPaths.push_back(ProgramSpvPath);

    POCL_UNLOCK_OBJ(input_programs[i]);
  }

  pocl_cache_create_program_cachedir(program, device_i, SpvConcatBinary.data(),
                                     SpvConcatBinary.size(), ProgramBcPath);
  convertProgramBcToSpv(ProgramBcPath, ProgramSpvPath);

  char ProgramSpvPathTemp[POCL_FILENAME_LENGTH];
  pocl_cache_tempname(ProgramSpvPathTemp, ".spv", NULL);

  int err = linkWithSpirvLink(program, device_i, ProgramSpvPathTemp,
                              SpvBinaryPaths, create_library);
  if (err != CL_SUCCESS) {
    return err;
  }

  pocl_rename(ProgramSpvPathTemp, ProgramSpvPath);
  readProgramSpv(program, device_i, ProgramSpvPath);
  assert(program->program_il != nullptr);
  assert(program->program_il_size > 0);

  if (create_library == 0) {
    return Device->createProgram(program, device_i);
  } else {
    // only final (linked) programs have  ZE module
    assert(program->data[device_i] == nullptr);
    return CL_SUCCESS;
  }
}

int pocl_level0_free_program(cl_device_id device, cl_program program,
                             unsigned program_device_i) {
  cl_device_id dev = program->devices[program_device_i];
  Level0Device *Device = (Level0Device *)dev->data;
  /* module can be NULL if compilation fails */
  Device->freeProgram(program, program_device_i);
  return 0;
}

static constexpr unsigned MaxKernelsPerProgram = 4096;

int pocl_level0_setup_metadata(cl_device_id device, cl_program program,
                               unsigned program_device_i) {
  assert(program->data[program_device_i] != NULL);

  Level0ProgramSPtr *ProgramSPtr =
      (Level0ProgramSPtr *)program->data[program_device_i];
  ze_module_handle_t ModuleH = ProgramSPtr->get()->getAnyHandle();

  const char *NameArray[MaxKernelsPerProgram];
  uint32_t NameCount = MaxKernelsPerProgram;
  ze_result_t res = zeModuleGetKernelNames(ModuleH, &NameCount, NameArray);
  assert(res == ZE_RESULT_SUCCESS);
  if (NameCount == MaxKernelsPerProgram) {
    POCL_MSG_ERR("Too many kernels found in the program\n");
    return 0;
  }
  if (NameCount == 0) {
    POCL_MSG_WARN("No kernels found in program.\n");
    return 1;
  }
  program->num_kernels = NameCount;

  program->kernel_meta = (pocl_kernel_metadata_t *)calloc(
      program->num_kernels, sizeof(pocl_kernel_metadata_t));

  // TODO this is using program_il as source
  int32_t *Stream = (int32_t *)program->program_il;
  size_t StreamSize = program->program_il_size / 4;
  OpenCLFunctionInfoMap KernelInfoMap;
  poclParseSPIRV(Stream, StreamSize, KernelInfoMap);

  if (NameCount != KernelInfoMap.size()) {
    POCL_MSG_ERR("ZE reports %u kernel names, "
                 "but SPIRV parser reports %zu\n",
                 NameCount, KernelInfoMap.size());
    for (auto &KInfo : KernelInfoMap) {
      std::string Name = KInfo.first;
      POCL_MSG_ERR("SPIRV kernel: %s\n", Name.c_str());
    }
    for (unsigned i = 0; i < NameCount; ++i) {
      POCL_MSG_ERR("ZE kernel: %s\n", NameArray[i]);
    }
    return 0;
  }

  uint32_t i = 0;
  for (auto &I : KernelInfoMap) {
    std::string Name = I.first;
    OCLFuncInfo *FI = I.second.get();

    pocl_kernel_metadata_t *meta = &program->kernel_meta[i];
    meta->data = (void **)calloc(program->num_devices, sizeof(void *));
    meta->num_args = FI->ArgTypeInfo.size();
    meta->name = strdup(Name.c_str());

    // Level zero driver handles the static locals
    meta->num_locals = 0;
    meta->local_sizes = nullptr;

    meta->max_subgroups =
        (size_t *)calloc(program->num_devices, sizeof(size_t));
    meta->compile_subgroups =
        (size_t *)calloc(program->num_devices, sizeof(size_t));
    meta->max_workgroup_size =
        (size_t *)calloc(program->num_devices, sizeof(size_t));
    meta->preferred_wg_multiple =
        (size_t *)calloc(program->num_devices, sizeof(size_t));
    meta->local_mem_size =
        (cl_ulong *)calloc(program->num_devices, sizeof(cl_ulong));
    meta->private_mem_size =
        (cl_ulong *)calloc(program->num_devices, sizeof(cl_ulong));
    meta->spill_mem_size =
        (cl_ulong *)calloc(program->num_devices, sizeof(cl_ulong));

    // ZE kernel metadata
    {
      ze_kernel_handle_t hKernel = nullptr;
      ze_kernel_desc_t KernelDesc = {
          ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr,
          0, // flags | ZE_KERNEL_FLAG_FORCE_RESIDENCY
          meta->name};
      LEVEL0_CHECK_RET(0, zeKernelCreate(ModuleH, &KernelDesc, &hKernel));

      ze_kernel_preferred_group_size_properties_t PrefGroupSize = {
          ZE_STRUCTURE_TYPE_KERNEL_PREFERRED_GROUP_SIZE_PROPERTIES, NULL, 0};
      ze_kernel_properties_t KernelProps{};
      KernelProps.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
      KernelProps.pNext = (void *)&PrefGroupSize;
      LEVEL0_CHECK_RET(0, zeKernelGetProperties(hKernel, &KernelProps));

      assert(meta->num_args == KernelProps.numKernelArgs);
      meta->reqd_wg_size[0] = KernelProps.requiredGroupSizeX;
      meta->reqd_wg_size[1] = KernelProps.requiredGroupSizeY;
      meta->reqd_wg_size[2] = KernelProps.requiredGroupSizeZ;
      // TODO: setup of these attributes is missing
      // meta->vec_type_hint
      // meta->wg_size_hint

      uint32_t AttrSize = 0;
      char *AttrString = nullptr;
      LEVEL0_CHECK_RET(0, zeKernelGetSourceAttributes(hKernel, &AttrSize,
                                                      &AttrString));
      if (AttrSize > 0) {
        meta->attributes = strdup(AttrString);
      }

      LEVEL0_CHECK_RET(0, zeKernelDestroy(hKernel));

      meta->max_subgroups[program_device_i] = KernelProps.maxSubgroupSize;
      meta->compile_subgroups[program_device_i] =
          KernelProps.requiredSubgroupSize;
      meta->max_workgroup_size[program_device_i] = 0; // TODO
      meta->preferred_wg_multiple[program_device_i] =
          PrefGroupSize.preferredMultiple;
      meta->local_mem_size[program_device_i] = KernelProps.localMemSize;
      meta->private_mem_size[program_device_i] = KernelProps.privateMemSize;
      meta->spill_mem_size[program_device_i] = KernelProps.spillMemSize;

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

    // ARGUMENTS
    if (meta->num_args != 0u) {
      meta->arg_info = (struct pocl_argument_info *)calloc(
          meta->num_args, sizeof(struct pocl_argument_info));

      for (uint32_t j = 0; j < meta->num_args; ++j) {
        cl_kernel_arg_address_qualifier Addr;
        cl_kernel_arg_access_qualifier Access;
        Addr = CL_KERNEL_ARG_ADDRESS_PRIVATE;
        Access = CL_KERNEL_ARG_ACCESS_NONE;
        meta->arg_info[j].name = strdup(FI->ArgTypeInfo[j].Name.c_str());
        meta->arg_info[j].type_name = nullptr;
        switch (FI->ArgTypeInfo[j].Type) {
        case OCLType::POD: {
          meta->arg_info[j].type = POCL_ARG_TYPE_NONE;
          meta->arg_info[j].type_size = FI->ArgTypeInfo[j].Size;
          break;
        }
        case OCLType::Pointer: {
          meta->arg_info[j].type = POCL_ARG_TYPE_POINTER;
          meta->arg_info[j].type_size = sizeof(cl_mem);
          switch (FI->ArgTypeInfo[j].Space) {
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
          meta->arg_info[j].type = POCL_ARG_TYPE_IMAGE;
          meta->arg_info[j].type_size = sizeof(cl_mem);
          Addr = CL_KERNEL_ARG_ADDRESS_GLOBAL;
          bool Readable = FI->ArgTypeInfo[j].Attrs.ReadableImg;
          bool Writable = FI->ArgTypeInfo[j].Attrs.WriteableImg;
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
          meta->arg_info[j].type = POCL_ARG_TYPE_SAMPLER;
          meta->arg_info[j].type_size = sizeof(cl_mem);
          break;
        }
        case OCLType::Opaque: {
          POCL_MSG_ERR("Unknown OCL type OPaque\n");
          meta->arg_info[j].type = POCL_ARG_TYPE_NONE;
          meta->arg_info[j].type_size = FI->ArgTypeInfo[j].Size;
          break;
        }
        }
        meta->arg_info[j].address_qualifier = Addr;
        meta->arg_info[j].access_qualifier = Access;
        meta->arg_info[j].type_qualifier = CL_KERNEL_ARG_TYPE_NONE;
        if (FI->ArgTypeInfo[j].Attrs.Constant) {
          meta->arg_info[j].type_qualifier = CL_KERNEL_ARG_TYPE_CONST;
        }
        if (FI->ArgTypeInfo[j].Attrs.Restrict) {
          meta->arg_info[j].type_qualifier = CL_KERNEL_ARG_TYPE_RESTRICT;
        }
        if (FI->ArgTypeInfo[j].Attrs.Volatile) {
          meta->arg_info[j].type_qualifier = CL_KERNEL_ARG_TYPE_VOLATILE;
        }
      }

      // TODO: POCL_HAS_KERNEL_ARG_TYPE_NAME missing
      meta->has_arg_metadata = POCL_HAS_KERNEL_ARG_ACCESS_QUALIFIER |
                               POCL_HAS_KERNEL_ARG_ADDRESS_QUALIFIER |
                               POCL_HAS_KERNEL_ARG_TYPE_QUALIFIER |
                               POCL_HAS_KERNEL_ARG_NAME;
    }

    ++i;
  }

  return 1;
}

int pocl_level0_create_kernel(cl_device_id device, cl_program program,
                              cl_kernel kernel, unsigned program_device_i) {
  assert(program->data[program_device_i] != nullptr);
  Level0ProgramSPtr *ProgramSPtr =
      (Level0ProgramSPtr *)program->data[program_device_i];
  Level0Kernel *Ker = ProgramSPtr->get()->createKernel(kernel->name);
  assert(Ker != nullptr);
  kernel->data[program_device_i] = Ker;
  return 0;
}

int pocl_level0_free_kernel(cl_device_id device, cl_program program,
                            cl_kernel kernel, unsigned program_device_i) {

  assert(program->data[program_device_i] != nullptr);
  Level0ProgramSPtr *ProgramSPtr =
      (Level0ProgramSPtr *)program->data[program_device_i];
  assert(kernel->data[program_device_i] != nullptr);
  Level0Kernel *Ker = (Level0Kernel *)kernel->data[program_device_i];

  bool Res = ProgramSPtr->get()->releaseKernel(Ker);
  assert(Res == true);

  return 0;
}

int pocl_level0_build_poclbinary(cl_program program, cl_uint device_i) {

  assert(program->build_status == CL_BUILD_SUCCESS);
  if (program->num_kernels == 0) {
    return CL_SUCCESS;
  }

  /* For binaries of other than Executable type (libraries, compiled but
   * not linked programs, etc), do not attempt to compile the kernels. */
  if (program->binary_type != CL_PROGRAM_BINARY_TYPE_EXECUTABLE) {
    return CL_SUCCESS;
  }

  assert(program->binaries[device_i]);

  return CL_SUCCESS;
}


void pocl_level0_submit(_cl_command_node *node, cl_command_queue cq) {
  node->ready = 1;
  if (pocl_command_is_ready(node->sync.event.event) != 0) {
    pocl_update_event_submitted(node->sync.event.event);
    Level0Device *Device = (Level0Device *)cq->device->data;
    Device->pushCommand(node);
  }
  POCL_UNLOCK_OBJ(node->sync.event.event);
}

int pocl_level0_init_queue(cl_device_id dev, cl_command_queue queue) {
  queue->data =
      pocl_aligned_malloc(HOST_CPU_CACHELINE_SIZE, sizeof(pocl_cond_t));
  pocl_cond_t *cond = (pocl_cond_t *)queue->data;
  POCL_INIT_COND(*cond);
  return CL_SUCCESS;
}

int pocl_level0_free_queue(cl_device_id dev, cl_command_queue queue) {
  pocl_cond_t *cond = (pocl_cond_t *)queue->data;
  POCL_DESTROY_COND(*cond);
  POCL_MEM_FREE(queue->data);
  return CL_SUCCESS;
}

void pocl_level0_notify_cmdq_finished(cl_command_queue cq) {
  /* must be called with CQ already locked.
   * this must be a broadcast since there could be multiple
   * user threads waiting on the same command queue
   * in pthread_scheduler_wait_cq(). */
  pocl_cond_t *cq_cond = (pocl_cond_t *)cq->data;
  POCL_BROADCAST_COND(*cq_cond);
}

void pocl_level0_notify_event_finished(cl_event event) {
  pocl_cond_t *event_cond = (pocl_cond_t *)event->data;
  POCL_BROADCAST_COND(*event_cond);
}

void pocl_level0_free_event_data(cl_event event) {
  if (event->data == nullptr) {
    return;
  }
  pocl_cond_t *event_cond = (pocl_cond_t *)event->data;
  POCL_DESTROY_COND(*event_cond);
  POCL_MEM_FREE(event->data);
}

void pocl_level0_join(cl_device_id device, cl_command_queue cq) {
  POCL_LOCK_OBJ(cq);
  pocl_cond_t *cq_cond = (pocl_cond_t *)cq->data;
  while (true) {
    if (cq->command_count == 0) {
      POCL_UNLOCK_OBJ(cq);
      return;
    } else {
      PTHREAD_CHECK(pthread_cond_wait(cq_cond, &cq->pocl_lock));
    }
  }
}

void pocl_level0_flush(cl_device_id device, cl_command_queue cq) {}

void pocl_level0_notify(cl_device_id device, cl_event event,
                        cl_event finished) {
  _cl_command_node *node = event->command;

  if (finished->status < CL_COMPLETE) {
    pocl_update_event_failed(event);
    return;
  }

  if (node->ready == 0) {
    return;
  }

  POCL_MSG_PRINT_LEVEL0("notify on event %zu \n", event->id);

  if (pocl_command_is_ready(node->sync.event.event) != 0) {
    pocl_update_event_submitted(event);
    Level0Device *Device = (Level0Device *)device->data;
    Device->pushCommand(node);
  }
}

void pocl_level0_update_event(cl_device_id device, cl_event event) {
  if (event->data == nullptr) {
    pocl_cond_t *event_cond = (pocl_cond_t *)malloc(sizeof(pocl_cond_t));
    assert(event_cond);
    POCL_INIT_COND(*event_cond);
    event->data = (void *)event_cond;
  }
  if (event->status == CL_QUEUED) {
    event->time_queue = pocl_gettimemono_ns();
  }
  if (event->status == CL_SUBMITTED) {
    event->time_submit = pocl_gettimemono_ns();
  }
}

void pocl_level0_wait_event(cl_device_id device, cl_event event) {
  POCL_MSG_PRINT_LEVEL0("device->wait_event on event %zu\n", event->id);
  pocl_cond_t *event_cond = (pocl_cond_t *)event->data;

  POCL_LOCK_OBJ(event);
  while (event->status > CL_COMPLETE) {
    POCL_WAIT_COND(*event_cond, event->pocl_lock);
  }
  POCL_UNLOCK_OBJ(event);
}


int pocl_level0_alloc_mem_obj(cl_device_id device, cl_mem mem, void *host_ptr) {
  Level0Device *Device = (Level0Device *)device->data;
  pocl_mem_identifier *p = &mem->device_ptrs[device->global_mem_id];

  assert(p->mem_ptr == NULL);

  p->extra = 0;
  /* for Images, ze_image_handler_t */
  p->extra_ptr = NULL;

  /* won't preallocate host-visible memory for images,
   * only for buffers */
  if (((mem->flags & CL_MEM_ALLOC_HOST_PTR) != 0u) &&
      (mem->mem_host_ptr == NULL) && (mem->is_image != 0u)) {
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  }

  void *Allocation = nullptr;
  // special handling for clCreateBuffer called on SVM pointer
  if (((mem->flags & CL_MEM_USE_HOST_PTR) != 0u) &&
      (mem->mem_host_ptr_is_svm != 0)) {
    p->mem_ptr = mem->mem_host_ptr;
    p->version = mem->mem_host_ptr_version;
  } else {
    //    void *all = Device->allocDeviceMem(mem->size);
    Allocation = Device->allocSharedMem(mem->size);
    if (Allocation == nullptr) {
      return CL_MEM_OBJECT_ALLOCATION_FAILURE;
    }
    p->mem_ptr = Allocation;
    p->version = 0;
  }

  if (mem->is_image != 0u) {
    // image attributes must be already set up
    assert(mem->image_channel_data_type != 0);
    assert(mem->image_channel_order != 0);
    ze_image_handle_t Image = Device->allocImage(
        mem->image_channel_data_type, mem->image_channel_order, mem->type,
        mem->flags, mem->image_width, mem->image_height, mem->image_depth);
    if (Image == nullptr) {
      if (Allocation != nullptr) {
        Device->freeMem(Allocation);
      }
      p->mem_ptr = nullptr;
      p->version = 0;
      return CL_MEM_OBJECT_ALLOCATION_FAILURE;
    } else {
      p->extra_ptr = (void *)Image;
    }
  }

  // since we allocate shared memory, use it for mem_host_ptr
  if (mem->mem_host_ptr == nullptr) {
    assert((mem->flags & CL_MEM_USE_HOST_PTR) == 0);
    mem->mem_host_ptr = Allocation;
    mem->mem_host_ptr_version = 0;
    ++mem->mem_host_ptr_refcount;
  }

  POCL_MSG_PRINT_MEMORY("level0 DEVICE ALLOC | MEM_HOST_PTR %p SIZE %zu | "
                        "level0 DEV BUF %p | STA BUF %p | EXTRA_PTR %p \n",
                        mem->mem_host_ptr, mem->size, p->mem_ptr,
                        (void *)p->extra, p->extra_ptr);

  return CL_SUCCESS;
}

void pocl_level0_free(cl_device_id device, cl_mem mem) {
  Level0Device *Device = (Level0Device *)device->data;
  pocl_mem_identifier *p = &mem->device_ptrs[device->global_mem_id];

  POCL_MSG_PRINT_MEMORY("level0 DEVICE FREE | PTR %p SIZE %zu \n", p->mem_ptr,
                        mem->size);

  if (mem->is_image != 0u) {
    assert(p->extra_ptr != nullptr);
    ze_image_handle_t Image = (ze_image_handle_t)p->extra_ptr;
    pocl::Level0Device::freeImage(Image);
  }

  // special handling for clCreateBuffer called on SVM pointer
  if (((mem->flags & CL_MEM_USE_HOST_PTR) != 0u) &&
      (mem->mem_host_ptr_is_svm != 0)) {
    p->mem_ptr = nullptr;
    p->version = 0;
  } else {
    Device->freeMem(p->mem_ptr);
  }

  if (mem->mem_host_ptr != nullptr && mem->mem_host_ptr == p->mem_ptr) {
    assert((mem->flags & CL_MEM_USE_HOST_PTR) == 0);
    mem->mem_host_ptr = nullptr;
    mem->mem_host_ptr_version = 0;
    --mem->mem_host_ptr_refcount;
    // TODO refcounting
    // assert(mem->mem_host_ptr_refcount == 0);
  }

  p->mem_ptr = nullptr;
  p->version = 0;
  p->extra_ptr = nullptr;
  p->extra = 0;
}

cl_int pocl_level0_get_mapping_ptr(void *data, pocl_mem_identifier *mem_id,
                                   cl_mem mem, mem_mapping_t *map) {
  /* assume buffer is allocated */
  assert(mem_id->mem_ptr != NULL);

  if (mem->is_image != 0u) {
    map->host_ptr = (char *)mem_id->mem_ptr + map->offset;
  } else if ((mem->flags & CL_MEM_USE_HOST_PTR) != 0u) {
    map->host_ptr = (char *)mem->mem_host_ptr + map->offset;
  } else {
    map->host_ptr = (char *)mem_id->mem_ptr + map->offset;
  }
  /* POCL_MSG_ERR ("map HOST_PTR: %p | SIZE %zu | OFFS %zu | DEV PTR: %p \n",
                  map->host_ptr, map->size, map->offset, mem_id->mem_ptr); */
  assert(map->host_ptr);
  return CL_SUCCESS;
}

cl_int pocl_level0_free_mapping_ptr(void *data, pocl_mem_identifier *mem_id,
                                    cl_mem mem, mem_mapping_t *map) {
  map->host_ptr = NULL;
  return CL_SUCCESS;
}

int pocl_level0_create_sampler(cl_device_id device, cl_sampler samp,
                               unsigned context_device_i) {
  Level0Device *Device = (Level0Device *)device->data;
  ze_sampler_handle_t hSampler = Device->allocSampler(
      samp->addressing_mode, samp->filter_mode, samp->normalized_coords);
  if (hSampler == nullptr) {
    return CL_FAILED;
  }
  samp->device_data[device->dev_id] = hSampler;
  return CL_SUCCESS;
}

int pocl_level0_free_sampler(cl_device_id device, cl_sampler samp,
                             unsigned context_device_i) {

  Level0Device *Device = (Level0Device *)device->data;
  ze_sampler_handle_t hSampler =
      (ze_sampler_handle_t)samp->device_data[device->dev_id];
  if (hSampler != nullptr) {
    pocl::Level0Device::freeSampler(hSampler);
  }
  return CL_SUCCESS;
}

void pocl_level0_svm_free(cl_device_id dev, void *svm_ptr) {
  Level0Device *Device = (Level0Device *)dev->data;
  Device->freeMem(svm_ptr);
}

void *pocl_level0_svm_alloc(cl_device_id dev, cl_svm_mem_flags flags,
                            size_t size)

{
  Level0Device *Device = (Level0Device *)dev->data;
  return Device->allocSharedMem(size);
}

cl_int pocl_level0_get_device_info_ext(cl_device_id dev,
                                       cl_device_info param_name,
                                       size_t param_value_size,
                                       void *param_value,
                                       size_t *param_value_size_ret) {
  Level0Device *Device = (Level0Device *)dev->data;
  const std::vector<size_t> &SupportedSGSizes =
      Device->getSupportedSubgroupSizes();

  switch (param_name) {
  case CL_DEVICE_SUB_GROUP_SIZES_INTEL: {
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
