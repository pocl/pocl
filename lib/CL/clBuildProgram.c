/* OpenCL runtime library: clBuildProgram()

   Copyright (c) 2011-2013 Universidad Rey Juan Carlos,
                 2011-2014 Pekka Jääskeläinen / Tampere Univ. of Technology
   
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

#include "pocl_cl.h"
#include <assert.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#ifndef _MSC_VER
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif
#include "pocl_llvm.h"
#include "pocl_util.h"
#include "pocl_file_util.h"
#include "pocl_cache.h"
#include "config.h"
#include "pocl_runtime_config.h"
#include "pocl_binary.h"

/* supported compiler parameters which should pass to the frontend directly
   by using -Xclang */
static const char cl_parameters[] =
  "-cl-single-precision-constant "
  "-cl-fp32-correctly-rounded-divide-sqrt "
  "-cl-opt-disable "
  "-cl-mad-enable "
  "-cl-unsafe-math-optimizations "
  "-cl-finite-math-only "
  "-cl-fast-relaxed-math "
  "-cl-std=CL1.2 "
  "-cl-std=CL1.1 "
  "-cl-std=CL2.0 "
  "-cl-kernel-arg-info "
  "-w "
  "-g "
  "-Werror ";

static const char cl_parameters_not_yet_supported_by_clang[] =
  "-cl-strict-aliasing "
  "-cl-denorms-are-zero "
  "-cl-no-signed-zeros ";

#define MEM_ASSERT(x, err_jmp) do{ if (x){errcode = CL_OUT_OF_HOST_MEMORY;goto err_jmp;}} while(0)

// append token, growing modded_options, if necessary, by max(strlen(token)+1, 256)
#define APPEND_TOKEN() do { \
  size_t needed = strlen(token) + 1; \
  if (size <= (i + needed)) { \
    size_t grow_by = needed > 256 ? needed : 256; \
    char *grown_ptr = (char *)realloc(modded_options, size + grow_by); \
    if (grown_ptr == NULL) { \
      /* realloc failed, free modded_options and return */ \
      errcode = CL_OUT_OF_HOST_MEMORY; \
      goto ERROR_CLEAN_OPTIONS; \
    } \
    modded_options = grown_ptr; \
    size += grow_by; \
  } \
  i += needed; \
  strcat (modded_options, token); \
  strcat (modded_options, " "); \
} while (0)

#define APPEND_TO_MAIN_BUILD_LOG(...)  \
  POCL_MSG_ERR(__VA_ARGS__);   \
  {                            \
    size_t l = strlen(program->main_build_log); \
    snprintf(program->main_build_log + l, (640 - l), __VA_ARGS__); \
  }

cl_int
program_compile_dynamic_wg_binaries(cl_program program)
{
  unsigned i, device_i;
  cl_int errcode = CL_SUCCESS;
  _cl_command_node cmd;

  assert(program->num_kernels);
  assert(program->build_status == CL_BUILD_SUCCESS);

  memset(&cmd, 0, sizeof(_cl_command_node));
  cmd.type = CL_COMMAND_NDRANGE_KERNEL;
  char cachedir[POCL_FILENAME_LENGTH];
  cmd.command.run.tmp_dir = cachedir;
  POCL_LOCK_OBJ(program);

  /* build the dynamic WG sized parallel.bc and device specific code,
 * for each kernel & device combo */
  for (device_i = 0; device_i < program->num_devices; ++device_i)
    {
      cl_device_id device = program->devices[device_i];

      /* program may not be built for some of its devices */
      if (program->pocl_binaries[device_i] || (!program->binaries[device_i]))
        continue;

      cmd.device = device;

      for (i=0; i < program->num_kernels; i++)
        {
          pocl_cache_make_kernel_cachedir_path(cachedir, program, device,
                                               program->default_kernels[i],
                                               0,0,0);

          errcode = pocl_llvm_generate_workgroup_function(device,
                                                          program->default_kernels[i],
                                                          0,0,0);
          if (errcode != CL_SUCCESS)
            {
              POCL_MSG_ERR("Failed to generate workgroup function for "
                           "kernel %s for device %s\n",
                           program->kernel_names[i], device->short_name);
              goto RET;
            }
          cmd.command.run.kernel = program->default_kernels[i];
          device->ops->compile_kernel(&cmd, program->default_kernels[i],
                                      device);
        }

    }

RET:
  POCL_UNLOCK_OBJ(program);
  return errcode;
}

CL_API_ENTRY cl_int CL_API_CALL
POname(clBuildProgram)(cl_program program,
                       cl_uint num_devices,
                       const cl_device_id *device_list,
                       const char *options,
                       void (CL_CALLBACK *pfn_notify) (cl_program program, 
                                                       void *user_data),
                       void *user_data) 
CL_API_SUFFIX__VERSION_1_0
{
  char program_bc_path[POCL_FILENAME_LENGTH];
  int errcode;
  size_t i;
  int error;
  uint64_t fsize;
  cl_device_id * unique_devlist = NULL;
  char *binary = NULL;
  unsigned device_i = 0, actually_built = 0;
  const char *user_options = "";
  char *temp_options = NULL;
  char *modded_options = NULL;
  char *token = NULL;
  char *saveptr = NULL;
  void* write_cache_lock = NULL;

  POCL_RETURN_ERROR_COND((program == NULL), CL_INVALID_PROGRAM);

  POCL_RETURN_ERROR_COND((num_devices > 0 && device_list == NULL), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND((num_devices == 0 && device_list != NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND((pfn_notify == NULL && user_data != NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_ON(program->kernels, CL_INVALID_OPERATION, "Program already has kernels\n");

  POCL_RETURN_ERROR_ON((program->source == NULL && program->binaries == NULL),
    CL_INVALID_PROGRAM, "Program doesn't have sources or binaries! You need "
                        "to call clCreateProgramWith{Binary|Source} first\n");

  POCL_LOCK_OBJ(program);

  program->main_build_log[0] = 0;

  if (options != NULL)
    {
      size_t size = 512;
      size_t i = 1; /* terminating char */
      modded_options = (char*) calloc (size, 1);

      if (strstr(options, "-cl-kernel-arg-info"))
        temp_options = strdup (options);
      else
        {
          temp_options = 
            calloc (1, strlen(options) + 1 + strlen(" -cl-kernel-arg-info"));
          strcat (temp_options, options);
          strcat (temp_options, " -cl-kernel-arg-info");
        }
      token = strtok_r (temp_options, " ", &saveptr);
      while (token != NULL)
        {
          /* check if parameter is supported compiler parameter */
          if (memcmp (token, "-cl", 3) == 0 || memcmp (token, "-w", 2) == 0 
              || memcmp(token, "-g", 2) == 0 || memcmp(token, "-Werror", 7) == 0)
            {
              if (strstr (cl_parameters, token))
                {
                  /* the LLVM API call pushes the parameters directly to the 
                     frontend without using -Xclang */
                }
              else if (strstr (cl_parameters_not_yet_supported_by_clang, token))
                {
                  APPEND_TO_MAIN_BUILD_LOG("Build option isnt yet supported by clang: %s\n", token);
                  token = strtok_r (NULL, " ", &saveptr);  
                  continue;
                }
              else
                {
                  APPEND_TO_MAIN_BUILD_LOG("Invalid build option: %s\n", token);
                  errcode = CL_INVALID_BUILD_OPTIONS;
                  goto ERROR_CLEAN_OPTIONS;
                }
            }
          else if (memcmp (token, "-D", 2) == 0 || memcmp (token, "-I", 2) == 0)
            {
              APPEND_TOKEN();
              /* if there is a space in between, then next token is part 
                 of the option */
              if (strlen (token) == 2)
                token = strtok_r (NULL, " ", &saveptr);
              else
                {
                  token = strtok_r (NULL, " ", &saveptr);
                  continue;
                }
            }
          else
            {
              APPEND_TO_MAIN_BUILD_LOG("Invalid build option: %s\n", token);
              errcode = CL_INVALID_BUILD_OPTIONS;
              goto ERROR_CLEAN_OPTIONS;
            }
          APPEND_TOKEN();
          token = strtok_r (NULL, " ", &saveptr);
        }
      POCL_MEM_FREE(temp_options);
      user_options = modded_options;
      program->compiler_options = strdup (modded_options);
    }
  else
    {
      program->compiler_options = calloc (1, strlen("-cl-kernel-arg-info")+1);
      strcat (program->compiler_options, "-cl-kernel-arg-info");
    }

  if (num_devices == 0)
    {
      num_devices = program->num_devices;
      device_list = program->devices;
    }
  else
    {
      // convert subdevices to devices and remove duplicates
      cl_uint real_num_devices = 0;
      unique_devlist = pocl_unique_device_list(device_list, num_devices, &real_num_devices);
      num_devices = real_num_devices;
      device_list = unique_devlist;
    }

  POCL_MSG_PRINT_INFO("building program with options %s\n",
                      user_options != NULL ? user_options : "");

  /* Build the fully linked non-parallel bitcode for all
         devices. */
  for (device_i = 0; device_i < program->num_devices; ++device_i)
    {
      cl_device_id device = program->devices[device_i];

      /* find the device in the supplied devices-to-build-for list */
      int found = 0;
      for (i = 0; i < num_devices; ++i)
          if (device_list[i] == device) found = 1;
      if (!found) continue;

      actually_built++;

      /* clCreateProgramWithSource */
      if (program->source)
        {
          error = pocl_llvm_build_program(program, device_i,
                                          user_options, program_bc_path);
          POCL_GOTO_ERROR_ON((error != 0), CL_BUILD_PROGRAM_FAILURE,
                             "pocl_llvm_build_program() failed\n");
        }
      /* clCreateProgramWithBinaries */
      else if (program->binaries[device_i])
        {
          error = pocl_cache_create_program_cachedir(program, device_i,
                                                     NULL, 0, program_bc_path);
          POCL_GOTO_ERROR_ON((error != 0), CL_BUILD_PROGRAM_FAILURE,
                             "Could not create program cachedir");
          write_cache_lock = pocl_cache_acquire_writer_lock_i(program, device_i);
          assert(write_cache_lock);
          errcode = pocl_write_file(program_bc_path, (char*)program->binaries[device_i],
                          (uint64_t)program->binary_sizes[device_i], 0, 0);
          POCL_GOTO_ERROR_ON(errcode, CL_BUILD_PROGRAM_FAILURE,
                             "Failed to write binaries to program.bc\n");
        }
      else if (program->pocl_binaries[device_i])
        /* TODO pocl_binaries[i] might contain program.bc */
        continue;
      /* fail */
      else
        POCL_GOTO_ERROR_ON(1, CL_INVALID_BINARY, "Don't have sources and also no "
                           "binaries for device %s - can't build the program\n",
                           device->short_name);

      /* Read binaries from program.bc to memory */
      if (program->binaries[device_i] == NULL)
        {
          if (!write_cache_lock)
            write_cache_lock = pocl_cache_acquire_writer_lock_i(program, device_i);
          assert(write_cache_lock);
          errcode = pocl_read_file(program_bc_path, &binary, &fsize);
          POCL_GOTO_ERROR_ON(errcode, CL_BUILD_ERROR,
                             "Failed to read binaries from program.bc to "
                             "memory: %s\n", program_bc_path);

          program->binary_sizes[device_i] = (size_t)fsize;
          program->binaries[device_i] = (unsigned char *)binary;
        }

      if (program->llvm_irs[device_i] == NULL)
        {
          if (!write_cache_lock)
            write_cache_lock = pocl_cache_acquire_writer_lock_i(program, device_i);
          assert(write_cache_lock);
          pocl_update_program_llvm_irs(program, device_i, device);
        }

      /* Maintain a 'last_accessed' file in every program's
       * cache directory. Will be useful for cache pruning script
       * that flushes old directories based on LRU */
      pocl_cache_update_program_last_access(program, device_i);

      if (write_cache_lock)
        {
          pocl_cache_release_lock(write_cache_lock);
          write_cache_lock = NULL;
        }

    }

  POCL_GOTO_ERROR_ON((actually_built < num_devices), CL_BUILD_PROGRAM_FAILURE,
                     "Some of the devices on the argument-supplied list are"
                     "not available for the program, or do not exist\n");

  /* TODO probably wrong to assume */
  assert(program->num_kernels == 0);
  for (i=0; i < program->num_devices; i++)
    {
      if (program->binaries[i])
        {
          program->num_kernels = pocl_llvm_get_kernel_count(program);
          if (program->num_kernels)
            {
              program->kernel_names = calloc(program->num_kernels, sizeof(char*));
              pocl_llvm_get_kernel_names(program,
                                         program->kernel_names,
                                         program->num_kernels);
            }
          break;
        }
      if (program->pocl_binaries[i])
        {
          program->num_kernels =
              pocl_binary_get_kernel_count(program->pocl_binaries[i]);
          if (program->num_kernels)
            {
              program->kernel_names = calloc(program->num_kernels, sizeof(char*));
              pocl_binary_get_kernel_names(program->pocl_binaries[i],
                                           program->kernel_names,
                                           program->num_kernels);
            }
          break;
        }
    }
  POCL_GOTO_ERROR_ON((i >= program->num_devices),
                     CL_INVALID_BINARY,
                     "Could not set kernel number / names from the binary\n");

  POCL_MEM_FREE(unique_devlist);
  program->build_status = CL_BUILD_SUCCESS;
  POCL_UNLOCK_OBJ(program);

  /* set up all program kernels */
  /* TODO should not have to unlock program while adding default kernels */
  assert(program->default_kernels == NULL);
  program->default_kernels = calloc(program->num_kernels, sizeof(cl_kernel));

  for (i=0; i < program->num_kernels; i++)
    {
      program->default_kernels[i] =
          POname(clCreateKernel)(program,
                                 program->kernel_names[i],
                                 &error);
      POCL_GOTO_ERROR_ON((error != CL_SUCCESS),
                         CL_BUILD_PROGRAM_FAILURE,
                         "Failed to create default kernels\n");
    }

  return CL_SUCCESS;

  /* Set pointers to NULL during cleanup so that clProgramRelease won't
   * cause a double free. */

ERROR:
  for(i = 0; i < program->num_devices; i++)
  {
    POCL_MEM_FREE(program->binaries[i]);
    pocl_cache_release_lock(program->read_locks[i]);
    program->read_locks[i] = NULL;
  }
  if (program->num_kernels && program->kernel_names)
    {
      for (i=0; i < program->num_kernels; i++)
        POCL_MEM_FREE(program->kernel_names[i]);
      POCL_MEM_FREE(program->kernel_names);
    }
  if (program->default_kernels)
    {
      for (i=0; i < program->num_kernels; i++)
        if (program->default_kernels[i])
          POname(clReleaseKernel)(program->default_kernels[i]);
      POCL_MEM_FREE(program->default_kernels);
      POCL_LOCK_OBJ(program);
    }
  POCL_MEM_FREE(program->binaries);
  POCL_MEM_FREE(program->binary_sizes);
  POCL_MEM_FREE(unique_devlist);
  pocl_cache_release_lock(write_cache_lock);
ERROR_CLEAN_OPTIONS:
  POCL_MEM_FREE(modded_options);
  program->build_status = CL_BUILD_ERROR;

  POCL_UNLOCK_OBJ(program);
  return errcode;
}
POsym(clBuildProgram)
