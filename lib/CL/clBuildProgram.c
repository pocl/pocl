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
#include "pocl.h"
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
#ifdef OCS_AVAILABLE
#include "pocl_llvm.h"
#endif
#include "pocl_util.h"
#include "pocl_file_util.h"
#include "pocl_cache.h"
#include "config.h"
#include "pocl_runtime_config.h"
#include "pocl_binary.h"
#include "pocl_shared.h"

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

static const char cl_parameters_supported_after_clang_3_9[] =
  "-cl-strict-aliasing " /* deprecated after OCL1.0 */
  "-cl-denorms-are-zero "
  "-cl-no-signed-zeros ";

static const char cl_parameters_not_yet_supported_by_clang[] =
  "-cl-uniform-work-group-size ";

#define MEM_ASSERT(x, err_jmp) do{ if (x){errcode = CL_OUT_OF_HOST_MEMORY;goto err_jmp;}} while(0)

// append token, growing modded_options, if necessary, by max(strlen(token)+1, 256)
#define APPEND_TOKEN() do {          \
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

#ifdef OCS_AVAILABLE
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

  /* Build the dynamic WG sized parallel.bc and device specific code,
     for each kernel & device combo.  */
  for (device_i = 0; device_i < program->num_devices; ++device_i)
    {
      cl_device_id device = program->devices[device_i];

      /* program may not be built for some of its devices */
      if (program->pocl_binaries[device_i] || (!program->binaries[device_i]))
        continue;

      cmd.device = device;

      for (i=0; i < program->num_kernels; i++)
        {
          cl_kernel kernel = program->default_kernels[i];
          size_t local_x = 0, local_y = 0, local_z = 0;
          if (kernel->reqd_wg_size != NULL &&
              kernel->reqd_wg_size[0] > 0 &&
              kernel->reqd_wg_size[1] > 0 &&
              kernel->reqd_wg_size[2] > 0)
            {
              local_x = kernel->reqd_wg_size[0];
              local_y = kernel->reqd_wg_size[1];
              local_z = kernel->reqd_wg_size[2];
            }
          cmd.command.run.local_x = local_x;
          cmd.command.run.local_y = local_y;
          cmd.command.run.local_z = local_z;
          cmd.command.run.kernel = kernel;
          pocl_cache_kernel_cachedir_path (cachedir, program, device_i, kernel,
                                           "", local_x, local_y, local_z);
          device->ops->compile_kernel (&cmd, kernel, device);
        }
    }

  POCL_UNLOCK_OBJ(program);
  return errcode;
}

#endif

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
  int error;
  uint64_t fsize;
  cl_device_id * unique_devlist = NULL;
  char *binary = NULL;
  unsigned device_i = 0, actually_built = 0;
  char *temp_options = NULL;
  char *modded_options = NULL;
  char *token = NULL;
  char *saveptr = NULL;
  void* write_cache_lock = NULL;
  build_program_callback_t *callback = NULL;

  POCL_RETURN_ERROR_COND((program == NULL), CL_INVALID_PROGRAM);

  POCL_RETURN_ERROR_COND((num_devices > 0 && device_list == NULL), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND((num_devices == 0 && device_list != NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND((pfn_notify == NULL && user_data != NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_ON(program->kernels, CL_INVALID_OPERATION, "Program already has kernels\n");

  POCL_RETURN_ERROR_ON((program->source == NULL && program->binaries == NULL),
    CL_INVALID_PROGRAM, "Program doesn't have sources or binaries! You need "
                        "to call clCreateProgramWith{Binary|Source} first\n");

  POCL_LOCK_OBJ(program);

  if (pfn_notify)
    {
      POCL_MEM_FREE (program->buildprogram_callback);
      callback = (build_program_callback_t*) malloc (sizeof(build_program_callback_t));
      if (callback == NULL)
        {
          POCL_UNLOCK_OBJ(program);
          return CL_OUT_OF_HOST_MEMORY;
        }

      callback->callback_function = pfn_notify;
      callback->user_data = user_data;
      program->buildprogram_callback = callback;
    }

  program->main_build_log[0] = 0;

  size_t i = 1; /* terminating char */
  modded_options = (char*) calloc (512, 1);

  if (options != NULL)
    {
      size_t size = 512;
      size_t i = 1; /* terminating char */
      temp_options = strdup(options);

      token = strtok_r (temp_options, " ", &saveptr);
      while (token != NULL)
        {
          /* check if parameter is supported compiler parameter */
          if (memcmp (token, "-cl", 3) == 0 || memcmp (token, "-w", 2) == 0 
              || memcmp(token, "-Werror", 7) == 0)
            {
              if (strstr (cl_parameters, token))
                {
                  /* the LLVM API call pushes the parameters directly to the 
                     frontend without using -Xclang */
                }
              else if (strstr (cl_parameters_supported_after_clang_3_9, token))
                {
#ifndef LLVM_OLDER_THAN_3_9
                  /* the LLVM API call pushes the parameters directly to the
                   * frontend without using -Xclang*/
#else
                  APPEND_TO_MAIN_BUILD_LOG("This build option is supported after clang3.9: %s\n", token);
                  token = strtok_r (NULL, " ", &saveptr);  
                  continue;
#endif
                }
              else if (strstr (cl_parameters_not_yet_supported_by_clang, token))
                {
                  APPEND_TO_MAIN_BUILD_LOG("This build option is not yet supported by clang: %s\n", token);
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
          else if (memcmp(token, "-g", 2) == 0)
            {
#ifndef LLVM_OLDER_THAN_3_8
              token = "-debug-info-kind=line-tables-only";
#endif
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
          else if (memcmp (token, "-x", 2) == 0 && strlen (token) == 2)
            {
              /* only "-x spir" is valid for the "-x" option */
              token = strtok_r (NULL, " ", &saveptr);
              if (!token || memcmp (token, "spir", 4) != 0)
                {
                  APPEND_TO_MAIN_BUILD_LOG("Invalid parameter to -x build option\n");
                  errcode = CL_INVALID_BUILD_OPTIONS;
                  goto ERROR_CLEAN_OPTIONS;
                }
              /* "-x spir" is not valid if we are building from source */
              else if (program->source)
                {
                  APPEND_TO_MAIN_BUILD_LOG("\"-x spir\" is not valid when building from source\n");
                  errcode = CL_INVALID_BUILD_OPTIONS;
                  goto ERROR_CLEAN_OPTIONS;
                }
              token = strtok_r (NULL, " ", &saveptr);
              continue;
            }
          else if (memcmp (token, "-spir-std=1.2", 13) == 0)
            {
              /* "-spir-std=" flags are not valid when building from source */
              if (program->source)
                {
                  APPEND_TO_MAIN_BUILD_LOG("\"-spir-std=\" flag is not valid when building from source\n");
                  errcode = CL_INVALID_BUILD_OPTIONS;
                  goto ERROR_CLEAN_OPTIONS;
                }
              token = strtok_r (NULL, " ", &saveptr);
              continue;
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
    }

  POCL_MEM_FREE(program->compiler_options);
  program->compiler_options = modded_options;

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
                       program->compiler_options);

  /* if we're rebuilding the program, release the kernels and reset log/status
   */
  if ((program->build_status != CL_BUILD_NONE) || program->num_kernels > 0)
    {
      cl_kernel k;
      for (k = program->kernels; k != NULL; k = k->next)
        {
          k->program = NULL;
          --program->pocl_refcount;
        }
      program->kernels = NULL;
      if (program->num_kernels)
        {
          program->operating_on_default_kernels = 1;
          for (i = 0; i < program->num_kernels; i++)
            {
              if (program->kernel_names)
                POCL_MEM_FREE (program->kernel_names[i]);
              if (program->default_kernels && program->default_kernels[i])
                POname (clReleaseKernel) (program->default_kernels[i]);
            }
          POCL_MEM_FREE (program->kernel_names);
          POCL_MEM_FREE (program->default_kernels);
          program->operating_on_default_kernels = 0;
        }
      program->num_kernels = 0;
      program->build_status = CL_BUILD_NONE;
      if (program->build_log)
        for (i = 0; i < program->num_devices; ++i)
          {
            POCL_MEM_FREE (program->build_log[i]);
            memset (program->build_hash[i], 0, sizeof (SHA1_digest_t));
          }
    }

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
          POCL_MSG_PRINT_INFO("building from sources for device %d\n", device_i);
#ifdef OCS_AVAILABLE
          error = pocl_llvm_build_program(program, device_i,
                                          program->compiler_options,
                                          program_bc_path, 0, NULL, NULL);
          POCL_GOTO_ERROR_ON((error != 0), CL_BUILD_PROGRAM_FAILURE,
                             "pocl_llvm_build_program() failed\n");
#else
          strcpy(program->main_build_log,
                 "Cannot build a program from sources with pocl "
                 "that does not have online compiler support\n");
          POCL_GOTO_ERROR_ON(1, CL_COMPILER_NOT_AVAILABLE,
                             "%s", program->main_build_log);
#endif
        }
      /* clCreateProgramWithBinaries */
      else if (program->binaries[device_i])
        {
          POCL_MSG_PRINT_INFO("building from a BC binary for device %d\n", device_i);

#ifdef OCS_AVAILABLE
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
#else
          if (!program->pocl_binaries[device_i])
            {
              strcpy(program->main_build_log,
                     "Cannot build program from LLVM IR binaries with "
                     "pocl that does not have online compiler support\n");
              POCL_GOTO_ERROR_ON(1, CL_COMPILER_NOT_AVAILABLE,
                                 "%s", program->main_build_log);
            }
          else
            continue;
#endif
        }
      else if (program->pocl_binaries[device_i])
        {
          POCL_MSG_PRINT_INFO("having a poclbinary for device %d\n", device_i);
          /* TODO pocl_binaries[i] might contain program.bc */
          continue;
          /* fail */
        }
      else
        {
          POCL_MSG_PRINT_INFO("no sources nor binaries to build for device %d\n",
                              device_i);
          /* TODO pocl_binaries[i] might contain program.bc */

          POCL_GOTO_ERROR_ON(1, CL_INVALID_BINARY,
                             "No sources nor binaries for device %s - can't "
                             "build the program\n", device->short_name);
        }

#ifdef OCS_AVAILABLE
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
#endif

    }

  POCL_GOTO_ERROR_ON((actually_built < num_devices), CL_BUILD_PROGRAM_FAILURE,
                     "Some of the devices on the argument-supplied list are"
                     "not available for the program, or do not exist\n");

  assert(program->num_kernels == 0);
  for (i=0; i < program->num_devices; i++)
    {
#ifdef OCS_AVAILABLE
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
#endif
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

  if (program->buildprogram_callback)
    program->buildprogram_callback->callback_function (program,
                                  program->buildprogram_callback->user_data);

  /* Set up all program kernels.  */
  /* TODO: Should not have to unlock program while adding default kernels.  */
  assert (program->default_kernels == NULL);
  program->operating_on_default_kernels = 1;
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

  program->operating_on_default_kernels = 0;

  return CL_SUCCESS;

  /* Set pointers to NULL during cleanup so that clProgramRelease won't
   * cause a double free. */

ERROR:
  if (program->buildprogram_callback)
    {
      program->buildprogram_callback->callback_function(program,
                         program->buildprogram_callback->user_data);
      POCL_MEM_FREE(program->buildprogram_callback);
    }
  program->kernels = 0;
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
  program->build_status = CL_BUILD_ERROR;

  POCL_UNLOCK_OBJ(program);
  return errcode;
}
POsym(clBuildProgram)
