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
#include "pocl_file_util.h"
#include "pocl_cache.h"
#include "config.h"
#include "pocl_runtime_config.h"

/* supported compiler parameters which should pass to the frontend directly
   by using -Xclang */
static char cl_parameters[] = 
  "-cl-single-precision-constant "
  "-cl-fp32-correctly-rounded-divide-sqrt "
  "-cl-opt-disable "
  "-cl-mad-enable "
  "-cl-unsafe-math-optimizations "
  "-cl-finite-math-only "
  "-cl-fast-relaxed-math "
  "-cl-std=CL1.2 "
  "-cl-std=CL1.1 "
  "-cl-kernel-arg-info "
  "-w "
  "-g ";

static char cl_parameters_not_yet_supported_by_clang[] = 
  "-cl-strict-aliasing "
  "-cl-denorms-are-zero "
  "-cl-no-signed-zeros ";

#define MEM_ASSERT(x, err_jmp) do{ if (x){errcode = CL_OUT_OF_HOST_MEMORY;goto err_jmp;}} while(0)

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
  size_t length;
  uint64_t fsize;
  char *binary = NULL;
  unsigned real_num_devices;
  const cl_device_id *real_device_list;
  /* The default build script for .cl files. */
  unsigned device_i = 0;
  const char *user_options = "";
  char *temp_options;
  char *modded_options = NULL;
  char *token;
  char *saveptr = NULL;

  POCL_GOTO_ERROR_COND((program == NULL), CL_INVALID_PROGRAM);

  POCL_GOTO_ERROR_COND((pfn_notify == NULL && user_data != NULL), CL_INVALID_VALUE);

  POCL_LOCK_OBJ(program);

  POCL_GOTO_ERROR_ON(program->kernels, CL_INVALID_OPERATION, "Program already has kernels\n");
  
  if (options != NULL)
    {
      size_t size = 512;
      size_t i = 1; /* terminating char */
      char *swap_tmp;
      modded_options = (char*) calloc (size, 1);
      temp_options = strdup (options);
      token = strtok_r (temp_options, " ", &saveptr);
      while (token != NULL)
        {
          /* check if parameter is supported compiler parameter */
          if (memcmp (token, "-cl", 3) == 0 || memcmp (token, "-w", 2) == 0 
              || memcmp(token, "-g", 2) == 0)
            {
              if (strstr (cl_parameters, token))
                {
                  /* the LLVM API call pushes the parameters directly to the 
                     frontend without using -Xclang */
                }
              else if (strstr (cl_parameters_not_yet_supported_by_clang, token))
                {
                  POCL_MSG_ERR("Build option isnt yet supported by clang: %s\n", token);
                  token = strtok_r (NULL, " ", &saveptr);  
                  continue;
                }
              else
                {
                  POCL_MSG_ERR("Invalid build option: %s\n", token);
                  errcode = CL_INVALID_BUILD_OPTIONS;
                  goto ERROR_CLEAN_OPTIONS;
                }
            }
          else if (memcmp (token, "-D", 2) == 0 || memcmp (token, "-I", 2) == 0)
            {
              if (size <= (i + strlen (token) + 1))
                {
                  swap_tmp = modded_options;
                  modded_options = (char*) malloc (size + 256);
                  if (modded_options == NULL)
                    return CL_OUT_OF_HOST_MEMORY;
                  memcpy (modded_options, swap_tmp, size);
                  POCL_MEM_FREE(swap_tmp);
                  size += 256;
                }
              i += strlen (token) + 1;
              strcat (modded_options, token);
              strcat (modded_options, " ");
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
              POCL_MSG_ERR("Invalid build option: %s\n", token);
              errcode = CL_INVALID_BUILD_OPTIONS;
              goto ERROR_CLEAN_OPTIONS;
            }
          if (size <= (i + strlen (token) + 1))
            {
              swap_tmp = modded_options;
              modded_options = (char*) malloc (size + 256);
              if (modded_options == NULL)
                return CL_OUT_OF_HOST_MEMORY;
              memcpy (modded_options, swap_tmp, size); 
              POCL_MEM_FREE(swap_tmp);
              size += 256;
            }
          i += strlen (token) + 1;
          strcat (modded_options, token);
          strcat (modded_options, " ");
          token = strtok_r (NULL, " ", &saveptr);  
        }
      POCL_MEM_FREE(temp_options);
      user_options = modded_options;
      program->compiler_options = strdup (modded_options);
    }
  else
    {
      POCL_MEM_FREE(program->compiler_options);
    }  

  POCL_GOTO_ERROR_ON((program->source == NULL && program->binaries == NULL),
    CL_INVALID_PROGRAM, "Program doesn't have sources or binaries! You need "
                        "to call clCreateProgramWith{Binary|Source} first\n");

  POCL_GOTO_ERROR_COND((num_devices > 0 && device_list == NULL), CL_INVALID_VALUE);
  POCL_GOTO_ERROR_COND((num_devices == 0 && device_list != NULL), CL_INVALID_VALUE);
      
  if (num_devices == 0)
    {
      real_num_devices = program->num_devices;
      real_device_list = program->devices;
    } else
    {
      real_num_devices = num_devices;
      real_device_list = device_list;
    }

  pocl_cache_create_program_cachedir(program);

  if (program->source)
    {
      /* Realloc for every clBuildProgram call
       * since clBuildProgram can be called multiple times
       * with different options and device count
       */
      length = sizeof(size_t) * real_num_devices;
      program->binary_sizes = (size_t *) realloc(program->binary_sizes, length);
      MEM_ASSERT(program->binary_sizes == NULL, ERROR_CLEAN_PROGRAM);
      memset(program->binary_sizes, 0, length);

      length = sizeof(unsigned char*) * real_num_devices;
      program->binaries = (unsigned char**) realloc(program->binaries, length);
      MEM_ASSERT(program->binaries == NULL, ERROR_CLEAN_PROGRAM);
      memset(program->binaries, 0, length);

      length = sizeof(void*) * real_num_devices;
      program->llvm_irs = (void**) realloc (program->llvm_irs, length);
      MEM_ASSERT(program->llvm_irs == NULL, ERROR_CLEAN_PROGRAM);
      memset(program->llvm_irs, 0, length);
    }

  POCL_MSG_PRINT_INFO("building program with options %s\n",
                      options != NULL ? options : "");

  /* Build the fully linked non-parallel bitcode for all
         devices. */
  for (device_i = 0; device_i < real_num_devices; ++device_i)
    {
      cl_device_id device = real_device_list[device_i];

      pocl_cache_make_device_cachedir(program, device);

      pocl_cache_program_bc_path(program_bc_path, program, device);

      // clCreateProgramWithSource
      if (program->source)
        {
          error = pocl_llvm_build_program(program, device, device_i, user_options);
          if (error != 0)
            {
              errcode = CL_BUILD_PROGRAM_FAILURE;
              goto ERROR_CLEAN_BINARIES;
            }
        }
      else
        if (program->binaries[device_i]) {
            errcode = pocl_write_file(program_bc_path, program->binaries[device_i],
                          (uint64_t)program->binary_sizes[device_i], 0, 0);
            POCL_GOTO_ERROR_ON(errcode, CL_BUILD_ERROR, "Failed to write binaries to program.bc\n");
        }

      /* Read binaries from program.bc to memory */
      if (program->binaries[device_i] == NULL)
        {
          errcode = pocl_read_file(program_bc_path, &binary, &fsize);
          POCL_GOTO_ERROR_ON(errcode, CL_BUILD_ERROR, "Failed to read binaries from program.bc to memory: %s\n", program_bc_path);

          program->binary_sizes[device_i] = (size_t)fsize;
          program->binaries[device_i] = binary;
        }

      if (program->llvm_irs[device_i] == NULL)
        {
          pocl_update_program_llvm_irs(program,
                                       device, program_bc_path);
        }
    }

  /* Maintain a 'last_accessed' file in every program's
   * cache directory. Will be useful for cache pruning script
   * that flushes old directories based on LRU */
  pocl_cache_update_program_last_access(program);

  program->build_status = CL_BUILD_SUCCESS;
  POCL_UNLOCK_OBJ(program);
  return CL_SUCCESS;

  /* Set pointers to NULL during cleanup so that clProgramRelease won't
   * cause a double free. */

ERROR_CLEAN_BINARIES:
  for(i = 0; i < device_i; i++)
  {
    POCL_MEM_FREE(program->binaries[i]);
  }
ERROR_CLEAN_PROGRAM:
  POCL_MEM_FREE(program->binaries);
  POCL_MEM_FREE(program->binary_sizes);
ERROR_CLEAN_OPTIONS:
  POCL_MEM_FREE(modded_options);
ERROR:
  program->build_status = CL_BUILD_ERROR;
  POCL_UNLOCK_OBJ(program);
  return errcode;
}
POsym(clBuildProgram)
