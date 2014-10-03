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
#include <unistd.h>
#include <sys/stat.h>
#include "pocl_llvm.h"
#include "pocl_hash.h"
#include "pocl_util.h"

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
#define COMMAND_LENGTH 4096

static inline void
build_program_compute_hash(cl_program program)
{
  SHA1_CTX hash_ctx;
  int total_binary_size, i;

  pocl_SHA1_Init(&hash_ctx);

  if (program->source)
    {
      pocl_SHA1_Update(&hash_ctx, program->source, strlen(program->source));
    }
  else  /* Program was created with clCreateProgramWithBinary() */
    {
      total_binary_size = 0;
      for (i = 0; i < program->num_devices; ++i)
        total_binary_size += program->binary_sizes[i];

      /* Binaries are stored in continuous chunk of memory starting from binaries[0] */
      pocl_SHA1_Update(&hash_ctx, program->binaries[0], total_binary_size);
    }

  if (program->compiler_options)
    pocl_SHA1_Update(&hash_ctx, program->compiler_options, strlen(program->compiler_options));

  pocl_SHA1_Final(&hash_ctx, program->build_hash);
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
  char device_tmpdir[POCL_FILENAME_LENGTH];
  char binary_file_name[POCL_FILENAME_LENGTH];
  char buildlog_file_name[POCL_FILENAME_LENGTH];
  FILE *binary_file;
  size_t n;
  int errcode;
  int i;
  int error;
  size_t length;
  unsigned char *binary;
  unsigned real_num_devices;
  const cl_device_id *real_device_list;
  /* The default build script for .cl files. */
  int device_i = 0;
  const char *user_options = "";
  char *temp_options;
  char *modded_options = NULL;
  char *token;
  char *saveptr;
  char *str = NULL;

  if (program == NULL)
  {
    errcode = CL_INVALID_PROGRAM;
    goto ERROR;
  }

  if (pfn_notify == NULL && user_data != NULL)
  {
    errcode = CL_INVALID_VALUE;
    goto ERROR;
  }

  POCL_LOCK_OBJ(program);
  if (program->kernels)
  {
    errcode = CL_INVALID_OPERATION;
    goto ERROR;
  }
  
  if (options != NULL)
    {
      int size = 512; 
      int i = 1; /* terminating char */
      char *swap_tmp;
      modded_options = calloc (size, 1);
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
                  token = strtok_r (NULL, " ", &saveptr);  
                  continue;
                }
              else
                {
                  errcode = CL_INVALID_BUILD_OPTIONS;
                  goto ERROR_CLEAN_OPTIONS;
                }
            }
          else if (memcmp (token, "-D", 2) == 0 || memcmp (token, "-I", 2) == 0)
            {
              if (size <= (i + strlen (token) + 1))
                {
                  swap_tmp = modded_options;
                  modded_options = malloc (size + 256);
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
              errcode = CL_INVALID_BUILD_OPTIONS;
              goto ERROR_CLEAN_OPTIONS;
            }
          if (size <= (i + strlen (token) + 1))
            {
              swap_tmp = modded_options;
              modded_options = malloc (size + 256);
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

  if (program->source == NULL && program->binaries == NULL)
  {
    errcode = CL_INVALID_PROGRAM;
    goto ERROR;
  }

  if ((num_devices > 0 && device_list == NULL) ||
      (num_devices == 0 && device_list != NULL))
  {
    errcode = CL_INVALID_VALUE;
    goto ERROR;
  }
      
  if (num_devices == 0)
    {
      real_num_devices = program->num_devices;
      real_device_list = program->devices;
    } else
    {
      real_num_devices = num_devices;
      real_device_list = device_list;
    }

  build_program_compute_hash(program);
  program->temp_dir = pocl_create_progam_cache_dir(program);

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

  /* Build the fully linked non-parallel bitcode for all
         devices. */
  for (device_i = 0; device_i < real_num_devices; ++device_i)
    {
      cl_device_id device = real_device_list[device_i];
      snprintf(device_tmpdir, POCL_FILENAME_LENGTH, "%s/%s",
               program->temp_dir, device->short_name);

      if (access (device_tmpdir, F_OK) != 0)
        mkdir(device_tmpdir, S_IRWXU);

      pocl_check_and_invalidate_cache(program, device_i, device_tmpdir);

      snprintf(binary_file_name, POCL_FILENAME_LENGTH, "%s/%s",
               device_tmpdir, POCL_PROGRAM_BC_FILENAME);
      snprintf(buildlog_file_name, POCL_FILENAME_LENGTH, "%s/%s",
               program->temp_dir, POCL_BUILDLOG_FILENAME);

      /* First call to clBuildProgram. Cache not filled yet */
      if (access(binary_file_name, F_OK) != 0)
        {
          if (program->source)
            {
              error = pocl_llvm_build_program(program, device, device_i,
                        program->temp_dir, binary_file_name, device_tmpdir, user_options);

              if (error != 0)
                {
                  errcode = CL_BUILD_PROGRAM_FAILURE;
                  goto ERROR_CLEAN_BINARIES;
                }
            }

          if (program->binaries[device_i])
            {
              binary_file = fopen(binary_file_name, "w");
              MEM_ASSERT(binary_file == NULL, ERROR_CLEAN_PROGRAM);

              fwrite(program->binaries[device_i], 1,
                     program->binary_sizes[device_i], binary_file);

              fclose (binary_file);
            }
        }
      else if (pocl_read_text_file(buildlog_file_name, &str))
        {
          fprintf(stderr, str);
          POCL_MEM_FREE(str);
        }

      /* Read binaries from program.bc to memory */
      if (program->binaries[device_i] == NULL)
        {
          binary_file = fopen(binary_file_name, "r");
          MEM_ASSERT(binary_file == NULL, ERROR_CLEAN_PROGRAM);

          fseek(binary_file, 0, SEEK_END);
          program->binary_sizes[device_i] = ftell(binary_file);
          fseek(binary_file, 0, SEEK_SET);

          binary = (unsigned char *) malloc(program->binary_sizes[device_i]);
          MEM_ASSERT(binary == NULL, ERROR_CLEAN_PROGRAM);

          n = fread(binary, 1, program->binary_sizes[device_i], binary_file);
          MEM_ASSERT((n < program->binary_sizes[device_i]), ERROR_CLEAN_PROGRAM);
          program->binaries[device_i] = binary;
        }

      if (program->llvm_irs[device->dev_id] == NULL)
        {
          pocl_update_program_llvm_irs(program,
                                       device, binary_file_name);
        }
    }

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
