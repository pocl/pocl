/* OpenCL runtime library: clBuildProgram()

   Copyright (c) 2011-2013 Universidad Rey Juan Carlos,
                           Pekka Jääskeläinen / Tampere Univ. of Technology
   
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
#include "install-paths.h"
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include "pocl_llvm.h"

/* supported compiler parameters */
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
  "-cl-kernel-arg-info ";

static char cl_parameters_not_yet_supported_by_clang[] = 
  "-cl-strict-aliasing "
  "-cl-denorms-are-zero "
  "-cl-no-signed-zeros ";

#define MEM_ASSERT(x, err_jmp) do{ if (x){errcode = CL_OUT_OF_HOST_MEMORY;goto err_jmp;}} while(0)
#define COMMAND_LENGTH 4096

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
  char tmpdir[POCL_FILENAME_LENGTH];
  char device_tmpdir[POCL_FILENAME_LENGTH];
  char source_file_name[POCL_FILENAME_LENGTH], binary_file_name[POCL_FILENAME_LENGTH];
  FILE *source_file, *binary_file;
  size_t n;
  struct stat buf;
  char command[COMMAND_LENGTH];
  int errcode;
  int i;
  int error;
  unsigned char *binary;
  unsigned real_num_devices;
  const cl_device_id *real_device_list;
  /* The default build script for .cl files. */
  char *pocl_build_script;
  int device_i = 0;
  const char *user_options = "";
  char *temp_options;
  char *modded_options = NULL;
  char *token;
  char *saveptr;

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

  if (program->kernels)
  {
    errcode = CL_INVALID_OPERATION;
    goto ERROR;
  }
  
  if (options != NULL)
    {
      modded_options = calloc (512, 1);
      temp_options = strdup (options);
      token = strtok_r (temp_options, " ", &saveptr);
      while (token != NULL)
        {
          /* check if parameter is supported compiler parameter */
          if (strstr (token, "-cl"))
            {
              if (strstr (cl_parameters, token))
                strcat (modded_options, "-Xclang ");
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
          else if (strstr (token, "-D") || strstr (token, "-I"))
            {
              strcat (modded_options, token);
              strcat (modded_options, " ");
              /* if there is a space in between, then next token is part 
                 of the option */
              if (strlen (token) == 2)
                token = strtok_r (NULL, " ", &saveptr);
            }
          else
            {
              errcode = CL_INVALID_BUILD_OPTIONS;
              goto ERROR_CLEAN_OPTIONS;
            }
          strcat (modded_options, token);
          strcat (modded_options, " ");
          token = strtok_r (NULL, " ", &saveptr);  
        }
      free (temp_options);
      
      user_options = modded_options;
      program->compiler_options = strdup (modded_options);
    }
  else
    {
      free(program->compiler_options);
      program->compiler_options = NULL;        
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

  if (program->binaries == NULL)
    {
      snprintf (tmpdir, POCL_FILENAME_LENGTH, "%s/", program->temp_dir);
      mkdir (tmpdir, S_IRWXU);

      if (((program->binary_sizes =
           (size_t *) malloc (sizeof (size_t) * real_num_devices)) == NULL) 
              || (program->binaries = 
           (unsigned char**) calloc( real_num_devices, sizeof (unsigned char*))) == NULL)
      {
        errcode = CL_OUT_OF_HOST_MEMORY;
        goto ERROR_CLEAN_BINARIES;
      }

      snprintf 
        (source_file_name, POCL_FILENAME_LENGTH, "%s/%s", tmpdir, 
         POCL_PROGRAM_CL_FILENAME);

      source_file = fopen(source_file_name, "w+");
      if (source_file == NULL)
      {
        errcode = CL_OUT_OF_HOST_MEMORY;
        goto ERROR_CLEAN_BINARIES;
      }

      n = fwrite (program->source, 1,
                  strlen(program->source), source_file);
      fclose(source_file);

      if (n < strlen(program->source))
      {
        errcode = CL_OUT_OF_HOST_MEMORY;
        goto ERROR_CLEAN_BINARIES;
      }


      /* Build the fully linked non-parallel bitcode for all
         devices. */
      for (device_i = 0; device_i < real_num_devices; ++device_i)
        {
          program->binaries[device_i] = NULL;
          cl_device_id device = real_device_list[device_i];
          snprintf (device_tmpdir, POCL_FILENAME_LENGTH, "%s/%s", 
                    program->temp_dir, device->short_name);
          mkdir (device_tmpdir, S_IRWXU);

          snprintf 
            (binary_file_name, POCL_FILENAME_LENGTH, "%s/%s", 
             device_tmpdir, POCL_PROGRAM_BC_FILENAME);

          error = call_pocl_build( device, source_file_name,
                                   binary_file_name, device_tmpdir,
                                   user_options );     

          if (error != 0)
          {
            errcode = CL_BUILD_PROGRAM_FAILURE;
            goto ERROR_CLEAN_BINARIES;
          }

          binary_file = fopen(binary_file_name, "r");
          if (binary_file == NULL)
          {
            errcode = CL_OUT_OF_HOST_MEMORY;
            goto ERROR_CLEAN_BINARIES;
          }

          fseek(binary_file, 0, SEEK_END);

          program->binary_sizes[device_i] = ftell(binary_file);
          fseek(binary_file, 0, SEEK_SET);

          binary = (unsigned char *) malloc(program->binary_sizes[device_i]);
          if (binary == NULL)
          {
              errcode = CL_OUT_OF_HOST_MEMORY;
              goto ERROR_CLEAN_BINARIES;
          }

          n = fread(binary, 1, program->binary_sizes[device_i], binary_file);
          if (n < program->binary_sizes[device_i])
            {
                errcode = CL_OUT_OF_HOST_MEMORY;
                goto ERROR_CLEAN_BINARIES;
            }
          program->binaries[device_i] = binary;
        }
    }
  else
    {
      /* Build from a binary. The "binaries" (LLVM bitcodes) are loaded to
         memory in the clBuildWithBinary(). Dump them to the files. */
      for (device_i = 0; device_i < real_num_devices; ++device_i)
        {
          int count;
          count = snprintf (device_tmpdir, POCL_FILENAME_LENGTH, "%s/%s", 
                    program->temp_dir, real_device_list[device_i]->short_name);
          MEM_ASSERT(count >= POCL_FILENAME_LENGTH, ERROR_CLEAN_PROGRAM);

          error = mkdir (device_tmpdir, S_IRWXU);
          MEM_ASSERT(error, ERROR_CLEAN_PROGRAM);

          count = snprintf 
            (binary_file_name, POCL_FILENAME_LENGTH, "%s/%s", 
             device_tmpdir, POCL_PROGRAM_BC_FILENAME);
          MEM_ASSERT(count >= POCL_FILENAME_LENGTH, ERROR_CLEAN_PROGRAM);

          binary_file = fopen(binary_file_name, "w");
          MEM_ASSERT(binary_file == NULL, ERROR_CLEAN_PROGRAM);

          fwrite (program->binaries[device_i], 1, program->binary_sizes[device_i],
                  binary_file);

          fclose (binary_file);
        }      
    }

  return CL_SUCCESS;

  /* Set pointers to NULL during cleanup so that clProgramRelease won't
   * cause a double free. */

ERROR_CLEAN_BINARIES:
  for(i = 0; i < device_i; i++)
  {
    free(program->binaries[i]);
    program->binaries[i] = NULL;
  }
ERROR_CLEAN_PROGRAM:
  free(program->binaries);
  program->binaries = NULL;
  free(program->binary_sizes);
  program->binary_sizes = NULL;
ERROR_CLEAN_OPTIONS:
  free (modded_options);
ERROR:
  return errcode;
}
POsym(clBuildProgram)
