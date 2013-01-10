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

#define COMMAND_LENGTH 1024

CL_API_ENTRY cl_int CL_API_CALL
POname(clBuildProgram)(cl_program program,
               cl_uint num_devices,
               const cl_device_id *device_list,
               const char *options,
               void (CL_CALLBACK *pfn_notify)(cl_program program, void *user_data),
               void *user_data) CL_API_SUFFIX__VERSION_1_0
{
  char tmpdir[POCL_FILENAME_LENGTH];
  char device_tmpdir[POCL_FILENAME_LENGTH];
  char source_file_name[POCL_FILENAME_LENGTH], binary_file_name[POCL_FILENAME_LENGTH];
  FILE *source_file, *binary_file;
  size_t n;
  struct stat buf;
  char command[COMMAND_LENGTH];
  int error;
  unsigned char *binary;
  int device_i;
  unsigned real_num_devices;
  cl_device_id *real_device_list;
  /* The default build script for .cl files. */
  char *pocl_build_script;
  char *user_options = "";

  if (program == NULL)
    return CL_INVALID_PROGRAM;

  if (pfn_notify == NULL && user_data != NULL)
    return CL_INVALID_VALUE;

  if (options != NULL)
    {
      user_options = options;
      program->compiler_options = strdup(options);
    }
  else
    {
      free(program->compiler_options);
      program->compiler_options = NULL;        
    }  

  if (program->source == NULL && program->binaries == NULL)
    return CL_INVALID_PROGRAM;

  if ((num_devices > 0 && device_list == NULL) ||
      (num_devices == 0 && device_list != NULL))
    return CL_INVALID_VALUE;
      
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

      if ((program->binary_sizes =
           (size_t *) malloc (sizeof (size_t) * real_num_devices)) == NULL)
        return CL_OUT_OF_HOST_MEMORY;

      if ((program->binaries = 
           (unsigned char**) 
           calloc( real_num_devices, sizeof (unsigned char*))) == NULL)
        {
          free (program->binary_sizes);
          program->binary_sizes = NULL;
          return CL_OUT_OF_HOST_MEMORY;
        }
     
      snprintf 
        (source_file_name, POCL_FILENAME_LENGTH, "%s/%s", tmpdir, 
         POCL_PROGRAM_CL_FILENAME);

      source_file = fopen(source_file_name, "w+");
      if (source_file == NULL)
        return CL_OUT_OF_HOST_MEMORY;

      n = fwrite (program->source, 1,
                  strlen(program->source), source_file);
      if (n < strlen(program->source))
        return CL_OUT_OF_HOST_MEMORY;

      fclose(source_file);

      if (getenv("POCL_BUILDING") != NULL)
        pocl_build_script = BUILDDIR "/scripts/" POCL_BUILD;
      else if (access(PKGDATADIR "/" POCL_BUILD, X_OK) == 0)
        pocl_build_script = PKGDATADIR "/" POCL_BUILD;
      else
        pocl_build_script = POCL_BUILD;

      /* Build the fully linked non-parallel bitcode for all
         devices. */
      for (device_i = 0; device_i < real_num_devices; ++device_i)
        {
          cl_device_id device = real_device_list[device_i];
          snprintf (device_tmpdir, POCL_FILENAME_LENGTH, "%s/%s", 
                    program->temp_dir, device->name);
          mkdir (device_tmpdir, S_IRWXU);

          snprintf 
            (binary_file_name, POCL_FILENAME_LENGTH, "%s/%s", 
             device_tmpdir, POCL_PROGRAM_BC_FILENAME);

          if (real_device_list[device_i]->llvm_target_triplet != NULL)
            {
              error = snprintf(command, COMMAND_LENGTH,
                               "USER_OPTIONS=\"%s\" %s -t %s -o %s %s", 
                               user_options,
                               pocl_build_script,
                               device->llvm_target_triplet,                               
                               binary_file_name, source_file_name);
            }
          else 
            {
              error = snprintf(command, COMMAND_LENGTH,
                               "USER_OPTIONS=\"%s\" %s -o %s %s", 
                               user_options,
                               pocl_build_script,
                               binary_file_name, source_file_name);
            }
          
          if (error < 0)
            return CL_OUT_OF_HOST_MEMORY;

          /* call the customized build command, if needed for the
             device driver */
          if (device->build_program != NULL)
            {
              error = device->build_program 
                (device->data, source_file_name, binary_file_name, 
                 command, device_tmpdir);
            }
          else
            {
              error = system(command);
            }

          if (error != 0)
            return CL_BUILD_PROGRAM_FAILURE;

          binary_file = fopen(binary_file_name, "r");
          if (binary_file == NULL)
            return CL_OUT_OF_HOST_MEMORY;

          fseek(binary_file, 0, SEEK_END);

          program->binary_sizes[device_i] = ftell(binary_file);
          fseek(binary_file, 0, SEEK_SET);

          binary = (unsigned char *) malloc(program->binary_sizes[device_i]);
          if (binary == NULL)
            return CL_OUT_OF_HOST_MEMORY;

          n = fread(binary, 1, program->binary_sizes[device_i], binary_file);
          if (n < program->binary_sizes[device_i])
            {
              free (binary);
              return CL_OUT_OF_HOST_MEMORY;
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
          snprintf (device_tmpdir, POCL_FILENAME_LENGTH, "%s/%s", 
                    program->temp_dir, real_device_list[device_i]->name);
          mkdir (device_tmpdir, S_IRWXU);

          snprintf 
            (binary_file_name, POCL_FILENAME_LENGTH, "%s/%s", 
             device_tmpdir, POCL_PROGRAM_BC_FILENAME);

          binary_file = fopen(binary_file_name, "w");
          fwrite (program->binaries[device_i], 1, program->binary_sizes[device_i],
                  binary_file);

          fclose (binary_file);
          
        }      
    }

  return CL_SUCCESS;
}
POsym(clBuildProgram)
