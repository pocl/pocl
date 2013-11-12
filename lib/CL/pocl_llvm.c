/* pocl_llvm.c: C wrappers for calling the scripts that direct the kernel
   compilation process (by calling clang and opt from the shell)

   Copyright (c) 2013 Kalle Raiskila
   
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
#include "install-paths.h"
#include "pocl_llvm.h"
#include <assert.h>
#include <sys/stat.h>
#include <unistd.h>
#include "pocl_cl.h"


// TODO: copies...
#define COMMAND_LENGTH 2048
#define WORKGROUP_STRING_LENGTH 128

#if !defined(USE_LLVM_API) || USE_LLVM_API != 1

int call_pocl_build( cl_device_id device, 
                     const char* source_file_name,
                     const char* binary_file_name,
                     const char* device_tmpdir,
                     const char* user_options )
{
  int error;
  const char *pocl_build_script;
  char command[COMMAND_LENGTH];

  if (getenv("POCL_BUILDING") != NULL)
    pocl_build_script = BUILDDIR "/scripts/" POCL_BUILD;
  else if (access(PKGDATADIR "/" POCL_BUILD, X_OK) == 0)
    pocl_build_script = PKGDATADIR "/" POCL_BUILD;
  else
    pocl_build_script = POCL_BUILD;

  if (device->llvm_target_triplet != NULL)
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
         command, user_options, device_tmpdir);
    }
  else
    {
      error = system(command);
    }

  return error;
}

/* call pocl_kenel script.
 *
 * device_i is the index into the device and binaries-arrays 
 * in the 'program' struct that point to the device we are building
 * for. No safety checks in this function!
 */
int call_pocl_kernel(cl_program program, 
                     cl_kernel kernel,
                     int device_i,     
                     const char* kernel_name,
                     const char* device_tmpdir, 
                     char* descriptor_filename,
                     int *errcode )
{
  int error;
  char* pocl_kernel_fmt;
  FILE *binary_file;
  size_t n;
  char tmpdir[POCL_FILENAME_LENGTH];
  char binary_filename[POCL_FILENAME_LENGTH];
  char command[COMMAND_LENGTH];

  if (getenv("POCL_BUILDING") != NULL)
    pocl_kernel_fmt = BUILDDIR "/scripts/" POCL_KERNEL " -k %s -t %s -o %s %s";
  else if (access(PKGDATADIR "/" POCL_KERNEL, X_OK) == 0)
    pocl_kernel_fmt = PKGDATADIR "/" POCL_KERNEL " -k %s -t %s -o %s %s";
  else
    pocl_kernel_fmt = POCL_KERNEL " -k %s -t %s -o %s %s";


  snprintf (tmpdir, POCL_FILENAME_LENGTH, "%s/%s", 
            device_tmpdir, kernel_name);
  mkdir (tmpdir, S_IRWXU);

  error = snprintf(binary_filename, POCL_FILENAME_LENGTH,
                   "%s/kernel.bc",
                   tmpdir);
  if (error < 0)
  {
    *errcode = CL_OUT_OF_HOST_MEMORY;
    return -1;
  }

  binary_file = fopen(binary_filename, "w+");
  if (binary_file == NULL)
  {
    *errcode = CL_OUT_OF_HOST_MEMORY;
    return -1;
  }

  n = fwrite(program->binaries[device_i], 1,
             program->binary_sizes[device_i], binary_file);
  fclose(binary_file);

  if (n < program->binary_sizes[device_i])
  {
    *errcode = CL_OUT_OF_HOST_MEMORY;
    return -1;
  }


  error |= snprintf(descriptor_filename, POCL_FILENAME_LENGTH,
                   "%s/%s/descriptor.so", device_tmpdir, kernel_name);

  error |= snprintf(command, COMMAND_LENGTH,
                   pocl_kernel_fmt,
                   kernel_name,
                   program->devices[device_i]->llvm_target_triplet,
                   descriptor_filename,
                   binary_filename);
  if (error < 0)
  {
    *errcode = CL_OUT_OF_HOST_MEMORY;
    return -1;
  }

  error = system(command);
  if (error != 0)
  {
    *errcode = CL_INVALID_KERNEL_NAME;
    return -1;
  }
  
  *errcode=CL_SUCCESS;
  return 0;
}

#endif

/* The WG generation does not yet work through the API. 
   Always call the script version for now. */

int call_pocl_workgroup( char* function_name, 
                    size_t local_x, size_t local_y, size_t local_z,
                    const char* llvm_target_triplet, 
                    const char* parallel_filename,
                    const char* kernel_filename )
{
  int error;
  char *pocl_wg_script;
  char command[COMMAND_LENGTH];

      if (getenv("POCL_BUILDING") != NULL)
        pocl_wg_script = BUILDDIR "/scripts/" POCL_WORKGROUP;
      else if (access(PKGDATADIR "/" POCL_WORKGROUP, X_OK) == 0)
        pocl_wg_script = PKGDATADIR "/" POCL_WORKGROUP;
      else
        pocl_wg_script = POCL_WORKGROUP;

      if (llvm_target_triplet != NULL) 
        {
          error = snprintf
            (command, COMMAND_LENGTH,
             "%s -k %s -x %zu -y %zu -z %zu -t %s -o %s %s",
             pocl_wg_script,
             function_name,
             local_x, local_y, local_z,
             llvm_target_triplet,
             parallel_filename, kernel_filename);
        }
      else
        {
          error = snprintf
            (command, COMMAND_LENGTH,
             "%s -k %s -x %zu -y %zu -z %zu -o %s %s",
             pocl_wg_script,
             function_name,
             local_x, local_y, local_z,
             parallel_filename, kernel_filename);
        }

      if (error < 0)
        return CL_OUT_OF_HOST_MEMORY;

      error = system (command);
      if (error != 0)
        return CL_OUT_OF_RESOURCES;

      return 0;
}
