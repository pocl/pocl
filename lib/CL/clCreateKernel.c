/* OpenCL runtime library: clCreateKernel()

   Copyright (c) 2011 Universidad Rey Juan Carlos and
                 2012 Pekka Jääskeläinen / Tampere Univ. of Technology
   
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
#include "pocl_llvm.h"
#include "install-paths.h"
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

#define COMMAND_LENGTH 1024

CL_API_ENTRY cl_kernel CL_API_CALL
POname(clCreateKernel)(cl_program program,
               const char *kernel_name,
               cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  cl_kernel kernel;
  char device_tmpdir[POCL_FILENAME_LENGTH];
  char descriptor_filename[POCL_FILENAME_LENGTH];
  int errcode;
  int error;
  lt_dlhandle dlhandle = NULL;
  int i;
  int device_i;
  
  if (program == NULL || program->num_devices == 0)
  {
    errcode = CL_INVALID_PROGRAM;
    goto ERROR;
  }

  if (program->binaries == NULL || program->binary_sizes == NULL)
  {
    errcode = CL_INVALID_PROGRAM_EXECUTABLE;
    goto ERROR;
  }

  kernel = (cl_kernel) malloc(sizeof(struct _cl_kernel));
  if (kernel == NULL)
  {
    errcode = CL_OUT_OF_HOST_MEMORY;
    goto ERROR;
  }

  POCL_INIT_OBJECT (kernel);

  for (device_i = 0; device_i < program->num_devices; ++device_i)
    {
      if (device_i > 0)
        POname(clRetainKernel) (kernel);

      snprintf (device_tmpdir, POCL_FILENAME_LENGTH, "%s/%s", 
                program->temp_dir, program->devices[device_i]->short_name);

      /* If there is no device dir for this device, the program was
         not built for that device in clBuildProgram. This seems to
         be OK by the standard. */
      if (access (device_tmpdir, F_OK) != 0) continue;
 
      error = pocl_llvm_get_kernel_metadata 
          (program, kernel, device_i, kernel_name, 
           device_tmpdir, descriptor_filename, &errcode);

      if (error)
        {
          goto ERROR_CLEAN_KERNEL;
        } 

      /* when using the API, there is no descriptor file */
    }

  /* TODO: one of these two could be eliminated?  */
  kernel->function_name = strdup(kernel_name);
  kernel->name = strdup(kernel_name);

  kernel->context = program->context;
  kernel->program = program;
  kernel->next = NULL;

  cl_kernel k = program->kernels;
  program->kernels = kernel;
  kernel->next = k;

  POCL_RETAIN_OBJECT(program);

  if (errcode_ret != NULL)
    *errcode_ret = CL_SUCCESS;
  return kernel;

ERROR_CLEAN_KERNEL:
  free(kernel);
ERROR:
  if(errcode_ret != NULL)
  {
    *errcode_ret = errcode;
  }
  return NULL;
}
POsym(clCreateKernel)
