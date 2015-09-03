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
#include "pocl_file_util.h"
#include "pocl_cache.h"
#include "pocl_llvm.h"
#include <string.h>
#include <sys/stat.h>
#ifndef _MSC_VER
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#define COMMAND_LENGTH 1024

CL_API_ENTRY cl_kernel CL_API_CALL
POname(clCreateKernel)(cl_program program,
               const char *kernel_name,
               cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  cl_kernel kernel = NULL;
  int errcode;
  int error;
  unsigned device_i;
  void* cache_lock = NULL;

  POCL_GOTO_ERROR_COND((kernel_name == NULL), CL_INVALID_VALUE);

  POCL_GOTO_ERROR_COND((program == NULL), CL_INVALID_PROGRAM);
  
  POCL_GOTO_ERROR_ON((program->num_devices == 0),
    CL_INVALID_PROGRAM, "Invalid program (has no devices assigned)\n");

  POCL_GOTO_ERROR_ON((program->build_status == CL_BUILD_NONE),
    CL_INVALID_PROGRAM_EXECUTABLE, "You must call clBuildProgram first!"
      " (even for programs created with binaries)\n");

  POCL_GOTO_ERROR_ON((program->build_status != CL_BUILD_SUCCESS),
    CL_INVALID_PROGRAM_EXECUTABLE, "Last BuildProgram() was not successful\n")

  POCL_GOTO_ERROR_ON((program->llvm_irs == NULL),
    CL_INVALID_PROGRAM_EXECUTABLE, "No built binaries in program "
    "(this shouldn't happen...)\n");

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

      /* If there is no device dir for this device, the program was
         not built for that device in clBuildProgram. This seems to
         be OK by the standard. */
      if (!pocl_cache_device_cachedir_exists(program, device_i))
          continue;

      cache_lock = pocl_cache_acquire_writer_lock_i(program, device_i);
      assert(cache_lock);

      error = pocl_llvm_get_kernel_metadata(program,
                      kernel, device_i, kernel_name, &errcode);

      pocl_cache_release_lock(cache_lock);

      if (error)
        {
          POCL_MSG_ERR("Failed to get kernel metadata "
            "for kernel %s on device %s\n", kernel_name,
              program->devices[device_i]->short_name);
          goto ERROR;
        } 

    }

  kernel->name = strdup(kernel_name);
  POCL_GOTO_ERROR_ON((kernel->name == NULL), CL_OUT_OF_HOST_MEMORY,
                     "clCreateKernel couldn't allocate memory");

  kernel->context = program->context;
  kernel->program = program;
  kernel->next = NULL;

  POCL_LOCK_OBJ (program);
  cl_kernel k = program->kernels;
  program->kernels = kernel;
  POCL_UNLOCK_OBJ (program);
  kernel->next = k;

  POCL_RETAIN_OBJECT(program);

  errcode = CL_SUCCESS;
  goto SUCCESS;

ERROR:
  POCL_MEM_FREE(kernel);
  kernel = NULL;

SUCCESS:
  if(errcode_ret != NULL)
  {
    *errcode_ret = errcode;
  }
  return kernel;
}
POsym(clCreateKernel)
