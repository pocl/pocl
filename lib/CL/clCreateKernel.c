/* OpenCL runtime library: clCreateKernel()

   Copyright (c) 2011 Universidad Rey Juan Carlos and
                 2012-2019 Pekka Jääskeläinen

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
#ifdef OCS_AVAILABLE
#include "pocl_llvm.h"
#endif
#include "pocl_binary.h"
#include "pocl_util.h"
#include <string.h>
#include <sys/stat.h>
#ifndef _MSC_VER
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

CL_API_ENTRY cl_kernel CL_API_CALL
POname(clCreateKernel)(cl_program program,
               const char *kernel_name,
               cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  cl_kernel kernel = NULL;
  int errcode = CL_SUCCESS;
  size_t i;

  POCL_GOTO_ERROR_COND((kernel_name == NULL), CL_INVALID_VALUE);

  POCL_GOTO_ERROR_COND((program == NULL), CL_INVALID_PROGRAM);

  POCL_GOTO_ERROR_ON((program->num_devices == 0),
    CL_INVALID_PROGRAM, "Invalid program (has no devices assigned)\n");

  POCL_GOTO_ERROR_ON((program->build_status == CL_BUILD_NONE),
    CL_INVALID_PROGRAM_EXECUTABLE, "You must call clBuildProgram first!"
      " (even for programs created with binaries)\n");

  POCL_GOTO_ERROR_ON((program->build_status != CL_BUILD_SUCCESS),
    CL_INVALID_PROGRAM_EXECUTABLE, "Last BuildProgram() was not successful\n");

  POCL_GOTO_ERROR_ON (
      (program->builtin_kernel_names == NULL && program->llvm_irs == NULL),
      CL_INVALID_PROGRAM_EXECUTABLE,
      "No built binaries nor built-in kernels in program "
      "(this shouldn't happen...)\n");

  kernel = (cl_kernel) calloc(1, sizeof(struct _cl_kernel));
  POCL_GOTO_ERROR_ON((kernel == NULL), CL_OUT_OF_HOST_MEMORY,
                     "clCreateKernel couldn't allocate memory");

  POCL_INIT_OBJECT (kernel);

  for (i = 0; i < program->num_kernels; ++i)
    if (strcmp (program->kernel_meta[i].name, kernel_name) == 0)
      break;

  POCL_GOTO_ERROR_ON ((i >= program->num_kernels), CL_INVALID_KERNEL_NAME,
                      "Can't find a kernel with name %s in this program\n",
                      kernel_name);

  kernel->meta = &program->kernel_meta[i];
  kernel->name = kernel->meta->name;
  kernel->context = program->context;
  kernel->program = program;

  kernel->dyn_arguments
      = calloc ((kernel->meta->num_args), sizeof (struct pocl_argument));
  POCL_GOTO_ERROR_COND ((kernel->dyn_arguments == NULL),
                        CL_OUT_OF_HOST_MEMORY);

  POCL_LOCK_OBJ (program);
  cl_kernel k = program->kernels;
  program->kernels = kernel;
  kernel->next = k;
  POCL_RETAIN_OBJECT_UNLOCKED (program);
  POCL_UNLOCK_OBJ (program);

  errcode = CL_SUCCESS;
  goto SUCCESS;

ERROR:
  if (kernel)
    POCL_MEM_FREE (kernel->dyn_arguments);
  POCL_MEM_FREE (kernel);
  kernel = NULL;

SUCCESS:
  if(errcode_ret != NULL)
  {
    *errcode_ret = errcode;
  }
  return kernel;
}
POsym(clCreateKernel)
