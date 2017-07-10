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

#define COMMAND_LENGTH 1024

CL_API_ENTRY cl_kernel CL_API_CALL
POname(clCreateKernel)(cl_program program,
               const char *kernel_name,
               cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  cl_kernel kernel = NULL;
  int errcode = CL_SUCCESS;
  unsigned device_i, i;

  POCL_GOTO_ERROR_COND((kernel_name == NULL), CL_INVALID_VALUE);

  POCL_GOTO_ERROR_COND((program == NULL), CL_INVALID_PROGRAM);

  POCL_GOTO_ERROR_ON((program->num_devices == 0),
    CL_INVALID_PROGRAM, "Invalid program (has no devices assigned)\n");

  POCL_GOTO_ERROR_ON((program->build_status == CL_BUILD_NONE),
    CL_INVALID_PROGRAM_EXECUTABLE, "You must call clBuildProgram first!"
      " (even for programs created with binaries)\n");

  POCL_GOTO_ERROR_ON((program->build_status != CL_BUILD_SUCCESS),
    CL_INVALID_PROGRAM_EXECUTABLE, "Last BuildProgram() was not successful\n");

  POCL_GOTO_ERROR_ON((program->llvm_irs == NULL),
    CL_INVALID_PROGRAM_EXECUTABLE, "No built binaries in program "
    "(this shouldn't happen...)\n");

  kernel = (cl_kernel) calloc(1, sizeof(struct _cl_kernel));
  POCL_GOTO_ERROR_ON((kernel == NULL), CL_OUT_OF_HOST_MEMORY,
                     "clCreateKernel couldn't allocate memory");

  POCL_INIT_OBJECT (kernel);

  kernel->name = strdup(kernel_name);
  POCL_GOTO_ERROR_ON((kernel->name == NULL), CL_OUT_OF_HOST_MEMORY,
                     "clCreateKernel couldn't allocate memory");

  kernel->context = program->context;
  kernel->program = program;
  kernel->next = NULL;

  for (device_i = 0; device_i < program->num_devices; ++device_i)
    {
      if (program->binaries[device_i] &&
          pocl_cache_device_cachedir_exists(program, device_i))
        {
#ifdef OCS_AVAILABLE
          pocl_llvm_get_kernel_metadata (program, kernel, device_i,
                                         kernel_name, &errcode);
          cl_device_id device = program->devices[device_i];
          if (device->spmd)
            {
              char cachedir[POCL_FILENAME_LENGTH];
              _cl_command_node cmd;
              memset (&cmd, 0, sizeof(_cl_command_node));
              cmd.type = CL_COMMAND_NDRANGE_KERNEL;
              cmd.command.run.tmp_dir = cachedir;
              cmd.command.run.kernel = kernel;
              cmd.device = device;
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
              pocl_cache_kernel_cachedir_path (cachedir, program, device_i,
                                               kernel, "", local_x,
                                               local_y, local_z);

              device->ops->compile_kernel (&cmd, kernel, device);
            }
#endif
        }
      /* If the program was created with a pocl binary, we won't be able to
         get the metadata for the cl_kernel from an IR file, so we call pocl
         binary function to initialize the cl_kernel data */
      else if (program->pocl_binaries[device_i])
        {
          errcode
            = pocl_binary_get_kernel_metadata (program->pocl_binaries[device_i],
                                               kernel_name, kernel,
                                               program->devices[device_i]);
        }
      else
        /* If there is no device dir for this device, the program was
           not built for that device in clBuildProgram. This seems to
           be OK by the standard. */
        continue;

      if (errcode != CL_SUCCESS)
        {
          POCL_MSG_ERR( "Failed to get kernel metadata "
                        "for kernel %s on device %s\n", kernel_name,
                        program->devices[device_i]->short_name);
          goto ERROR;
        }
    }

  /* default kernels don't go on the program-kernels linked list,
   * and they don't increase the program refcount. */
  if (!program->operating_on_default_kernels)
    {
      POCL_LOCK_OBJ (program);
      cl_kernel k = program->kernels;
      program->kernels = kernel;
      POCL_UNLOCK_OBJ (program);
      kernel->next = k;
      POCL_RETAIN_OBJECT (program);
    }

  errcode = CL_SUCCESS;
  goto SUCCESS;

ERROR:
  if (kernel)
    {
      if (kernel->arg_info)
        for (i = 0; i < kernel->num_args; i++)
          {
            POCL_MEM_FREE (kernel->arg_info[i].name);
            POCL_MEM_FREE (kernel->arg_info[i].type_name);
          }

      if (kernel->dyn_arguments)
        for (i = 0; i < (kernel->num_args + kernel->num_locals); i++)
          {
            pocl_aligned_free (kernel->dyn_arguments[i].value);
          }
      POCL_MEM_FREE(kernel->reqd_wg_size);
      POCL_MEM_FREE(kernel->dyn_arguments);
      POCL_MEM_FREE(kernel->arg_info);
      POCL_MEM_FREE(kernel);
    }
  kernel = NULL;

SUCCESS:
  if(errcode_ret != NULL)
  {
    *errcode_ret = errcode;
  }
  return kernel;
}
POsym(clCreateKernel)
