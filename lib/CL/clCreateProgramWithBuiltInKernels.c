/* OpenCL runtime library: clCreateProgramWithBuiltInKernels()

   Copyright (c) 2017 Michal Babej / Tampere University
                 2019 Pekka Jääskeläinen / Tampere University

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
#include "pocl_shared.h"
#include "pocl_util.h"

#include <string.h>

#define MAX_KERNELS 1024

CL_API_ENTRY cl_program CL_API_CALL
POname (clCreateProgramWithBuiltInKernels) (cl_context context,
                                            cl_uint num_devices,
                                            const cl_device_id *device_list,
                                            const char *kernel_names,
                                            cl_int *errcode_ret)
CL_API_SUFFIX__VERSION_1_2
{
  cl_int errcode;
  char *kernel_names_copy = NULL;
  cl_program program = NULL;
  char **builtin_names = NULL;
  unsigned num_kernels = 0;
  unsigned i, j, supported_devices = 0;
  char *save_ptr;
  char *token;

  POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (context)), CL_INVALID_CONTEXT);

  POCL_GOTO_ERROR_COND ((device_list == NULL), CL_INVALID_VALUE);
  POCL_GOTO_ERROR_COND ((num_devices == 0), CL_INVALID_VALUE);

  POCL_GOTO_ERROR_COND ((kernel_names == NULL), CL_INVALID_VALUE);

  builtin_names = (char **)calloc (MAX_KERNELS, sizeof (char *));
  POCL_GOTO_ERROR_COND ((builtin_names == NULL), CL_OUT_OF_HOST_MEMORY);
  kernel_names_copy = strdup (kernel_names);
  token = strtok_r (kernel_names_copy, ";", &save_ptr);
  for (i = 0; ((i < MAX_KERNELS) && (token != NULL)); ++i)
    {
      builtin_names[i] = strdup (token);
      token = strtok_r (NULL, ";", &save_ptr);
      ++num_kernels;
    }
  POCL_MEM_FREE (kernel_names_copy);

  for (i = 0; i < num_devices; ++i)
    {
      unsigned num_supported_kernels = 0;
      cl_device_id dev = device_list[i];
      for (j = 0; j < num_kernels; ++j)
        {
          if (pocl_device_supports_builtin_kernel (dev, builtin_names[j]))
            ++num_supported_kernels;
        }
      if (num_supported_kernels == num_kernels)
        ++supported_devices;
    }

  POCL_GOTO_ERROR_ON ((supported_devices == 0), CL_INVALID_VALUE,
                      "None of the devices in context supports all "
                      "requested builtin kernels!\n");

  program = create_program_skeleton (context, num_devices, device_list, NULL,
                                     NULL, NULL, &errcode, 1);
  if (program == NULL)
    goto ERROR;

  program->num_builtin_kernels = num_kernels;
  program->builtin_kernel_names = builtin_names;
  program->concated_builtin_names = strdup (kernel_names);

  if (errcode_ret != NULL)
    *errcode_ret = CL_SUCCESS;

  return program;

ERROR:
  POCL_MEM_FREE (kernel_names_copy);
  POCL_MEM_FREE (builtin_names);
  if (errcode_ret)
    *errcode_ret = errcode;
  return NULL;
}
POsym (clCreateProgramWithBuiltInKernels)
