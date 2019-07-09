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

CL_API_ENTRY cl_program CL_API_CALL
POname (clCreateProgramWithBuiltInKernels) (cl_context context,
                                            cl_uint num_devices,
                                            const cl_device_id *device_list,
                                            const char *kernel_names,
                                            cl_int *errcode_ret)
CL_API_SUFFIX__VERSION_1_2
{
  int errcode;
  POCL_GOTO_ERROR_COND ((context == NULL), CL_INVALID_CONTEXT);

  POCL_GOTO_ERROR_COND ((device_list == NULL), CL_INVALID_VALUE);
  POCL_GOTO_ERROR_COND ((num_devices == 0), CL_INVALID_VALUE);

  POCL_GOTO_ERROR_COND ((kernel_names == NULL), CL_INVALID_VALUE);

  // check for invalid devices in device_list[].
  for (int i = 0; i < num_devices; i++)
    {
      int found = 0;
      for (int j = 0; j < context->num_devices; j++)
        {
          found |= context->devices[j] == device_list[i];
        }
      POCL_GOTO_ERROR_ON (
          (!found), CL_INVALID_DEVICE,
          "device not found in the device list of the context\n");
    }

  char *save_ptr;
  char *token;
  char *kernel_names_copy = strdup (kernel_names);
  size_t num_kernels = 0;

  token = strtok_r (kernel_names_copy, ";", &save_ptr);
  while (token != NULL)
    {
      int num_supported = 0;
      for (int i = 0; i < num_devices; ++i)
        {
          cl_device_id dev = device_list[i];
          if (dev->ops->supports_builtin_kernel (dev->data, token))
            {
              num_supported++;
            }
        }
      POCL_GOTO_ERROR_COND ((num_supported == 0), CL_INVALID_VALUE);
      ++num_kernels;
      token = strtok_r (NULL, ";", &save_ptr);
    }
  free (kernel_names_copy);

  cl_program program = (cl_program)calloc (1, sizeof (struct _cl_program));
  if (program == NULL)
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR;
    }
  cl_device_id *devs = malloc (sizeof (cl_device_id) * num_devices);
  memcpy (devs, device_list, sizeof (cl_device_id) * num_devices);
  POCL_INIT_OBJECT (program);

  program->num_builtin_kernels = num_kernels;
  program->builtin_kernel_names = calloc (num_kernels, sizeof (char *));
  if (program->builtin_kernel_names == NULL)
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR;
    }

  kernel_names_copy = strdup (kernel_names);
  token = strtok_r (kernel_names_copy, ";", &save_ptr);
  for (int i = 0; token != NULL; ++i)
    {
      program->builtin_kernel_names[i] = strdup (token);
      token = strtok_r (NULL, ";", &save_ptr);
    }
  free (kernel_names_copy);

  program->num_devices = num_devices;
  program->devices = devs;
  program->context = context;
  program->build_status = CL_BUILD_NONE;

  if ((program->build_log = (char **)calloc (num_devices, sizeof (char *)))
      == NULL)
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      POCL_MEM_FREE (program);
      goto ERROR;
    }

  POCL_RETAIN_OBJECT (context);
  if (errcode_ret != NULL)
    *errcode_ret = CL_SUCCESS;

  return program;

ERROR:
  if (program->builtin_kernel_names)
    {
      for (int i = 0; i < num_kernels; ++i)
        POCL_MEM_FREE (program->builtin_kernel_names[i]);
    }
  if (errcode_ret)
    *errcode_ret = errcode;
  return NULL;
}
POsym (clCreateProgramWithBuiltInKernels)
