/* OpenCL runtime library: clCreateProgramWithDefinedBuiltInKernels()

   Copyright (c) 2024 Michal Babej / Intel Finland Oy

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include "pocl_builtin_kernels.h"
#include "pocl_cl.h"
#include "pocl_shared.h"
#include "pocl_util.h"

#include <string.h>

/*
 * num_kernel - number of kernels (K)
 * kernel_ids - array of K enum values, must be valid cl_dbk_id_exp
 * kernel_names - array of K strings, these are user-chosen kernel names for
 * each kernel, must be unique within program
 * kernel_attributes - array of K structs that contain attrs
 * specific to each kernel
 */

CL_API_ENTRY cl_program CL_API_CALL
POname (clCreateProgramWithDefinedBuiltInKernelsEXP) (
  cl_context context,
  cl_uint num_devices,
  const cl_device_id *device_list,
  cl_uint num_kernels,
  const cl_dbk_id_exp *kernel_ids,
  const char **kernel_names,
  const void **kernel_attributes,
  cl_int *device_support,
  cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_2
{
  cl_int errcode;
  cl_program program = NULL;
  unsigned i, j, supported_devices = 0;
  char **builtin_names = NULL;
  size_t concated_kernel_names_size = 0;
  char *concated_kernel_names = NULL;
  cl_dbk_id_exp *builtin_kernel_ids = NULL;
  void **builtin_kernel_attrs = NULL;

  POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (context)), CL_INVALID_CONTEXT);

  POCL_GOTO_ERROR_COND ((device_list == NULL), CL_INVALID_VALUE);
  POCL_GOTO_ERROR_COND ((num_devices == 0), CL_INVALID_VALUE);

  POCL_GOTO_ERROR_COND ((kernel_names == NULL), CL_INVALID_VALUE);
  POCL_GOTO_ERROR_COND ((num_kernels == 0), CL_INVALID_VALUE);

  builtin_names = (char **)calloc (num_kernels, sizeof (char *));
  POCL_GOTO_ERROR_COND ((builtin_names == NULL), CL_OUT_OF_HOST_MEMORY);

  for (i = 0; i < num_kernels; ++i)
    {
      POCL_GOTO_ERROR_COND ((kernel_names[i] == NULL), CL_INVALID_VALUE);
      concated_kernel_names_size += strlen (kernel_names[i]) + 1;
      for (j = 0; j < i; ++j)
        {
          POCL_GOTO_ERROR_ON ((strcmp (kernel_names[j], kernel_names[i]) == 0),
                              CL_INVALID_VALUE,
                              "Kernel names at index"
                              "%u and %u are identical (%s)\n",
                              i, j, kernel_names[i]);
        }
    }
  concated_kernel_names = (char *)malloc (concated_kernel_names_size);
  POCL_GOTO_ERROR_COND ((concated_kernel_names == NULL),
                        CL_OUT_OF_HOST_MEMORY);

  builtin_kernel_ids
    = (cl_dbk_id_exp *)calloc (num_kernels, sizeof (cl_dbk_id_exp));
  POCL_GOTO_ERROR_COND ((builtin_kernel_ids == NULL), CL_OUT_OF_HOST_MEMORY);

  builtin_kernel_attrs = (void **)calloc (num_kernels, sizeof (void *));
  POCL_GOTO_ERROR_COND ((builtin_kernel_attrs == NULL), CL_OUT_OF_HOST_MEMORY);

  for (i = 0; i < num_kernels; ++i)
    {
      POCL_GOTO_ERROR_COND ((kernel_ids[i] >= POCL_CDBI_LAST),
                            CL_INVALID_VALUE);
      errcode = pocl_validate_dbk_attributes (kernel_ids[i],
                                              kernel_attributes[i], NULL);
      POCL_GOTO_ERROR_ON ((errcode != CL_SUCCESS), CL_INVALID_ARG_VALUE,
                          "DefinedBuiltinKernel attributes for kernel %u "
                          "are invalid",
                          i);
    }

  for (i = 0; i < num_devices; ++i)
    {
      unsigned num_supported_kernels = 0;
      cl_device_id dev = device_list[i];
      if (dev->ops->supports_dbk == NULL)
        {
          if (device_support)
            device_support[i] = CL_DBK_UNSUPPORTED_EXP;
          continue;
        }
      for (j = 0; j < num_kernels; ++j)
        {
          errcode = dev->ops->supports_dbk (dev, kernel_ids[j],
                                            kernel_attributes[j]);
          if (device_support)
            device_support[i] = errcode;
          if (errcode == CL_SUCCESS)
            ++num_supported_kernels;
          else
            break;
        }
      if (num_supported_kernels == num_kernels)
        ++supported_devices;
    }

  POCL_GOTO_ERROR_ON ((supported_devices == 0), CL_INVALID_VALUE,
                      "None of the devices in context supports all of the "
                      "requested builtin kernels!\n");

  char *concat_ptr = concated_kernel_names;
  /* create a copy of the arguments */
  for (i = 0; i < num_kernels; ++i)
    {
      if (i > 0)
        strcpy (concat_ptr++, ";");

      strcpy (concat_ptr, kernel_names[i]);
      concat_ptr += strlen (kernel_names[i]);

      builtin_names[i] = strdup (kernel_names[i]);
      POCL_GOTO_ERROR_COND ((builtin_names[i] == NULL), CL_OUT_OF_HOST_MEMORY);

      builtin_kernel_ids[i] = kernel_ids[i];

      if (kernel_attributes[i])
        {
          void *attrs = pocl_copy_defined_builtin_attributes (
            kernel_ids[i], kernel_attributes[i]);
          POCL_GOTO_ERROR_COND ((attrs == NULL), CL_OUT_OF_HOST_MEMORY);
          builtin_kernel_attrs[i] = attrs;
        }
    }

  program = create_program_skeleton (context, num_devices, device_list, NULL,
                                     NULL, NULL, &errcode, 1);
  if (program == NULL)
    goto ERROR;

  program->num_builtin_kernels = num_kernels;
  program->builtin_kernel_names = builtin_names;
  program->concated_builtin_names = concated_kernel_names;
  program->builtin_kernel_ids = builtin_kernel_ids;
  program->builtin_kernel_attributes = builtin_kernel_attrs;

  if (errcode_ret != NULL)
    *errcode_ret = CL_SUCCESS;

  return program;

ERROR:
  POCL_MEM_FREE (concated_kernel_names);
  POCL_MEM_FREE (builtin_kernel_ids);
  if (builtin_names)
    {
      for (i = 0; i < num_kernels; ++i)
        POCL_MEM_FREE (builtin_names[i]);
      POCL_MEM_FREE (builtin_names);
    }
  if (builtin_kernel_attrs)
    {
      for (i = 0; i < num_kernels; ++i)
        {
          if (builtin_kernel_attrs[i])
            pocl_release_defined_builtin_attributes (kernel_ids[i],
                                                     builtin_kernel_attrs[i]);
          builtin_kernel_attrs[i] = NULL;
        }
      POCL_MEM_FREE (builtin_kernel_attrs);
    }
  if (errcode_ret)
    *errcode_ret = errcode;
  return NULL;
}
POsym (clCreateProgramWithDefinedBuiltInKernelsEXP)
