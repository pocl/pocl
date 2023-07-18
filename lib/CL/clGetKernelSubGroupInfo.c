/* OpenCL runtime library: clGetKernelSubGroupInfo()

   Copyright (c) 2021 Väinö Liukko
                 2022 Pekka Jääskeläinen / Intel Finland Oy

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

#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_ENTRY POname (clGetKernelSubGroupInfo) (
    cl_kernel kernel, cl_device_id device, cl_kernel_sub_group_info param_name,
    size_t input_value_size, const void *input_value, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_2_1
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (kernel)), CL_INVALID_KERNEL);

  /* Check that kernel is associated with device, or that there is no
   risk of confusion. */
  if (device != NULL)
    {
      POCL_RETURN_ERROR_ON (
          !(pocl_device_is_associated_with_kernel (device, kernel)),
          CL_INVALID_DEVICE,
          "could not find the device supplied in argument\n");
    }
  else
    {
      POCL_RETURN_ERROR_ON ((kernel->context->num_devices > 1),
                            CL_INVALID_DEVICE,
                            "No device given and context has > 1 device\n");
      device = kernel->context->devices[0];
    }

  /* Check device for subgroup support */
  POCL_RETURN_ERROR_ON ((device->max_num_sub_groups == 0),
                        CL_INVALID_OPERATION,
                        "device does not support any subgroup sizes\n");

  // find device index
  cl_uint dev_i = CL_UINT_MAX;
  cl_device_id realdev = pocl_real_dev (device);
  for (unsigned i = 0; i < kernel->program->num_devices; ++i)
    {
      if (kernel->program->devices[i] == realdev)
        dev_i = i;
    }
  POCL_RETURN_ERROR_ON ((dev_i == CL_UINT_MAX), CL_INVALID_KERNEL,
                        "the kernel was not built for this device\n");

  if (param_name == CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE)
    {
      POCL_RETURN_ERROR_ON ((input_value == NULL
                             || input_value_size < sizeof (size_t)
                             || input_value_size > sizeof (size_t) * 3),
                            CL_INVALID_VALUE, "NDRange not given.");
    }

  POCL_RETURN_ERROR_ON (
      (strstr (realdev->extensions, "cl_khr_subgroup") == NULL),
      CL_INVALID_OPERATION,
      "device does not support any subgroup extensions\n");

  /************************************************************************/

  switch (param_name)
    {
    /* TODO: this should be a device ops callback */
    case CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE:
      {
        /* For now assume SG == WG_x. */
        POCL_RETURN_GETINFO (size_t, ((size_t *)input_value)[0]);
      }
    case CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE:
      {
        /* For now assume SG == WG_x and thus we have WG_size_y*WG_size_z of
           them per WG. */
        POCL_RETURN_GETINFO (size_t,
                             min (device->max_num_sub_groups,
                                  (input_value_size > sizeof (size_t)
                                       ? ((size_t *)input_value)[1]
                                       : 1)
                                      * (input_value_size > sizeof (size_t) * 2
                                             ? ((size_t *)input_value)[2]
                                             : 1)));
      }
    case CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT:
      {
        POCL_RETURN_ERROR_ON ((input_value == NULL), CL_INVALID_VALUE,
                              "SG size wish not given.");
        size_t n_wish = *(size_t *)input_value;
        /* For now assume SG == WG_x and the simplest way of looping only at
           y dimension. Use magic number 32 as the preferred SG size for now.
         */
        size_t nd[3];
        if (n_wish > device->max_num_sub_groups ||
            (n_wish > 1 && param_value_size / sizeof(size_t) == 1))
          {
            nd[0] = nd[1] = nd[2] = 0;
            POCL_RETURN_GETINFO_ARRAY (size_t,
                                       param_value_size / sizeof(size_t),
                                       nd);
          }
        else
          {
            nd[0] = device->max_work_group_size / n_wish;
            nd[1] = n_wish;
            nd[2] = 1;
            POCL_RETURN_GETINFO_ARRAY (size_t,
                                       param_value_size / sizeof(size_t),
                                       nd);
          }
      }

    /************ these are NOT dependent on NDRANGE ***********************/
    case CL_KERNEL_MAX_NUM_SUB_GROUPS:
      if (kernel->meta->max_subgroups)
        POCL_RETURN_GETINFO (size_t, kernel->meta->max_subgroups[dev_i]);
      else
        POCL_RETURN_GETINFO (size_t, 0);

    case CL_KERNEL_COMPILE_NUM_SUB_GROUPS:
      if (kernel->meta->compile_subgroups)
        POCL_RETURN_GETINFO (size_t, kernel->meta->compile_subgroups[dev_i]);
      else
        POCL_RETURN_GETINFO (size_t, 0);

    default:
      POCL_MSG_ERR ("clGetKernelSubGroupInfo for param_name value %u "
                    "is not implemented\n",
                    param_name);
      return CL_INVALID_VALUE;
    }
}
POsym(clGetKernelSubGroupInfo)
