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
      POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (device)),
                              CL_INVALID_DEVICE);

      POCL_RETURN_ERROR_ON (
        !(pocl_device_is_associated_with_kernel (device, kernel)),
        CL_INVALID_DEVICE, "The kernel is not associated with the device\n");
    }
  else
    {
      POCL_RETURN_ERROR_ON ((kernel->context->num_devices > 1),
                            CL_INVALID_DEVICE,
                            "No device given and context has > 1 device\n");
      device = kernel->context->devices[0];
    }

  cl_device_id realdev = pocl_real_dev (device);
  /* Check device for subgroup support */
  POCL_RETURN_ERROR_ON (
    (strstr (realdev->extensions, "cl_khr_subgroup") == NULL),
    CL_INVALID_OPERATION, "device does not support any subgroup extensions\n");
  POCL_RETURN_ERROR_ON ((realdev->max_num_sub_groups == 0),
                        CL_INVALID_OPERATION,
                        "device does not support any subgroup sizes\n");

  // find device index
  cl_uint dev_i = CL_UINT_MAX;
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

  /************************************************************************/

  switch (param_name)
    {
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
      POCL_RETURN_ERROR_ON ((realdev->ops->get_subgroup_info_ext == NULL),
                            CL_INVALID_VALUE,
                            "clGetKernelSubGroupInfo for param_name value %u "
                            "is not implemented\n",
                            (unsigned)param_name);
      return realdev->ops->get_subgroup_info_ext (
        realdev, kernel, dev_i, param_name, input_value_size, input_value,
        param_value_size, param_value, param_value_size_ret);
    }
}
POsym(clGetKernelSubGroupInfo)
