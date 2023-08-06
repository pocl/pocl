/* OpenCL runtime library: clGetKernelWorkGroupInfo()

   Copyright (c) 2011 Universidad Rey Juan Carlos and
                 2012-2018 Pekka Jääskeläinen / Tampere University of Technology

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

#include "devices/devices.h"
#include "pocl_util.h"

extern CL_API_ENTRY cl_int CL_API_CALL
POname(clGetKernelWorkGroupInfo)
(cl_kernel kernel,
 cl_device_id device,
 cl_kernel_work_group_info param_name,
 size_t param_value_size,
 void *param_value,
 size_t * param_value_size_ret)
  CL_API_SUFFIX__VERSION_1_0
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
      POCL_RETURN_ERROR_ON((kernel->context->num_devices > 1), CL_INVALID_DEVICE,
        "No device given and context has > 1 device\n");
      device = kernel->context->devices[0];
    }

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

  POCL_RETURN_ERROR_COND ((*(device->available) == CL_FALSE),
                          CL_DEVICE_NOT_AVAILABLE);
  /************************************************************************/

  switch (param_name)
    {
    case CL_KERNEL_GLOBAL_WORK_SIZE:
      {
        /* this parameter is only for custom devices & builtin kernels. */
        POCL_RETURN_ERROR_ON (
            (!kernel->meta->builtin_kernel
             && (device->type != CL_DEVICE_TYPE_CUSTOM)),
            CL_INVALID_VALUE,
            "only valid for custom devices or builtin kernels\n");
        POCL_RETURN_GETINFO (size_t_3, kernel->meta->builtin_max_global_work);
      }
    case CL_KERNEL_WORK_GROUP_SIZE:
      {
        if (kernel->meta->max_workgroup_size
            && kernel->meta->max_workgroup_size[dev_i])
          POCL_RETURN_GETINFO (size_t,
                               kernel->meta->max_workgroup_size[dev_i]);
        else // fallback to device's CL_DEVICE_MAX_WORK_GROUP_SIZE
          return POname (clGetDeviceInfo) (
              device, CL_DEVICE_MAX_WORK_GROUP_SIZE, param_value_size,
              param_value, param_value_size_ret);
      }
    case CL_KERNEL_COMPILE_WORK_GROUP_SIZE:
    {
        POCL_MSG_PRINT_GENERAL (
            "### reqd wg sizes %zu %zu %zu\n", kernel->meta->reqd_wg_size[0],
            kernel->meta->reqd_wg_size[1], kernel->meta->reqd_wg_size[2]);
        POCL_RETURN_GETINFO (size_t_3,
                             *(size_t_3 *)kernel->meta->reqd_wg_size);
    }
    case CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE:
      {
        if (kernel->meta->preferred_wg_multiple)
          POCL_RETURN_GETINFO (size_t,
                               kernel->meta->preferred_wg_multiple[dev_i]);
        else // fallback to device's preferred WG size multiple
          POCL_RETURN_GETINFO (size_t, device->preferred_wg_size_multiple);
      }
    case CL_KERNEL_LOCAL_MEM_SIZE:
    {
      if (kernel->meta->local_mem_size)
        POCL_RETURN_GETINFO (size_t, kernel->meta->local_mem_size[dev_i]);
      else
        POCL_RETURN_GETINFO (cl_ulong, 0);
    }
    case CL_KERNEL_PRIVATE_MEM_SIZE:
      {
        if (kernel->meta->private_mem_size)
          POCL_RETURN_GETINFO (size_t, kernel->meta->private_mem_size[dev_i]);
        else
          POCL_RETURN_GETINFO (cl_ulong, 0);
      }
    case CL_KERNEL_SPILL_MEM_SIZE_INTEL:
      {
        if (kernel->meta->spill_mem_size)
          POCL_RETURN_GETINFO (size_t, kernel->meta->spill_mem_size[dev_i]);
        else
          POCL_RETURN_GETINFO (cl_ulong, 0);
      }
    default:
      return CL_INVALID_VALUE;
    }
  return CL_SUCCESS;
}
POsym(clGetKernelWorkGroupInfo)
