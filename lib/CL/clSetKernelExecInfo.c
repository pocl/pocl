/* OpenCL runtime library: clSetKernelExecInfo()

   Copyright (c) 2015 Michal Babej / Tampere University of Technology
                 2024 Pekka Jääskeläinen / Intel Finland Oy

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

#include "pocl_cl.h"
#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clSetKernelExecInfo)(cl_kernel kernel,
                            cl_kernel_exec_info param_name,
                            size_t param_value_size,
                            const void *param_value) CL_API_SUFFIX__VERSION_1_0
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (kernel)), CL_INVALID_KERNEL);

  switch (param_name)
    {
    case CL_KERNEL_EXEC_INFO_SVM_PTRS:
    case CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM:
      {
        cl_device_id dev = kernel->context->svm_allocdev;
        POCL_RETURN_ERROR_ON ((dev == NULL), CL_INVALID_OPERATION,
                              "no devices in the context associated with"
                              " kernel support SVM\n");
        cl_device_id realdev = pocl_real_dev (dev);
        cl_uint program_device_i = CL_UINT_MAX;
        for (unsigned i = 0; i < kernel->program->num_devices; ++i)
          {
            if (kernel->program->devices[i] == realdev)
              {
                program_device_i = i;
                break;
              }
          }
        POCL_RETURN_ERROR_ON ((program_device_i == CL_UINT_MAX),
                              CL_INVALID_KERNEL,
                              "Can't find the kernel for this device\n");
        POCL_RETURN_ERROR_ON (
            (realdev->ops->set_kernel_exec_info_ext == NULL),
            CL_INVALID_OPERATION,
            "This device doesn't support clSetKernelExecInfo\n");
        return realdev->ops->set_kernel_exec_info_ext (
            realdev, program_device_i, kernel, param_name, param_value_size,
            param_value);
      }

    case CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL:
    case CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL:
    case CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL:
    case CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL:
      {
        cl_device_id dev = kernel->context->usm_allocdev;
        POCL_RETURN_ERROR_ON ((dev == NULL), CL_INVALID_OPERATION,
                              "no devices in the context associated with"
                              " kernel support USM\n");
        /* Inform the device responsible for USM allocations. Note:
           the device might not have the kernel built, thus it might
           not be found in the kernel's program's devices since it's
           possible to build only for selected devices in the context. */
        POCL_RETURN_ERROR_ON (
            (dev->ops->set_kernel_exec_info_ext == NULL), CL_INVALID_OPERATION,
            "This USM allocator device doesn't support clSetKernelExecInfo\n");
        return dev->ops->set_kernel_exec_info_ext (
            dev, 0, kernel, param_name, param_value_size, param_value);
      }

    /* For the cl_ext_buffer_device_address extension. Just error checking
       until there are device drivers that need to actually migrate buffers
       because of this. */
    case CL_KERNEL_EXEC_INFO_DEVICE_PTRS_EXT:
      {
        void **ptrs = (void **)param_value;

        for (size_t i = 0; i < param_value_size / sizeof (void *); ++i)
          {
            void *dev_ptr = ptrs[i];
            pocl_raw_ptr *raw_ptr_info
                = pocl_find_raw_ptr_with_dev_ptr (kernel->context, dev_ptr);
            POCL_RETURN_ERROR_ON ((raw_ptr_info == NULL), CL_INVALID_VALUE,
                                  "the device pointer %p was not found\n",
                                  dev_ptr);
          }

        /* USM implements the device ptrs via USM device. We can
           delegate to USM in that case to avoid additional notifcation
           hooks for device ptrs. */
        cl_device_id usm_dev = kernel->context->usm_allocdev;
        if (usm_dev != NULL)
          {
            cl_device_id realdev = pocl_real_dev (usm_dev);
            cl_uint program_device_i = CL_UINT_MAX;
            for (unsigned i = 0; i < kernel->program->num_devices; ++i)
              {
                if (kernel->program->devices[i] == realdev)
                  {
                    program_device_i = i;
                    break;
                  }
              }
            if (program_device_i == CL_UINT_MAX)
              return CL_SUCCESS;
            POCL_RETURN_ERROR_ON (
                (realdev->ops->set_kernel_exec_info_ext == NULL),
                CL_INVALID_OPERATION,
                "This device doesn't support clSetKernelExecInfo\n");
            return realdev->ops->set_kernel_exec_info_ext (
                realdev, program_device_i, kernel,
                CL_KERNEL_EXEC_INFO_DEVICE_PTRS_EXT, param_value_size,
                param_value);
          }
        return CL_SUCCESS;
      }

    default:
      POCL_RETURN_ERROR (CL_INVALID_VALUE,
                         "Given param_name(%u) is not valid\n", param_name);
    }
}
POsym(clSetKernelExecInfo)
