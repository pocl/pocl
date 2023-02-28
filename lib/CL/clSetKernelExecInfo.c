/* OpenCL runtime library: clSetKernelExecInfo()

   Copyright (c) 2015 Michal Babej / Tampere University of Technology

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
#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clSetKernelExecInfo)(cl_kernel  kernel ,
                            cl_kernel_exec_info  param_name ,
                            size_t  param_value_size ,
                            const void  *param_value) CL_API_SUFFIX__VERSION_1_0
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (kernel)), CL_INVALID_KERNEL);

  POCL_RETURN_ERROR_ON (
      (kernel->context->svm_allocdev == NULL
       && kernel->context->usm_allocdev == NULL),
      CL_INVALID_OPERATION,
      "None of the devices in this context is SVM or USM capable\n");

  switch (param_name)
    {
    case CL_KERNEL_EXEC_INFO_SVM_PTRS:
    case CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM:
      {
        cl_device_id dev = kernel->context->svm_allocdev;
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
    default:
      {
        cl_device_id dev = kernel->context->usm_allocdev;
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
    }
}
POsym(clSetKernelExecInfo)
