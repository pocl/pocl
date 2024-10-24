/* OpenCL runtime library: clGetExtensionFunctionAddressForPlatform()

   Copyright (c) 2017 Michal Babej / Tampere University of Technology
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

#include <string.h>

CL_API_ENTRY void * CL_API_CALL
POname (clGetExtensionFunctionAddressForPlatform) (cl_platform_id  platform,
                                                   const char *func_name)
CL_API_SUFFIX__VERSION_1_2
{
  cl_platform_id pocl_platform;
  cl_uint actual_num = 0;
  POname (clGetPlatformIDs) (1, &pocl_platform, &actual_num);
  if (actual_num != 1)
    {
      POCL_MSG_WARN ("Couldn't get the platform ID of PoCL platform\n");
      return NULL;
    }

  assert (pocl_platform);
  if (platform != pocl_platform)
    {
      POCL_MSG_WARN ("Requested Function Address not "
                     "for PoCL platform, ignoring\n");
      return NULL;
    }

#ifdef BUILD_ICD
  if (strcmp (func_name, "clIcdGetPlatformIDsKHR") == 0)
    return (void *)&POname(clIcdGetPlatformIDsKHR);
#endif

#ifdef BUILD_PROXY
  if (strcmp (func_name, "clGetGLContextInfoKHR") == 0)
    return (void *)&POname (clGetGLContextInfoKHR);
#endif

  if (strcmp (func_name, "clSetContentSizeBufferPoCL") == 0)
    return (void *)&POname (clSetContentSizeBufferPoCL);

  if (strcmp (func_name, "clGetPlatformInfo") == 0)
    return (void *)&POname(clGetPlatformInfo);

  if (strcmp (func_name, "clCreateProgramWithILKHR") == 0)
    return (void *)&POname(clCreateProgramWithIL);

  /* cl_khr_command_buffer */
  if (strcmp (func_name, "clCreateCommandBufferKHR") == 0)
    return (void *)&POname (clCreateCommandBufferKHR);

  if (strcmp (func_name, "clRetainCommandBufferKHR") == 0)
    return (void *)&POname (clRetainCommandBufferKHR);

  if (strcmp (func_name, "clReleaseCommandBufferKHR") == 0)
    return (void *)&POname (clReleaseCommandBufferKHR);

  if (strcmp (func_name, "clFinalizeCommandBufferKHR") == 0)
    return (void *)&POname (clFinalizeCommandBufferKHR);

  if (strcmp (func_name, "clEnqueueCommandBufferKHR") == 0)
    return (void *)&POname (clEnqueueCommandBufferKHR);

  if (strcmp (func_name, "clCommandBarrierWithWaitListKHR") == 0)
    return (void *)&POname (clCommandBarrierWithWaitListKHR);

  if (strcmp (func_name, "clCommandCopyBufferKHR") == 0)
    return (void *)&POname (clCommandCopyBufferKHR);

  if (strcmp (func_name, "clCommandCopyBufferRectKHR") == 0)
    return (void *)&POname (clCommandCopyBufferRectKHR);

  if (strcmp (func_name, "clCommandCopyBufferToImageKHR") == 0)
    return (void *)&POname (clCommandCopyBufferToImageKHR);

  if (strcmp (func_name, "clCommandCopyImageKHR") == 0)
    return (void *)&POname (clCommandCopyImageKHR);

  if (strcmp (func_name, "clCommandCopyImageToBufferKHR") == 0)
    return (void *)&POname (clCommandCopyImageToBufferKHR);

  if (strcmp (func_name, "clCommandFillBufferKHR") == 0)
    return (void *)&POname (clCommandFillBufferKHR);

  if (strcmp (func_name, "clCommandFillImageKHR") == 0)
    return (void *)&POname (clCommandFillImageKHR);

  if (strcmp (func_name, "clCommandNDRangeKernelKHR") == 0)
    return (void *)&POname (clCommandNDRangeKernelKHR);

  if (strcmp (func_name, "clGetCommandBufferInfoKHR") == 0)
    return (void *)&POname (clGetCommandBufferInfoKHR);
  /* end of cl_khr_command_buffer */

  /* cl_khr_command_buffer_multi_device */
  if (strcmp (func_name, "clRemapCommandBufferKHR") == 0)
    return (void *)&POname (clRemapCommandBufferKHR);
  /* end of cl_khr_command_buffer_multi_device */

  /* cl_intel_unified_shared_memory */
  if (strcmp (func_name, "clHostMemAllocINTEL") == 0)
    return (void *)&POname (clHostMemAllocINTEL);

  if (strcmp (func_name, "clDeviceMemAllocINTEL") == 0)
    return (void *)&POname (clDeviceMemAllocINTEL);

  if (strcmp (func_name, "clSharedMemAllocINTEL") == 0)
    return (void *)&POname (clSharedMemAllocINTEL);

  if (strcmp (func_name, "clMemFreeINTEL") == 0)
    return (void *)&POname (clMemFreeINTEL);

  if (strcmp (func_name, "clMemBlockingFreeINTEL") == 0)
    return (void *)&POname (clMemBlockingFreeINTEL);

  if (strcmp (func_name, "clGetMemAllocInfoINTEL") == 0)
    return (void *)&POname (clGetMemAllocInfoINTEL);

  if (strcmp (func_name, "clSetKernelArgMemPointerINTEL") == 0)
    return (void *)&POname (clSetKernelArgMemPointerINTEL);

  if (strcmp (func_name, "clEnqueueMemFillINTEL") == 0)
    return (void *)&POname (clEnqueueMemFillINTEL);

  if (strcmp (func_name, "clEnqueueMemcpyINTEL") == 0)
    return (void *)&POname (clEnqueueMemcpyINTEL);

  if (strcmp (func_name, "clEnqueueMigrateMemINTEL") == 0)
    return (void *)&POname (clEnqueueMigrateMemINTEL);

  if (strcmp (func_name, "clEnqueueMemAdviseINTEL") == 0)
    return (void *)&POname (clEnqueueMemAdviseINTEL);
  /* end of cl_intel_unified_shared_memory */

  if (strcmp (func_name, "clCommandSVMMemFillKHR") == 0)
    return (void *)&POname (clCommandSVMMemFillKHR);

  if (strcmp (func_name, "clCommandSVMMemcpyKHR") == 0)
    return (void *)&POname (clCommandSVMMemcpyKHR);

  /* cl_pocl_command_buffer_svm */
  if (strcmp (func_name, "clCommandSVMMemcpyPOCL") == 0)
    return (void *)&POname (clCommandSVMMemcpyPOCL);

  if (strcmp (func_name, "clCommandSVMMemcpyRectPOCL") == 0)
    return (void *)&POname (clCommandSVMMemcpyRectPOCL);

  if (strcmp (func_name, "clCommandSVMMemfillPOCL") == 0)
    return (void *)&POname (clCommandSVMMemfillPOCL);

  if (strcmp (func_name, "clCommandSVMMemfillRectPOCL") == 0)
    return (void *)&POname (clCommandSVMMemfillRectPOCL);

  /* cl_pocl_command_buffer_host_buffer */
  if (strcmp (func_name, "clCommandReadBufferPOCL") == 0)
    return (void *)&POname (clCommandReadBufferPOCL);

  if (strcmp (func_name, "clCommandReadBufferRectPOCL") == 0)
    return (void *)&POname (clCommandReadBufferRectPOCL);

  if (strcmp (func_name, "clCommandReadImagePOCL") == 0)
    return (void *)&POname (clCommandReadImagePOCL);

  if (strcmp (func_name, "clCommandWriteBufferPOCL") == 0)
    return (void *)&POname (clCommandWriteBufferPOCL);

  if (strcmp (func_name, "clCommandWriteBufferRectPOCL") == 0)
    return (void *)&POname (clCommandWriteBufferRectPOCL);

  if (strcmp (func_name, "clCommandWriteImagePOCL") == 0)
    return (void *)&POname (clCommandWriteImagePOCL);

  /* cl_pocl_svm_rect */
  if (strcmp (func_name, "clEnqueueSVMMemcpyRectPOCL") == 0)
    return (void *)&POname (clEnqueueSVMMemcpyRectPOCL);

  if (strcmp (func_name, "clEnqueueSVMMemFillRectPOCL") == 0)
    return (void *)&POname (clEnqueueSVMMemFillRectPOCL);

  /* cl_ext_buffer_device_address */
  if (strcmp (func_name, "clSetKernelArgDevicePointerEXT") == 0)
    return (void *)&POname (clSetKernelArgDevicePointerEXT);

  /* cl_intel_create_buffer_with_properties */
  /* Some applications want to use clCreateBufferWithPropertiesINTEL even
     with 2.0+ OpenCL targets. */
  if (strcmp (func_name, "clCreateBufferWithPropertiesINTEL") == 0)
    return (void *)&POname (clCreateBufferWithProperties);

  if (strcmp (func_name, "clCreateCommandQueueWithPropertiesKHR") == 0)
    return (void *)&POname (clCreateCommandQueueWithProperties);

  if (strcmp (func_name, "clUpdateMutableCommandsKHR") == 0)
    return (void *)&POname (clUpdateMutableCommandsKHR);

  if (strcmp (func_name, "clGetMutableCommandInfoKHR") == 0)
    return (void *)&POname (clGetMutableCommandInfoKHR);

  if (strcmp (func_name, "clCreateProgramWithDefinedBuiltInKernels") == 0)
    return (void *)&POname (clCreateProgramWithDefinedBuiltInKernels);

  POCL_MSG_ERR ("unknown platform extension requested: %s\n", func_name);
  return NULL;
}
POsymAlways (clGetExtensionFunctionAddressForPlatform)
