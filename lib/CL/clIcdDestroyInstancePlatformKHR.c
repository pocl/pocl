/* OpenCL runtime library: clSetPlatformDispatchDataKHR()

   Copyright (c) 2023 Brice Videau / Argonne National Laboratory

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
#include "pocl_cl.h"

#ifdef BUILD_ICD
POCL_EXPORT CL_API_ENTRY cl_int CL_API_CALL
POname (clIcdDestroyInstancePlatformKHR) (cl_platform_id platform)
{
  cl_platform_id pocl_platform;
  POCL_RETURN_ERROR_COND ((platform == NULL), CL_INVALID_PLATFORM);
  POname (clGetPlatformIDs) (1, &pocl_platform, NULL);
  POCL_RETURN_ERROR_ON ((!POCL_PLATFORM_VALID (platform, pocl_platform)),
                        CL_INVALID_PLATFORM,
                        "Can only release instance of the POCL platform\n");
  POCL_RETURN_ERROR_ON ((!platform->instance), CL_INVALID_PLATFORM,
                        "Can only release instance of the POCL platform\n");
  for (unsigned i = 0; i < platform->num_devices; i++)
    {
      cl_device_id device = platform->devices[i];
      POCL_DESTROY_OBJECT (device);
      POCL_MEM_FREE (device->builtin_kernel_list);
      POCL_MEM_FREE (device->builtin_kernels_with_version);
      POCL_MEM_FREE (device);
    }
  POCL_MEM_FREE (platform->devices);
  POCL_MEM_FREE (platform);

  return CL_SUCCESS;
}
POsymICD (clIcdDestroyInstancePlatformKHR)
#endif
