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

#include "pocl_cl.h"
#include "devices/devices.h"

#ifdef BUILD_ICD
POCL_EXPORT CL_API_ENTRY cl_int CL_API_CALL
POname (clSetPlatformDispatchDataKHR) (cl_platform_id  platform,
                                       void *disp_data)
{
  cl_platform_id pocl_platform;

  POCL_RETURN_ERROR_COND((platform == NULL), CL_INVALID_PLATFORM);
  POname (clGetPlatformIDs) (1, &pocl_platform, NULL);
  POCL_RETURN_ERROR_ON ((platform != pocl_platform), CL_INVALID_PLATFORM,
                        "Can only set dispatch data of the POCL platform\n");
  platform->disp_data = disp_data;
  pocl_set_devices_dispatch_data(disp_data);
  return CL_SUCCESS;
}
POsymICD(clSetPlatformDispatchDataKHR)
#endif
