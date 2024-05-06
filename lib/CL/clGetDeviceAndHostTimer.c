/* OpenCL runtime library: clGetDeviceAndHostTimer()

   Copyright (c) 2021 Väinö Liukko

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

CL_API_ENTRY cl_int CL_API_ENTRY POname (clGetDeviceAndHostTimer) (
    cl_device_id device, cl_ulong *device_timestamp,
    cl_ulong *host_timestamp) CL_API_SUFFIX__VERSION_2_1
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (device)), CL_INVALID_DEVICE);

  POCL_RETURN_ERROR_COND ((*(device->available) == CL_FALSE),
                          CL_DEVICE_NOT_AVAILABLE);

  POCL_RETURN_ERROR_COND (device_timestamp == NULL, CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND (host_timestamp == NULL, CL_INVALID_VALUE);

  if (device->ops->get_synchronized_timestamps)
    return device->ops->get_synchronized_timestamps (device, device_timestamp,
                                                     host_timestamp);
  POCL_RETURN_ERROR_ON(1, CL_INVALID_OPERATION, "Selected device "
                       "does not support timestamp synchronization\n");
}
POsym(clGetDeviceAndHostTimer)
