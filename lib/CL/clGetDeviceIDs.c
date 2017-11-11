/* OpenCL runtime library: clGetDeviceIDs()

   Copyright (c) 2011 Kalle Raiskila

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
#include <string.h>

CL_API_ENTRY cl_int CL_API_CALL
POname(clGetDeviceIDs)(cl_platform_id   platform,
               cl_device_type   device_type,
               cl_uint          num_entries,
               cl_device_id *   devices,
               cl_uint *        num_devices) CL_API_SUFFIX__VERSION_1_0
{
  int total_num = 0;
  int devices_added = 0;
  cl_platform_id tmp_platform;

  /* TODO: OpenCL API specification allows implementation dependent
     behaviour if platform == NULL. Should we just allow it? */
  POCL_RETURN_ERROR_COND((platform == NULL), CL_INVALID_PLATFORM);

  POCL_RETURN_ERROR_COND((num_entries == 0 && devices != NULL), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND((num_devices == NULL && devices == NULL), CL_INVALID_VALUE);

  POname (clGetPlatformIDs) (1, &tmp_platform, NULL);
  POCL_RETURN_ERROR_ON ((platform != tmp_platform), CL_INVALID_PLATFORM,
                        "Can only return devices from the POCL platform\n");

  int err = pocl_init_devices();
  if (err)
    return err;

  total_num = pocl_get_device_type_count(device_type);

  if (total_num == 0)
      return CL_DEVICE_NOT_FOUND;

  if (devices != NULL)
    devices_added = pocl_get_devices(device_type, devices, num_entries);

  if (num_devices != NULL)
    *num_devices = total_num;

  if (devices_added > 0 || num_entries == 0)
    return CL_SUCCESS;
  else
    return CL_DEVICE_NOT_FOUND;
}
POsym(clGetDeviceIDs)
