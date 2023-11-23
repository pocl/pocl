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
POCL_EXPORT CL_API_ENTRY cl_platform_id CL_API_CALL
POname (clIcdCreateInstancePlatformKHR) (cl_platform_id platform,
                                         cl_int *errcode_ret)
{
  int errcode = CL_SUCCESS;
  cl_platform_id pocl_platform;
  cl_platform_id new_platform = NULL;

  POCL_GOTO_ERROR_COND ((platform == NULL), CL_INVALID_PLATFORM);
  POname (clGetPlatformIDs) (1, &pocl_platform, NULL);
  POCL_GOTO_ERROR_ON ((!POCL_PLATFORM_VALID (platform, pocl_platform)),
                      CL_INVALID_PLATFORM,
                      "Can only create instance of the POCL platform\n");

  errcode = pocl_init_devices (platform);
  if (errcode)
    goto ERROR;

  new_platform = (cl_platform_id)calloc (1, sizeof (struct _cl_platform_id));
  POCL_GOTO_ERROR_COND ((new_platform == NULL), CL_OUT_OF_HOST_MEMORY);

  errcode = pocl_get_instance_devices (platform, &new_platform->num_devices,
                                       &new_platform->devices);
  if (errcode)
    goto ERROR;

  *errcode_ret = CL_SUCCESS;
  return new_platform;

ERROR:
  POCL_MEM_FREE (new_platform);
  if (errcode_ret)
    {
      *errcode_ret = errcode;
    }

  return NULL;
}
POsymICD (clIcdCreateInstancePlatformKHR)
#endif
