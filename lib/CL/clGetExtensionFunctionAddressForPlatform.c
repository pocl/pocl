/* OpenCL runtime library: clGetExtensionFunctionAddressForPlatform()

   Copyright (c) 2017 Michal Babej / Tampere University of Technology

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

#include <string.h>

CL_API_ENTRY void * CL_API_CALL
POname (clGetExtensionFunctionAddressForPlatform) (cl_platform_id  platform,
                                                   const char *func_name)
CL_EXT_SUFFIX__VERSION_1_2
{
  cl_platform_id pocl_platform;
  cl_uint actual_num = 0;
  POname (clGetPlatformIDs) (1, &pocl_platform, &actual_num);
  if (actual_num != 1)
    {
      POCL_MSG_WARN ("Couldn't get the platform ID of Pocl platform\n");
      return NULL;
    }

  assert (pocl_platform);
  if (platform != pocl_platform)
    {
      POCL_MSG_PRINT_INFO ("Requested Function Address not "
                           "for Pocl platform, ignoring\n");
      return NULL;
    }

#ifdef BUILD_ICD
  if (strcmp (func_name, "clIcdGetPlatformIDsKHR") == 0)
    return (void *)&POname(clIcdGetPlatformIDsKHR);
#endif
  if (strcmp (func_name, "clGetPlatformInfo") == 0)
    return (void *)&POname(clGetPlatformInfo);

  return NULL;
}
POsymAlways (clGetExtensionFunctionAddressForPlatform)
