/* OpenCL runtime library: clGetPlatformInfo()

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

#include "pocl_util.h"

    
CL_API_ENTRY cl_int CL_API_CALL 
POname(clGetPlatformInfo)(cl_platform_id   platform,
                  cl_platform_info param_name,
                  size_t           param_value_size, 
                  void *           param_value,
                  size_t *         param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
  const char *ret;
  int retlen;
  cl_platform_id tmp_platform;

  // TODO: if we don't have ICD in use, platform==NULL should be valid & point to pocl
  if (platform == NULL)
    return CL_INVALID_PLATFORM;
	
  POname(clGetPlatformIDs)(1, &tmp_platform, NULL);
  if (platform != tmp_platform)
    return CL_INVALID_PLATFORM;

  switch (param_name)
  {
    case CL_PLATFORM_PROFILE:
      // TODO: figure this out depending on the native execution host.
      // assume FULL_PROFILE for now.
      POCL_RETURN_GETINFO_STR("FULL_PROFILE");

    case CL_PLATFORM_VERSION:
      POCL_RETURN_GETINFO_STR("OpenCL 1.2 pocl " PACKAGE_VERSION);

    case CL_PLATFORM_NAME:
      POCL_RETURN_GETINFO_STR("Portable Computing Language");

    case CL_PLATFORM_VENDOR:
      POCL_RETURN_GETINFO_STR("The pocl project");

    case CL_PLATFORM_EXTENSIONS:
      // TODO: do we want to list all supported extensions *here*, or in some header?.
      // TODO: yes, it is better here: available through ICD Loader and headers can be the ones from Khronos
#ifdef BUILD_ICD
      POCL_RETURN_GETINFO_STR("cl_khr_icd");
#else
      POCL_RETURN_GETINFO_STR("");
#endif

    case CL_PLATFORM_ICD_SUFFIX_KHR:
      POCL_RETURN_GETINFO_STR("POCL");

    default: 
      return CL_INVALID_VALUE;
  }

  // the OpenCL API docs *seem* to count the trailing NULL
  retlen = strlen(ret) + 1;
	
  if (param_value != NULL)
  {
    if (param_value_size < retlen)
      return CL_INVALID_VALUE;
    
    memcpy(param_value, ret, retlen); 
  }
	
  if (param_value_size_ret != NULL)
    *param_value_size_ret = retlen;
	
  return CL_SUCCESS;
}
POsym(clGetPlatformInfo)
