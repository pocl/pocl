/* OpenCL runtime library: clGetPlatformInfo()
 *
 * Copyright - Kalle Raiskila 2011.
 *
 * This is file is licencsed under a "Free Beer" type license:
 * You can do whatever you want with this stuff. If we meet some day,
 * and you think this stuff is worth it, you can buy me a beer in return.
 */

#include <string.h>
#include "pocl_cl.h"

CL_API_ENTRY cl_int CL_API_CALL 
clGetPlatformInfo(cl_platform_id   platform,
                  cl_platform_info param_name,
                  size_t           param_value_size, 
                  void *           param_value,
                  size_t *         param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
  char *ret=0;
  int retlen;	

  if (platform == NULL || (platform->magic != 42))
    return CL_INVALID_PLATFORM;
	
  switch (param_name)
	{
    case CL_PLATFORM_PROFILE:
      // TODO: figure this out depending on the native execution host.
      // assume FULL_PROFILE for now.
      ret = "FULL_PROFILE";
      break;
    case CL_PLATFORM_VERSION:
      ret = "OpenCL 1.2";
      break;
    case CL_PLATFORM_NAME:
      ret = "Portable OpenCL";
      break;
    case CL_PLATFORM_VENDOR:
      ret = "The POCL project";
      break;
    case CL_PLATFORM_EXTENSIONS:
      // TODO: list all suppoted extensions here.
      ret = "";
      break;
    default: 
      return CL_INVALID_VALUE;
	}

  // Specs say (section 4.1) to "ignore param_value" should it be NULL
  if (param_value == NULL)
    return CL_SUCCESS;	
	
  // the OpenCL API docs *seem* to count the trailing NULL
  retlen = strlen(ret) + 1;
	
  if (param_value_size < retlen)
    return CL_INVALID_VALUE;

  strncpy(param_value, ret, retlen); 
	
  if (param_value_size_ret != NULL)
    *param_value_size_ret=retlen;
	
  return CL_SUCCESS;

}
