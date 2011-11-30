/* OpenCL runtime library: clGetProgramBuildInfo()
 *
 * Copyright - Kalle Raiskila 2011.
 *
 * This is file is licencsed under a "Free Beer" type license:
 * You can do whatever you want with this stuff. If we meet some day,
 * and you think this stuff is worth it, you can buy me a beer in return.
 */

#include "pocl_cl.h"
#include <string.h>

CL_API_ENTRY cl_int CL_API_CALL
clGetProgramBuildInfo(cl_program            program,
                      cl_device_id          device,
                      cl_program_build_info param_name,
                      size_t                param_value_size,
                      void *                param_value,
                      size_t *              param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
  char *retval = "";
  int retlen;

  if (program == NULL)
    return CL_INVALID_PROGRAM;

  // currently - just stub-implmement this.
  // there doesn't seem to exist an "CL_INTERNAL_ERROR" return code :(
  if (param_name != CL_PROGRAM_BUILD_LOG)
    return CL_INVALID_OPERATION;

  retlen = strlen(retval) + 1;
	
  if (param_value == NULL)
    return CL_SUCCESS; 	

  if (param_value_size < retlen)
    return CL_INVALID_VALUE;
	
  strncpy(param_value, retval, retlen);
	
  if (param_value_size_ret != NULL)
    *param_value_size_ret = retlen;

  return CL_SUCCESS;
}


