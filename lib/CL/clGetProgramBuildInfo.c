/* OpenCL runtime library: clGetProgramBuildInfo()

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


