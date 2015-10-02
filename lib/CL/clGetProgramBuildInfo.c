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
#include "pocl_util.h"
#include "pocl_file_util.h"
#include "pocl_cache.h"
#include <string.h>

CL_API_ENTRY cl_int CL_API_CALL
POname(clGetProgramBuildInfo)(cl_program            program,
                      cl_device_id          device,
                      cl_program_build_info param_name,
                      size_t                param_value_size,
                      void *                param_value,
                      size_t *              param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
  const char *empty_str = "";      /* dummy return value */
  const char *str;

  POCL_RETURN_ERROR_COND((program == NULL), CL_INVALID_PROGRAM);

  int device_i = pocl_cl_device_to_index(program, device);
  POCL_RETURN_ERROR_ON((device_i < 0), CL_INVALID_DEVICE, "Program was not "
    "built for this device\n")

  switch (param_name) {
  case CL_PROGRAM_BUILD_STATUS:
    {
      POCL_RETURN_GETINFO(cl_build_status, program->build_status);
    }
    
  case CL_PROGRAM_BUILD_OPTIONS:
    {
      str = (program->compiler_options)? program->compiler_options: empty_str;
      POCL_RETURN_GETINFO_STR(str);
    }
    
  case CL_PROGRAM_BUILD_LOG:
    {
      char *build_log;
      if (program->main_build_log[0])
          build_log = strdup(program->main_build_log);
      else if (program->build_log[device_i])
          build_log = strdup(program->build_log[device_i]);
      else
          build_log = pocl_cache_read_buildlog(program, device_i);
      if (program->build_status == CL_BUILD_NONE)
          build_log = empty_str;
      POCL_RETURN_ERROR_ON((build_log==NULL), CL_OUT_OF_HOST_MEMORY, "failed to read build log");

      size_t const value_size = strlen(build_log) + 1;
      if (param_value)
      {
        if (param_value_size < value_size)
        {
            POCL_MEM_FREE(build_log);
            return CL_INVALID_VALUE;
        }
        memcpy(param_value, build_log, value_size);
      }
      POCL_MEM_FREE(build_log);
      if (param_value_size_ret)
        *param_value_size_ret = value_size;
      return CL_SUCCESS;
    }
  }
  
  return CL_INVALID_VALUE;
}
POsym(clGetProgramBuildInfo)
