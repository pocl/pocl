/* OpenCL runtime library: clGetProgramBuildInfo()

   Copyright (c) 2011 Kalle Raiskila
                 2011-2019 Pekka Jääskeläinen / Tampere University

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
  const char *str;

  POCL_RETURN_ERROR_COND((program == NULL), CL_INVALID_PROGRAM);

  POCL_RETURN_ERROR_COND ((device == NULL), CL_INVALID_DEVICE);

  int device_i = pocl_cl_device_to_index(program, device);
  POCL_RETURN_ERROR_ON((device_i < 0), CL_INVALID_DEVICE, "Program does not have "
    "this device in it's device list\n");

  switch (param_name) {
  case CL_PROGRAM_BUILD_STATUS:
    {
      POCL_RETURN_GETINFO(cl_build_status, program->build_status);
    }
    
  case CL_PROGRAM_BUILD_OPTIONS:
    {
      str = (program->compiler_options)? program->compiler_options: "";
      POCL_RETURN_GETINFO_STR(str);
    }
    
  case CL_PROGRAM_BUILD_LOG:
    {
      POCL_RETURN_ERROR_ON((program->build_status == CL_BUILD_NONE),
                           CL_INVALID_PROGRAM,
                           "Program was not built");
      if (program->builtin_kernel_names != NULL)
        {
          POCL_RETURN_GETINFO_STR ("");
        }
      if (program->main_build_log[0])
        {
          POCL_RETURN_GETINFO_STR (program->main_build_log);
        }
      else if (program->build_log[device_i])
        {
          POCL_RETURN_GETINFO_STR (program->build_log[device_i]);
        }
      else
        {
          char *build_log = pocl_cache_read_buildlog (program, device_i);
          if (build_log)
            POCL_RETURN_GETINFO_STR_FREE (build_log);
        }

      POCL_RETURN_GETINFO_STR ("");
    }
  case CL_PROGRAM_BINARY_TYPE:
    {
      POCL_RETURN_GETINFO(cl_program_binary_type, program->binary_type);
    }
  }
  
  return CL_INVALID_VALUE;
}
POsym(clGetProgramBuildInfo)
