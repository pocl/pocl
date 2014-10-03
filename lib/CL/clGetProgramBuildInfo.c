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
#include <string.h>

CL_API_ENTRY cl_int CL_API_CALL
POname(clGetProgramBuildInfo)(cl_program            program,
                      cl_device_id          device,
                      cl_program_build_info param_name,
                      size_t                param_value_size,
                      void *                param_value,
                      size_t *              param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
  int i;
  cl_bool found;

  found = CL_FALSE;
  for (i = 0; i < program->num_devices; i++)
    if (device == program->devices[i]) found = CL_TRUE;

  if (found == CL_FALSE) return CL_INVALID_DEVICE;

  switch (param_name) {
  case CL_PROGRAM_BUILD_STATUS:
    {
      size_t const value_size = sizeof(cl_build_status);
      if (param_value)
      {
        if (param_value_size < value_size) return CL_INVALID_VALUE;
        memcpy(param_value, &(program->build_status), value_size);
      }
      if (param_value_size_ret)
        *param_value_size_ret = value_size;

      return CL_SUCCESS;
    }
    
  case CL_PROGRAM_BUILD_OPTIONS:
    {
      size_t const value_size = strlen(program->compiler_options) + 1;
      if (param_value && program->compiler_options)
      {
        if (param_value_size < value_size) return CL_INVALID_VALUE;
        memcpy(param_value, program->compiler_options, value_size);
      }
      if (param_value_size_ret)
        *param_value_size_ret = value_size;
      return CL_SUCCESS;
    }
    
  case CL_PROGRAM_BUILD_LOG:
    {
      char *build_log = NULL;
      char buildlog_file_name[POCL_FILENAME_LENGTH];
      snprintf(buildlog_file_name, POCL_FILENAME_LENGTH, "%s/%s",
               program->temp_dir, POCL_BUILDLOG_FILENAME);

      size_t const value_size = pocl_read_text_file(buildlog_file_name, &build_log) + 1;
      if (param_value && build_log)
      {
        if (param_value_size < value_size) return CL_INVALID_VALUE;
        memcpy(param_value, build_log, value_size);
        POCL_MEM_FREE(build_log);
      }
      if (param_value_size_ret)
        *param_value_size_ret = value_size;
      return CL_SUCCESS;
    }
  }
  
  return CL_INVALID_VALUE;
}
POsym(clGetProgramBuildInfo)
