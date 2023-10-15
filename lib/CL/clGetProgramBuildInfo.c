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

static int
pocl_cl_device_assoc_index (cl_program program, cl_device_id device)
{
  unsigned i;
  assert (program);
  for (i = 0; i < program->associated_num_devices; i++)
    if (program->associated_devices[i] == device
        || program->associated_devices[i] == device->parent_device)
      return i;
  return -1;
}

static int
pocl_cl_device_built_index (cl_program program, cl_device_id device)
{
  unsigned i;
  assert (program);
  for (i = 0; i < program->num_devices; i++)
    if (program->devices[i] == device
        || program->devices[i] == device->parent_device)
      return i;
  return -1;
}

CL_API_ENTRY cl_int CL_API_CALL
POname(clGetProgramBuildInfo)(cl_program            program,
                      cl_device_id          device,
                      cl_program_build_info param_name,
                      size_t                param_value_size,
                      void *                param_value,
                      size_t *              param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
  const char *str;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (program)), CL_INVALID_PROGRAM);

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (device)), CL_INVALID_DEVICE);

  /*  Returns CL_INVALID_DEVICE if device is not in the list
   *  of devices associated with program. */
  int device_i = pocl_cl_device_assoc_index (program, device);
  POCL_RETURN_ERROR_ON ((device_i < 0), CL_INVALID_DEVICE,
                        "Device is not in the list of devices"
                        " associated with the program\n");

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
      if (program->builtin_kernel_names != NULL)
        {
          POCL_RETURN_GETINFO_STR ("");
        }
      if (program->main_build_log[0])
        {
          POCL_RETURN_GETINFO_STR (program->main_build_log);
        }
      else
        {
          /*  If build status of program for device is CL_BUILD_NONE,
           *  an empty string is returned. */
          device_i = pocl_cl_device_built_index (program, device);
          if (device_i < 0)
            POCL_RETURN_GETINFO_STR ("");
          if (program->build_log[device_i])
            {
              POCL_RETURN_GETINFO_STR (program->build_log[device_i]);
            }
          else
            {
              char *build_log = pocl_cache_read_buildlog (program, device_i);
              if (build_log)
                POCL_RETURN_GETINFO_STR_FREE (build_log);
            }
        }
      POCL_RETURN_GETINFO_STR ("");
    }

  case CL_PROGRAM_BINARY_TYPE:
    {
      POCL_RETURN_GETINFO(cl_program_binary_type, program->binary_type);
    }

  case CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE:
    {
      POCL_RETURN_GETINFO (size_t, program->global_var_total_size[device_i]);
    }
  }
  
  return CL_INVALID_VALUE;
}
POsym(clGetProgramBuildInfo)
