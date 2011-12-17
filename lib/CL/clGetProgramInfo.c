/* OpenCL runtime library: clGetProgramInfo()

   Copyright (c) 2011 Erik Schnetter
   
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
clGetProgramInfo(cl_program program,
                 cl_program_info param_name,
                 size_t param_value_size,
                 void *param_value,
                 size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
  switch (param_name)
  {
  case CL_PROGRAM_REFERENCE_COUNT:
    return CL_INVALID_VALUE;    /* not yet implemented */
    
  case CL_PROGRAM_CONTEXT:
    return CL_INVALID_VALUE;    /* not yet implemented */
    
  case CL_PROGRAM_NUM_DEVICES:
    return CL_INVALID_VALUE;    /* not yet implemented */
    
  case CL_PROGRAM_DEVICES:
    return CL_INVALID_VALUE;    /* not yet implemented */
    
  case CL_PROGRAM_SOURCE:
    {
      size_t const value_size = strlen(program->source) + 1;
      if (param_value && param_value_size >= value_size)
        memcpy(param_value, program->source, value_size);
      if (param_value_size_ret)
        *param_value_size_ret = value_size;
      return CL_SUCCESS;
    }
    
  case CL_PROGRAM_BINARY_SIZES:
    {
      size_t const value_size = sizeof(size_t);
      if (param_value && param_value_size >= value_size)
        *(size_t*)param_value = program->binary_size;
      if (param_value_size_ret)
        *param_value_size_ret = value_size;
      return CL_SUCCESS;
    }
    
  case CL_PROGRAM_BINARIES:
    {
      size_t const value_size = sizeof(unsigned char *);
      if (param_value && param_value_size >= value_size)
        *(unsigned char **)param_value = program->binary;
      if (param_value_size_ret)
        *param_value_size_ret = value_size;
      return CL_SUCCESS;
    }
  }
  
  return CL_INVALID_VALUE;
}
