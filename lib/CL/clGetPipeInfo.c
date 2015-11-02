/* OpenCL runtime library: clGetPipeInfo()

   Copyright (c) 2015 Michal Babej / Tampere University of Technology

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

CL_API_ENTRY cl_int CL_API_CALL
POname(clGetPipeInfo)(cl_mem  pipe,
                      cl_pipe_info  param_name,
                      size_t  param_value_size,
                      void  *param_value,
                      size_t  *param_value_size_ret) CL_API_SUFFIX__VERSION_2_0
{
#ifndef BUILD_HSA
  POCL_RETURN_ERROR_ON(1, CL_INVALID_CONTEXT, "This pocl was not built with HSA\n");
#else
  size_t value_size;

  POCL_RETURN_ERROR_ON((pipe->type != CL_MEM_OBJECT_PIPE),
                       CL_INVALID_MEM_OBJECT,
                       "Argument is not a pipe\n");

  switch (param_name) {
    case CL_PIPE_PACKET_SIZE:
      POCL_RETURN_GETINFO(cl_uint, pipe->packet_size);
    case CL_PIPE_MAX_PACKETS:
      POCL_RETURN_GETINFO(cl_uint, pipe->max_packets);
    default:
      return CL_INVALID_VALUE;
    }

  return CL_SUCCESS;
#endif
}
POsym(clGetPipeInfo)
