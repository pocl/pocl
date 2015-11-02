/* OpenCL runtime library: clCreatePipe()

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

CL_API_ENTRY cl_mem CL_API_CALL
POname(clCreatePipe)(cl_context context,
                     cl_mem_flags flags,
                     cl_uint pipe_packet_size,
                     cl_uint pipe_max_packets,
                     const cl_pipe_properties * properties,
                     cl_int *errcode_ret) CL_API_SUFFIX__VERSION_2_0
{
  int errcode;
#ifndef BUILD_HSA
  POCL_MSG_PRINT_INFO("This pocl was not built with HSA\n");
  errcode = CL_INVALID_CONTEXT;
#else

  POCL_GOTO_ERROR_COND((context == NULL), CL_INVALID_CONTEXT);

  POCL_GOTO_ERROR_COND((properties != NULL), CL_INVALID_VALUE);

  //CL_INVALID_VALUE if values specified in flags are not as defined above.

  //CL_INVALID_PIPE_SIZE if pipe_packet_size is 0
  //or if pipe_max_packets is 0.
  POCL_GOTO_ERROR_COND((pipe_packet_size == 0), CL_INVALID_PIPE_SIZE);
  POCL_GOTO_ERROR_COND((pipe_max_packets == 0), CL_INVALID_PIPE_SIZE);

  // or the pipe_packet_size exceeds CL_DEVICE_PIPE_MAX_PACKET_SIZE value specified in table 4.3
  // (see clGetDeviceInfo) for all devices in context
  for (unsigned i = 0; i < context->num_devices; i++)
    POCL_GOTO_ERROR_COND(context->devices[i]->max_pipe_packet_size < pipe_packet_size,
                         CL_INVALID_PIPE_SIZE)

  //CL_MEM_OBJECT_ALLOCATION_FAILURE if there is a failure to allocate memory for the pipe object.

  *errcode_ret = CL_SUCCESS;
  return NULL;

#endif

ERROR:
  if (errcode_ret)
    *errcode_ret = errcode;
  return NULL;
}
