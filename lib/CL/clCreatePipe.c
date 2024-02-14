/* OpenCL runtime library: clCreatePipe

   Copyright (c) 2021 Väinö Liukko

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include "pocl_shared.h"
#include "pocl_util.h"

/*  Minimal OpenCL 3.0 conformant implemenation of clCreatePipe where only
    error handling has been implemented.
*/
CL_API_ENTRY cl_mem CL_API_CALL POname (clCreatePipe) (
    cl_context context, cl_mem_flags flags, cl_uint pipe_packet_size,
    cl_uint pipe_max_packets, const cl_pipe_properties *properties,
    cl_int *errcode_ret) CL_API_SUFFIX__VERSION_2_0
{
  if (!IS_CL_OBJECT_VALID (context))
    {
      POCL_ERROR (CL_INVALID_CONTEXT);
    }

  // Check if any device within the context supports pipes.
  unsigned i;
  cl_bool context_pipe_support = CL_FALSE;
  for (i = 0; i < context->num_devices; i++)
    {
      if (context->devices[i]->pipe_support == CL_TRUE)
        {
          context_pipe_support = CL_TRUE;
          break;
        }
    }

  if (!context_pipe_support)
    {
      POCL_ERROR (CL_INVALID_OPERATION);
    }

  int errcode = CL_SUCCESS;
  /* validate flags */
  POCL_GOTO_ERROR_ON ((flags > (1 << 10) - 1), CL_INVALID_VALUE,
                      "Flags must "
                      "be < 1024 (there are only 10 flags)\n");

  POCL_GOTO_ERROR_ON (
      (flags
       & ~((flags & CL_MEM_READ_WRITE) | (flags & CL_MEM_HOST_NO_ACCESS))),
      CL_INVALID_VALUE,
      "Only CL_MEM_READ_WRITE and CL_MEM_HOST_NO_ACCESS can be specified when "
      "creating a pipe object");

  /* Currently pipes do not have any properties */
  if (properties != NULL)
    {
      POCL_ERROR (CL_INVALID_VALUE);
    }

  cl_mem mem = NULL;
  mem = pocl_create_memobject (context, flags, pipe_max_packets,
                               CL_MEM_OBJECT_PIPE, NULL, NULL, 0, &errcode);
  if (mem == NULL)
    goto ERROR;

  mem->pipe_packet_size = pipe_packet_size;
  mem->pipe_max_packets = pipe_max_packets;
  mem->num_properties = 0;

ERROR:
  if (errcode_ret)
    {
      *errcode_ret = errcode;
    }
  return mem;
}
POsym (clCreatePipe)
