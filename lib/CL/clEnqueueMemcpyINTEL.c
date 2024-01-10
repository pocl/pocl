/* OpenCL runtime library: clEnqueueMemcpyINTEL()

   Copyright (c) 2023 Michal Babej / Intel Finland Oy
                 2024 Pekka Jääskeläinen / Intel Finland Oy

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

#include "pocl_cl.h"
#include "pocl_shared.h"
#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_CALL
POname (clEnqueueMemcpyINTEL) (cl_command_queue command_queue,
                               cl_bool blocking, void *dst_ptr,
                               const void *src_ptr, size_t size,
                               cl_uint num_events_in_wait_list,
                               const cl_event *event_wait_list,
                               cl_event *event) CL_API_SUFFIX__VERSION_2_0
{
  cl_int errcode;
  _cl_command_node *cmd = NULL;

  errcode = pocl_svm_memcpy_common (
      NULL, command_queue, CL_COMMAND_MEMCPY_INTEL, dst_ptr, src_ptr, size,
      num_events_in_wait_list, event_wait_list, event, NULL, NULL, &cmd);
  if (errcode != CL_SUCCESS)
    return errcode;

  if (cmd != NULL)
    pocl_command_enqueue (command_queue, cmd);

  if (blocking)
    POname (clFinish) (command_queue);

  return CL_SUCCESS;
}
POsym (clEnqueueMemcpyINTEL)
