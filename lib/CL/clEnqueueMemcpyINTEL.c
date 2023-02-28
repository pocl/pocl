/* OpenCL runtime library: clEnqueueMemcpyINTEL()

   Copyright (c) 2023 Michal Babej / Intel Finland Oy

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
  return pocl_svm_memcpy_common (
      CL_COMMAND_MEMCPY_INTEL, command_queue, blocking, dst_ptr, src_ptr, size,
      num_events_in_wait_list, event_wait_list, event);
}
POsym (clEnqueueMemcpyINTEL)
