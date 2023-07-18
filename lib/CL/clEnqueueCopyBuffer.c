/* OpenCL runtime library: clEnqueueCopyBuffer()

   Copyright (c) 2011-2015 Universidad Rey Juan Carlos and
                           Pekka Jääskeläinen / Tampere Univ. of Tech.
                           Ville Korhonen / Tampere Univ. of Tech.

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

#include "pocl_shared.h"
#include "pocl_util.h"
#include "utlist.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueCopyBuffer)(cl_command_queue command_queue,
                            cl_mem src_buffer,
                            cl_mem dst_buffer,
                            size_t src_offset,
                            size_t dst_offset,
                            size_t size,
                            cl_uint num_events_in_wait_list,
                            const cl_event *event_wait_list,
                            cl_event *event) 
CL_API_SUFFIX__VERSION_1_0
{
  _cl_command_node *cmd = NULL;
  int errcode;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  errcode = pocl_copy_buffer_common (
      NULL, command_queue, src_buffer, dst_buffer, src_offset, dst_offset,
      size, num_events_in_wait_list, event_wait_list, event, NULL, NULL, &cmd);
  if (errcode != CL_SUCCESS)
    return errcode;

  pocl_command_enqueue(command_queue, cmd);

  return CL_SUCCESS;
}
POsym(clEnqueueCopyBuffer)
