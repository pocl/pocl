/* OpenCL runtime library: clEnqueueFillBufferRect()

   Copyright (c) 2022-2024 Michal Babej / Intel Finland Oy

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

extern CL_API_ENTRY cl_int CL_API_CALL
POname (clEnqueueFillBufferRect) (cl_command_queue command_queue,
                                  cl_mem buffer,
                                  const size_t *origin,
                                  const size_t *region,
                                  size_t row_pitch,
                                  size_t slice_pitch,
                                  const void *pattern,
                                  size_t pattern_size,
                                  cl_uint num_events_in_wait_list,
                                  const cl_event *event_wait_list,
                                  cl_event *event) CL_API_SUFFIX__VERSION_1_2
{
  int errcode = CL_SUCCESS;
  _cl_command_node *cmd = NULL;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  errcode = pocl_fill_buffer_rect_common (
      NULL, command_queue, buffer, origin, region, row_pitch, slice_pitch,
      pattern, pattern_size, num_events_in_wait_list, event_wait_list, event,
      NULL, NULL, &cmd);
  if (errcode != CL_SUCCESS)
    return errcode;

  pocl_command_enqueue (command_queue, cmd);

  return CL_SUCCESS;
}
POsym (clEnqueueFillBuffer)
