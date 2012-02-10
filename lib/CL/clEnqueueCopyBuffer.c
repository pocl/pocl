/* OpenCL runtime library: clEnqueueCopyBuffer()

   Copyright (c) 2011 Universidad Rey Juan Carlos
   
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
#include <assert.h>

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyBuffer(cl_command_queue command_queue,
                    cl_mem src_buffer,
                    cl_mem dst_buffer,
                    size_t src_offset,
                    size_t dst_offset,
                    size_t cb, 
                    cl_uint num_events_in_wait_list,
                    const cl_event *event_wait_list,
                    cl_event *event) CL_API_SUFFIX__VERSION_1_0
{
  cl_device_id device_id;
  unsigned i;

  if (command_queue == NULL)
    return CL_INVALID_COMMAND_QUEUE;

  if ((src_buffer == NULL) || (dst_buffer == NULL))
    return CL_INVALID_MEM_OBJECT;

  if ((command_queue->context != src_buffer->context) ||
      (command_queue->context != dst_buffer->context))
    return CL_INVALID_CONTEXT;

  if ((src_offset + cb > src_buffer->size) ||
      (dst_offset + cb > dst_buffer->size))
    return CL_INVALID_VALUE;

  device_id = command_queue->device;
  for (i = 0; i < command_queue->context->num_devices; ++i)
    {
      if (command_queue->context->devices[i] == device_id)
        break;
    }

  assert(i < command_queue->context->num_devices);

  // TODO: check the intended semantics of buffer->device_ptrs[i] - can we just add a offset like this?
  device_id->copy(device_id->data, src_buffer->device_ptrs[i]+src_offset, dst_buffer->device_ptrs[i]+dst_offset, cb);

  return CL_SUCCESS;
}
