/* OpenCL runtime library: clEnqueueCopyBufferRect()

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
#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueCopyBufferRect)(cl_command_queue command_queue,
                                cl_mem src_buffer,
                                cl_mem dst_buffer,
                                const size_t *src_origin,
                                const size_t *dst_origin, 
                                const size_t *region,
                                size_t src_row_pitch,
                                size_t src_slice_pitch,
                                size_t dst_row_pitch,
                                size_t dst_slice_pitch,
                                cl_uint num_events_in_wait_list,
                                const cl_event *event_wait_list,
                                cl_event *event) CL_API_SUFFIX__VERSION_1_1
{
  cl_device_id device_id;
  unsigned i;
  int errcode;

  if (command_queue == NULL)
    return CL_INVALID_COMMAND_QUEUE;

  if ((src_buffer == NULL) || (dst_buffer == NULL))
    return CL_INVALID_MEM_OBJECT;

  if ((command_queue->context != src_buffer->context) ||
      (command_queue->context != dst_buffer->context))
    return CL_INVALID_CONTEXT;

  if ((src_origin == NULL) ||
      (dst_origin == NULL) ||
      (region == NULL))
    return CL_INVALID_VALUE;

  if ((region[0]*region[1]*region[2] > 0) &&
      (src_origin[0] + region[0]-1 +
       src_row_pitch * (src_origin[1] + region[1]-1) +
       src_slice_pitch * (src_origin[2] + region[2]-1) >= src_buffer->size))
    return CL_INVALID_VALUE;
  if ((region[0]*region[1]*region[2] > 0) &&
      (dst_origin[0] + region[0]-1 +
       dst_row_pitch * (dst_origin[1] + region[1]-1) +
       dst_slice_pitch * (dst_origin[2] + region[2]-1) >= dst_buffer->size))
    return CL_INVALID_VALUE;

  device_id = command_queue->device;

  for (i = 0; i < command_queue->context->num_devices; ++i)
    {
      if (command_queue->context->devices[i] == device_id)
        break;
    }
  assert(i < command_queue->context->num_devices);

  /* execute directly */
  /* TODO: enqueue the read_rect if this is a non-blocking read (see
     clEnqueueReadBuffer) */
  if (command_queue->properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
    {
      /* wait for the event in event_wait_list to finish */
      POCL_ABORT_UNIMPLEMENTED();
    }
  else
    {
      /* in-order queue - all previously enqueued commands must 
       * finish before this read */
      // ensure our buffer is not freed yet
      POname(clRetainMemObject) (src_buffer);
      POname(clRetainMemObject) (dst_buffer);
      POname(clFinish)(command_queue);
    }
  POCL_UPDATE_EVENT_SUBMITTED(event, command_queue);
  POCL_UPDATE_EVENT_RUNNING(event, command_queue);

  /* TODO: offset computation doesn't work in case the ptr is not 
     a direct pointer */
  device_id->ops->copy_rect(device_id->data,
                       src_buffer->device_ptrs[device_id->dev_id].mem_ptr, 
                       dst_buffer->device_ptrs[device_id->dev_id].mem_ptr,
                       src_origin, dst_origin, region,
                       src_row_pitch, src_slice_pitch,
                       dst_row_pitch, dst_slice_pitch);

  POCL_UPDATE_EVENT_COMPLETE(event, command_queue);

  POname(clReleaseMemObject) (src_buffer);
  POname(clReleaseMemObject) (dst_buffer);

  return CL_SUCCESS;
}
POsym(clEnqueueCopyBufferRect)
