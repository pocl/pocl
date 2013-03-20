/* OpenCL runtime library: clEnqueueCopyBuffer()

   Copyright (c) 2011-2012 Universidad Rey Juan Carlos and
                           Pekka Jääskeläinen / Tampere Univ. of Tech.
   
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
#include "utlist.h"
#include <assert.h>

CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueCopyBuffer)(cl_command_queue command_queue,
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

  if (event != NULL)
    {
      *event = (cl_event)malloc(sizeof(struct _cl_event));
      if (*event == NULL)
        return CL_OUT_OF_HOST_MEMORY; 
      POCL_INIT_OBJECT(*event);
      (*event)->queue = command_queue;
      POname(clRetainCommandQueue) (command_queue);
      (*event)->command_type = CL_COMMAND_COPY_BUFFER;
      POCL_UPDATE_EVENT_QUEUED;
    }


  _cl_command_node * cmd = malloc(sizeof(_cl_command_node));
  if (cmd == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  cmd->command.copy.src_buffer = src_buffer;
  cmd->command.copy.dst_buffer = dst_buffer;
  
  POname(clRetainMemObject)(src_buffer);
  POname(clRetainMemObject)(dst_buffer);

  cmd->type = CL_COMMAND_COPY_BUFFER;
  cmd->command.copy.data = device_id->data;
  /* TODO: call device->buf_offset() or similar as device_ptrs might not be
     actual buffer pointers but pointers to a book keeping structure. */
  cmd->command.copy.src_ptr = src_buffer->device_ptrs[device_id->dev_id] + src_offset;
  cmd->command.copy.dst_ptr = dst_buffer->device_ptrs[device_id->dev_id] + dst_offset;
  cmd->command.copy.cb = cb;
  cmd->next = NULL;
  cmd->event = event ? *event : NULL;

  LL_APPEND(command_queue->root, cmd);

  return CL_SUCCESS;
}
POsym(clEnqueueCopyBuffer)
