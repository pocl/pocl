/* OpenCL runtime library: clEnqueueReadBuffer()

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
#include "utlist.h"
#include <assert.h>

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadBuffer(cl_command_queue command_queue,
                    cl_mem buffer,
                    cl_bool blocking_read,
                    size_t offset,
                    size_t cb, 
                    void *ptr,
                    cl_uint num_events_in_wait_list,
                    const cl_event *event_wait_list,
                    cl_event *event) CL_API_SUFFIX__VERSION_1_0
{
  cl_device_id device;

  if (command_queue == NULL)
    return CL_INVALID_COMMAND_QUEUE;

  if (buffer == NULL)
    return CL_INVALID_MEM_OBJECT;

  if (command_queue->context != buffer->context)
    return CL_INVALID_CONTEXT;

  if ((ptr == NULL) ||
      (offset + cb > buffer->size))
    return CL_INVALID_VALUE;

  device = command_queue->device;

  if (event != NULL)
    {
      *event = (cl_event)malloc(sizeof(struct _cl_event));
      if (*event == NULL)
        return CL_OUT_OF_HOST_MEMORY; 
      POCL_INIT_OBJECT(*event);
      (*event)->queue = command_queue;
    }


  /* enqueue the read, or execute directly */
  if (blocking_read)
    {
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
          clRetainMemObject (buffer);
          clFinish(command_queue);
        }
      /* TODO: offset computation doesn't work in case the ptr is not 
         a direct pointer */
      device->read(device->data, ptr, buffer->device_ptrs[device->dev_id]+offset, cb);
      clReleaseMemObject (buffer);
    }
  else
  {
    _cl_command_node * cmd = malloc(sizeof(_cl_command_node));
    if (cmd == NULL)
      return CL_OUT_OF_HOST_MEMORY;
    
    cmd->type = CL_COMMAND_TYPE_READ;
    cmd->command.read.data = device->data;
    cmd->command.read.host_ptr = ptr;
    cmd->command.read.device_ptr = buffer->device_ptrs[device->dev_id]+offset;
    cmd->command.read.cb = cb;
    cmd->next = NULL;
    cmd->command.read.buffer = buffer;
    clRetainMemObject (buffer);
    LL_APPEND(command_queue->root, cmd );
  }

  return CL_SUCCESS;
}
