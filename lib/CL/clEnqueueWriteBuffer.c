/* OpenCL runtime library: clEnqueueWriteBuffer()

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
#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueWriteBuffer)(cl_command_queue command_queue,
                     cl_mem buffer,
                     cl_bool blocking_write,
                     size_t offset,
                     size_t cb, 
                     const void *ptr,
                     cl_uint num_events_in_wait_list,
                     const cl_event *event_wait_list,
                     cl_event *event) CL_API_SUFFIX__VERSION_1_0
{
  cl_device_id device;
  unsigned i;
  _cl_command_node *cmd = NULL;
  int errcode;

  if (command_queue == NULL)
    return CL_INVALID_COMMAND_QUEUE;

  if (buffer == NULL)
    return CL_INVALID_MEM_OBJECT;

  if (command_queue->context != buffer->context)
    return CL_INVALID_CONTEXT;

  if ((ptr == NULL) ||
      (offset + cb > buffer->size))
    return CL_INVALID_VALUE;

  if (num_events_in_wait_list > 0 && event_wait_list == NULL)
    return CL_INVALID_EVENT_WAIT_LIST;

  if (num_events_in_wait_list == 0 && event_wait_list != NULL)
    return CL_INVALID_EVENT_WAIT_LIST;

  for(i=0; i<num_events_in_wait_list; i++)
    {
      if (event_wait_list[i] == NULL)
        return CL_INVALID_EVENT_WAIT_LIST;
    }
  
  device = command_queue->device;

  for (i = 0; i < command_queue->context->num_devices; ++i)
    {
        if (command_queue->context->devices[i] == device)
            break;
    }
  assert(i < command_queue->context->num_devices);

  errcode = pocl_create_command (&cmd, command_queue, 
                                 CL_COMMAND_WRITE_BUFFER, 
                                 event, num_events_in_wait_list, 
                                 event_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;
  
  cmd->command.write.host_ptr = ptr;
  cmd->command.write.device_ptr = buffer->device_ptrs[i].mem_ptr+offset;
  cmd->command.write.cb = cb;
  cmd->command.write.buffer = buffer;
  POname(clRetainMemObject) (buffer);
  pocl_command_enqueue(command_queue, cmd);
  
  if (blocking_write)
    POname(clFinish) (command_queue);

  return CL_SUCCESS;
}
POsym(clEnqueueWriteBuffer)
