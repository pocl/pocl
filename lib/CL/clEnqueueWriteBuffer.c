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
                     size_t size,
                     const void *ptr,
                     cl_uint num_events_in_wait_list,
                     const cl_event *event_wait_list,
                     cl_event *event) CL_API_SUFFIX__VERSION_1_0
{
  cl_device_id device;
  unsigned i;
  _cl_command_node *cmd = NULL;
  int errcode;

  POCL_RETURN_ERROR_COND((command_queue == NULL), CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_COND((buffer == NULL), CL_INVALID_MEM_OBJECT);

  POCL_RETURN_ON_SUB_MISALIGN (buffer, command_queue);

  POCL_RETURN_ERROR_ON((command_queue->context != buffer->context),
    CL_INVALID_CONTEXT, "buffer and command_queue are not from the same context\n");

  POCL_RETURN_ERROR_ON (
      (buffer->flags & (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS)),
      CL_INVALID_OPERATION,
      "buffer has been created with CL_MEM_HOST_READ_ONLY "
      "or CL_MEM_HOST_NO_ACCESS\n");

  POCL_RETURN_ERROR_COND((ptr == NULL), CL_INVALID_VALUE);

  if (pocl_buffer_boundcheck (buffer, offset, size) != CL_SUCCESS)
    return CL_INVALID_VALUE;

  POCL_CONVERT_SUBBUFFER_OFFSET (buffer, offset);

  POCL_RETURN_ERROR_ON((buffer->size > command_queue->device->max_mem_alloc_size),
                        CL_OUT_OF_RESOURCES,
                        "buffer is larger than device's MAX_MEM_ALLOC_SIZE\n");

  errcode = pocl_check_event_wait_list (command_queue, num_events_in_wait_list,
                                        event_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;

  POCL_CHECK_DEV_IN_CMDQ;

  errcode = pocl_create_command (&cmd, command_queue, 
                                 CL_COMMAND_WRITE_BUFFER, 
                                 event, num_events_in_wait_list, 
                                 event_wait_list, 1, &buffer);
  if (errcode != CL_SUCCESS)
    return errcode;

  cmd->command.write.src_host_ptr = ptr;
  cmd->command.write.dst_mem_id = &buffer->device_ptrs[device->dev_id];
  cmd->command.write.offset = offset;
  cmd->command.write.size = size;

  POname(clRetainMemObject) (buffer);
  buffer->owning_device = command_queue->device;

  pocl_command_enqueue (command_queue, cmd);

  if (blocking_write)
    POname(clFinish) (command_queue);

  return CL_SUCCESS;
}
POsym(clEnqueueWriteBuffer)
