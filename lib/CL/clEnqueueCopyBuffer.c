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

#include "utlist.h"
#include <assert.h>
#include "pocl_util.h"

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
  cl_device_id device;
  unsigned i;
  _cl_command_node *cmd = NULL;
  int errcode;

  POCL_RETURN_ERROR_COND((command_queue == NULL), CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_COND((src_buffer == NULL), CL_INVALID_MEM_OBJECT);

  POCL_RETURN_ERROR_COND((dst_buffer == NULL), CL_INVALID_MEM_OBJECT);

  POCL_RETURN_ERROR_ON((src_buffer->type != CL_MEM_OBJECT_BUFFER),
      CL_INVALID_MEM_OBJECT, "src_buffer is not a CL_MEM_OBJECT_BUFFER\n");
  POCL_RETURN_ERROR_ON((dst_buffer->type != CL_MEM_OBJECT_BUFFER),
      CL_INVALID_MEM_OBJECT, "dst_buffer is not a CL_MEM_OBJECT_BUFFER\n");

  POCL_RETURN_ON_SUB_MISALIGN (src_buffer, command_queue);

  POCL_RETURN_ON_SUB_MISALIGN (dst_buffer, command_queue);

  POCL_CONVERT_SUBBUFFER_OFFSET (src_buffer, src_offset);

  POCL_CONVERT_SUBBUFFER_OFFSET (dst_buffer, dst_offset);

  POCL_RETURN_ERROR_ON((src_buffer->size > command_queue->device->max_mem_alloc_size),
                        CL_OUT_OF_RESOURCES,
                        "src is larger than device's MAX_MEM_ALLOC_SIZE\n");

  POCL_RETURN_ERROR_ON((dst_buffer->size > command_queue->device->max_mem_alloc_size),
                        CL_OUT_OF_RESOURCES,
                        "src is larger than device's MAX_MEM_ALLOC_SIZE\n");

  POCL_RETURN_ERROR_ON(((command_queue->context != src_buffer->context) ||
      (command_queue->context != dst_buffer->context)), CL_INVALID_CONTEXT,
      "src_buffer, dst_buffer and command_queue are not from the same context\n");

  POCL_RETURN_ERROR_COND((size == 0), CL_INVALID_VALUE);

  errcode = pocl_check_event_wait_list (command_queue, num_events_in_wait_list,
                                        event_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;

  if (pocl_buffers_boundcheck(src_buffer, dst_buffer, src_offset,
        dst_offset, size) != CL_SUCCESS) return CL_INVALID_VALUE;

  if (pocl_buffers_overlap(src_buffer, dst_buffer, src_offset,
        dst_offset, size) != CL_SUCCESS) return CL_MEM_COPY_OVERLAP;

  POCL_CHECK_DEV_IN_CMDQ;

  cl_mem buffers[2] = { src_buffer, dst_buffer };

  errcode = pocl_create_command (&cmd, command_queue, CL_COMMAND_COPY_BUFFER, 
                                 event, num_events_in_wait_list, 
                                 event_wait_list, 2, buffers);
  if (errcode != CL_SUCCESS)
    return errcode;

  cmd->command.copy.src_mem_id = &src_buffer->device_ptrs[device->dev_id];
  cmd->command.copy.src_offset = src_offset;

  cmd->command.copy.dst_mem_id = &dst_buffer->device_ptrs[device->dev_id];
  cmd->command.copy.dst_offset = dst_offset;
  cmd->command.copy.size = size;

  POname(clRetainMemObject)(src_buffer);
  src_buffer->owning_device = command_queue->device;
  POname(clRetainMemObject)(dst_buffer);
  dst_buffer->owning_device = command_queue->device;
  
  pocl_command_enqueue(command_queue, cmd);

  return CL_SUCCESS;
}
POsym(clEnqueueCopyBuffer)
