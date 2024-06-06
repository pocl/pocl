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

cl_int
pocl_validate_copy_buffer (cl_command_queue command_queue,
                           cl_mem src_buffer,
                           cl_mem dst_buffer,
                           size_t src_offset,
                           size_t dst_offset,
                           size_t size)
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (src_buffer)),
                          CL_INVALID_MEM_OBJECT);

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (dst_buffer)),
                          CL_INVALID_MEM_OBJECT);

  POCL_RETURN_ERROR_ON ((src_buffer->type != CL_MEM_OBJECT_BUFFER),
                        CL_INVALID_MEM_OBJECT,
                        "src_buffer is not a CL_MEM_OBJECT_BUFFER\n");
  POCL_RETURN_ERROR_ON ((dst_buffer->type != CL_MEM_OBJECT_BUFFER),
                        CL_INVALID_MEM_OBJECT,
                        "dst_buffer is not a CL_MEM_OBJECT_BUFFER\n");

  POCL_RETURN_ERROR_ON (((command_queue->context != src_buffer->context)
                         || (command_queue->context != dst_buffer->context)),
                        CL_INVALID_CONTEXT,
                        "src_buffer, dst_buffer and command_queue are not "
                        "from the same context\n");

  POCL_RETURN_ON_SUB_MISALIGN (src_buffer, command_queue);

  POCL_RETURN_ON_SUB_MISALIGN (dst_buffer, command_queue);

  POCL_RETURN_ERROR_COND ((size == 0), CL_INVALID_VALUE);

  return CL_SUCCESS;
}

cl_int
pocl_copy_buffer_common (cl_command_buffer_khr command_buffer,
                         cl_command_queue command_queue,
                         cl_mem src_buffer,
                         cl_mem dst_buffer,
                         size_t src_offset,
                         size_t dst_offset,
                         size_t size,
                         cl_uint num_items_in_wait_list,
                         const cl_event *event_wait_list,
                         cl_event *event,
                         const cl_sync_point_khr *sync_point_wait_list,
                         cl_sync_point_khr *sync_point,
                         _cl_command_node **cmd)
{
  POCL_VALIDATE_WAIT_LIST_PARAMS;

  unsigned i;
  cl_device_id device;
  POCL_CHECK_DEV_IN_CMDQ;

  cl_int errcode = pocl_validate_copy_buffer (
      command_queue, src_buffer, dst_buffer, src_offset, dst_offset, size);
  if (errcode != CL_SUCCESS)
    return errcode;

  POCL_RETURN_ERROR_ON (
      (src_buffer->size > command_queue->device->max_mem_alloc_size),
      CL_OUT_OF_RESOURCES, "src is larger than device's MAX_MEM_ALLOC_SIZE\n");
  if (pocl_buffers_boundcheck (src_buffer, dst_buffer, src_offset, dst_offset,
                               size)
      != CL_SUCCESS)
    return CL_INVALID_VALUE;

  POCL_RETURN_ERROR_ON (
      (dst_buffer->size > command_queue->device->max_mem_alloc_size),
      CL_OUT_OF_RESOURCES, "src is larger than device's MAX_MEM_ALLOC_SIZE\n");
  if (pocl_buffers_overlap (src_buffer, dst_buffer, src_offset, dst_offset,
                            size)
      != CL_SUCCESS)
    return CL_MEM_COPY_OVERLAP;

  pocl_buffer_migration_info *migr_infos
    = pocl_append_unique_migration_info (NULL, src_buffer, 1);
  pocl_append_unique_migration_info (migr_infos, dst_buffer, 0);

  if (src_buffer->size_buffer != NULL)
    pocl_append_unique_migration_info (migr_infos, src_buffer->size_buffer, 1);

  if (command_buffer == NULL)
    {
      errcode = pocl_check_event_wait_list (
          command_queue, num_items_in_wait_list, event_wait_list);
      if (errcode != CL_SUCCESS)
        return errcode;
      errcode = pocl_create_command (
        cmd, command_queue, CL_COMMAND_COPY_BUFFER, event,
        num_items_in_wait_list, event_wait_list, migr_infos);
    }
  else
    {
      errcode = pocl_create_recorded_command (
        cmd, command_buffer, command_queue, CL_COMMAND_COPY_BUFFER,
        num_items_in_wait_list, sync_point_wait_list, migr_infos);
    }
  if (errcode != CL_SUCCESS)
    return errcode;

  _cl_command_node *c = *cmd;

  c->command.copy.src_offset = src_offset;
  c->command.copy.src = src_buffer;

  c->command.copy.dst_offset = dst_offset;
  c->command.copy.dst = dst_buffer;

  c->command.copy.size = size;
  if (src_buffer->size_buffer != ((void *)0))
    {
      c->command.copy.src_content_size = src_buffer->size_buffer;
      c->command.copy.src_content_size_mem_id
          = &src_buffer->size_buffer->device_ptrs[device->dev_id];
    }

  return CL_SUCCESS;
}

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

  POCL_RETURN_ERROR_COND ((*(command_queue->device->available) == CL_FALSE),
                          CL_DEVICE_NOT_AVAILABLE);

  errcode = pocl_copy_buffer_common (
      NULL, command_queue, src_buffer, dst_buffer, src_offset, dst_offset,
      size, num_events_in_wait_list, event_wait_list, event, NULL, NULL, &cmd);

  if (errcode != CL_SUCCESS)
    return errcode;

  pocl_command_enqueue (command_queue, cmd);

  return CL_SUCCESS;
}
POsym(clEnqueueCopyBuffer)
