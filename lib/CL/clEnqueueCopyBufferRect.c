/* OpenCL runtime library: clEnqueueCopyBufferRect()

   Copyright (c) 2011 Universidad Rey Juan Carlos

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

#include "pocl_cl.h"
#include "pocl_mem_management.h"
#include "pocl_shared.h"
#include "pocl_util.h"
#include <assert.h>

cl_int
pocl_copy_buffer_rect_common (cl_command_buffer_khr command_buffer,
                              cl_command_queue command_queue,
                              cl_mem src_buffer,
                              cl_mem dst_buffer,
                              const size_t *src_origin,
                              const size_t *dst_origin,
                              const size_t *region,
                              size_t src_row_pitch,
                              size_t src_slice_pitch,
                              size_t dst_row_pitch,
                              size_t dst_slice_pitch,
                              cl_uint num_items_in_wait_list,
                              const cl_event *event_wait_list,
                              cl_event *event,
                              const cl_sync_point_khr *sync_point_wait_list,
                              cl_sync_point_khr *sync_point,
                              _cl_command_node **cmd)
{
  cl_int errcode = pocl_rect_copy (
      command_buffer, command_queue, CL_COMMAND_COPY_BUFFER_RECT, src_buffer,
      CL_FALSE, dst_buffer, CL_FALSE, src_origin, dst_origin, region,
      &src_row_pitch, &src_slice_pitch, &dst_row_pitch, &dst_slice_pitch,
      num_items_in_wait_list, event_wait_list, event, sync_point_wait_list,
      sync_point, cmd);

  if (errcode != CL_SUCCESS)
    return errcode;

  size_t src_offset = 0;
  POCL_GOTO_ERROR_ON (
      (src_buffer->size > command_queue->device->max_mem_alloc_size),
      CL_OUT_OF_RESOURCES, "src is larger than device's MAX_MEM_ALLOC_SIZE\n");

  size_t dst_offset = 0;
  POCL_GOTO_ERROR_ON (
      (dst_buffer->size > command_queue->device->max_mem_alloc_size),
      CL_OUT_OF_RESOURCES, "dst is larger than device's MAX_MEM_ALLOC_SIZE\n");

  _cl_command_node *c = *cmd;
  cl_device_id dev = command_queue->device;

  c->command.copy_rect.src_mem_id
      = &src_buffer->device_ptrs[dev->global_mem_id];
  c->command.copy_rect.src = src_buffer;
  c->command.copy_rect.dst_mem_id
      = &dst_buffer->device_ptrs[dev->global_mem_id];
  c->command.copy_rect.dst = dst_buffer;

  c->command.copy_rect.src_origin[0] = src_offset + src_origin[0];
  c->command.copy_rect.src_origin[1] = src_origin[1];
  c->command.copy_rect.src_origin[2] = src_origin[2];
  c->command.copy_rect.dst_origin[0] = dst_offset + dst_origin[0];
  c->command.copy_rect.dst_origin[1] = dst_origin[1];
  c->command.copy_rect.dst_origin[2] = dst_origin[2];
  c->command.copy_rect.region[0] = region[0];
  c->command.copy_rect.region[1] = region[1];
  c->command.copy_rect.region[2] = region[2];

  c->command.copy_rect.src_row_pitch = src_row_pitch;
  c->command.copy_rect.src_slice_pitch = src_slice_pitch;
  c->command.copy_rect.dst_row_pitch = dst_row_pitch;
  c->command.copy_rect.dst_slice_pitch = dst_slice_pitch;

  return CL_SUCCESS;

ERROR:
  pocl_mem_manager_free_command (*cmd);
  return errcode;
}

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
  _cl_command_node *cmd = NULL;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_COND ((*(command_queue->device->available) == CL_FALSE),
                          CL_DEVICE_NOT_AVAILABLE);

  cl_int errcode = pocl_copy_buffer_rect_common (
      NULL, command_queue, src_buffer, dst_buffer, src_origin, dst_origin,
      region, src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch,
      num_events_in_wait_list, event_wait_list, event, NULL, NULL, &cmd);

  if (errcode != CL_SUCCESS)
    return errcode;

  pocl_command_enqueue (command_queue, cmd);

  return CL_SUCCESS;
}
POsym(clEnqueueCopyBufferRect)
