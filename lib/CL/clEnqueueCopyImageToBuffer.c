/* OpenCL runtime library: clEnqueueCopyImageToBuffer()

   Copyright (c) 2011-2023 pocl developers

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

#include "pocl_image_util.h"
#include "pocl_mem_management.h"
#include "pocl_shared.h"
#include "pocl_util.h"

cl_int
pocl_copy_image_to_buffer_common (
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    cl_mem src_image,
    cl_mem dst_buffer,
    const size_t *src_origin,
    const size_t *region,
    size_t dst_offset,
    cl_uint num_items_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event,
    const cl_sync_point_khr *sync_point_wait_list,
    cl_sync_point_khr *sync_point,
    cl_mutable_command_khr *mutable_handle,
    _cl_command_node **cmd)
{
  cl_int errcode;
  const size_t dst_origin[3] = { dst_offset, 0, 0 };
  size_t src_row_pitch = 0, src_slice_pitch = 0, dst_row_pitch = 0,
         dst_slice_pitch = 0;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (src_image)),
                          CL_INVALID_MEM_OBJECT);

  if (IS_IMAGE1D_BUFFER (src_image))
    {
      /* If src_image is a 1D image or 1D image buffer object, src_origin[1]
       * and src_origin[2] must be 0 If src_image is a 1D image or 1D image
       * buffer object, region[1] and region[2] must be 1. */
      IMAGE1D_ORIG_REG_TO_BYTES (src_image, src_origin, region);
      if (command_buffer == NULL)
        {
          return POname (clEnqueueCopyBufferRect) (
              command_queue, src_image->buffer, dst_buffer, i1d_origin,
              dst_origin, i1d_region, src_image->image_row_pitch, 0,
              src_image->image_row_pitch, 0, num_items_in_wait_list,
              event_wait_list, event);
        }
      else
        {
          return POname (clCommandCopyBufferRectKHR) (
              command_buffer, command_queue, src_image->buffer, dst_buffer,
              i1d_origin, dst_origin, i1d_region, src_image->image_row_pitch,
              0, src_image->image_row_pitch, 0, num_items_in_wait_list,
              sync_point_wait_list, sync_point, mutable_handle);
        }
    }

  POCL_RETURN_ON_SUB_MISALIGN (dst_buffer, command_queue);

  errcode = pocl_rect_copy (
      command_buffer, command_queue, CL_COMMAND_COPY_IMAGE_TO_BUFFER,
      src_image, CL_TRUE, dst_buffer, CL_FALSE, src_origin, dst_origin, region,
      &src_row_pitch, &src_slice_pitch, &dst_row_pitch, &dst_slice_pitch,
      num_items_in_wait_list, event_wait_list, event, sync_point_wait_list,
      sync_point, cmd);

  if (errcode != CL_SUCCESS)
    return errcode;

  POCL_CONVERT_SUBBUFFER_OFFSET (dst_buffer, dst_offset);

  POCL_GOTO_ERROR_ON (
      (dst_buffer->size > command_queue->device->max_mem_alloc_size),
      CL_OUT_OF_RESOURCES, "src is larger than device's MAX_MEM_ALLOC_SIZE\n");

  _cl_command_node *c = *cmd;
  cl_device_id dev = command_queue->device;
  c->command.read_image.src_mem_id
      = &src_image->device_ptrs[dev->global_mem_id];
  c->command.read_image.src = src_image;
  c->command.read_image.dst_host_ptr = ((void *)0);
  c->command.read_image.dst = dst_buffer;
  c->command.read_image.dst_mem_id
      = &dst_buffer->device_ptrs[dev->global_mem_id];
  c->command.read_image.origin[0] = src_origin[0];
  c->command.read_image.origin[1] = src_origin[1];
  c->command.read_image.origin[2] = src_origin[2];
  c->command.read_image.region[0] = region[0];
  c->command.read_image.region[1] = region[1];
  c->command.read_image.region[2] = region[2];
  c->command.read_image.dst_row_pitch = dst_row_pitch;
  c->command.read_image.dst_slice_pitch = dst_slice_pitch;
  c->command.read_image.dst_offset = dst_offset;

  return CL_SUCCESS;

ERROR:
  pocl_mem_manager_free_command (*cmd);
  return errcode;
}

CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueCopyImageToBuffer)(cl_command_queue  command_queue ,
                           cl_mem            src_image ,
                           cl_mem            dst_buffer ,
                           const size_t *    src_origin ,
                           const size_t *    region ,
                           size_t            dst_offset ,
                           cl_uint           num_events_in_wait_list ,
                           const cl_event *  event_wait_list ,
                           cl_event *        event ) CL_API_SUFFIX__VERSION_1_0
{
  _cl_command_node *cmd = NULL;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_COND ((*(command_queue->device->available) == CL_FALSE),
                          CL_DEVICE_NOT_AVAILABLE);

  cl_int errcode = pocl_copy_image_to_buffer_common (
      NULL, command_queue, src_image, dst_buffer, src_origin, region,
      dst_offset, num_events_in_wait_list, event_wait_list, event, NULL, NULL,
      NULL, &cmd);
  if (errcode != CL_SUCCESS)
    return errcode;

  if (cmd)
    pocl_command_enqueue (command_queue, cmd);

  return CL_SUCCESS;
}
POsym(clEnqueueCopyImageToBuffer)


