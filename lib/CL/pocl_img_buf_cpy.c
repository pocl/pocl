/* pocl_img_buf_cpy.c: common parts of image and buffer copying

   Copyright (c) 2011 Universidad Rey Juan Carlos
   Copyright (c) 2015 Giuseppe Bilotta
   
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

#include <assert.h>

#include "pocl_cl.h"
#include "pocl_image_util.h"
#include "pocl_mem_management.h"
#include "pocl_util.h"

/* Copies between images and rectangular buffer copies share most of the code,
   with specializations only needed for specific checks. The actual API calls
   thus defer to this function, with the additional information of which of src
   and/or dst is an image and which is a buffer.
 */

cl_int
pocl_validate_rect_copy (cl_command_queue command_queue,
                         cl_command_type command_type, cl_mem src,
                         cl_int src_is_image, cl_mem dst, cl_int dst_is_image,
                         const size_t *src_origin, const size_t *dst_origin,
                         const size_t *region, size_t *src_row_pitch,
                         size_t *src_slice_pitch, size_t *dst_row_pitch,
                         size_t *dst_slice_pitch)
{
  cl_int errcode;

  POCL_RETURN_ERROR_ON (
      ((command_queue->context != src->context)
       || (command_queue->context != dst->context)),
      CL_INVALID_CONTEXT,
      "src, dst and command_queue are not from the same context\n");

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (src)), CL_INVALID_MEM_OBJECT);
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (dst)), CL_INVALID_MEM_OBJECT);
  POCL_RETURN_ERROR_COND((src_origin == NULL), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND((dst_origin == NULL), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND((region == NULL), CL_INVALID_VALUE);

  if (src_is_image)
    {
      POCL_RETURN_ERROR_ON ((!src->is_image), CL_INVALID_MEM_OBJECT,
                            "src is not an image\n");
      POCL_RETURN_ERROR_ON ((src->is_gl_texture), CL_INVALID_MEM_OBJECT,
                            "src is a GL texture\n");
      POCL_RETURN_ON_UNSUPPORTED_IMAGE (src, command_queue->device);
      POCL_RETURN_ERROR_ON((src->type == CL_MEM_OBJECT_IMAGE2D && src_origin[2] != 0),
        CL_INVALID_VALUE, "src_origin[2] must be 0 for 2D src_image\n");
    }
  else
    {
      POCL_RETURN_ERROR_ON((src->type != CL_MEM_OBJECT_BUFFER),
        CL_INVALID_MEM_OBJECT, "src is not a CL_MEM_OBJECT_BUFFER\n");
      POCL_RETURN_ON_SUB_MISALIGN (src, command_queue);
      POCL_RETURN_ERROR_ON((src->size > command_queue->device->max_mem_alloc_size),
                           CL_OUT_OF_RESOURCES,
                           "src is larger than device's MAX_MEM_ALLOC_SIZE\n");
    }

  if (dst_is_image)
    {
      POCL_RETURN_ERROR_ON ((!dst->is_image), CL_INVALID_MEM_OBJECT,
                            "dst is not an image\n");
      POCL_RETURN_ERROR_ON ((dst->is_gl_texture), CL_INVALID_MEM_OBJECT,
                            "dst is a GL texture\n");
      POCL_RETURN_ON_UNSUPPORTED_IMAGE (dst, command_queue->device);
      POCL_RETURN_ERROR_ON((dst->type == CL_MEM_OBJECT_IMAGE2D && dst_origin[2] != 0),
        CL_INVALID_VALUE, "dst_origin[2] must be 0 for 2D dst_image\n");
    }
  else
    {
      POCL_RETURN_ERROR_ON((dst->type != CL_MEM_OBJECT_BUFFER),
        CL_INVALID_MEM_OBJECT, "dst is not a CL_MEM_OBJECT_BUFFER\n");
      POCL_RETURN_ON_SUB_MISALIGN (dst, command_queue);
      POCL_RETURN_ERROR_ON((dst->size > command_queue->device->max_mem_alloc_size),
                           CL_OUT_OF_RESOURCES,
                           "dst is larger than device's MAX_MEM_ALLOC_SIZE\n");
    }

  if (src_is_image && dst_is_image)
    {
      POCL_RETURN_ERROR_ON((src->image_channel_order != dst->image_channel_order),
        CL_IMAGE_FORMAT_MISMATCH, "src and dst have different image channel order\n");
      POCL_RETURN_ERROR_ON((src->image_channel_data_type != dst->image_channel_data_type),
        CL_IMAGE_FORMAT_MISMATCH, "src and dst have different image channel data type\n");
      POCL_RETURN_ERROR_ON((
          (dst->type == CL_MEM_OBJECT_IMAGE2D || src->type == CL_MEM_OBJECT_IMAGE2D) &&
          region[2] != 1),
        CL_INVALID_VALUE, "for any 2D image copy, region[2] must be 1\n");
   }

  /* Images need to recompute the regions in bytes for checking */
  size_t mod_region[3], mod_src_origin[3], mod_dst_origin[3];
  memcpy(mod_region, region, 3*sizeof(size_t));
  memcpy(mod_src_origin, src_origin, 3*sizeof(size_t));
  memcpy(mod_dst_origin, dst_origin, 3*sizeof(size_t));

  /* NOTE: 1D image array has row_pitch == slice_pitch;
   * need to zero it for bufferbound checks.
   */
  if (src_is_image)
    {
      mod_region[0] *= src->image_elem_size * src->image_channels;
      mod_src_origin[0] *= src->image_elem_size * src->image_channels;
      *src_row_pitch = src->image_row_pitch;
      if (src->type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
        *src_slice_pitch = 0;
      else
        *src_slice_pitch = src->image_slice_pitch;
    }
  if (dst_is_image)
    {
      if (!src_is_image)
        mod_region[0] *= dst->image_elem_size * dst->image_channels;
      mod_dst_origin[0] *= dst->image_elem_size * dst->image_channels;
      *dst_row_pitch = dst->image_row_pitch;
      if (dst->type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
        *dst_slice_pitch = 0;
      else
        *dst_slice_pitch = dst->image_slice_pitch;
    }

  if (pocl_buffer_boundcheck_3d (src->size, mod_src_origin, mod_region,
                                 src_row_pitch, src_slice_pitch, "src_")
      != CL_SUCCESS)
    return CL_INVALID_VALUE;

  if (pocl_buffer_boundcheck_3d (dst->size, mod_dst_origin, mod_region,
                                 dst_row_pitch, dst_slice_pitch, "dst_")
      != CL_SUCCESS)
    return CL_INVALID_VALUE;

  if (src == dst)
    {
      POCL_RETURN_ERROR_ON((*src_slice_pitch != *dst_slice_pitch),
        CL_INVALID_VALUE, "src and dst are the same object,"
        " but the given dst & src slice pitch differ\n");

      POCL_RETURN_ERROR_ON((*src_row_pitch != *dst_row_pitch),
        CL_INVALID_VALUE, "src and dst are the same object,"
        " but the given dst & src row pitch differ\n");
      // TODO
      POCL_RETURN_ERROR_ON (
          (check_copy_overlap (mod_src_origin, mod_dst_origin, mod_region,
                               *src_row_pitch, *src_slice_pitch)),
          CL_MEM_COPY_OVERLAP,
          "src and dst are the same object,"
          "and source and destination regions overlap\n");
    }

  return CL_SUCCESS;
}

cl_int
pocl_rect_copy (cl_command_buffer_khr command_buffer,
                cl_command_queue command_queue, cl_command_type command_type,
                cl_mem src, cl_int src_is_image, cl_mem dst,
                cl_int dst_is_image, const size_t *src_origin,
                const size_t *dst_origin, const size_t *region,
                size_t *src_row_pitch, size_t *src_slice_pitch,
                size_t *dst_row_pitch, size_t *dst_slice_pitch,
                cl_uint num_items_in_wait_list,
                const cl_event *event_wait_list, cl_event *event,
                const cl_sync_point_khr *sync_point_wait_list,
                cl_sync_point_khr *sync_point, _cl_command_node **cmd)
{
  cl_int errcode;
  cl_mem buffers[2] = { src, dst };

  if (command_buffer == NULL)
    {
      assert (sync_point_wait_list == NULL);
      POCL_RETURN_ERROR_COND (
          (event_wait_list == NULL && num_items_in_wait_list > 0),
          CL_INVALID_EVENT_WAIT_LIST);
      POCL_RETURN_ERROR_COND (
          (event_wait_list != NULL && num_items_in_wait_list == 0),
          CL_INVALID_EVENT_WAIT_LIST);
    }
  else
    {
      assert (event_wait_list == NULL && event == NULL);
      /* sync point wait list is validated in pocl_create_recorded_command */
    }

  unsigned i;
  cl_device_id device;
  POCL_CHECK_DEV_IN_CMDQ;

  errcode = pocl_validate_rect_copy (
      command_queue, command_type, src, src_is_image, dst, dst_is_image,
      src_origin, dst_origin, region, src_row_pitch, src_slice_pitch,
      dst_row_pitch, dst_slice_pitch);

  if (errcode != CL_SUCCESS)
    return errcode;

  char rdonly[] = { 1, 0 };

  if (command_buffer == NULL)
    {
      errcode = pocl_check_event_wait_list (
          command_queue, num_items_in_wait_list, event_wait_list);
      if (errcode != CL_SUCCESS)
        return errcode;

      errcode = pocl_create_command (cmd, command_queue, command_type, event,
                                     num_items_in_wait_list, event_wait_list,
                                     2, buffers, rdonly);
    }
  else
    {
      errcode = pocl_create_recorded_command (
          cmd, command_buffer, command_queue, command_type,
          num_items_in_wait_list, sync_point_wait_list, 2, buffers, rdonly);
    }

  return errcode;
}

cl_int
pocl_validate_copy_buffer (cl_command_queue command_queue, cl_mem src_buffer,
                           cl_mem dst_buffer, size_t src_offset,
                           size_t dst_offset, size_t size)
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

/* Validate parameters of cl*CopyImage* that do not get checked by
 * pocl_rect_copy */
cl_int
pocl_validate_copy_image (cl_mem src_image, cl_mem dst_image)
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (src_image)),
                          CL_INVALID_MEM_OBJECT);

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (dst_image)),
                          CL_INVALID_MEM_OBJECT);

  /* src_image, dst_image: Can be 1D, 2D, 3D image or a 1D, 2D image array
   * objects allowing us to perform the following actions */
  POCL_RETURN_ERROR_ON (
      (IS_IMAGE1D_BUFFER (src_image) || IS_IMAGE1D_BUFFER (dst_image)),
      CL_INVALID_MEM_OBJECT,
      "clEnqueueCopyImage cannot be called on image 1D buffers!\n");

  return CL_SUCCESS;
}

cl_int
pocl_copy_buffer_common (cl_command_buffer_khr command_buffer,
                         cl_command_queue command_queue, cl_mem src_buffer,
                         cl_mem dst_buffer, size_t src_offset,
                         size_t dst_offset, size_t size,
                         cl_uint num_items_in_wait_list,
                         const cl_event *event_wait_list, cl_event *event,
                         const cl_sync_point_khr *sync_point_wait_list,
                         cl_sync_point_khr *sync_point, _cl_command_node **cmd)
{
  if (command_buffer == NULL)
    {
      assert (sync_point_wait_list == NULL);
      POCL_RETURN_ERROR_COND (
          (event_wait_list == NULL && num_items_in_wait_list > 0),
          CL_INVALID_EVENT_WAIT_LIST);
      POCL_RETURN_ERROR_COND (
          (event_wait_list != NULL && num_items_in_wait_list == 0),
          CL_INVALID_EVENT_WAIT_LIST);
    }
  else
    {
      assert (event_wait_list == NULL && event == NULL);
      /* sync point wait list is validated in pocl_create_recorded_command */
    }

  unsigned i;
  cl_device_id device;
  POCL_CHECK_DEV_IN_CMDQ;

  cl_int errcode = pocl_validate_copy_buffer (
      command_queue, src_buffer, dst_buffer, src_offset, dst_offset, size);
  if (errcode != CL_SUCCESS)
    return errcode;

  POCL_CONVERT_SUBBUFFER_OFFSET (src_buffer, src_offset);
  POCL_RETURN_ERROR_ON (
      (src_buffer->size > command_queue->device->max_mem_alloc_size),
      CL_OUT_OF_RESOURCES, "src is larger than device's MAX_MEM_ALLOC_SIZE\n");
  if (pocl_buffers_boundcheck (src_buffer, dst_buffer, src_offset, dst_offset,
                               size)
      != CL_SUCCESS)
    return CL_INVALID_VALUE;
  POCL_CONVERT_SUBBUFFER_OFFSET (dst_buffer, dst_offset);
  POCL_RETURN_ERROR_ON (
      (dst_buffer->size > command_queue->device->max_mem_alloc_size),
      CL_OUT_OF_RESOURCES, "src is larger than device's MAX_MEM_ALLOC_SIZE\n");
  if (pocl_buffers_overlap (src_buffer, dst_buffer, src_offset, dst_offset,
                            size)
      != CL_SUCCESS)
    return CL_MEM_COPY_OVERLAP;

  cl_mem buffers[3] = { src_buffer, dst_buffer, NULL };
  char rdonly[] = { 1, 0, 1 };
  int n_bufs = 2;
  if (src_buffer->size_buffer != NULL)
    {
      n_bufs = 3;
      buffers[2] = src_buffer->size_buffer;
    }

  if (command_buffer == NULL)
    {
      errcode = pocl_check_event_wait_list (
          command_queue, num_items_in_wait_list, event_wait_list);
      if (errcode != CL_SUCCESS)
        return errcode;
      errcode = pocl_create_command (
          cmd, command_queue, CL_COMMAND_COPY_BUFFER, event,
          num_items_in_wait_list, event_wait_list, n_bufs, buffers, rdonly);
    }
  else
    {
      errcode = pocl_create_recorded_command (
          cmd, command_buffer, command_queue, CL_COMMAND_COPY_BUFFER,
          num_items_in_wait_list, sync_point_wait_list, n_bufs, buffers,
          rdonly);
    }
  if (errcode != CL_SUCCESS)
    return errcode;

  _cl_command_node *c = *cmd;
  cl_device_id dev = command_queue->device;

  c->command.copy.src_mem_id = &src_buffer->device_ptrs[dev->global_mem_id];
  c->command.copy.src_offset = src_offset;
  c->command.copy.src = src_buffer;

  c->command.copy.dst_mem_id = &dst_buffer->device_ptrs[dev->global_mem_id];
  c->command.copy.dst_offset = dst_offset;
  c->command.copy.dst = dst_buffer;

  c->command.copy.size = size;
  if (src_buffer->size_buffer != ((void *)0))
    {
      c->command.copy.src_content_size = src_buffer->size_buffer;
      c->command.copy.src_content_size_mem_id
          = &src_buffer->size_buffer->device_ptrs[dev->dev_id];
    }

  return CL_SUCCESS;
}

cl_int
pocl_copy_buffer_rect_common (
    cl_command_buffer_khr command_buffer, cl_command_queue command_queue,
    cl_mem src_buffer, cl_mem dst_buffer, const size_t *src_origin,
    const size_t *dst_origin, const size_t *region, size_t src_row_pitch,
    size_t src_slice_pitch, size_t dst_row_pitch, size_t dst_slice_pitch,
    cl_uint num_items_in_wait_list, const cl_event *event_wait_list,
    cl_event *event, const cl_sync_point_khr *sync_point_wait_list,
    cl_sync_point_khr *sync_point, _cl_command_node **cmd)
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
  POCL_CONVERT_SUBBUFFER_OFFSET (src_buffer, src_offset);
  POCL_GOTO_ERROR_ON (
      (src_buffer->size > command_queue->device->max_mem_alloc_size),
      CL_OUT_OF_RESOURCES, "src is larger than device's MAX_MEM_ALLOC_SIZE\n");

  size_t dst_offset = 0;
  POCL_CONVERT_SUBBUFFER_OFFSET (dst_buffer, dst_offset);
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

cl_int
pocl_copy_buffer_to_image_common (
    cl_command_buffer_khr command_buffer, cl_command_queue command_queue,
    cl_mem src_buffer, cl_mem dst_image, size_t src_offset,
    const size_t *dst_origin, const size_t *region,
    cl_uint num_items_in_wait_list, const cl_event *event_wait_list,
    cl_event *event, const cl_sync_point_khr *sync_point_wait_list,
    cl_sync_point_khr *sync_point, cl_mutable_command_khr *mutable_handle,
    _cl_command_node **cmd)
{
  cl_int errcode;
  size_t src_row_pitch = 0, src_slice_pitch = 0, dst_row_pitch = 0,
         dst_slice_pitch = 0;

  /* pass src_origin through in a format pocl_record_rect_copy understands */
  const size_t src_origin[3] = { src_offset, 0, 0 };

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (src_buffer)),
                          CL_INVALID_MEM_OBJECT);

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (dst_image)),
                          CL_INVALID_MEM_OBJECT);

  if (IS_IMAGE1D_BUFFER (dst_image))
    {
      /* If src_image is a 1D image or 1D image buffer object, src_origin[1]
       * and src_origin[2] must be 0 If src_image is a 1D image or 1D image
       * buffer object, region[1] and region[2] must be 1. */
      IMAGE1D_ORIG_REG_TO_BYTES (dst_image, dst_origin, region);
      if (command_buffer == NULL)
        {
          return POname (clEnqueueCopyBufferRect) (
              command_queue, src_buffer, dst_image->buffer, src_origin,
              i1d_origin, i1d_region, dst_image->image_row_pitch, 0,
              dst_image->image_row_pitch, 0, num_items_in_wait_list,
              event_wait_list, event);
        }
      else
        {
          return POname (clCommandCopyBufferRectKHR) (
              command_buffer, command_queue, src_buffer, dst_image->buffer,
              src_origin, i1d_origin, i1d_region, dst_image->image_row_pitch,
              0, dst_image->image_row_pitch, 0, num_items_in_wait_list,
              sync_point_wait_list, sync_point, mutable_handle);
        }
    }

  POCL_RETURN_ON_SUB_MISALIGN (src_buffer, command_queue);

  errcode = pocl_rect_copy (
      command_buffer, command_queue, CL_COMMAND_COPY_BUFFER_TO_IMAGE,
      src_buffer, CL_FALSE, dst_image, CL_TRUE, src_origin, dst_origin, region,
      &src_row_pitch, &src_slice_pitch, &dst_row_pitch, &dst_slice_pitch,
      num_items_in_wait_list, event_wait_list, event, sync_point_wait_list,
      sync_point, cmd);

  if (errcode != CL_SUCCESS)
    return errcode;

  POCL_CONVERT_SUBBUFFER_OFFSET (src_buffer, src_offset);

  POCL_GOTO_ERROR_ON (
      (src_buffer->size > command_queue->device->max_mem_alloc_size),
      CL_OUT_OF_RESOURCES, "src is larger than device's MAX_MEM_ALLOC_SIZE\n");

  _cl_command_node *c = *cmd;
  cl_device_id dev = command_queue->device;
  c->command.write_image.dst_mem_id
      = &dst_image->device_ptrs[dev->global_mem_id];
  c->command.write_image.dst = dst_image;
  c->command.write_image.src_host_ptr = ((void *)0);
  c->command.write_image.src_mem_id
      = &src_buffer->device_ptrs[dev->global_mem_id];
  c->command.write_image.src = src_buffer;
  c->command.write_image.src_row_pitch = src_row_pitch;
  c->command.write_image.src_slice_pitch = src_slice_pitch;
  c->command.write_image.src_offset = src_offset;
  c->command.write_image.origin[0] = dst_origin[0];
  c->command.write_image.origin[1] = dst_origin[1];
  c->command.write_image.origin[2] = dst_origin[2];
  c->command.write_image.region[0] = region[0];
  c->command.write_image.region[1] = region[1];
  c->command.write_image.region[2] = region[2];

  return CL_SUCCESS;

ERROR:
  pocl_mem_manager_free_command (*cmd);
  return errcode;
}

cl_int
pocl_copy_image_to_buffer_common (
    cl_command_buffer_khr command_buffer, cl_command_queue command_queue,
    cl_mem src_image, cl_mem dst_buffer, const size_t *src_origin,
    const size_t *region, size_t dst_offset, cl_uint num_items_in_wait_list,
    const cl_event *event_wait_list, cl_event *event,
    const cl_sync_point_khr *sync_point_wait_list,
    cl_sync_point_khr *sync_point, cl_mutable_command_khr *mutable_handle,
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

cl_int
pocl_copy_image_common (cl_command_buffer_khr command_buffer,
                        cl_command_queue command_queue, cl_mem src_image,
                        cl_mem dst_image, const size_t *src_origin,
                        const size_t *dst_origin, const size_t *region,
                        cl_uint num_items_in_wait_list,
                        const cl_event *event_wait_list, cl_event *event,
                        const cl_sync_point_khr *sync_point_wait_list,
                        cl_sync_point_khr *sync_point, _cl_command_node **cmd)
{
  cl_int errcode;
  size_t src_row_pitch = 0, src_slice_pitch = 0, dst_row_pitch = 0,
         dst_slice_pitch = 0;
  errcode = pocl_validate_copy_image (src_image, dst_image);
  if (errcode != CL_SUCCESS)
    return errcode;

  errcode = pocl_rect_copy (
      command_buffer, command_queue, CL_COMMAND_COPY_IMAGE, src_image, CL_TRUE,
      dst_image, CL_TRUE, src_origin, dst_origin, region, &src_row_pitch,
      &src_slice_pitch, &dst_row_pitch, &dst_slice_pitch,
      num_items_in_wait_list, event_wait_list, event, sync_point_wait_list,
      sync_point, cmd);
  if (errcode != CL_SUCCESS)
    return errcode;

  _cl_command_node *c = *cmd;
  cl_device_id dev = command_queue->device;
  c->command.copy_image.src_mem_id
      = &src_image->device_ptrs[dev->global_mem_id];
  c->command.copy_image.src = src_image;
  c->command.copy_image.dst_mem_id
      = &dst_image->device_ptrs[dev->global_mem_id];
  c->command.copy_image.dst = dst_image;
  c->command.copy_image.src_origin[0] = src_origin[0];
  c->command.copy_image.src_origin[1] = src_origin[1];
  c->command.copy_image.src_origin[2] = src_origin[2];
  c->command.copy_image.dst_origin[0] = dst_origin[0];
  c->command.copy_image.dst_origin[1] = dst_origin[1];
  c->command.copy_image.dst_origin[2] = dst_origin[2];
  c->command.copy_image.region[0] = region[0];
  c->command.copy_image.region[1] = region[1];
  c->command.copy_image.region[2] = region[2];

  return CL_SUCCESS;
}
