/* pocl_fill_memobj.c: helpers for FillBuffer and FillImage commands

   Copyright (c) 2022 Jan Solanti / Tampere University

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
#include "pocl_image_util.h"
#include "pocl_util.h"
#include "pocl_shared.h"

cl_int
pocl_validate_fill_buffer (cl_command_queue command_queue, cl_mem buffer,
                           const void *pattern, size_t pattern_size,
                           size_t offset, size_t size)
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (buffer)),
                          CL_INVALID_MEM_OBJECT);

  POCL_RETURN_ERROR_ON ((buffer->type != CL_MEM_OBJECT_BUFFER),
                        CL_INVALID_MEM_OBJECT,
                        "buffer is not a CL_MEM_OBJECT_BUFFER\n");

  POCL_RETURN_ERROR_ON (
      (command_queue->context != buffer->context), CL_INVALID_CONTEXT,
      "buffer and command_queue are not from the same context\n");

  cl_int errcode = pocl_buffer_boundcheck (buffer, offset, size);
  if (errcode != CL_SUCCESS)
    return errcode;

  /* CL_INVALID_VALUE if pattern is NULL or if pattern_size is 0
   * or if pattern_size is not one of {1, 2, 4, 8, 16, 32, 64, 128}. */
  POCL_RETURN_ERROR_COND ((pattern == NULL), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND ((pattern_size == 0), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND ((pattern_size > 128), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_ON (
      (__builtin_popcount (pattern_size) > 1), CL_INVALID_VALUE,
      "pattern_size(%zu) must be a power-of-two value", pattern_size);

  /* CL_INVALID_VALUE if offset and size are not a multiple of pattern_size. */
  POCL_RETURN_ERROR_ON (
      (offset % pattern_size), CL_INVALID_VALUE,
      "offset(%zu) must be a multiple of pattern_size(%zu)\n", offset,
      pattern_size);
  POCL_RETURN_ERROR_ON ((size % pattern_size), CL_INVALID_VALUE,
                        "size(%zu) must be a multiple of pattern_size(%zu)\n",
                        size, pattern_size);

  POCL_RETURN_ON_SUB_MISALIGN (buffer, command_queue);

  return CL_SUCCESS;
}

cl_int
pocl_fill_buffer_common (cl_command_buffer_khr command_buffer,
                         cl_command_queue command_queue, cl_mem buffer,
                         const void *pattern, size_t pattern_size,
                         size_t offset, size_t size,
                         cl_uint num_items_in_wait_list,
                         const cl_event *event_wait_list, cl_event *event,
                         const cl_sync_point_khr *sync_point_wait_list,
                         cl_sync_point_khr *sync_point, _cl_command_node **cmd)
{
  cl_int errcode;
  POCL_VALIDATE_WAIT_LIST_PARAMS;

  errcode = pocl_validate_fill_buffer (command_queue, buffer, pattern,
                                       pattern_size, offset, size);
  if (errcode != CL_SUCCESS)
    return errcode;

  POCL_RETURN_ERROR_ON (
      (buffer->size > command_queue->device->max_mem_alloc_size),
      CL_OUT_OF_RESOURCES,
      "buffer is larger than device's MAX_MEM_ALLOC_SIZE\n");

  char rdonly = 0;
  if (command_buffer == NULL)
    {
      errcode = pocl_check_event_wait_list (
          command_queue, num_items_in_wait_list, event_wait_list);
      if (errcode != CL_SUCCESS)
        return errcode;
      errcode = pocl_create_command (
        cmd, command_queue, CL_COMMAND_FILL_BUFFER, event,
        num_items_in_wait_list, event_wait_list,
        pocl_append_unique_migration_info (NULL, buffer, rdonly));
    }
  else
    {
      errcode = pocl_create_recorded_command (
        cmd, command_buffer, command_queue, CL_COMMAND_FILL_BUFFER,
        num_items_in_wait_list, sync_point_wait_list,
        pocl_append_unique_migration_info (NULL, buffer, rdonly));
    }
  if (errcode != CL_SUCCESS)
    return errcode;

  _cl_command_node *c = *cmd;
  c->command.memfill.dst_mem_id
      = &buffer->device_ptrs[command_queue->device->global_mem_id];
  c->command.memfill.size = size;
  c->command.memfill.offset = offset;
  void *p = pocl_aligned_malloc (pattern_size, pattern_size);
  memcpy (p, pattern, pattern_size);
  c->command.memfill.pattern = p;
  c->command.memfill.pattern_size = pattern_size;
  c->command.memfill.dst = buffer;

  return CL_SUCCESS;
}

cl_int
pocl_validate_fill_image (cl_command_queue command_queue, cl_mem image,
                          const void *fill_color, const size_t *origin,
                          const size_t *region)
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (image)),
                          CL_INVALID_MEM_OBJECT);
  POCL_RETURN_ERROR_COND ((origin == NULL), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND ((region == NULL), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND ((fill_color == NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_ON (
      (command_queue->context != image->context), CL_INVALID_CONTEXT,
      "image and command_queue are not from the same context\n");

  POCL_RETURN_ERROR_ON ((!image->is_image), CL_INVALID_MEM_OBJECT,
                        "image argument is not an image\n");
  POCL_RETURN_ERROR_ON ((image->is_gl_texture), CL_INVALID_MEM_OBJECT,
                        "image is a GL texture\n");
  POCL_RETURN_ON_UNSUPPORTED_IMAGE (image, command_queue->device);

  return pocl_check_image_origin_region (image, origin, region);
}

cl_int
pocl_fill_image_common (cl_command_buffer_khr command_buffer,
                        cl_command_queue command_queue, cl_mem image,
                        const void *fill_color, const size_t *origin,
                        const size_t *region, cl_uint num_items_in_wait_list,
                        const cl_event *event_wait_list, cl_event *event,
                        const cl_sync_point_khr *sync_point_wait_list,
                        cl_sync_point_khr *sync_point,
                        cl_mutable_command_khr *mutable_handle,
                        _cl_command_node **cmd)
{
  cl_int errcode;

  POCL_VALIDATE_WAIT_LIST_PARAMS;

  errcode = pocl_validate_fill_image (command_queue, image, fill_color, origin,
                                      region);
  if (errcode != CL_SUCCESS)
    return errcode;

  cl_uint4 fill_color_vec;
  memcpy(&fill_color_vec, fill_color, 16);

  size_t px = image->image_elem_size * image->image_channels;
  char fill_pattern[16];
  pocl_write_pixel_zero (fill_pattern, fill_color_vec,
                         image->image_channel_order, image->image_elem_size,
                         image->image_channel_data_type);

  /* The fill color is:
   *
   * a four component RGBA floating-point color value if the image channel
   * data type is NOT an unnormalized signed and unsigned integer type,
   *
   * a four component signed integer value if the image channel data type
   * is an unnormalized signed integer type and
   *
   * a four component unsigned integer value if the image channel data type
   * is an unormalized unsigned integer type.
   *
   * The fill color will be converted to the appropriate
   * image channel format and order associated with image.
   */

  if (IS_IMAGE1D_BUFFER (image))
    {
      if (command_buffer == NULL)
        {
          return POname (clEnqueueFillBuffer) (
              command_queue, image->buffer, fill_pattern, px, origin[0] * px,
              region[0] * px, num_items_in_wait_list, event_wait_list, event);
        }
      else
        {
          return POname (clCommandFillBufferKHR) (
              command_buffer, command_queue, image->buffer, fill_pattern, px,
              origin[0] * px, region[0] * px, num_items_in_wait_list,
              sync_point_wait_list, sync_point, mutable_handle);
        }
    }

  char rdonly = 0;
  if (command_buffer == NULL)
    {
      errcode = pocl_check_event_wait_list (
          command_queue, num_items_in_wait_list, event_wait_list);
      if (errcode != CL_SUCCESS)
        return errcode;
      errcode = pocl_create_command (
        cmd, command_queue, CL_COMMAND_FILL_IMAGE, event,
        num_items_in_wait_list, event_wait_list,
        pocl_append_unique_migration_info (NULL, image, rdonly));
    }
  else
    {
      errcode = pocl_create_recorded_command (
        cmd, command_buffer, command_queue, CL_COMMAND_FILL_IMAGE,
        num_items_in_wait_list, sync_point_wait_list,
        pocl_append_unique_migration_info (NULL, image, rdonly));
    }
  if (errcode != CL_SUCCESS)
    return errcode;

  _cl_command_node *c = *cmd;
  memcpy (c->command.fill_image.fill_pixel, fill_pattern, 16);
  c->command.fill_image.orig_pixel = fill_color_vec;
  c->command.fill_image.pixel_size = px;
  c->command.fill_image.mem_id
      = &image->device_ptrs[command_queue->device->global_mem_id];
  c->command.fill_image.origin[0] = origin[0];
  c->command.fill_image.origin[1] = origin[1];
  c->command.fill_image.origin[2] = origin[2];
  c->command.fill_image.region[0] = region[0];
  c->command.fill_image.region[1] = region[1];
  c->command.fill_image.region[2] = region[2];
  c->command.fill_image.dst = image;

  return CL_SUCCESS;
}
