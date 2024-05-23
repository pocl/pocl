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

#include "pocl_cl.h"
#include "pocl_image_util.h"
#include "pocl_mem_management.h"
#include "pocl_shared.h"
#include "pocl_util.h"

/* Copies between images and rectangular buffer copies share most of the code,
   with specializations only needed for specific checks. The actual API calls
   thus defer to this function, with the additional information of which of src
   and/or dst is an image and which is a buffer.
 */

cl_int
pocl_validate_rect_copy (cl_command_queue command_queue,
                         cl_command_type command_type,
                         cl_mem src,
                         cl_int src_is_image,
                         cl_mem dst,
                         cl_int dst_is_image,
                         const size_t *src_origin,
                         const size_t *dst_origin,
                         const size_t *region,
                         size_t *src_row_pitch,
                         size_t *src_slice_pitch,
                         size_t *dst_row_pitch,
                         size_t *dst_slice_pitch,
                         size_t mod_region[3],
                         size_t mod_src_origin[3],
                         size_t mod_dst_origin[3])
{
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
      POCL_RETURN_ERROR_ON (((src->type == CL_MEM_OBJECT_IMAGE2D
                              || src->type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
                             && src_origin[2] != 0),
                            CL_INVALID_VALUE,
                            "src_origin[2] must be 0 for 2D src_image\n");
      POCL_RETURN_ERROR_ON (
        ((src->type == CL_MEM_OBJECT_IMAGE1D
          || src->type == CL_MEM_OBJECT_IMAGE1D_BUFFER)
         && (src_origin[2] != 0 || src_origin[1] != 0)),
        CL_INVALID_VALUE,
        "src_origin[2] & src_origin[1] must be 0 for 1D src_image\n");
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
      POCL_RETURN_ERROR_ON (((dst->type == CL_MEM_OBJECT_IMAGE2D
                              || dst->type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
                             && dst_origin[2] != 0),
                            CL_INVALID_VALUE,
                            "dst_origin[2] must be 0 for 2D dst_image\n");
      POCL_RETURN_ERROR_ON (
        ((dst->type == CL_MEM_OBJECT_IMAGE1D
          || dst->type == CL_MEM_OBJECT_IMAGE1D_BUFFER)
         && (dst_origin[2] != 0 || dst_origin[1] != 0)),
        CL_INVALID_VALUE,
        "dst_origin[2] & dst_origin[1] must be 0 for 1D dst_image\n");
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
      POCL_RETURN_ERROR_ON (((dst->type == CL_MEM_OBJECT_IMAGE2D
                              || src->type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
                             && region[2] != 1),
                            CL_INVALID_VALUE,
                            "for any 2D image copy, region[2] must be 1\n");
      POCL_RETURN_ERROR_ON (
        ((dst->type == CL_MEM_OBJECT_IMAGE1D
          || src->type == CL_MEM_OBJECT_IMAGE1D_BUFFER)
         && (region[2] != 1 || region[1] != 1)),
        CL_INVALID_VALUE,
        "for any 1D image copy, region[2] and region[1] must be 1\n");
   }

  /* Images need to recompute the regions in bytes for checking */
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
  unsigned i;
  cl_device_id device;
  *cmd = NULL;

  POCL_VALIDATE_WAIT_LIST_PARAMS;
  POCL_CHECK_DEV_IN_CMDQ;

  size_t mod_region[3], mod_src_origin[3], mod_dst_origin[3];
  errcode = pocl_validate_rect_copy (
    command_queue, command_type, src, src_is_image, dst, dst_is_image,
    src_origin, dst_origin, region, src_row_pitch, src_slice_pitch,
    dst_row_pitch, dst_slice_pitch, mod_region, mod_src_origin,
    mod_dst_origin);

  if (errcode != CL_SUCCESS)
    return errcode;

  if (IS_IMAGE1D_BUFFER (src) && IS_IMAGE1D_BUFFER (dst))
    {
      src = src->buffer;
      dst = dst->buffer;
      // TODO handle command buffer
      return POname (clEnqueueCopyBuffer) (
        command_queue, src, dst, mod_src_origin[0], mod_dst_origin[0],
        mod_region[0], num_items_in_wait_list, event_wait_list, event);
    }

  if (IS_IMAGE1D_BUFFER (src))
    {
      src = src->buffer;
      // TODO handle command buffer
      return POname (clEnqueueCopyBufferToImage) (
        command_queue, src, dst, mod_src_origin[0], dst_origin, region,
        num_items_in_wait_list, event_wait_list, event);
    }

  if (IS_IMAGE1D_BUFFER (dst))
    {
      dst = dst->buffer;
      // TODO handle command buffer
      return POname (clEnqueueCopyImageToBuffer) (
        command_queue, src, dst, src_origin, region, mod_dst_origin[0],
        num_items_in_wait_list, event_wait_list, event);
    }

  cl_mem buffers[3] = { src, dst, NULL };
  char rdonly[] = { 1, 0, 1 };
  int n_bufs = 2;
  if (src->size_buffer != NULL)
    {
      n_bufs = 3;
      buffers[2] = src->size_buffer;
    }

  if (command_buffer == NULL)
    {
      errcode = pocl_check_event_wait_list (
          command_queue, num_items_in_wait_list, event_wait_list);
      if (errcode != CL_SUCCESS)
        return errcode;

      errcode = pocl_create_command (cmd, command_queue, command_type, event,
                                     num_items_in_wait_list, event_wait_list,
                                     n_bufs, buffers, rdonly);
    }
  else
    {
      errcode = pocl_create_recorded_command (
          cmd, command_buffer, command_queue, command_type,
          num_items_in_wait_list, sync_point_wait_list, n_bufs, buffers,
          rdonly);
    }

  return errcode;
}

