/* OpenCL runtime library: clCommandCopyImageToBufferKHR()

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
#include "pocl_shared.h"
#include "pocl_util.h"

extern CL_API_ENTRY cl_int CL_API_CALL
POname (clCommandCopyImageToBufferKHR) (
    cl_command_buffer_khr command_buffer, cl_command_queue command_queue,
    cl_mem src_image, cl_mem dst_buffer, const size_t *src_origin,
    const size_t *region, size_t dst_offset,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr *sync_point_wait_list,
    cl_sync_point_khr *sync_point,
    cl_mutable_command_khr *mutable_handle) CL_API_SUFFIX__VERSION_1_2
{
  cl_int errcode;
  _cl_recorded_command *cmd = NULL;
  /* pass src_origin through in a format pocl_record_rect_copy understands */
  const size_t dst_origin[3] = { dst_offset, 0, 0 };

  CMDBUF_VALIDATE_COMMON_HANDLES;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (src_image)),
                          CL_INVALID_MEM_OBJECT);

  if (IS_IMAGE1D_BUFFER (src_image))
    {
      /* If src_image is a 1D image or 1D image buffer object, src_origin[1]
       * and src_origin[2] must be 0 If src_image is a 1D image or 1D image
       * buffer object, region[1] and region[2] must be 1. */
      IMAGE1D_ORIG_REG_TO_BYTES (src_image, src_origin, region);
      return POname (clCommandCopyBufferRectKHR) (
          command_buffer, command_queue, src_image->buffer, dst_buffer,
          i1d_origin, dst_origin, i1d_region, src_image->image_row_pitch, 0,
          src_image->image_row_pitch, 0, num_sync_points_in_wait_list,
          sync_point_wait_list, sync_point, mutable_handle);
    }

  POCL_RETURN_ON_SUB_MISALIGN (dst_buffer, command_queue);

  errcode = pocl_record_rect_copy (
      command_buffer->queues[0], CL_COMMAND_COPY_IMAGE_TO_BUFFER, src_image,
      CL_TRUE, dst_buffer, CL_FALSE, src_origin, dst_origin, region, 0, 0, 0,
      0, num_sync_points_in_wait_list, sync_point_wait_list, &cmd,
      command_buffer);

  if (errcode != CL_SUCCESS)
    goto ERROR;

  POCL_CONVERT_SUBBUFFER_OFFSET (dst_buffer, dst_offset);

  POCL_RETURN_ERROR_ON (
      (dst_buffer->size > command_queue->device->max_mem_alloc_size),
      CL_OUT_OF_RESOURCES, "src is larger than device's MAX_MEM_ALLOC_SIZE\n");

  POCL_FILL_COMMAND_COPY_IMAGE_TO_BUFFER;

  errcode = pocl_command_record (command_buffer, cmd, sync_point);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  return CL_SUCCESS;

ERROR:
  pocl_free_recorded_command (cmd);
  return errcode;
}
POsym (clCommandCopyImageToBufferKHR)
