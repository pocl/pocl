/* OpenCL runtime library: clCommandFillImageKHR()

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

#include <CL/cl_ext.h>

#include "pocl_cl.h"
#include "pocl_image_util.h"
#include "pocl_shared.h"
#include "pocl_util.h"

CL_API_ENTRY cl_int
POname (clCommandFillImageKHR) (
    cl_command_buffer_khr command_buffer, cl_command_queue command_queue,
    cl_mem image, const void *fill_color, const size_t *origin,
    const size_t *region, cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr *sync_point_wait_list,
    cl_sync_point_khr *sync_point,
    cl_mutable_command_khr *mutable_handle) CL_API_SUFFIX__VERSION_1_2
{
  cl_int errcode;
  _cl_recorded_command *cmd = NULL;

  CMDBUF_VALIDATE_COMMON_HANDLES;

  errcode = pocl_validate_fill_image (command_queue, image, fill_color, origin,
                                      region);
  if (errcode != CL_SUCCESS)
    return errcode;

  cl_uint4 fill_color_vec = *(const cl_uint4 *)fill_color;

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
      return POname (clCommandFillBufferKHR) (
          command_buffer, command_queue, image->buffer, fill_pattern, px,
          origin[0] * px, region[0] * px, num_sync_points_in_wait_list,
          sync_point_wait_list, sync_point, mutable_handle);
    }

  char rdonly = 0;
  errcode = pocl_create_recorded_command (
      &cmd, command_buffer, command_queue, CL_COMMAND_FILL_IMAGE,
      num_sync_points_in_wait_list, sync_point_wait_list, 1, &image, &rdonly);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  POCL_FILL_COMMAND_FILL_IMAGE;

  errcode = pocl_command_record (command_buffer, cmd, sync_point);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  POname (clRetainMemObject) (image);

  return CL_SUCCESS;

ERROR:
  pocl_free_recorded_command (cmd);
  return errcode;
}
POsym (clCommandFillImageKHR)
