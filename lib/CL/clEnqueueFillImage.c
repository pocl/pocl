/* OpenCL runtime library: clEnqueueFillImage()

   Copyright (c) 2013 Ville Korhonen / Tampere Univ. of Tech.
   
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

#include <CL/cl.h>
#include <string.h>

#include "pocl_image_util.h"
#include "pocl_shared.h"
#include "pocl_util.h"

extern CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueFillImage)(cl_command_queue  command_queue,
                           cl_mem            image,
                           const void *      fill_color,
                           const size_t*     origin, 
                           const size_t *    region,
                           cl_uint           num_events_in_wait_list,
                           const cl_event*   event_wait_list,
                           cl_event*         event) 
CL_API_SUFFIX__VERSION_1_2
{
  int errcode;
  _cl_command_node *cmd = NULL;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  errcode = pocl_check_event_wait_list (command_queue, num_events_in_wait_list,
                                        event_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;

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
      return POname (clEnqueueFillBuffer) (
          command_queue, image->buffer, fill_pattern, px, origin[0] * px,
          region[0] * px, num_events_in_wait_list, event_wait_list, event);
    }

  char rdonly = 0;
  errcode = pocl_create_command (&cmd, command_queue, CL_COMMAND_FILL_IMAGE,
                                 event, num_events_in_wait_list,
                                 event_wait_list, 1, &image, &rdonly);
  if (errcode != CL_SUCCESS)
    return errcode;

  POCL_FILL_COMMAND_FILL_IMAGE;

  pocl_command_enqueue(command_queue, cmd);

  return CL_SUCCESS;
}
POsym(clEnqueueFillImage)
