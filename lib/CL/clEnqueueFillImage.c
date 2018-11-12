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

#include "pocl_util.h"
#include "pocl_image_util.h"
#include "cl_platform.h"
#include <string.h>

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
  int errcode = CL_SUCCESS;
  _cl_command_node *cmd = NULL;
  void *fill_pixel = NULL;

  POCL_RETURN_ERROR_COND((command_queue == NULL), CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_COND((image == NULL), CL_INVALID_MEM_OBJECT);
  POCL_RETURN_ERROR_COND((origin == NULL), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND((region == NULL), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND((fill_color == NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_ON((command_queue->context != image->context), CL_INVALID_CONTEXT,
      "image and command_queue are not from the same context\n");

  POCL_RETURN_ERROR_ON ((!image->is_image), CL_INVALID_MEM_OBJECT,
                        "image argument is not an image\n");
  POCL_RETURN_ON_UNSUPPORTED_IMAGE (image, command_queue->device);

  errcode = pocl_check_event_wait_list (command_queue, num_events_in_wait_list,
                                        event_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;

  errcode = pocl_check_image_origin_region (image, origin, region);
  if (errcode != CL_SUCCESS)
    return errcode;

  fill_pixel = malloc (4 * sizeof(int));
  if (fill_pixel == NULL)
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR_CLEAN;
    }

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
  pocl_write_pixel_zero (fill_pixel, fill_color, image->image_channel_order,
                         image->image_elem_size,
                         image->image_channel_data_type);

  size_t px = image->image_elem_size * image->image_channels;

  if (IS_IMAGE1D_BUFFER (image))
    {
      return POname (clEnqueueFillBuffer) (
          command_queue,
          image->buffer,
          fill_pixel, 16,
          origin[0] * px,
          region[0] * px,
          num_events_in_wait_list, event_wait_list, event);
    }

  errcode = pocl_create_command (&cmd, command_queue, CL_COMMAND_FILL_IMAGE,
                                 event, num_events_in_wait_list,
                                 event_wait_list, 1, &image);
  if (errcode != CL_SUCCESS)
    goto ERROR_CLEAN;

  cmd->command.fill_image.fill_pixel = fill_pixel;
  cmd->command.fill_image.pixel_size = px;

  cmd->command.fill_image.mem_id
      = &image->device_ptrs[command_queue->device->dev_id];

  cmd->command.fill_image.origin[0] = origin[0];
  cmd->command.fill_image.origin[1] = origin[1];
  cmd->command.fill_image.origin[2] = origin[2];
  cmd->command.fill_image.region[0] = region[0];
  cmd->command.fill_image.region[1] = region[1];
  cmd->command.fill_image.region[2] = region[2];

  POname(clRetainMemObject) (image);
  image->owning_device = command_queue->device;
  pocl_command_enqueue(command_queue, cmd);

  return errcode;
  
 ERROR_CLEAN:
  POCL_MEM_FREE(fill_pixel);
  return errcode;
}
POsym(clEnqueueFillImage)
