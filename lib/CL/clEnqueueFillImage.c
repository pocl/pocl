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
  cl_image_format *supported_image_formats = NULL;
  void *fill_pixel = NULL;
/*  size_t tuned_origin[3]; */

  POCL_RETURN_ERROR_COND((command_queue == NULL), CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_ON((!command_queue->device->image_support), CL_INVALID_OPERATION,
    "Device %s does not support images\n", command_queue->device->long_name);

  POCL_RETURN_ERROR_COND((image == NULL), CL_INVALID_MEM_OBJECT);
  POCL_RETURN_ERROR_COND((origin == NULL), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND((region == NULL), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND((fill_color == NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_ON((command_queue->context != image->context), CL_INVALID_CONTEXT,
      "image and command_queue are not from the same context\n");

  POCL_RETURN_ERROR_ON((!image->is_image), CL_INVALID_MEM_OBJECT,
                                                "image argument is not an image\n");

  errcode = pocl_check_event_wait_list (command_queue, num_events_in_wait_list,
                                        event_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;

  errcode = pocl_check_image_origin_region (image, origin, region);
  if (errcode != CL_SUCCESS)
    return errcode;

  errcode = pocl_check_device_supports_image(image, command_queue);
  if (errcode != CL_SUCCESS)
    return errcode;

  fill_pixel = malloc (4 * sizeof(int));
  if (fill_pixel == NULL)
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR_CLEAN;
    }

  /* TODO: channel order, saturating data type conversion */
  if (image->image_elem_size == 1)
    {
      ((cl_char4*)fill_pixel)->s[0] = ((cl_int4*)fill_color)->s[0];
      ((cl_char4*)fill_pixel)->s[1] = ((cl_int4*)fill_color)->s[1];
      ((cl_char4*)fill_pixel)->s[2] = ((cl_int4*)fill_color)->s[2];
      ((cl_char4*)fill_pixel)->s[3] = ((cl_int4*)fill_color)->s[3];
    }
  if (image->image_elem_size == 2)
    {
      ((cl_short4*)fill_pixel)->s[0] = ((cl_int4*)fill_color)->s[0];
      ((cl_short4*)fill_pixel)->s[1] = ((cl_int4*)fill_color)->s[1];
      ((cl_short4*)fill_pixel)->s[2] = ((cl_int4*)fill_color)->s[2];
      ((cl_short4*)fill_pixel)->s[3] = ((cl_int4*)fill_color)->s[3];
    }
 if (image->image_elem_size == 4)
    {
      memcpy (fill_pixel, fill_color, sizeof (cl_int4));      
    }

  /* POCL uses top-left corner as origin for images and AMD SDK ImageOverlap 
     test uses bottom-left corner as origin. Because of this we need to modify 
     y-coordinate so the fill goes in the right place.
  tuned_origin[0] = origin[0];
  tuned_origin[1] = image->image_height - region[1] - origin[1];
  tuned_origin[2] = origin[2];
  */
  errcode = pocl_create_command (&cmd, command_queue, CL_COMMAND_FILL_IMAGE, 
                                 event, num_events_in_wait_list, 
                                 event_wait_list, 1, &image);
  if (errcode != CL_SUCCESS)
    goto ERROR_CLEAN;

  cmd->command.fill_image.data = command_queue->device->data;
  cmd->command.fill_image.device_ptr = 
    image->device_ptrs[command_queue->device->dev_id].mem_ptr;
  memcpy (&(cmd->command.fill_image.buffer_origin), origin, 
          3*sizeof(size_t));
  memcpy (&(cmd->command.fill_image.region), region, 3*sizeof(size_t));
  cmd->command.fill_image.rowpitch = image->image_row_pitch;
  cmd->command.fill_image.slicepitch = image->image_slice_pitch;
  cmd->command.fill_image.fill_pixel = fill_pixel;
  cmd->command.fill_image.pixel_size = image->image_elem_size * image->image_channels;

  POname(clRetainMemObject) (image);
  image->owning_device = command_queue->device;
  pocl_command_enqueue(command_queue, cmd);
  
  POCL_MEM_FREE(supported_image_formats);
  return errcode;
  
 ERROR_CLEAN:
  POCL_MEM_FREE(supported_image_formats);
  POCL_MEM_FREE(fill_pixel);
  return errcode;
}
POsym(clEnqueueFillImage)
