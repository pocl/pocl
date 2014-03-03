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

#include "pocl_cl.h"
#include "utlist.h"
#include "pocl_image_util.h"
#include "pocl_util.h"
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
  cl_uint num_entries = 0;
  _cl_command_node *cmd = NULL;
  cl_image_format *supported_image_formats = NULL;
  int i;
  void *fill_pixel = NULL;
  int num_image_channels;
  int image_elem_size; 
  size_t tuned_origin[3];

  if (command_queue == NULL)
    return CL_INVALID_COMMAND_QUEUE;
  
  if (image == NULL)
    return CL_INVALID_MEM_OBJECT;

  errcode = pocl_check_image_origin_region (image, origin, region);
  if (errcode != CL_SUCCESS)
    return errcode;

  if (command_queue->context != image->context)
    return CL_INVALID_CONTEXT;
  
  if (fill_color == NULL || origin == NULL || region == NULL) 
    return CL_INVALID_VALUE;
  
  if (event_wait_list == NULL && num_events_in_wait_list > 0)
    return CL_INVALID_EVENT_WAIT_LIST;

  /* TODO: handle 1D image buffer size check. 
     needs new attribute to device struct */
  if (image->type == CL_MEM_OBJECT_IMAGE1D ||
      image->type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
    {
      if (image->image_width > command_queue->device->image2d_max_width)
        return CL_INVALID_IMAGE_SIZE;
    }

  if (image->type == CL_MEM_OBJECT_IMAGE2D || 
      image->type == CL_MEM_OBJECT_IMAGE2D_ARRAY)
    {
      if (image->image_width > command_queue->device->image2d_max_width ||
          image->image_height > command_queue->device->image2d_max_height)
        return CL_INVALID_IMAGE_SIZE;
    }
  
  if (image->type == CL_MEM_OBJECT_IMAGE3D)
    {
      if (image->image_width > command_queue->device->image3d_max_width ||
          image->image_height > command_queue->device->image3d_max_height ||
          image->image_depth > command_queue->device->image3d_max_depth)
        return CL_INVALID_IMAGE_SIZE;
    }
  
  /* check if image format is supported */
  errcode = POname(clGetSupportedImageFormats)
    (command_queue->context, 0, image->type, 0, NULL, 
     &num_entries);
  
  if (errcode != CL_SUCCESS || num_entries == 0) 
    return errcode;
  
  supported_image_formats = malloc (num_entries * sizeof(cl_image_format));
  if (supported_image_formats == NULL)
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR_CLEAN;
    }
  
  errcode = POname(clGetSupportedImageFormats)
    (command_queue->context, 0, image->type, num_entries, 
     supported_image_formats, NULL);
  
  for (i = 0; i < num_entries; i++)
    {
      if (supported_image_formats[i].image_channel_order == 
          image->image_channel_order &&
          supported_image_formats[i].image_channel_data_type ==
          image->image_channel_data_type)
        {
          goto TYPE_SUPPORTED;
        }
    }
  errcode = CL_INVALID_VALUE;
  goto ERROR_CLEAN;

 TYPE_SUPPORTED: 
 
  fill_pixel = malloc (4 * sizeof(int));
  if (fill_pixel == NULL)
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR_CLEAN;
    }

  pocl_get_image_information (image->image_channel_order, 
                              image->image_channel_data_type, 
                              &num_image_channels, 
                              &image_elem_size);

  /* TODO: channel order, saturating data type conversion */
  if (image_elem_size == 1)
    {
      ((cl_char4*)fill_pixel)->x = ((cl_int4*)fill_color)->x;
      ((cl_char4*)fill_pixel)->y = ((cl_int4*)fill_color)->y;
      ((cl_char4*)fill_pixel)->z = ((cl_int4*)fill_color)->z;
      ((cl_char4*)fill_pixel)->w = ((cl_int4*)fill_color)->w;
    }
  if (image_elem_size == 2)
    {
      ((cl_short4*)fill_pixel)->x = ((cl_int4*)fill_color)->x;
      ((cl_short4*)fill_pixel)->y = ((cl_int4*)fill_color)->y;
      ((cl_short4*)fill_pixel)->z = ((cl_int4*)fill_color)->z;
      ((cl_short4*)fill_pixel)->w = ((cl_int4*)fill_color)->w;
    }
 if (image_elem_size == 4)
    {
      memcpy (fill_pixel, fill_color, sizeof (cl_int4));      
    }

  /* POCL uses top-left corner as origin for images and AMD SDK ImageOverlap 
     test uses bottom-left corner as origin. Because of this we need to modify 
     y-coordinate so the fill goes in the right place. */
  tuned_origin[0] = origin[0];
  tuned_origin[1] = image->image_height - region[1] - origin[1];
  tuned_origin[2] = origin[2];

  errcode = pocl_create_command (&cmd, command_queue, CL_COMMAND_FILL_IMAGE, 
                                 event, num_events_in_wait_list, 
                                 event_wait_list);
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
  cmd->command.fill_image.pixel_size = image_elem_size * num_image_channels;
  pocl_command_enqueue(command_queue, cmd);
  
  free (supported_image_formats);
  return errcode;
  
 ERROR_CLEAN:
  free (supported_image_formats);
  if (*event)
    free (*event);
  free (fill_pixel);
  return errcode;
}
POsym(clEnqueueFillImage)
