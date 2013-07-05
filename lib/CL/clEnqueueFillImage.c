/* OpenCL runtime library: clEnqueueFillImage()

   Copyright (c) 2013 Ville Korhonen
   
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
  int num_entries = 0;
  cl_image_format *supported_image_formats;
  int i;
  void *fill_ptr;
 
  if (command_queue == NULL)
    return CL_INVALID_COMMAND_QUEUE;
  
  if (image == NULL)
    return CL_INVALID_MEM_OBJECT;

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
  
  if (event != NULL)
    {
      *event = (cl_event)malloc (sizeof(struct _cl_event));
      if (event == NULL)
        {
          errcode = CL_OUT_OF_HOST_MEMORY;
          goto ERROR_CLEAN;
        }
      POCL_INIT_OBJECT(*event);
      (*event)->queue = command_queue;
      POname(clRetainCommandQueue) (command_queue);
      (*event)->command_type = CL_COMMAND_FILL_IMAGE;
      POCL_UPDATE_EVENT_QUEUED;
    }

  _cl_command_node *cmd = malloc (sizeof(_cl_command_node));
  if (cmd == NULL)
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR_CLEAN;
    }

  cmd->command.fill_image.data = command_queue->device->data;
  cmd->command.fill_image.host_ptr = fill_ptr;
  cmd->command.fill_image.device_ptr = 
    image->device_ptrs[command_queue->device->dev_id];
  cmd->command.fill_image.origin = origin;
  cmd->command.fill_image.region = region;
  cmd->command.fill_image.rowpitch = image->image_row_pitch;
  cmd->command.fill_image.slicepitch = image->image_slice_pitch;
  

 ERROR_CLEAN:
  free (supported_image_formats);
  
  return errcode;
}
POsym(clEnqueueFillImage)

