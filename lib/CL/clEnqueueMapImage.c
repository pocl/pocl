/* OpenCL runtime library: clEnqueueMapImage()

   Copyright (c) 2011 Ville Korhonen / Tampere Univ. of Tech.
   
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
#include <stdlib.h>
#include "pocl_cl.h"
#include "pocl_image_util.h"
#include "pocl_util.h"
#include "utlist.h"
#include <stdlib.h>

CL_API_ENTRY void * CL_API_CALL
POname(clEnqueueMapImage)(cl_command_queue   command_queue,
                          cl_mem             image,
                          cl_bool            blocking_map, 
                          cl_map_flags       map_flags, 
                          const size_t *     origin,
                          const size_t *     region,
                          size_t *           image_row_pitch,
                          size_t *           image_slice_pitch,
                          cl_uint            num_events_in_wait_list,
                          const cl_event *   event_wait_list,
                          cl_event *         event,
                          cl_int *           errcode_ret ) 
CL_API_SUFFIX__VERSION_1_0
{
  cl_int errcode;
  int elem_size;
  int num_channels;
  int map_size;
  void *map = NULL;
  cl_device_id device;
  _cl_command_node *cmd = NULL;
  mem_mapping_t *mapping_info = NULL;

  if (command_queue == NULL)
    {
      errcode =  CL_INVALID_COMMAND_QUEUE;
      goto ERROR;
    }
  if (image == NULL)
    {
      errcode = CL_INVALID_MEM_OBJECT;
      goto ERROR;
    }
  if (command_queue->context != image->context)
    {
      errcode = CL_INVALID_CONTEXT;
      goto ERROR;
    }
  if (event_wait_list == NULL && num_events_in_wait_list != 0 ||
      event_wait_list != NULL && num_events_in_wait_list == 0)
    {
      errcode = CL_INVALID_EVENT_WAIT_LIST;
      goto ERROR;
    }

  errcode = pocl_check_image_origin_region(image, origin, region);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  if (image_row_pitch == NULL)
    {
      errcode = CL_INVALID_VALUE;
      goto ERROR;
    }

  if (image_slice_pitch == NULL && 
      (image->type == CL_MEM_OBJECT_IMAGE3D || 
       image->type == CL_MEM_OBJECT_IMAGE1D_ARRAY ||
       image->type == CL_MEM_OBJECT_IMAGE2D_ARRAY))
    {
      errcode = CL_INVALID_VALUE;
      goto ERROR;
    }

  /* TODO: more error checks */

  pocl_get_image_information(image->image_channel_order, 
                             image->image_channel_data_type, 
                             &num_channels, &elem_size);

  map_size = elem_size * num_channels * region[0] * region[1] * region[2];
  map = malloc (map_size);
  if (map == NULL)
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR;
    }

  size_t tuned_origin[3] = {origin[0] * elem_size * num_channels, origin[1], 
                            origin[2]};
  size_t tuned_region[3] = {region[0] * elem_size * num_channels, region[1], 
                            region[2]};
  size_t map_origin[3] = {0, 0, 0}; 

  if (event != NULL)
    {
      errcode = pocl_create_event (event, command_queue, CL_COMMAND_MAP_IMAGE);
      if (errcode != CL_SUCCESS)
        goto ERROR;
    }
  
  mapping_info = (mem_mapping_t*) malloc (sizeof (mem_mapping_t));
  if (mapping_info == NULL)
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR;
    }
  mapping_info->host_ptr = map;
  mapping_info->offset = 0;
  mapping_info->size = map_size;
  DL_APPEND (image->mappings, mapping_info);
  
  device = command_queue->device;
  if (blocking_map)
    {
      device->read_rect (device->data, map, image->device_ptrs[device->dev_id],
                         tuned_origin, map_origin, tuned_region, 
                         image->image_row_pitch, image->image_slice_pitch, 
                         image->image_row_pitch, image->image_slice_pitch);
      POCL_UPDATE_EVENT_COMPLETE;
    }
  else
    {
      errcode = pocl_create_command (&cmd, command_queue, CL_COMMAND_MAP_IMAGE, 
                                     event, num_events_in_wait_list, 
                                     event_wait_list);
      if (errcode != CL_SUCCESS)
        goto ERROR;
      
      cmd->command.map_image.data = command_queue->device->data;
      cmd->command.map_image.device_ptr = 
        image->device_ptrs[command_queue->device->dev_id];
      cmd->command.map_image.map_ptr = map;
      memcpy ((cmd->command.map_image.origin), tuned_origin, 3*sizeof (size_t));
      memcpy ((cmd->command.map_image.region), tuned_region, 3*sizeof (size_t));
      cmd->command.map_image.rowpitch = image->image_row_pitch;
      cmd->command.map_image.slicepitch = image->image_slice_pitch;
      LL_APPEND(command_queue->root, cmd);
      POCL_UPDATE_EVENT_QUEUED;
    }
  *image_row_pitch = image->image_row_pitch;
  if (image_slice_pitch)
    *image_slice_pitch = image->image_slice_pitch;

  return map;
 
 ERROR:
  free (map);
  free (cmd);
  free (mapping_info);
  if (event != NULL)
    free (*event);
  if(errcode_ret != NULL)
    (*errcode_ret) = errcode;
      
  return NULL;
}
POsym(clEnqueueMapImage)
