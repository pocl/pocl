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
#include <string.h>

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
                          cl_event *         event_ret,
                          cl_int *           errcode_ret )
CL_API_SUFFIX__VERSION_1_0
{
  cl_int errcode;
  size_t offset;
  void *map = NULL;
  int using_host_buffer = 0;
  cl_device_id device;
  cl_event event = NULL;
  _cl_command_node *cmd = NULL;
  mem_mapping_t *mapping_info = NULL;

  POCL_GOTO_ERROR_COND((command_queue == NULL), CL_INVALID_COMMAND_QUEUE);

  device = command_queue->device;

  POCL_GOTO_ERROR_ON((!command_queue->device->image_support), CL_INVALID_OPERATION,
    "Device %s does not support images\n", command_queue->device->long_name);

  POCL_GOTO_ERROR_ON((command_queue->context != image->context),
    CL_INVALID_CONTEXT, "image and command_queue are not from the same context\n");

  POCL_GOTO_ERROR_COND((image == NULL), CL_INVALID_MEM_OBJECT);

  POCL_GOTO_ERROR_ON((!image->is_image), CL_INVALID_MEM_OBJECT,
    "image argument is not an image type cl_mem\n");

  POCL_GOTO_ERROR_COND((event_wait_list == NULL && num_events_in_wait_list > 0),
    CL_INVALID_EVENT_WAIT_LIST);

  POCL_GOTO_ERROR_COND((event_wait_list != NULL && num_events_in_wait_list == 0),
    CL_INVALID_EVENT_WAIT_LIST);

  errcode = pocl_check_device_supports_image(image, command_queue);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  POCL_GOTO_ERROR_COND((image_row_pitch == NULL), CL_INVALID_VALUE)

  errcode = pocl_check_image_origin_region(image, origin, region);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  POCL_GOTO_ERROR_ON((image_slice_pitch == NULL &&
      (image->type == CL_MEM_OBJECT_IMAGE3D || 
       image->type == CL_MEM_OBJECT_IMAGE1D_ARRAY ||
       image->type == CL_MEM_OBJECT_IMAGE2D_ARRAY)), CL_INVALID_VALUE,
       "For a 3D image, 1D, and 2D image array, "
       "image_slice_pitch must be a non-NULL value\n");

  /* TODO: more error checks */
  
  offset = image->image_channels * image->image_elem_size * origin[0];
  
  mapping_info = (mem_mapping_t*) malloc (sizeof (mem_mapping_t));
  if (mapping_info == NULL)
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR;
    }

  errcode = pocl_create_command (&cmd, command_queue, CL_COMMAND_MAP_IMAGE,
                                 &event, num_events_in_wait_list,
                                 event_wait_list);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  if (image->flags & CL_MEM_USE_HOST_PTR)
    {
      /* In this case it should use the given host_ptr + offset as
         the mapping area in the host memory. */
      assert (image->mem_host_ptr != NULL);
      map = (char*)image->mem_host_ptr + offset;
      device->ops->read_rect (device->data, map,
                              image->device_ptrs[device->dev_id].mem_ptr,
                              origin, origin, region,
                              image->image_row_pitch, image->image_slice_pitch,
                              image->image_row_pitch, image->image_slice_pitch);
      POCL_UPDATE_EVENT_COMPLETE(&event);
    }
  else
    {
      /* The first call to the device driver's map mem tells where
         the mapping will be stored (the last argument is NULL) in
         the host memory. When the last argument is non-NULL, the
         buffer will be mapped there (assumed it will succeed).  */

      map = device->ops->map_mem 
        (device->data, 
         image->device_ptrs[device->dev_id].mem_ptr, 
         offset, 0/*size*/, NULL);
    }

  if (map == NULL)
    {
      errcode = CL_MAP_FAILURE;
      goto ERROR;
    }

  mapping_info->host_ptr = map;
  mapping_info->offset = offset;
  mapping_info->size = 0;/* not needed ?? */
  POCL_LOCK_OBJ (image);
  DL_APPEND (image->mappings, mapping_info);
  POCL_UNLOCK_OBJ (image);

  cmd->command.map.buffer = image;
  cmd->command.map.mapping = mapping_info;
  pocl_command_enqueue(command_queue, cmd);

  *image_row_pitch = image->image_row_pitch;
  if (image_slice_pitch)
    *image_slice_pitch = image->image_slice_pitch;

  if (blocking_map)
    {
      POname(clFinish) (command_queue);
    }

  if (event_ret)
    *event_ret = event;
  if (errcode_ret != NULL)
    (*errcode_ret) = CL_SUCCESS;

  return map;

ERROR:
  POCL_MEM_FREE(event);
  assert(map == NULL);
  POCL_MEM_FREE(cmd);
  POCL_MEM_FREE(mapping_info);
  if(errcode_ret != NULL)
    (*errcode_ret) = errcode;

  return NULL;
}
POsym(clEnqueueMapImage)
