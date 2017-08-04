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
                          cl_event *         event,
                          cl_int *           errcode_ret ) 
CL_API_SUFFIX__VERSION_1_0
{
  cl_int errcode;
  size_t offset;
  void *map = NULL;
  cl_device_id device;
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

  errcode = pocl_check_event_wait_list (command_queue, num_events_in_wait_list,
                                        event_wait_list);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  POCL_GOTO_ERROR_COND((image_row_pitch == NULL), CL_INVALID_VALUE);

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
  size_t tuned_origin[3]
      = { origin[0] * image->image_elem_size * image->image_channels,
          origin[1], origin[2] };

  offset = tuned_origin[0] + tuned_origin[1] * image->image_row_pitch
           + tuned_origin[2] * image->image_slice_pitch;

  mapping_info = (mem_mapping_t*) malloc (sizeof (mem_mapping_t));
  if (mapping_info == NULL)
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR;
    }

  *image_row_pitch = image->image_row_pitch;
  if (image_slice_pitch)
    *image_slice_pitch = image->image_slice_pitch;

  HANDLE_IMAGE1D_BUFFER (image);

  /* CL_INVALID_OPERATION if buffer has been created with
   * CL_MEM_HOST_WRITE_ONLY
   * or CL_MEM_HOST_NO_ACCESS and CL_MAP_READ is set in map_flags or
   *
   * if buffer has been created with CL_MEM_HOST_READ_ONL or
   * CL_MEM_HOST_NO_ACCESS
   * and CL_MAP_WRITE or CL_MAP_WRITE_INVALIDATE_REGION is set in map_flags.
   */

  POCL_GOTO_ERROR_COND (
      ((map_flags & CL_MAP_READ)
       && (image->flags & (CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS))),
      CL_INVALID_OPERATION);

  POCL_GOTO_ERROR_COND (
      ((map_flags & CL_MAP_WRITE)
       && (image->flags & (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS))),
      CL_INVALID_OPERATION);

  if (image->flags & CL_MEM_USE_HOST_PTR)
    {
      /* In this case it should use the given host_ptr + offset as
         the mapping area in the host memory. */
      assert (image->mem_host_ptr != NULL);
      map = (char*)image->mem_host_ptr + offset;
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
      POCL_UPDATE_EVENT_COMPLETE (*event);
      errcode = CL_MAP_FAILURE;
      goto ERROR;
    }

  mapping_info->host_ptr = map;
  mapping_info->offset = offset;
  mapping_info->size = 0;/* not needed ?? */
  mapping_info->next = NULL;
  mapping_info->prev = NULL;
  POCL_LOCK_OBJ (image);
  DL_APPEND (image->mappings, mapping_info);
  POCL_UNLOCK_OBJ (image);

  POCL_MSG_PRINT_MEMORY ("Image %p, Mapping: host_ptr %p offset %zu\n", image,
                         mapping_info->host_ptr, mapping_info->offset);

  errcode = pocl_create_command (&cmd, command_queue, CL_COMMAND_MAP_IMAGE, 
                                 event, num_events_in_wait_list, 
                                 event_wait_list, 1, &image);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  cmd->command.map.buffer = image;
  cmd->command.map.mapping = mapping_info;
  POname(clRetainMemObject) (image);
  image->owning_device = command_queue->device;
  pocl_command_enqueue(command_queue, cmd);

  if (blocking_map)
    {
      POname(clFinish) (command_queue);
    }

  if (errcode_ret != NULL)
    (*errcode_ret) = CL_SUCCESS;

  return map;

 ERROR:
  POCL_MEM_FREE(map);
  POCL_MEM_FREE(cmd);
  POCL_MEM_FREE(mapping_info);
  if(errcode_ret != NULL)
    (*errcode_ret) = errcode;

  return NULL;
}
POsym(clEnqueueMapImage)
