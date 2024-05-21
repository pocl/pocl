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
  cl_int errcode = CL_SUCCESS;
  cl_device_id device = NULL;
  _cl_command_node *cmd = NULL;
  unsigned i;
  mem_mapping_t *mapping_info = NULL;
  /* need to release the memobject before returning? */
  int must_release = 0;

  POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                        CL_INVALID_COMMAND_QUEUE);

  POCL_GOTO_ERROR_COND ((*(command_queue->device->available) == CL_FALSE),
                        CL_DEVICE_NOT_AVAILABLE);

  POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (image)), CL_INVALID_MEM_OBJECT);

  device = command_queue->device;

  POCL_GOTO_ERROR_ON((command_queue->context != image->context),
    CL_INVALID_CONTEXT, "image and command_queue are not from the same context\n");

  if (IS_IMAGE1D_BUFFER (image))
    {
      IMAGE1D_ORIG_REG_TO_BYTES (image, origin, region)
      return POname (clEnqueueMapBuffer) (
          command_queue, image->buffer, blocking_map, map_flags, i1d_origin[0],
          i1d_region[0], num_events_in_wait_list, event_wait_list, event,
          errcode_ret);
    }

  POCL_GOTO_ERROR_ON ((!image->is_image), CL_INVALID_MEM_OBJECT,
                      "image argument is not an image\n");
  POCL_GOTO_ERROR_ON ((image->is_gl_texture), CL_INVALID_MEM_OBJECT,
                      "image is a GL texture\n");

  POCL_GOTO_ON_UNSUPPORTED_IMAGE (image, command_queue->device);

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

  POCL_CHECK_DEV_IN_CMDQ;

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

  mapping_info = (mem_mapping_t*) calloc (1, sizeof (mem_mapping_t));
  POCL_GOTO_ERROR_COND ((mapping_info == NULL), CL_OUT_OF_HOST_MEMORY);
  pocl_mem_identifier *mem_id = &image->device_ptrs[device->global_mem_id];

  POCL_LOCK_OBJ (image);
  /* increment refcount, actually twice (one happens in pocl_create_command):
     one is for duration of Map command, and is implicitly decreased
     after EnqueueMap is finished; the other one is for the duration
     of mapping, and is decreased by UnMap. */
  POCL_RETAIN_OBJECT_UNLOCKED (image);
  must_release = 1;

  *image_row_pitch = image->image_row_pitch;
  if (image_slice_pitch)
    *image_slice_pitch = image->image_slice_pitch;

  mapping_info->map_flags = map_flags;
  mapping_info->origin[0] = origin[0];
  mapping_info->origin[1] = origin[1];
  mapping_info->origin[2] = origin[2];
  mapping_info->region[0] = region[0];
  mapping_info->region[1] = region[1];
  mapping_info->region[2] = region[2];
  mapping_info->row_pitch = image->image_row_pitch;
  mapping_info->slice_pitch = image->image_slice_pitch;

  size_t px = image->image_elem_size * image->image_channels;
  size_t start_offset = origin[2] * mapping_info->slice_pitch
                        + origin[1] * mapping_info->row_pitch + origin[0] * px;
  size_t end_offset
      = px * (origin[0] + region[0] - 1)
        + mapping_info->row_pitch * (origin[1] + region[1] - 1)
        + mapping_info->slice_pitch * (origin[2] + region[2] - 1);

  mapping_info->offset = start_offset;
  assert (start_offset <= image->size);
  assert (start_offset <= end_offset);
  mapping_info->size = end_offset + 1 - start_offset;

  /* because cl_mems are per-context, PoCL delays allocation of
   * cl_mem backing memory until it knows which device will need it,
   * so the cl_mem might not yet be allocated on this device. However,
   * some drivers can avoid unnecessary host memory usage if we
   * allocate it now. We can assume it will be used on this device. */
  pocl_mem_identifier *p = &image->device_ptrs[device->global_mem_id];
  if (p->mem_ptr == NULL)
    errcode = device->ops->alloc_mem_obj (device, image, NULL);

  if (errcode == CL_SUCCESS)
    errcode = device->ops->get_mapping_ptr (device->data, mem_id, image,
                                            mapping_info);
  DL_APPEND (image->mappings, mapping_info);
  ++image->map_count;
  POCL_UNLOCK_OBJ (image);

  if (errcode != CL_SUCCESS)
    goto ERROR;

  char rdonly = (map_flags & CL_MAP_READ);

  errcode = pocl_create_command (&cmd, command_queue, CL_COMMAND_MAP_IMAGE,
                                 event, num_events_in_wait_list,
                                 event_wait_list, image, rdonly);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  cmd->command.map.mem_id = mem_id;
  cmd->command.map.mapping = mapping_info;

  POCL_MSG_PRINT_MEMORY ("Image %p, Mapping: host_ptr %p offset %zu\n", image,
                         mapping_info->host_ptr, mapping_info->offset);

  pocl_command_enqueue(command_queue, cmd);

  if (blocking_map)
    {
      POname(clFinish) (command_queue);
    }
  if (errcode_ret)
    *errcode_ret = errcode;

  return mapping_info->host_ptr;

ERROR:
  if (must_release)
    {
      POCL_LOCK_OBJ (image);
      assert (mapping_info);
      if (mapping_info->host_ptr)
        device->ops->free_mapping_ptr (device->data, mem_id, image,
                                       mapping_info);
      DL_DELETE (image->mappings, mapping_info);
      --image->map_count;
      POCL_UNLOCK_OBJ (image);
      POname (clReleaseMemObject) (image);
    }
  POCL_MEM_FREE (mapping_info);
  POCL_MEM_FREE (cmd);
  if (errcode_ret)
    *errcode_ret = errcode;

  return NULL;
}
POsym(clEnqueueMapImage)
