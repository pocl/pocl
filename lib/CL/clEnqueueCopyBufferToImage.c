#include "pocl_image_util.h"
#include "pocl_mem_management.h"
#include "pocl_shared.h"
#include "pocl_util.h"

cl_int
pocl_copy_buffer_to_image_common (
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    cl_mem src_buffer,
    cl_mem dst_image,
    size_t src_offset,
    const size_t *dst_origin,
    const size_t *region,
    cl_uint num_items_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event,
    const cl_sync_point_khr *sync_point_wait_list,
    cl_sync_point_khr *sync_point,
    cl_mutable_command_khr *mutable_handle,
    _cl_command_node **cmd)
{
  cl_int errcode;
  size_t src_row_pitch = 0, src_slice_pitch = 0, dst_row_pitch = 0,
         dst_slice_pitch = 0;

  /* pass src_origin through in a format pocl_record_rect_copy understands */
  const size_t src_origin[3] = { src_offset, 0, 0 };

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (src_buffer)),
                          CL_INVALID_MEM_OBJECT);

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (dst_image)),
                          CL_INVALID_MEM_OBJECT);

  if (IS_IMAGE1D_BUFFER (dst_image))
    {
      /* If src_image is a 1D image or 1D image buffer object, src_origin[1]
       * and src_origin[2] must be 0 If src_image is a 1D image or 1D image
       * buffer object, region[1] and region[2] must be 1. */
      IMAGE1D_ORIG_REG_TO_BYTES (dst_image, dst_origin, region);
      if (command_buffer == NULL)
        {
          return POname (clEnqueueCopyBufferRect) (
              command_queue, src_buffer, dst_image->buffer, src_origin,
              i1d_origin, i1d_region, dst_image->image_row_pitch, 0,
              dst_image->image_row_pitch, 0, num_items_in_wait_list,
              event_wait_list, event);
        }
      else
        {
          return POname (clCommandCopyBufferRectKHR) (
              command_buffer, command_queue, src_buffer, dst_image->buffer,
              src_origin, i1d_origin, i1d_region, dst_image->image_row_pitch,
              0, dst_image->image_row_pitch, 0, num_items_in_wait_list,
              sync_point_wait_list, sync_point, mutable_handle);
        }
    }

  POCL_RETURN_ON_SUB_MISALIGN (src_buffer, command_queue);

  errcode = pocl_rect_copy (
      command_buffer, command_queue, CL_COMMAND_COPY_BUFFER_TO_IMAGE,
      src_buffer, CL_FALSE, dst_image, CL_TRUE, src_origin, dst_origin, region,
      &src_row_pitch, &src_slice_pitch, &dst_row_pitch, &dst_slice_pitch,
      num_items_in_wait_list, event_wait_list, event, sync_point_wait_list,
      sync_point, cmd);

  if (errcode != CL_SUCCESS)
    return errcode;

  POCL_CONVERT_SUBBUFFER_OFFSET (src_buffer, src_offset);

  POCL_GOTO_ERROR_ON (
      (src_buffer->size > command_queue->device->max_mem_alloc_size),
      CL_OUT_OF_RESOURCES, "src is larger than device's MAX_MEM_ALLOC_SIZE\n");

  _cl_command_node *c = *cmd;
  cl_device_id dev = command_queue->device;
  c->command.write_image.dst_mem_id
      = &dst_image->device_ptrs[dev->global_mem_id];
  c->command.write_image.dst = dst_image;
  c->command.write_image.src_host_ptr = ((void *)0);
  c->command.write_image.src_mem_id
      = &src_buffer->device_ptrs[dev->global_mem_id];
  c->command.write_image.src = src_buffer;
  c->command.write_image.src_row_pitch = src_row_pitch;
  c->command.write_image.src_slice_pitch = src_slice_pitch;
  c->command.write_image.src_offset = src_offset;
  c->command.write_image.origin[0] = dst_origin[0];
  c->command.write_image.origin[1] = dst_origin[1];
  c->command.write_image.origin[2] = dst_origin[2];
  c->command.write_image.region[0] = region[0];
  c->command.write_image.region[1] = region[1];
  c->command.write_image.region[2] = region[2];

  return CL_SUCCESS;

ERROR:
  pocl_mem_manager_free_command (*cmd);
  return errcode;
}

extern CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueCopyBufferToImage)(cl_command_queue  command_queue,
                                   cl_mem            src_buffer,
                                   cl_mem            dst_image,
                                   size_t            src_offset,
                                   const size_t *    dst_origin, /*[3]*/
                                   const size_t *    region,  /*[3]*/
                                   cl_uint           num_events_in_wait_list,
                                   const cl_event *  event_wait_list,
                                   cl_event *        event )
CL_API_SUFFIX__VERSION_1_0
{
  cl_int errcode;
  _cl_command_node *cmd = NULL;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_COND ((*(command_queue->device->available) == CL_FALSE),
                          CL_DEVICE_NOT_AVAILABLE);

  errcode = pocl_copy_buffer_to_image_common (
      NULL, command_queue, src_buffer, dst_image, src_offset, dst_origin,
      region, num_events_in_wait_list, event_wait_list, event, NULL, NULL,
      NULL, &cmd);

  if (errcode != CL_SUCCESS)
    return errcode;

  if (cmd)
    pocl_command_enqueue (command_queue, cmd);

  return CL_SUCCESS;
}
POsym(clEnqueueCopyBufferToImage)
