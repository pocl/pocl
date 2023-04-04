#include "pocl_util.h"
#include "pocl_shared.h"
#include "pocl_image_util.h"

/* Validate parameters of cl*CopyImage* that do not get checked by
 * pocl_rect_copy */
cl_int
pocl_validate_copy_image (cl_mem src_image, cl_mem dst_image)
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (src_image)),
                          CL_INVALID_MEM_OBJECT);

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (dst_image)),
                          CL_INVALID_MEM_OBJECT);

  /* src_image, dst_image: Can be 1D, 2D, 3D image or a 1D, 2D image array
   * objects allowing us to perform the following actions */
  POCL_RETURN_ERROR_ON (
      (IS_IMAGE1D_BUFFER (src_image) || IS_IMAGE1D_BUFFER (dst_image)),
      CL_INVALID_MEM_OBJECT,
      "clEnqueueCopyImage cannot be called on image 1D buffers!\n");

  return CL_SUCCESS;
}

cl_int
pocl_copy_image_common (cl_command_buffer_khr command_buffer,
                        cl_command_queue command_queue,
                        cl_mem src_image,
                        cl_mem dst_image,
                        const size_t *src_origin,
                        const size_t *dst_origin,
                        const size_t *region,
                        cl_uint num_items_in_wait_list,
                        const cl_event *event_wait_list,
                        cl_event *event,
                        const cl_sync_point_khr *sync_point_wait_list,
                        cl_sync_point_khr *sync_point,
                        _cl_command_node **cmd)
{
  cl_int errcode;
  size_t src_row_pitch = 0, src_slice_pitch = 0, dst_row_pitch = 0,
         dst_slice_pitch = 0;
  errcode = pocl_validate_copy_image (src_image, dst_image);
  if (errcode != CL_SUCCESS)
    return errcode;

  errcode = pocl_rect_copy (
      command_buffer, command_queue, CL_COMMAND_COPY_IMAGE, src_image, CL_TRUE,
      dst_image, CL_TRUE, src_origin, dst_origin, region, &src_row_pitch,
      &src_slice_pitch, &dst_row_pitch, &dst_slice_pitch,
      num_items_in_wait_list, event_wait_list, event, sync_point_wait_list,
      sync_point, cmd);
  if (errcode != CL_SUCCESS)
    return errcode;

  _cl_command_node *c = *cmd;
  cl_device_id dev = command_queue->device;
  c->command.copy_image.src_mem_id
      = &src_image->device_ptrs[dev->global_mem_id];
  c->command.copy_image.src = src_image;
  c->command.copy_image.dst_mem_id
      = &dst_image->device_ptrs[dev->global_mem_id];
  c->command.copy_image.dst = dst_image;
  c->command.copy_image.src_origin[0] = src_origin[0];
  c->command.copy_image.src_origin[1] = src_origin[1];
  c->command.copy_image.src_origin[2] = src_origin[2];
  c->command.copy_image.dst_origin[0] = dst_origin[0];
  c->command.copy_image.dst_origin[1] = dst_origin[1];
  c->command.copy_image.dst_origin[2] = dst_origin[2];
  c->command.copy_image.region[0] = region[0];
  c->command.copy_image.region[1] = region[1];
  c->command.copy_image.region[2] = region[2];

  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueCopyImage)(cl_command_queue      command_queue ,
                   cl_mem                src_image ,
                   cl_mem                dst_image , 
                   const size_t *        src_origin ,
                   const size_t *        dst_origin ,
                   const size_t *        region , 
                   cl_uint               num_events_in_wait_list ,
                   const cl_event *      event_wait_list ,
                   cl_event *            event ) CL_API_SUFFIX__VERSION_1_0
{
  _cl_command_node *cmd = NULL;
  cl_int errcode;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_COND ((*(command_queue->device->available) == CL_FALSE),
                          CL_DEVICE_NOT_AVAILABLE);

  errcode = pocl_copy_image_common (NULL, command_queue, src_image, dst_image,
                                    src_origin, dst_origin, region,
                                    num_events_in_wait_list, event_wait_list,
                                    event, NULL, NULL, &cmd);
  if (errcode != CL_SUCCESS)
    return errcode;

  pocl_command_enqueue (command_queue, cmd);

  return CL_SUCCESS;
}
POsym(clEnqueueCopyImage)
