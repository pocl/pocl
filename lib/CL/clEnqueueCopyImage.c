#include "pocl_util.h"
#include "pocl_shared.h"
#include "pocl_image_util.h"

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
  cl_device_id device;
  unsigned i;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

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

  POCL_CHECK_DEV_IN_CMDQ;

  cl_int err = pocl_rect_copy(
    command_queue,
    CL_COMMAND_COPY_IMAGE,
    src_image, CL_TRUE,
    dst_image, CL_TRUE,
    src_origin, dst_origin, region,
    0, 0,
    0, 0,
    num_events_in_wait_list, event_wait_list,
    event,
    &cmd);

  if (err != CL_SUCCESS)
    return err;

  cmd->command.copy_image.src_mem_id = &src_image->device_ptrs[device->dev_id];
  cmd->command.copy_image.src = src_image;
  cmd->command.copy_image.dst_mem_id = &dst_image->device_ptrs[device->dev_id];
  cmd->command.copy_image.dst = dst_image;

  cmd->command.copy_image.src_origin[0] = src_origin[0];
  cmd->command.copy_image.src_origin[1] = src_origin[1];
  cmd->command.copy_image.src_origin[2] = src_origin[2];
  cmd->command.copy_image.dst_origin[0] = dst_origin[0];
  cmd->command.copy_image.dst_origin[1] = dst_origin[1];
  cmd->command.copy_image.dst_origin[2] = dst_origin[2];
  cmd->command.copy_image.region[0] = region[0];
  cmd->command.copy_image.region[1] = region[1];
  cmd->command.copy_image.region[2] = region[2];

  POname (clRetainMemObject) (src_image);
  src_image->owning_device = device;
  POname (clRetainMemObject) (dst_image);
  dst_image->owning_device = device;

  pocl_command_enqueue (command_queue, cmd);

  return CL_SUCCESS;
}
POsym(clEnqueueCopyImage)
