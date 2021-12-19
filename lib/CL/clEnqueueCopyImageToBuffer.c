#include "pocl_util.h"
#include "pocl_shared.h"
#include "pocl_image_util.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueCopyImageToBuffer)(cl_command_queue  command_queue ,
                           cl_mem            src_image ,
                           cl_mem            dst_buffer ,
                           const size_t *    src_origin ,
                           const size_t *    region ,
                           size_t            dst_offset ,
                           cl_uint           num_events_in_wait_list ,
                           const cl_event *  event_wait_list ,
                           cl_event *        event ) CL_API_SUFFIX__VERSION_1_0
{
  _cl_command_node *cmd = NULL;
  /* pass dst_origin through in a format pocl_rect_copy understands */
  const size_t dst_origin[3] = { dst_offset, 0, 0};

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (src_image)),
                          CL_INVALID_MEM_OBJECT);

  if (IS_IMAGE1D_BUFFER (src_image))
    {
      /* If src_image is a 1D image or 1D image buffer object, src_origin[1]
       * and src_origin[2] must be 0 If src_image is a 1D image or 1D image
       * buffer object, region[1] and region[2] must be 1. */
      IMAGE1D_ORIG_REG_TO_BYTES (src_image, src_origin, region);
      return POname (clEnqueueCopyBufferRect (
          command_queue, src_image->buffer, dst_buffer,
          i1d_origin, dst_origin, i1d_region,
          src_image->image_row_pitch, 0,
          src_image->image_row_pitch, 0,
          num_events_in_wait_list, event_wait_list, event));
    }

  POCL_RETURN_ON_SUB_MISALIGN (dst_buffer, command_queue);

  cl_int err = pocl_rect_copy(
    command_queue,
    CL_COMMAND_COPY_IMAGE_TO_BUFFER,
    src_image, CL_TRUE,
    dst_buffer, CL_FALSE,
    src_origin, dst_origin, region,
    0, 0,
    0, 0,
    num_events_in_wait_list, event_wait_list,
    event,
    &cmd);

  if (err != CL_SUCCESS)
    return err;

  POCL_CONVERT_SUBBUFFER_OFFSET (dst_buffer, dst_offset);

  POCL_RETURN_ERROR_ON((dst_buffer->size > command_queue->device->max_mem_alloc_size),
                        CL_OUT_OF_RESOURCES,
                        "src is larger than device's MAX_MEM_ALLOC_SIZE\n");

  cl_device_id dev = command_queue->device;

  cmd->command.read_image.src_mem_id = &src_image->device_ptrs[dev->global_mem_id];
  cmd->command.read_image.src = src_image;
  cmd->command.read_image.dst_host_ptr = NULL;
  cmd->command.read_image.dst = dst_buffer;
  cmd->command.read_image.dst_mem_id = &dst_buffer->device_ptrs[dev->global_mem_id];

  cmd->command.read_image.origin[0] = src_origin[0];
  cmd->command.read_image.origin[1] = src_origin[1];
  cmd->command.read_image.origin[2] = src_origin[2];
  cmd->command.read_image.region[0] = region[0];
  cmd->command.read_image.region[1] = region[1];
  cmd->command.read_image.region[2] = region[2];
  // TODO
  cmd->command.read_image.dst_row_pitch = 0;   // src_image->image_row_pitch;
  cmd->command.read_image.dst_slice_pitch = 0; // src_image->image_slice_pitch;
  cmd->command.read_image.dst_offset = dst_offset;

  pocl_command_enqueue (command_queue, cmd);

  return CL_SUCCESS;
}
POsym(clEnqueueCopyImageToBuffer)


