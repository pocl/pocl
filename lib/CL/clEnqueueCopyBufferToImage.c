#include "pocl_util.h"
#include "pocl_shared.h"
#include "pocl_image_util.h"

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
  _cl_command_node *cmd = NULL;
  /* pass src_origin through in a format pocl_rect_copy understands */
  const size_t src_origin[3] = { src_offset, 0, 0};

  POCL_RETURN_ERROR_COND ((command_queue == NULL), CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_COND ((dst_image == NULL), CL_INVALID_MEM_OBJECT);

  if (IS_IMAGE1D_BUFFER (dst_image))
    {
      /* If src_image is a 1D image or 1D image buffer object, src_origin[1]
       * and src_origin[2] must be 0 If src_image is a 1D image or 1D image
       * buffer object, region[1] and region[2] must be 1. */
      IMAGE1D_ORIG_REG_TO_BYTES (dst_image, dst_origin, region);
      return POname (clEnqueueCopyBufferRect (
          command_queue, src_buffer, dst_image->buffer,
          src_origin, i1d_origin, i1d_region,
          dst_image->image_row_pitch, 0,
          dst_image->image_row_pitch, 0,
          num_events_in_wait_list, event_wait_list, event));
    }

  POCL_RETURN_ON_SUB_MISALIGN (src_buffer, command_queue);

  cl_int err = pocl_rect_copy(
    command_queue,
    CL_COMMAND_COPY_BUFFER_TO_IMAGE,
    src_buffer, CL_FALSE,
    dst_image, CL_TRUE,
    src_origin, dst_origin, region,
    0, 0,
    0, 0,
    num_events_in_wait_list, event_wait_list,
    event,
    &cmd);

  if (err != CL_SUCCESS)
    return err;

  POCL_CONVERT_SUBBUFFER_OFFSET (src_buffer, src_offset);

  POCL_RETURN_ERROR_ON((src_buffer->size > command_queue->device->max_mem_alloc_size),
                        CL_OUT_OF_RESOURCES,
                        "src is larger than device's MAX_MEM_ALLOC_SIZE\n");

  cl_device_id dev = command_queue->device;

  cmd->command.write_image.dst_mem_id = &dst_image->device_ptrs[dev->dev_id];
  cmd->command.write_image.src_host_ptr = NULL;
  cmd->command.write_image.src_mem_id = &src_buffer->device_ptrs[dev->dev_id];

  // TODO
  cmd->command.write_image.src_row_pitch = 0;   // dst_image->image_row_pitch;
  cmd->command.write_image.src_slice_pitch = 0; // dst_image->image_slice_pitch;
  cmd->command.write_image.src_offset = src_offset;

  cmd->command.write_image.origin[0] = dst_origin[0];
  cmd->command.write_image.origin[1] = dst_origin[1];
  cmd->command.write_image.origin[2] = dst_origin[2];
  cmd->command.write_image.region[0] = region[0];
  cmd->command.write_image.region[1] = region[1];
  cmd->command.write_image.region[2] = region[2];

  POname (clRetainMemObject) (dst_image);
  dst_image->owning_device = dev;
  POname (clRetainMemObject) (src_buffer);
  src_buffer->owning_device = dev;

  pocl_command_enqueue (command_queue, cmd);

  return CL_SUCCESS;
}
POsym(clEnqueueCopyBufferToImage)
