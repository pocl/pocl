#include "pocl_cl.h"

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

  if (!src_image->is_image || !dst_image->is_image)
    return CL_INVALID_MEM_OBJECT;

  if (src_image->image_channel_order != dst_image->image_channel_order ||
        src_image->image_channel_data_type != dst_image->image_channel_data_type)
    return CL_IMAGE_FORMAT_MISMATCH;

  if (src_image->type == CL_MEM_OBJECT_IMAGE2D && src_origin[2] != 0)
    return CL_INVALID_VALUE;

  if (dst_image->type == CL_MEM_OBJECT_IMAGE2D && src_origin[2] != 0)
    return CL_INVALID_VALUE;

  if ((dst_image->type == CL_MEM_OBJECT_IMAGE2D || src_image->type == CL_MEM_OBJECT_IMAGE2D) &&
        region[2] != 1)
    return CL_INVALID_VALUE;

  /* TODO: overlap check */
  if (dst_image == src_image)
    POCL_ABORT_UNIMPLEMENTED();

  /* Adjust image pointers */
  size_t mod_region[3] = {region[0] * src_image->image_elem_size * src_image->image_channels,
                          region[1], region[2]};
  size_t mod_src_origin[3] = {src_origin[0] * src_image->image_elem_size * src_image->image_channels,
                              src_origin[1], src_origin[2]};
  size_t mod_dst_origin[3] = {dst_origin[0] * dst_image->image_elem_size * dst_image->image_channels,
                              dst_origin[1], dst_origin[2]};

  /* TODO: use copy buffer when possible (same width/height) */
  return POname(clEnqueueCopyBufferRect)(command_queue,
                                         src_image,
                                         dst_image,
                                         mod_src_origin,
                                         mod_dst_origin,
                                         mod_region,
                                         src_image->image_row_pitch,
                                         src_image->image_slice_pitch,
                                         dst_image->image_row_pitch,
                                         dst_image->image_slice_pitch,
                                         num_events_in_wait_list,
                                         event_wait_list,
                                         event);
}
POsym(clEnqueueCopyImage)
