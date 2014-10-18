#include "pocl_util.h"
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
  int errcode;

  POCL_RETURN_ERROR_COND((command_queue == NULL), CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_ON((!command_queue->device->image_support), CL_INVALID_OPERATION,
    "Device %s does not support images\n", command_queue->device->long_name);

  POCL_RETURN_ERROR_COND((src_image == NULL), CL_INVALID_MEM_OBJECT);
  POCL_RETURN_ERROR_COND((dst_image == NULL), CL_INVALID_MEM_OBJECT);
  POCL_RETURN_ERROR_COND((src_origin == NULL), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND((dst_origin == NULL), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND((region == NULL), CL_INVALID_VALUE);


  POCL_RETURN_ERROR_ON(((command_queue->context != src_image->context) ||
      (command_queue->context != dst_image->context)), CL_INVALID_CONTEXT,
      "src_image, dst_image and command_queue are not from the same context\n");

  POCL_RETURN_ERROR_ON((!src_image->is_image), CL_INVALID_MEM_OBJECT,
                                                "src_image is not an image\n");
  POCL_RETURN_ERROR_ON((!dst_image->is_image), CL_INVALID_MEM_OBJECT,
                                                "dst_image is not an image\n");

  POCL_RETURN_ERROR_COND((event_wait_list == NULL && num_events_in_wait_list > 0),
    CL_INVALID_EVENT_WAIT_LIST);

  POCL_RETURN_ERROR_COND((event_wait_list != NULL && num_events_in_wait_list == 0),
    CL_INVALID_EVENT_WAIT_LIST);


  POCL_RETURN_ERROR_ON((src_image->image_channel_order != dst_image->image_channel_order),
    CL_IMAGE_FORMAT_MISMATCH, "src_image and dst_image have different image channel order\n");

  POCL_RETURN_ERROR_ON((src_image->image_channel_data_type != dst_image->image_channel_data_type),
    CL_IMAGE_FORMAT_MISMATCH, "src_image and dst_image have different image channel data type\n");

  POCL_RETURN_ERROR_ON((src_image->type == CL_MEM_OBJECT_IMAGE2D && src_origin[2] != 0),
    CL_INVALID_VALUE, "src_origin[2] must be 0 for 2D src_image\n");

  POCL_RETURN_ERROR_ON((dst_image->type == CL_MEM_OBJECT_IMAGE2D && dst_origin[2] != 0),
    CL_INVALID_VALUE, "dst_origin[2] must be 0 for 2D dst_image\n");

  POCL_RETURN_ERROR_ON(((dst_image->type == CL_MEM_OBJECT_IMAGE2D ||
     src_image->type == CL_MEM_OBJECT_IMAGE2D) &&  region[2] != 1),
    CL_INVALID_VALUE, "for any 2D image copy, region[2] must be 1\n");

  errcode = pocl_check_device_supports_image(src_image, command_queue);
  if (errcode != CL_SUCCESS)
    return errcode;
  errcode = pocl_check_device_supports_image(dst_image, command_queue);
  if (errcode != CL_SUCCESS)
    return errcode;

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
