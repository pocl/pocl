#include "pocl_util.h"
#include "pocl_image_util.h"

extern CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueCopyBufferToImage)(cl_command_queue  command_queue,
                                   cl_mem            buffer,
                                   cl_mem            image, 
                                   size_t            src_offset,
                                   const size_t *    dst_origin, /*[3]*/
                                   const size_t *    region,  /*[3]*/
                                   cl_uint           num_events_in_wait_list,
                                   const cl_event *  event_wait_list,
                                   cl_event *        event ) 
CL_API_SUFFIX__VERSION_1_0
{
  int errcode;

  POCL_RETURN_ERROR_COND((command_queue == NULL), CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_ON((!command_queue->device->image_support), CL_INVALID_OPERATION,
    "Device %s does not support images\n", command_queue->device->long_name);

  POCL_RETURN_ERROR_COND((buffer == NULL), CL_INVALID_MEM_OBJECT);
  POCL_RETURN_ERROR_ON((buffer->type != CL_MEM_OBJECT_BUFFER),
    CL_INVALID_MEM_OBJECT, "buffer is not a CL_MEM_OBJECT_BUFFER\n");

  POCL_RETURN_ERROR_COND((image == NULL), CL_INVALID_MEM_OBJECT);

  POCL_RETURN_ERROR_ON((!image->is_image), CL_INVALID_MEM_OBJECT,
    "dst_image is not an image\n");

  POCL_RETURN_ERROR_COND((region == NULL), CL_INVALID_MEM_OBJECT);


  if (region[2] != 1) {
    POCL_ABORT_UNIMPLEMENTED("clEnqueueCopyBufferToImage on 3D images");
  }

  POCL_RETURN_ERROR_ON(((buffer->context != image->context) ||
    (buffer->context != command_queue->context)), CL_INVALID_CONTEXT, "src_buffer, "
    "dst_image and command_queue are not from the same context\n");

  POCL_RETURN_ERROR_COND((event_wait_list == NULL && num_events_in_wait_list > 0),
    CL_INVALID_EVENT_WAIT_LIST);

  POCL_RETURN_ERROR_COND((event_wait_list != NULL && num_events_in_wait_list == 0),
    CL_INVALID_EVENT_WAIT_LIST);

  errcode = pocl_check_device_supports_image(image, command_queue);
  if (errcode != CL_SUCCESS)
    return errcode;
    

  int host_elem_size;    
  int host_channels;
  pocl_get_image_information (image->image_channel_order,
                              image->image_channel_data_type,
                              &host_channels, &host_elem_size);
    
  char* temp = (char*) malloc (image->size);
    
  cl_device_id device_id = command_queue->device;

  device_id->ops->read
    (device_id->data, 
     temp, 
     image->device_ptrs[device_id->dev_id].mem_ptr, src_offset,
     image->size); 
            
  cl_int ret_code = pocl_write_image (image, command_queue->device, dst_origin,
                                      region, 0, 0, temp+src_offset);
    
  POCL_MEM_FREE(temp);
  POCL_UPDATE_EVENT_COMPLETE(event);
  return ret_code;
}
POsym(clEnqueueCopyBufferToImage) 
