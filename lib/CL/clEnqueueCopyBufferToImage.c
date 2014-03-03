#include "pocl_cl.h"
#include "pocl_image_util.h"
#include "pocl_util.h"

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

  if (region == NULL)
    return CL_INVALID_VALUE;
    
  if (region[2] != 1) //3D image
    POCL_ABORT_UNIMPLEMENTED();
    
  int host_elem_size;    
  int host_channels;
  pocl_get_image_information (image->image_channel_order,
                              image->image_channel_data_type,
                              &host_channels, &host_elem_size);
    
  void* temp = malloc (image->size);
    
  cl_device_id device_id = command_queue->device;

  device_id->ops->read
    (device_id->data, 
     temp, 
     image->device_ptrs[device_id->dev_id].mem_ptr, 
     image->size); 
            
  cl_int ret_code = pocl_write_image (image, command_queue->device, dst_origin,
                                      region, 0, 0, temp+src_offset);
    
  free (temp);
  POCL_UPDATE_EVENT_COMPLETE(event, command_queue);
  return ret_code;
}
POsym(clEnqueueCopyBufferToImage) 
