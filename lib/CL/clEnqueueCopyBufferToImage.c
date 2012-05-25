#include "pocl_cl.h"

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyBufferToImage(cl_command_queue  command_queue,
                           cl_mem            buffer,
                           cl_mem            image, 
                           size_t            src_offset,
                           const size_t *    dst_origin, /*[3]*/
                           const size_t *    region,  /*[3]*/
                           cl_uint           num_events_in_wait_list,
                           const cl_event *  event_wait_list,
                           cl_event *        event ) CL_API_SUFFIX__VERSION_1_0
  {
    if (region == NULL)
      return CL_INVALID_VALUE;
    
    if (region[2] != 1) //3D image
      POCL_ABORT_UNIMPLEMENTED();
    
    int dev_elem_size = sizeof(cl_float);
    int dev_channels = 4;
    
    int host_elem_size;    
    int host_channels;
    pocl_get_image_information (image, &host_channels, &host_elem_size);
    
    void* temp = malloc (image->size);
    
    cl_device_id device_id = command_queue->device;

    device_id->read
      (device_id->data, 
        temp, 
        image->device_ptrs[device_id->dev_id], 
        image->size); 
            
    cl_int ret_code = pocl_write_image   
      (image,
        command_queue->device,
        dst_origin,
        region,
        0,
        0, 
        temp+src_offset);
    
    free(temp);
    
    return ret_code;
  }
  