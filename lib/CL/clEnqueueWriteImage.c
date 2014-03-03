#include "pocl_cl.h"
#include "pocl_image_util.h"
#include "pocl_util.h"
#include "utlist.h"
#include <string.h>

extern CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueWriteImage)(cl_command_queue    command_queue,
                    cl_mem              image,
                    cl_bool             blocking_write, 
                    const size_t *      origin, /*[3]*/
                    const size_t *      region, /*[3]*/
                    size_t              host_row_pitch,
                    size_t              host_slice_pitch, 
                    const void *        ptr,
                    cl_uint             num_events_in_wait_list,
                    const cl_event *    event_wait_list,
                    cl_event *          event) CL_API_SUFFIX__VERSION_1_0
{
  cl_int status;
  int num_channels;
  int elem_size;
  _cl_command_node *cmd;

  if (image == NULL)
    return CL_INVALID_MEM_OBJECT;

  status = pocl_check_image_origin_region (image, origin, region);
  if (status != CL_SUCCESS)
    return status;

  if (command_queue == NULL)
    return CL_INVALID_COMMAND_QUEUE;

  if (command_queue->context != image->context)
    return CL_INVALID_CONTEXT;

  if (ptr == NULL)
    return CL_INVALID_VALUE;

  pocl_get_image_information(image->image_channel_order,
                             image->image_channel_data_type,
                             &num_channels, &elem_size);

  size_t tuned_origin[3] = {origin[0] * elem_size * num_channels, origin[1], 
                            origin[2]};
  size_t tuned_region[3] = {region[0] * elem_size * num_channels, region[1], 
                            region[2]};
  status = pocl_create_command (&cmd, command_queue, CL_COMMAND_WRITE_IMAGE, 
                                event, num_events_in_wait_list, 
                                event_wait_list);
  if (status != CL_SUCCESS)
    {
      if (event)
        free (*event);
      return status;
    }  

  cmd->command.rw_image.device_ptr = 
    image->device_ptrs[command_queue->device->dev_id].mem_ptr;
  cmd->command.rw_image.host_ptr = (void*) ptr;
  memcpy ((cmd->command.map_image.origin), tuned_origin, 3*sizeof (size_t));
  memcpy ((cmd->command.map_image.region), tuned_region, 3*sizeof (size_t));
  cmd->command.rw_image.rowpitch = image->image_row_pitch;
  cmd->command.rw_image.slicepitch = image->image_slice_pitch;
  cmd->command.rw_image.buffer = image;
  pocl_command_enqueue(command_queue, cmd);
  
  if (blocking_write)
    status = POname(clFinish) (command_queue);
    
  return status;
}
POsym(clEnqueueWriteImage)
