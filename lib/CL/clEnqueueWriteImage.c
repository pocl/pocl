#include "pocl_util.h"
#include "pocl_image_util.h"

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
  cl_int errcode;
  _cl_command_node *cmd;

  POCL_RETURN_ERROR_COND((command_queue == NULL), CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_COND((image == NULL), CL_INVALID_MEM_OBJECT);

  POCL_RETURN_ERROR_COND((ptr == NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_ON((command_queue->context != image->context),
    CL_INVALID_CONTEXT, "image and command_queue are not from the same context\n");

  POCL_RETURN_ERROR_COND((event_wait_list == NULL && num_events_in_wait_list > 0),
    CL_INVALID_EVENT_WAIT_LIST);

  POCL_RETURN_ERROR_COND((event_wait_list != NULL && num_events_in_wait_list == 0),
    CL_INVALID_EVENT_WAIT_LIST);

  errcode = pocl_check_device_supports_image(image, command_queue);
  if (errcode != CL_SUCCESS)
    return errcode;

  errcode = pocl_check_image_origin_region (image, origin, region);
  if (errcode != CL_SUCCESS)
    return errcode;

  size_t tuned_origin[3] = 
    {origin[0] * image->image_elem_size * image->image_channels, origin[1], 
     origin[2]};
  size_t tuned_region[3] = 
    {region[0] * image->image_elem_size * image->image_channels, region[1], 
     region[2]};

  errcode = pocl_create_command (&cmd, command_queue, CL_COMMAND_WRITE_IMAGE,
                                event, num_events_in_wait_list, 
                                event_wait_list, 1, &image);
  if (errcode != CL_SUCCESS)
    {
      return errcode;
    }  

  cmd->command.write_image.device_ptr = 
    image->device_ptrs[command_queue->device->dev_id].mem_ptr;
  cmd->command.write_image.host_ptr = (void*) ptr;
  memcpy ((cmd->command.write_image.origin), tuned_origin, 3*sizeof (size_t));
  memcpy ((cmd->command.write_image.region), tuned_region, 3*sizeof (size_t));
  cmd->command.write_image.b_rowpitch = image->image_row_pitch;
  cmd->command.write_image.b_slicepitch = image->image_slice_pitch;
  cmd->command.write_image.buffer = image;

  POname(clRetainMemObject) (image);
  pocl_command_enqueue(command_queue, cmd);

  if (blocking_write)
    errcode = POname(clFinish) (command_queue);

  return errcode;
}
POsym(clEnqueueWriteImage)
