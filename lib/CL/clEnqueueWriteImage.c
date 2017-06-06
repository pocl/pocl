#include "pocl_util.h"
#include "pocl_image_util.h"

extern CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueWriteImage)(cl_command_queue    command_queue,
                    cl_mem              image,
                    cl_bool             blocking_write, 
                    const size_t *      origin, /*[3]*/
                    const size_t *      region, /*[3]*/
                    size_t              input_row_pitch,
                    size_t              input_slice_pitch,
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

  POCL_RETURN_ERROR_ON (
      (!command_queue->device->image_support), CL_INVALID_OPERATION,
      "Device %s does not support images\n", command_queue->device->long_name);

  errcode = pocl_check_event_wait_list (command_queue, num_events_in_wait_list,
                                        event_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;

  if (image->buffer)
    POCL_RETURN_ERROR_ON (
        (image->buffer->flags
         & (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS)),
        CL_INVALID_OPERATION,
        "image buffer has been created with CL_MEM_HOST_READ_ONLY "
        "or CL_MEM_HOST_NO_ACCESS\n");

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
  cmd->command.write_image.h_rowpitch
      = (input_row_pitch ? input_row_pitch : tuned_region[0]);
  cmd->command.write_image.h_slicepitch
      = (input_slice_pitch ? input_slice_pitch
                           : (tuned_region[0] * region[1]));
  cmd->command.write_image.buffer = image;

  POname(clRetainMemObject) (image);
  image->owning_device = command_queue->device;
  pocl_command_enqueue(command_queue, cmd);

  if (blocking_write)
    errcode = POname(clFinish) (command_queue);

  return errcode;
}
POsym(clEnqueueWriteImage)
