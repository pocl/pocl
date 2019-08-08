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

  if (IS_IMAGE1D_BUFFER (image))
    {
      IMAGE1D_ORIG_REG_TO_BYTES (image, origin, region);
      return POname (clEnqueueWriteBuffer) (
          command_queue, image,
          blocking_write,
          i1d_origin[0], i1d_region[0],
          ptr,
          num_events_in_wait_list, event_wait_list, event);
    }

  POCL_RETURN_ERROR_ON((command_queue->context != image->context),
    CL_INVALID_CONTEXT, "image and command_queue are not from the same context\n");

  POCL_RETURN_ERROR_ON ((!image->is_image), CL_INVALID_MEM_OBJECT,
                        "image argument is not an image\n");
  POCL_RETURN_ON_UNSUPPORTED_IMAGE (image, command_queue->device);

  errcode = pocl_check_event_wait_list (command_queue, num_events_in_wait_list,
                                        event_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;

  POCL_RETURN_ERROR_ON (
      (image->flags & (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS)),
      CL_INVALID_OPERATION,
      "image buffer has been created with CL_MEM_HOST_READ_ONLY "
      "or CL_MEM_HOST_NO_ACCESS\n");

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

  errcode = pocl_create_command (&cmd, command_queue, CL_COMMAND_WRITE_IMAGE,
                                event, num_events_in_wait_list, 
                                event_wait_list, 1, &image);
  if (errcode != CL_SUCCESS)
    {
      return errcode;
    }  


  cl_device_id dev = command_queue->device;

  cmd->command.write_image.dst_mem_id = &image->gmem_ptrs[dev->global_mem_id];
  cmd->command.write_image.src_host_ptr = ptr;
  cmd->command.write_image.src_mem_id = NULL;

  cmd->command.write_image.origin[0] = origin[0];
  cmd->command.write_image.origin[1] = origin[1];
  cmd->command.write_image.origin[2] = origin[2];
  cmd->command.write_image.region[0] = region[0];
  cmd->command.write_image.region[1] = region[1];
  cmd->command.write_image.region[2] = region[2];

  cmd->command.write_image.src_row_pitch = input_row_pitch;
  cmd->command.write_image.src_slice_pitch = input_slice_pitch;
  cmd->command.write_image.src_offset = 0;

  POname(clRetainMemObject) (image);
  image->owning_device = command_queue->device;
  pocl_command_enqueue(command_queue, cmd);

  if (blocking_write)
    errcode = POname(clFinish) (command_queue);

  return errcode;
}
POsym(clEnqueueWriteImage)
