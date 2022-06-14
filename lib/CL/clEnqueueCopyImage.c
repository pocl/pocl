#include "pocl_util.h"
#include "pocl_shared.h"
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
  _cl_command_node *cmd = NULL;
  cl_device_id device;
  cl_int errcode;
  unsigned i;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  errcode = pocl_validate_copy_image (src_image, dst_image);
  if (errcode != CL_SUCCESS)
    return errcode;

  POCL_CHECK_DEV_IN_CMDQ;

  cl_int err = pocl_rect_copy(
    command_queue,
    CL_COMMAND_COPY_IMAGE,
    src_image, CL_TRUE,
    dst_image, CL_TRUE,
    src_origin, dst_origin, region,
    0, 0,
    0, 0,
    num_events_in_wait_list, event_wait_list,
    event,
    &cmd);

  if (err != CL_SUCCESS)
    return err;

  POCL_FILL_COMMAND_COPY_IMAGE;

  pocl_command_enqueue (command_queue, cmd);

  return CL_SUCCESS;
}
POsym(clEnqueueCopyImage)
