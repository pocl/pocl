#include "pocl_util.h"
#include "pocl_shared.h"
#include "pocl_image_util.h"

extern CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueCopyBufferToImage)(cl_command_queue  command_queue,
                                   cl_mem            src_buffer,
                                   cl_mem            dst_image,
                                   size_t            src_offset,
                                   const size_t *    dst_origin, /*[3]*/
                                   const size_t *    region,  /*[3]*/
                                   cl_uint           num_events_in_wait_list,
                                   const cl_event *  event_wait_list,
                                   cl_event *        event )
CL_API_SUFFIX__VERSION_1_0
{
  cl_int errcode;
  _cl_command_node *cmd = NULL;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  errcode = pocl_copy_buffer_to_image_common (
      NULL, command_queue, src_buffer, dst_image, src_offset, dst_origin,
      region, num_events_in_wait_list, event_wait_list, event, NULL, NULL,
      NULL, &cmd);
  if (errcode != CL_SUCCESS)
    return errcode;

  if (cmd)
    pocl_command_enqueue (command_queue, cmd);

  return CL_SUCCESS;
}
POsym(clEnqueueCopyBufferToImage)
