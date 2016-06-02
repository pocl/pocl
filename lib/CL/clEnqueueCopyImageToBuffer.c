#include "pocl_shared.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueCopyImageToBuffer)(cl_command_queue  command_queue ,
                           cl_mem            src_image ,
                           cl_mem            dst_buffer ,
                           const size_t *    src_origin ,
                           const size_t *    region ,
                           size_t            dst_offset ,
                           cl_uint           num_events_in_wait_list ,
                           const cl_event *  event_wait_list ,
                           cl_event *        event ) CL_API_SUFFIX__VERSION_1_0
{
  /* pass dst_origin through in a format pocl_rect_copy understands */
  const size_t dst_origin[3] = { dst_offset, 0, 0};
  return pocl_rect_copy(command_queue, CL_COMMAND_COPY_IMAGE_TO_BUFFER,
    src_image, CL_TRUE,
    dst_buffer, CL_FALSE,
    src_origin, dst_origin, region,
    0, 0,
    0, 0,
    num_events_in_wait_list, event_wait_list,
    event);
}
POsym(clEnqueueCopyImageToBuffer)


