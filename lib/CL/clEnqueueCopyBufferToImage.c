#include "pocl_shared.h"

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
  /* pass src_origin through in a format pocl_rect_copy understands */
  const size_t src_origin[3] = { src_offset, 0, 0};
  return pocl_rect_copy(command_queue, CL_COMMAND_COPY_BUFFER_TO_IMAGE,
    buffer, CL_FALSE,
    image, CL_TRUE,
    src_origin, dst_origin, region,
    0, 0,
    0, 0,
    num_events_in_wait_list, event_wait_list,
    event);
}
POsym(clEnqueueCopyBufferToImage)
