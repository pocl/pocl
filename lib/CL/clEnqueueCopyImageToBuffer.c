#include "pocl_cl.h"


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
  POCL_ABORT_UNIMPLEMENTED("The entire clEnqueueCopyImageToBuffer call");
  return CL_SUCCESS;
}
POsym(clEnqueueCopyImageToBuffer)
    

