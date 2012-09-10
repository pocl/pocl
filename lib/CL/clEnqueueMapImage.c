#include "pocl_cl.h"

CL_API_ENTRY void * CL_API_CALL
POclEnqueueMapImage(cl_command_queue   command_queue ,
                  cl_mem             image , 
                  cl_bool            blocking_map , 
                  cl_map_flags       map_flags , 
                  const size_t *     origin ,
                  const size_t *     region ,
                  size_t *           image_row_pitch ,
                  size_t *           image_slice_pitch ,
                  cl_uint            num_events_in_wait_list ,
                  const cl_event *   event_wait_list ,
                  cl_event *         event ,
                  cl_int *           errcode_ret ) CL_API_SUFFIX__VERSION_1_0
{
  POCL_ABORT_UNIMPLEMENTED();
  return CL_SUCCESS;
}
POsym(clEnqueueMapImage)
