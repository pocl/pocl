#include "pocl_cl.h"
#include "pocl_image_util.h"

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteImage(cl_command_queue    command_queue,
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
    return pocl_write_image   (image,
                      command_queue->device,
                      origin,
                      region,
                      host_row_pitch,
                      host_slice_pitch, 
                      ptr);
  }

