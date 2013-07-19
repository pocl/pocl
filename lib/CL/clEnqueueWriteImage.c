#include "pocl_cl.h"
#include "pocl_image_util.h"
#include "pocl_util.h"

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
  cl_int status;
  int errcode;

  if (image == NULL)
    return CL_INVALID_MEM_OBJECT;

  status = pocl_check_image_origin_region (image, origin, region);
  if (status != CL_SUCCESS)
    return status;

  if (command_queue == NULL)
    return CL_INVALID_COMMAND_QUEUE;

  if (command_queue->context != image->context)
    return CL_INVALID_CONTEXT;

  if (ptr == NULL)
    return CL_INVALID_VALUE;

  if (event != NULL)
    {
      errcode = pocl_create_event (event, command_queue, 
                                   CL_COMMAND_WRITE_IMAGE, 
                                   num_events_in_wait_list, 
                                   event_wait_list);
      if (errcode != CL_SUCCESS)
        return errcode;
      
      POCL_UPDATE_EVENT_QUEUED;
      POname(clRetainCommandQueue) (command_queue);
    }      
  if (blocking_write)
    {
      POCL_UPDATE_EVENT_SUBMITTED;
      POCL_UPDATE_EVENT_RUNNING;
      status = pocl_write_image(image, command_queue->device, origin, region,
                                host_row_pitch, host_slice_pitch, ptr);
      POCL_UPDATE_EVENT_COMPLETE;
    }
  else
    {
      POCL_ABORT_UNIMPLEMENTED();
    }

  return status;
}
POsym(clEnqueueWriteImage)
