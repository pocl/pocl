#include "pocl_cl.h"
#include "pocl_image_util.h"

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
  if (event != NULL)
    {
      *event = (cl_event)malloc(sizeof(struct _cl_event));
      if (*event == NULL)
	return CL_OUT_OF_HOST_MEMORY; 
      POCL_INIT_OBJECT(*event);
      (*event)->queue = command_queue;
      (*event)->command_type = CL_COMMAND_READ_BUFFER;
      
      POname(clRetainCommandQueue) (command_queue);
      POCL_UPDATE_EVENT_QUEUED;
      POCL_UPDATE_EVENT_SUBMITTED;
      POCL_UPDATE_EVENT_RUNNING;
    }
  status = pocl_write_image(image,
			    command_queue->device,
			    origin,
			    region,
			    host_row_pitch,
			    host_slice_pitch, 
			    ptr);
  POCL_UPDATE_EVENT_COMPLETE;
  return status;
}
POsym(clEnqueueWriteImage)
