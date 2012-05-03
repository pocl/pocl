#include "pocl_cl.h"

CL_API_ENTRY cl_int CL_API_CALL
clWaitForEvents(cl_uint              num_events ,
                const cl_event *     event_list ) CL_API_SUFFIX__VERSION_1_0
{
  int event_i;
  // dummy implementation, waits until *all* events have completed.
  for (event_i = 0; event_i < num_events; ++event_i)
    {
      clFinish(event_list[event_i]->queue);
    }
  return CL_SUCCESS;
}

