#include "pocl_cl.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clRetainEvent)(cl_event  event ) CL_API_SUFFIX__VERSION_1_0
{
  if (event == NULL || event->queue == NULL)
    return CL_INVALID_EVENT;

  POCL_RETAIN_OBJECT(event);

  return CL_SUCCESS;
}

POsym(clRetainEvent)
