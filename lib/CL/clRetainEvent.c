#include "pocl_cl.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clRetainEvent)(cl_event  event ) CL_API_SUFFIX__VERSION_1_0
{
  POCL_RETURN_ERROR_COND((event == NULL), CL_INVALID_EVENT);

  POCL_RETAIN_OBJECT(event);
  POCL_MSG_PRINT_REFCOUNTS ("Retain Event %p  : %d\n", event, event->pocl_refcount);

  return CL_SUCCESS;
}

POsym(clRetainEvent)
