#include "pocl_cl.h"
CL_API_ENTRY cl_int CL_API_CALL
POname(clSetUserEventStatus)(cl_event    event ,
                     cl_int      execution_status ) CL_API_SUFFIX__VERSION_1_1
{
  if(event == NULL)
    return CL_INVALID_EVENT;

  event->status = execution_status;
  
  return CL_SUCCESS;
}
POsym(clSetUserEventStatus)
