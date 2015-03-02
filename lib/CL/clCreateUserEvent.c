#include "pocl_cl.h"


CL_API_ENTRY cl_event CL_API_CALL
POname(clCreateUserEvent)(cl_context     context ,
                  cl_int *       errcode_ret ) CL_API_SUFFIX__VERSION_1_1 
{
  int error; 
  
  cl_event event;
  error = pocl_create_event (&event, 0, CL_COMMAND_USER);

  event->status = CL_QUEUED;

  if (error != CL_SUCCESS)
    {
      if (errcode_ret)
        *errcode_ret = error;

      return NULL;
    }
  
  return event;
}
POsym(clCreateUserEvent)
