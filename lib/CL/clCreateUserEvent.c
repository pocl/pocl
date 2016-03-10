#include "pocl_cl.h"
#include "pocl_util.h"
#include "pocl_timing.h"

CL_API_ENTRY cl_event CL_API_CALL
POname(clCreateUserEvent)(cl_context     context ,
                  cl_int *       errcode_ret ) CL_API_SUFFIX__VERSION_1_1
{
  int error; 
  cl_event event = NULL;

  error = pocl_create_event (&event, 0, CL_COMMAND_USER, 0, NULL, context);
  
  event->context = context;
  event->status = CL_QUEUED;

  if (error != CL_SUCCESS)
    {
      POCL_MEM_FREE(event);
    }
  else
    {
      event->status = CL_SUBMITTED;
    }

  event->time_queue = pocl_gettimemono_ns();
    
  if (errcode_ret)
    *errcode_ret = error;

  return event;
}
POsym(clCreateUserEvent)
