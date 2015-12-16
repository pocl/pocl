#include "pocl_cl.h"
#include "pocl_util.h"
#include <sys/time.h>

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

#ifndef _MSC_VER
  struct timeval current;
  gettimeofday(&current, NULL);
  event->time_queue = (current.tv_sec * 1000000 + current.tv_usec)*1000;
#else
  FILETIME ft;
  cl_ulong tmpres = 0;
  GetSystemTimeAsFileTime(&ft);
  tmpres |= ft.dwHighDateTime;
  tmpres <<= 32;
  tmpres |= ft.dwLowDateTime;
  tmpres -= 11644473600000000Ui64;
  tmpres /= 10;
  event->time_queue = tmpres;
#endif
    
  if (errcode_ret)
    *errcode_ret = error;

  return event;
}
POsym(clCreateUserEvent)
