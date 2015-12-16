#include "pocl_cl.h"
#ifndef _MSC_VER
#  include <unistd.h>
#  include <sys/time.h>
#else
#  include "vccompat.hpp"
#endif

CL_API_ENTRY cl_int CL_API_CALL
POname(clSetUserEventStatus)(cl_event event ,
                             cl_int   execution_status ) 
CL_API_SUFFIX__VERSION_1_1
{
  /* Must be a valid user event */
  if (event == NULL || event->command_type != CL_COMMAND_USER)
    return CL_INVALID_EVENT;
  /* Can only be set to CL_COMPLETE (0) or negative values */
  if (execution_status > CL_COMPLETE)
    return CL_INVALID_VALUE;
  /* Can only be done once */
  if (event->status <= CL_COMPLETE)
    return CL_INVALID_OPERATION;

  event->status = execution_status;
  if (execution_status == CL_COMPLETE)
    {
#ifndef _MSC_VER
      struct timeval current;
      gettimeofday(&current, NULL);
      event->time_end = (current.tv_sec * 1000000 + current.tv_usec)*1000;
#else
      FILETIME ft;
      cl_ulong tmpres = 0;
      GetSystemTimeAsFileTime(&ft);
      tmpres |= ft.dwHighDateTime;
      tmpres <<= 32;
      tmpres |= ft.dwLowDateTime;
      tmpres -= 11644473600000000Ui64;
      tmpres /= 10;
      time_end = tmpres;
#endif
      event->time_start = event->time_end;
      POCL_LOCK_OBJ (event);
      pocl_broadcast (event);
      POCL_UNLOCK_OBJ (event);
    }
  return CL_SUCCESS;
}
POsym(clSetUserEventStatus)
