#include "pocl_cl.h"
#include "pocl_timing.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clSetUserEventStatus)(cl_event event ,
                             cl_int   execution_status ) 
CL_API_SUFFIX__VERSION_1_1
{
  /* Must be a valid user event */
  POCL_RETURN_ERROR_COND((event == NULL), CL_INVALID_EVENT);
  POCL_RETURN_ERROR_COND((event->command_type != CL_COMMAND_USER), CL_INVALID_EVENT);
  /* Can only be set to CL_COMPLETE (0) or negative values */
  POCL_RETURN_ERROR_COND((execution_status > CL_COMPLETE), CL_INVALID_VALUE);
  /* Can only be done once */
  POCL_RETURN_ERROR_COND((event->status <= CL_COMPLETE), CL_INVALID_OPERATION);

  POCL_LOCK_OBJ (event);
  event->status = execution_status;
  if (execution_status == CL_COMPLETE)
    {
      pocl_broadcast (event);
      pocl_event_updated (event, CL_COMPLETE);
    }
  POCL_UNLOCK_OBJ (event);
  return CL_SUCCESS;
}
POsym(clSetUserEventStatus)
