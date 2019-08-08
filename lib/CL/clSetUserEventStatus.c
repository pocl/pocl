#include "pocl_cl.h"
#include "pocl_timing.h"
#include "pocl_tracing.h"
#include "common.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clSetUserEventStatus)(cl_event event ,
                             cl_int   execution_status ) 
CL_API_SUFFIX__VERSION_1_1
{
  int errcode = CL_SUCCESS;
  /* Must be a valid user event */
  POCL_RETURN_ERROR_COND ((event == NULL), CL_INVALID_EVENT);
  /* Can only be set to CL_COMPLETE (0) or negative values */
  POCL_RETURN_ERROR_COND ((execution_status > CL_COMPLETE), CL_INVALID_VALUE);

  POCL_LOCK_OBJ (event);

  POCL_GOTO_ERROR_COND ((event->command_type != CL_COMMAND_USER),
                        CL_INVALID_EVENT);
  /* Can only be done once */
  POCL_GOTO_ERROR_COND ((event->status <= CL_COMPLETE), CL_INVALID_OPERATION);

  event->status = execution_status;
  POCL_UNLOCK_OBJ (event);

  if (execution_status <= CL_COMPLETE)
    {
      POCL_MSG_PRINT_EVENTS ("User event %zu completed with status: %i\n",
                             event->id, execution_status);
      pocl_broadcast (event);
    }

  POCL_LOCK_OBJ (event);
  pocl_event_updated (event, execution_status);
  pocl_user_event_data *p = event->data;
  if (execution_status <= CL_COMPLETE)
    pthread_cond_broadcast (&p->wakeup_cond);

ERROR:
  POCL_UNLOCK_OBJ (event);
  return errcode;
}
POsym(clSetUserEventStatus)
