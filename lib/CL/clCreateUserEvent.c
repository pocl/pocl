#include "pocl_cl.h"
#include "pocl_util.h"
#include "pocl_timing.h"

CL_API_ENTRY cl_event CL_API_CALL
POname(clCreateUserEvent)(cl_context     context ,
                  cl_int *       errcode_ret ) CL_API_SUFFIX__VERSION_1_1
{
  int errcode;
  cl_event event = NULL;

  POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (context)), CL_INVALID_CONTEXT);

  errcode = pocl_create_event (&event, 0, CL_COMMAND_USER, 0, NULL, context);

  if (errcode != CL_SUCCESS)
    {
      POCL_MEM_FREE(event);
    }
  else
    {
      event->pocl_refcount = 1;
      event->status = CL_SUBMITTED;
      event->context = context;
      pocl_user_event_data *p
          = (pocl_user_event_data *)malloc (sizeof (pocl_user_event_data));
      assert (p);
      POCL_INIT_COND (p->wakeup_cond);
      event->data = p;
    }
ERROR:
  if (errcode_ret)
    *errcode_ret = errcode;

  return event;
}
POsym(clCreateUserEvent)
