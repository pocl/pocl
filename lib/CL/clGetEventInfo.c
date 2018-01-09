#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clGetEventInfo)(cl_event          event ,
                       cl_event_info     param_name ,
                       size_t            param_value_size ,
                       void *            param_value ,
                       size_t *          param_value_size_ret ) 
CL_API_SUFFIX__VERSION_1_0
{
  POCL_RETURN_ERROR_COND((event == NULL), CL_INVALID_EVENT);
  POCL_LOCK_OBJ (event);
  cl_int s = event->status;
  cl_command_queue q = event->queue;
  cl_command_type t = event->command_type;
  cl_uint r = event->pocl_refcount;
  cl_context c = event->context;
  POCL_UNLOCK_OBJ (event);

  switch (param_name)
    {
    case CL_EVENT_COMMAND_EXECUTION_STATUS:
      POCL_RETURN_GETINFO (cl_int, s);
    case CL_EVENT_COMMAND_QUEUE:
      POCL_RETURN_GETINFO (cl_command_queue, q);
    case CL_EVENT_COMMAND_TYPE:
      POCL_RETURN_GETINFO (cl_command_type, t);
    case CL_EVENT_REFERENCE_COUNT:
      POCL_RETURN_GETINFO (cl_uint, r);
    case CL_EVENT_CONTEXT:
      POCL_RETURN_GETINFO (cl_context, c);
    default:
      break;
    }
  return CL_INVALID_VALUE;
}
POsym(clGetEventInfo)
