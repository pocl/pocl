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

  switch (param_name)
    {
    case CL_EVENT_COMMAND_EXECUTION_STATUS:
      POCL_RETURN_GETINFO (cl_int, event->status);
    case CL_EVENT_COMMAND_QUEUE:
      POCL_RETURN_GETINFO(cl_command_queue, event->queue);
    case CL_EVENT_COMMAND_TYPE:
      POCL_RETURN_GETINFO(cl_command_type, event->command_type);
    case CL_EVENT_REFERENCE_COUNT:
      POCL_RETURN_GETINFO(cl_uint, event->pocl_refcount);
    case CL_EVENT_CONTEXT:
      POCL_RETURN_GETINFO(cl_context, event->context);
    default:
      break;
    }
  return CL_INVALID_VALUE;
}
POsym(clGetEventInfo)
