#include "pocl_cl.h"

#define POCL_RETURN_EVENT_INFO(__TYPE__, __VALUE__)                 \
  {                                                                 \
    size_t const value_size = sizeof(__TYPE__);                     \
    if (param_value)                                                \
      {                                                             \
        if (param_value_size < value_size) return CL_INVALID_VALUE; \
        *(__TYPE__*)param_value = __VALUE__;                        \
      }                                                             \
    if (param_value_size_ret)                                       \
      *param_value_size_ret = value_size;                           \
    return CL_SUCCESS;                                              \
  } 

CL_API_ENTRY cl_int CL_API_CALL
POclGetEventInfo(cl_event          event ,
               cl_event_info     param_name ,
               size_t            param_value_size ,
               void *            param_value ,
               size_t *          param_value_size_ret ) CL_API_SUFFIX__VERSION_1_0
{
  switch (param_name)
  {
  case CL_EVENT_COMMAND_EXECUTION_STATUS:
    POCL_RETURN_EVENT_INFO (cl_int, event->status);
  default:
    break;
  }
  POCL_ABORT_UNIMPLEMENTED();
  return CL_SUCCESS;
}
POsym(clGetEventInfo)
