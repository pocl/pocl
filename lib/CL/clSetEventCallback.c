#include "pocl_cl.h"

CL_API_ENTRY cl_int CL_API_CALL
clSetEventCallback( cl_event     event ,
                    cl_int       command_exec_callback_type ,
                    void (CL_CALLBACK *  pfn_notify )(cl_event, cl_int, void *),
                    void *       user_data ) CL_API_SUFFIX__VERSION_1_1
{
  POCL_ABORT_UNIMPLEMENTED();
  return CL_SUCCESS;
}

