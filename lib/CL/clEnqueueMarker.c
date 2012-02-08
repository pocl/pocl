#include "pocl_cl.h"
CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMarker(cl_command_queue     command_queue,
                cl_event *           event ) 
CL_API_SUFFIX__VERSION_1_0
{
  POCL_ABORT_UNIMPLEMENTED();
  return CL_SUCCESS;
}
 
