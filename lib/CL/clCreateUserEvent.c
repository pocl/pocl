#include "pocl_cl.h"


CL_API_ENTRY cl_event CL_API_CALL
POname(clCreateUserEvent)(cl_context     context ,
                  cl_int *       errcode_ret ) CL_API_SUFFIX__VERSION_1_1 
{
  POCL_ABORT_UNIMPLEMENTED("The entire clCreateUserEvent call");
  return CL_SUCCESS;
}
POsym(clCreateUserEvent)
