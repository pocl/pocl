#include "pocl_cl.h"
extern CL_API_ENTRY cl_int CL_API_CALL
POname(clReleaseSampler)(cl_sampler sampler)
CL_API_SUFFIX__VERSION_1_0
{
  return CL_SUCCESS;
}
POsym(clReleaseSampler)
