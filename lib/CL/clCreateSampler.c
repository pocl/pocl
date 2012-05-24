#include "pocl_cl.h"

CL_API_ENTRY cl_sampler CL_API_CALL
clCreateSampler(cl_context           context ,
                cl_bool              normalized_coords , 
                cl_addressing_mode   addressing_mode , 
                cl_filter_mode       filter_mode ,
                cl_int *             errcode_ret ) CL_API_SUFFIX__VERSION_1_0
{
  POCL_ABORT_UNIMPLEMENTED();
  return CL_SUCCESS;
}

