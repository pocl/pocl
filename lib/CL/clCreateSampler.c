#include "pocl_cl.h"
#include "pocl_icd.h"
extern CL_API_ENTRY cl_sampler CL_API_CALL
POname(clCreateSampler)(cl_context          context,
                cl_bool             normalized_coords, 
                cl_addressing_mode  addressing_mode, 
                cl_filter_mode      filter_mode,
                cl_int *            errcode_ret)
CL_API_SUFFIX__VERSION_1_0
{
  int errcode;
  cl_sampler sampler;

  POCL_GOTO_ERROR_COND((context == NULL), CL_INVALID_CONTEXT);
  
  sampler = (cl_sampler) malloc(sizeof(struct _cl_sampler));
  if (sampler == NULL)
  {
    errcode = CL_OUT_OF_HOST_MEMORY;
    goto ERROR;
  }
  
  if (normalized_coords == CL_TRUE)
    POCL_ABORT_UNIMPLEMENTED("clCreateSampler: normalized_coords");
  
  if (addressing_mode != CL_ADDRESS_CLAMP_TO_EDGE)
    POCL_ABORT_UNIMPLEMENTED("clCreateSampler: Addressing modes "
                              "other than CL_ADDRESS_CLAMP_TO_EDGE");
  
  if (filter_mode != CL_FILTER_NEAREST)
    POCL_ABORT_UNIMPLEMENTED("clCreateSampler: Filter modes other than "
                                    "CL_FILTER_NEAREST");
  
  POCL_INIT_ICD_OBJECT(sampler);
  sampler->normalized_coords = normalized_coords;
  sampler->addressing_mode = addressing_mode;
  sampler->filter_mode = filter_mode;
  
  return sampler;

ERROR:
  if(errcode_ret)
  {
    *errcode_ret = errcode;
  }
    return NULL;
}
POsym(clCreateSampler)
