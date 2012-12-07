#include "pocl_cl.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clCreateKernelsInProgram)(cl_program      program ,
                         cl_uint         num_kernels ,
                         cl_kernel *     kernels ,
                         cl_uint *       num_kernels_ret ) CL_API_SUFFIX__VERSION_1_0
{
  POCL_ABORT_UNIMPLEMENTED();
  return CL_SUCCESS;
}
POsym(clCreateKernelsInProgram)
