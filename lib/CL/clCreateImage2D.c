#include "pocl_cl.h"
CL_API_ENTRY cl_mem CL_API_CALL
clCreateImage2D(cl_context              context,
                cl_mem_flags            flags,
                const cl_image_format * image_format,
                size_t                  image_width,
                size_t                  image_height,
                size_t                  image_row_pitch, 
                void *                  host_ptr,
                cl_int *                errcode_ret)
CL_API_SUFFIX__VERSION_1_0
{
  POCL_ABORT_UNIMPLEMENTED();
  return CL_SUCCESS;
}
    
 
