#include "pocl_cl.h"

extern CL_API_ENTRY cl_mem CL_API_CALL
clCreateImage(cl_context              context,
              cl_mem_flags            flags,
              const cl_image_format * image_format,
              const cl_image_desc *   image_desc, 
              void *                  host_ptr,
              cl_int *                errcode_ret) 
CL_API_SUFFIX__VERSION_1_2
{
  if( image_desc->num_mip_levels != 0 
    || image_desc->num_samples != 0 )
    POCL_ABORT_UNIMPLEMENTED();
  
  if( image_desc->image_type != CL_MEM_OBJECT_IMAGE2D )
    POCL_ABORT_UNIMPLEMENTED();
  
  clCreateImage2D( context,
                   flags,
                   image_format,
                   image_desc->image_width,
                   image_desc->image_height,
                   image_desc->image_row_pitch,
                   host_ptr,
                   errcode_ret );
}
