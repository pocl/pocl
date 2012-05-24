#include "pocl_cl.h"

CL_API_ENTRY cl_int CL_API_CALL
clGetSupportedImageFormats(cl_context            context ,
                           cl_mem_flags          flags ,
                           cl_mem_object_type    image_type ,
                           cl_uint               num_entries ,
                           cl_image_format *     image_formats ,
                           cl_uint *             num_image_formats ) CL_API_SUFFIX__VERSION_1_0
{
  POCL_ABORT_UNIMPLEMENTED();
  return CL_SUCCESS;
}

