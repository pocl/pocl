#include "pocl_util.h"
#include "pocl_image_util.h"


CL_API_ENTRY cl_int CL_API_CALL
POname(clGetImageInfo)(cl_mem            image ,
                       cl_image_info     param_name , 
                       size_t            param_value_size ,
                       void *            param_value ,
                       size_t *          param_value_size_ret ) 
CL_API_SUFFIX__VERSION_1_0
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (image)),
                          CL_INVALID_MEM_OBJECT);

  cl_image_format image_format = {image->image_channel_order, 
                                  image->image_channel_data_type};
  switch (param_name)
    {
    case CL_IMAGE_FORMAT:
      POCL_RETURN_GETINFO (cl_image_format, image_format);
    case CL_IMAGE_ELEMENT_SIZE:
      POCL_RETURN_GETINFO (size_t, image->image_channels * image->image_elem_size);
    case CL_IMAGE_ROW_PITCH:
      POCL_RETURN_GETINFO (size_t, image->image_row_pitch);
    case CL_IMAGE_SLICE_PITCH:
      POCL_RETURN_GETINFO (size_t, image->image_slice_pitch);
    case CL_IMAGE_WIDTH:
      POCL_RETURN_GETINFO (size_t, image->image_width);
    case CL_IMAGE_HEIGHT:
      POCL_RETURN_GETINFO (size_t, image->image_height);
    case CL_IMAGE_DEPTH:
      POCL_RETURN_GETINFO (size_t, image->image_depth);
    case CL_IMAGE_ARRAY_SIZE:
      POCL_RETURN_GETINFO (size_t, image->image_array_size);
    case CL_IMAGE_BUFFER:
      POCL_RETURN_GETINFO (cl_mem, image->buffer);
    case CL_IMAGE_NUM_MIP_LEVELS:
      POCL_RETURN_GETINFO (cl_uint, image->num_mip_levels);
    case CL_IMAGE_NUM_SAMPLES:
      POCL_RETURN_GETINFO (cl_uint, image->num_samples);
    default:
      return CL_INVALID_VALUE;
    }

}
POsym(clGetImageInfo)
