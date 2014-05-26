#include "pocl_cl.h"
#include "pocl_image_util.h"

#define POCL_RETURN_IMAGE_INFO(__TYPE__, __VALUE__)                 \
  {                                                                 \
    size_t const value_size = sizeof(__TYPE__);                     \
    if (param_value)                                                \
      {                                                             \
        if (param_value_size < value_size) return CL_INVALID_VALUE; \
        *(__TYPE__*)param_value = __VALUE__;                        \
      }                                                             \
    if (param_value_size_ret)                                       \
      *param_value_size_ret = value_size;                           \
    return CL_SUCCESS;                                              \
  } 

CL_API_ENTRY cl_int CL_API_CALL
POname(clGetImageInfo)(cl_mem            image ,
                       cl_image_info     param_name , 
                       size_t            param_value_size ,
                       void *            param_value ,
                       size_t *          param_value_size_ret ) 
CL_API_SUFFIX__VERSION_1_0
{
  cl_image_format image_format = {image->image_channel_order, 
                                  image->image_channel_data_type};
  switch (param_name)
    {
    case CL_IMAGE_FORMAT:
      POCL_RETURN_IMAGE_INFO (cl_image_format, image_format);
    case CL_IMAGE_ELEMENT_SIZE:
      POCL_RETURN_IMAGE_INFO (size_t, image->image_channels * image->image_elem_size);
    case CL_IMAGE_ROW_PITCH:
      POCL_RETURN_IMAGE_INFO (size_t, image->image_row_pitch);
    case CL_IMAGE_SLICE_PITCH:
      POCL_RETURN_IMAGE_INFO (size_t, image->image_slice_pitch);
    case CL_IMAGE_WIDTH:
      POCL_RETURN_IMAGE_INFO (size_t, image->image_width);
    case CL_IMAGE_HEIGHT:
      POCL_RETURN_IMAGE_INFO (size_t, image->image_height);
    case CL_IMAGE_DEPTH:
      POCL_RETURN_IMAGE_INFO (size_t, image->image_depth);
    case CL_IMAGE_ARRAY_SIZE:
      POCL_RETURN_IMAGE_INFO (size_t, image->image_array_size);
    case CL_IMAGE_BUFFER:
      POCL_RETURN_IMAGE_INFO (cl_mem, image->buffer);
    case CL_IMAGE_NUM_MIP_LEVELS:
      POCL_RETURN_IMAGE_INFO (cl_uint, image->num_mip_levels);
    case CL_IMAGE_NUM_SAMPLES:
      POCL_RETURN_IMAGE_INFO (cl_uint, image->num_samples);
    default:
      return CL_INVALID_VALUE;
    }

}
POsym(clGetImageInfo)
