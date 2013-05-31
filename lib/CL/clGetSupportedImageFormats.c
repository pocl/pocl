#include "pocl_cl.h"

extern CL_API_ENTRY cl_int CL_API_CALL
POname(clGetSupportedImageFormats)(cl_context           context,
                           cl_mem_flags         flags,
                           cl_mem_object_type   image_type,
                           cl_uint              num_entries,
                           cl_image_format *    image_formats,
                           cl_uint *            num_image_formats) CL_API_SUFFIX__VERSION_1_0
{
  if (context == NULL)
    return CL_INVALID_CONTEXT;
  
  if (image_type != CL_MEM_OBJECT_IMAGE2D)
    return CL_INVALID_VALUE;
  
  if (num_entries==0 && image_formats!=NULL)
    return CL_INVALID_VALUE;
  
  int idx=0;
  
  const int supported_order_count = 2;
  cl_channel_order supported_orders[] = 
  {
    CL_RGBA,
    CL_R
  };
  
  const int supported_type_count = 2;
  cl_channel_type  supported_types[] =
  {
    CL_UNORM_INT8,
    CL_FLOAT
  };
  
  int i, j;
  for (i=0; i<supported_order_count; i++)
    for (j=0; j<supported_type_count; j++)
      {
        if (image_formats && idx < num_entries)
        {
          image_formats[idx].image_channel_order = supported_orders[i];
          image_formats[idx].image_channel_data_type = supported_types[j];
        }
        
        idx++;
      }
      
  /* Add special cases here if a channel order is supported with only some types or vice versa. */
  if (num_image_formats)
  {
    /* CL Standard:

       num_image_formats is the actual number of supported image formats for a
       specific context and values specified by flags. If num_image_formats is
       NULL, it is ignored. */
    *num_image_formats = idx;
  }
  
  return CL_SUCCESS;
}
POsym(clGetSupportedImageFormats)
