#include "pocl_cl.h"

extern CL_API_ENTRY cl_int CL_API_CALL
POname(clGetSupportedImageFormats)(cl_context           context,
                           cl_mem_flags         flags,
                           cl_mem_object_type   image_type,
                           cl_uint              num_entries,
                           cl_image_format *    image_formats,
                           cl_uint *            num_image_formats) CL_API_SUFFIX__VERSION_1_0
{


    int i, j, k;
    cl_device_id device_id;
    cl_image_format **dev_image_formats = 0;
    cl_uint *dev_num_image_formats = 0;
    int errcode = 0;
    
    cl_image_format reff;
    cl_image_format toReff;
    int reff_found;
    int formatCount = 0;
    
    if (context == NULL && context->num_devices == 0)
      return CL_INVALID_CONTEXT;
    
    if (image_type != CL_MEM_OBJECT_IMAGE2D)
      return CL_INVALID_VALUE;
    
    if (num_entries == 0 && image_formats != NULL)
      return CL_INVALID_VALUE;
    
    dev_image_formats = calloc ( context->num_devices, sizeof(void*) );
    dev_num_image_formats = calloc ( context->num_devices, sizeof(cl_uint));
    
    if (dev_image_formats == NULL || dev_num_image_formats == NULL)
      return CL_OUT_OF_HOST_MEMORY;

    for (i = 0; i < context->num_devices; ++i)
      {    
        device_id = context->devices[i];
        
        /* get num of entries */
        errcode = device_id->get_supported_image_formats(context, flags, 0, 
                                                         NULL, &dev_num_image_formats[i]);
        
        if ( errcode != CL_SUCCESS)
          goto CLEAN_MEM_N_RETURN;
        
        dev_image_formats[i] = malloc(dev_num_image_formats[i] * 
                                      sizeof(cl_image_format) );
        
        if( &dev_num_image_formats[i] == NULL)
          goto CLEAN_MEM_N_RETURN;

        /* get actual entries */
        errcode = device_id->get_supported_image_formats(context, flags, 
                                                      dev_num_image_formats[i], 
                                                      dev_image_formats[i], 
                                                      NULL);
        
        if (errcode != CL_SUCCESS)
          goto CLEAN_MEM_N_RETURN;
      }
    
    /* intersect of supported formats over devices */
    /* compare dev[0] sup. formats to all other devices sup. formats */
    for ( i = 0; i < dev_num_image_formats[0]; i++ )
      {
        reff_found = 1; /* init */
        reff = dev_image_formats[0][i];
        
        /* devices[1..*] */
        for (j = 1; i < context->num_devices && reff_found; j++)
          {
            reff_found = 0;
            /* sup. devices[j] image formats [0..*]   */
            for(k = 0; k < dev_num_image_formats[j]; k++)
              {
                toReff = dev_image_formats[j][k];
                if( reff.image_channel_order == toReff.image_channel_order &&
                    reff.image_channel_data_type == 
                    toReff.image_channel_data_type )
                  {
                    /* reff found in current device -> next device */
                    reff_found = 1;
                    break;
                  }
              }
          }
        if ( reff_found )
          { 
            /* if we get here reff is part of intersect */ 
            
            /* if second call */
            if ( image_formats != NULL && formatCount <= num_entries )
              image_formats[formatCount] = reff;
            
            ++formatCount;
          }   
      }
    
    if ( num_image_formats != NULL )
      {
        *num_image_formats = formatCount;
      }
    
    
CLEAN_MEM_N_RETURN:
    free ( dev_num_image_formats );
    for(i = 0; i < context->num_devices; i++)
      {
        free ( dev_image_formats[i] );
      }
    free ( dev_image_formats );
    return errcode;
    
} 

/*
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
        if (idx >= num_entries)
          return CL_SUCCESS;
        
        image_formats[idx].image_channel_order = supported_orders[i];
        image_formats[idx].image_channel_data_type = supported_types[j];
        
        idx++;
      }
      
   // Add special cases here if a channel order is supported with only some types or vice versa.
   *num_image_formats = idx;
   
   return CL_SUCCESS;
*/

POsym(clGetSupportedImageFormats)
