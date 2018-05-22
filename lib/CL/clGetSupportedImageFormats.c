/* OpenCL runtime library: clGetSupportedImageFormats()

   Copyright (c) 2013 Ville Korhonen / Tampere Univ. of Tech.
   
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "pocl_cl.h"

static int pocl_find_img_format (cl_image_format *toFind,
                                 const cl_image_format* list,
                                 int num_entries)
{
  int i;
  for (i = 0; i < num_entries; i++)
    {
      if (toFind->image_channel_order == list[i].image_channel_order &&
          toFind->image_channel_data_type == list[i].image_channel_data_type)
        {
          return 1;
        }
    }
  return 0;
}

extern CL_API_ENTRY cl_int CL_API_CALL
POname(clGetSupportedImageFormats) (cl_context           context,
                                    cl_mem_flags         flags,
                                    cl_mem_object_type   image_type,
                                    cl_uint              num_entries,
                                    cl_image_format *    image_formats,
                                    cl_uint *            num_image_formats) 
CL_API_SUFFIX__VERSION_1_0
{
  unsigned i, j;
  cl_device_id device_id;
  const cl_image_format **dev_image_formats = 0;
  unsigned *dev_num_image_formats = 0;
  int errcode = 0;
  
  cl_image_format reff;
  int reff_found;
  unsigned formatCount = 0;
  
  POCL_RETURN_ERROR_COND((context == NULL), CL_INVALID_CONTEXT);

  POCL_RETURN_ERROR_COND((context->num_devices == 0), CL_INVALID_CONTEXT);
  
  POCL_RETURN_ERROR_COND((num_entries == 0 && image_formats != NULL), CL_INVALID_VALUE);
  
  dev_image_formats = (const cl_image_format**) calloc (context->num_devices, sizeof(cl_image_format*));
  dev_num_image_formats = (unsigned*) calloc (context->num_devices, sizeof(unsigned));

  if (dev_image_formats == NULL || dev_num_image_formats == NULL)
    {
      /* one might have been allocated, but the other not */
      free(dev_image_formats);
      free(dev_num_image_formats);
      return CL_OUT_OF_HOST_MEMORY;
    }

  /* get supported image formats from devices */
  for (i = 0; i < context->num_devices; ++i)
    {    
      device_id = context->devices[i];
      if (device_id->image_support == CL_TRUE)
        {
          errcode = device_id->ops->get_supported_image_formats (
              flags, dev_image_formats + i,
              (cl_uint *)(dev_num_image_formats + i));
          if (errcode != CL_SUCCESS)
            goto CLEAN_MEM_AND_RETURN;
        }

      if (dev_num_image_formats[i] == 0) {
        /* this device supports no image formats. since we have to
         * present the intersection of all supported image formats
         * (see below), inform that we support none, and return early */
        if (num_image_formats != NULL)
          {
            *num_image_formats = 0;
          }
        goto CLEAN_MEM_AND_RETURN;
      }
    }
  
  /* intersect of supported image formats. TODO: should be union but 
     implementation does not support contexts where format is not supported 
     by every device in context */ 
  
  /* compare device[0] formats to all other devices */
  for (i = 0; i < dev_num_image_formats[0]; i++)
    {
      reff_found = 1; /* init */
      reff = dev_image_formats[0][i];
      
      /* devices[1..*] */
      for (j = 1; j < context->num_devices && reff_found; j++)
        {
          reff_found = 0;
          /* sup. devices[j] image formats [0..*]   */
          if (pocl_find_img_format (&reff, dev_image_formats[j], 
                                    dev_num_image_formats[j]))
            {
              reff_found = 1;
              continue;
            }
          break;
        }
      
      if (reff_found)
        { 
          /* if we get here reff is part of intersect */ 
          
          /* if second call */
          if (image_formats != NULL && formatCount <= num_entries)
            image_formats[formatCount] = reff;
          
          ++formatCount;
        }   
    }
  
  if (num_image_formats != NULL)
    {
      *num_image_formats = formatCount;
    }
  
 CLEAN_MEM_AND_RETURN:
  POCL_MEM_FREE(dev_num_image_formats);
  POCL_MEM_FREE(dev_image_formats);
  return errcode;
} 
POsym(clGetSupportedImageFormats)
