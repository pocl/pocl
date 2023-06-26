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
#include "pocl_image_util.h"

extern CL_API_ENTRY cl_int CL_API_CALL
POname(clGetSupportedImageFormats) (cl_context           context,
                                    cl_mem_flags         flags,
                                    cl_mem_object_type   image_type,
                                    cl_uint              num_entries,
                                    cl_image_format *    image_formats,
                                    cl_uint *            num_image_formats) 
CL_API_SUFFIX__VERSION_1_0
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (context)), CL_INVALID_CONTEXT);

  POCL_RETURN_ERROR_COND((context->num_devices == 0), CL_INVALID_CONTEXT);
  
  POCL_RETURN_ERROR_COND((num_entries == 0 && image_formats != NULL), CL_INVALID_VALUE);

  cl_int idx = pocl_opencl_image_type_to_index (image_type);

  POCL_RETURN_ERROR_ON ((idx < 0), CL_INVALID_VALUE,
                        "invalid image type\n");

#ifdef ENABLE_CONFORMANCE
  if (flags & CL_MEM_KERNEL_READ_AND_WRITE)
    {
      if (num_image_formats != NULL)
        *num_image_formats = 0;
      return CL_SUCCESS;
    }
#endif

  if (image_formats != NULL)
    {
      if (num_entries > context->num_image_formats[idx])
          num_entries = context->num_image_formats[idx];
      if (num_entries)
        memcpy (image_formats, context->image_formats[idx],
                sizeof (cl_image_format) * num_entries);
    }

  if (num_image_formats != NULL)
    {
      *num_image_formats = context->num_image_formats[idx];
    }

  return CL_SUCCESS;
} 
POsym(clGetSupportedImageFormats)
