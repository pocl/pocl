/* OpenCL runtime library: clCreateImage()

   Copyright (c) 2012 Timo Viitanen
   
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

extern CL_API_ENTRY cl_mem CL_API_CALL
POclCreateImage(cl_context              context,
              cl_mem_flags            flags,
              const cl_image_format * image_format,
              const cl_image_desc *   image_desc, 
              void *                  host_ptr,
              cl_int *                errcode_ret) 
CL_API_SUFFIX__VERSION_1_2
{
  if (image_desc->num_mip_levels != 0 || 
      image_desc->num_samples != 0 )
    POCL_ABORT_UNIMPLEMENTED();
  
  if (image_desc->image_type != CL_MEM_OBJECT_IMAGE2D)
    POCL_ABORT_UNIMPLEMENTED();
  
  POclCreateImage2D (context,
                   flags,
                   image_format,
                   image_desc->image_width,
                   image_desc->image_height,
                   image_desc->image_row_pitch,
                   host_ptr,
                   errcode_ret);
}
POsym(clCreateImage)
