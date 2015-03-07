/* OpenCL runtime library: clCreateImage3D()

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
CL_API_ENTRY cl_mem CL_API_CALL
POname(clCreateImage3D) (cl_context              context,
                         cl_mem_flags            flags,
                         const cl_image_format * image_format,
                         size_t                  image_width, 
                         size_t                  image_height,
                         size_t                  image_depth, 
                         size_t                  image_row_pitch, 
                         size_t                  image_slice_pitch, 
                         void *                  host_ptr,
                         cl_int *                errcode_ret) 
CL_API_SUFFIX__VERSION_1_0
{
  
  cl_image_desc img_desc;
  img_desc.image_type = CL_MEM_OBJECT_IMAGE3D;
  img_desc.image_width = image_width;
  img_desc.image_height = image_height; 
  img_desc.image_depth = image_depth;
  img_desc.image_array_size = 1;
  img_desc.image_row_pitch = image_row_pitch;
  img_desc.image_slice_pitch = image_slice_pitch;
  img_desc.num_mip_levels = 0;
  img_desc.num_samples = 0;
  img_desc.buffer = 0;
  
  return POname(clCreateImage) (context, flags, image_format, &img_desc,
                                host_ptr, errcode_ret);     
}
POsym(clCreateImage3D)
