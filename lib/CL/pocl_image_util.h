/* OpenCL runtime library: pocl_image_util image utility functions

   Copyright (c) 2012 Timo Viitanen / Tampere University of Technology
   
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

#ifndef POCL_IMAGE_UTIL_H
#define POCL_IMAGE_UTIL_H

#include "pocl_cl.h"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

extern cl_int 
pocl_check_image_origin_region (const cl_mem image, 
                                const size_t *origin, 
                                const size_t *region);

extern void
pocl_get_image_information (cl_channel_order  ch_order, 
                            cl_channel_type   ch_type,
                            cl_int*           host_channels,
                            cl_int*           host_elem_size);

extern cl_int pocl_check_device_supports_image (
    cl_device_id device, const cl_image_format *image_format,
    const cl_image_desc *image_desc, cl_image_format *supported_image_formats,
    cl_uint num_entries);

void pocl_write_pixel_zero (void *data, const void *color_ptr, int order,
                            int elem_size, int channel_type);

cl_char4 convert_char4_sat (cl_float4 x);
cl_char convert_char_sat (cl_float x);
cl_char4 convert_char4_sat_int (cl_int4 x);
cl_char convert_char_sat_int (cl_int x);
cl_uchar4 convert_uchar4_sat (cl_float4 x);
cl_uchar convert_uchar_sat (cl_float x);
cl_uchar4 convert_uchar4_sat_int (cl_uint4 x);
cl_uchar convert_uchar_sat_int (cl_uint x);

cl_short4 convert_short4_sat (cl_float4 x);
cl_short convert_short_sat (cl_float x);
cl_short4 convert_short4_sat_int (cl_int4 x);
cl_short convert_short_sat_int (cl_int x);
cl_ushort4 convert_ushort4_sat (cl_float4 x);
cl_ushort convert_ushort_sat (cl_float x);
cl_ushort4 convert_ushort4_sat_int (cl_uint4 x);
cl_ushort convert_ushort_sat_int (cl_uint x);

#ifdef __GNUC__
#pragma GCC visibility pop
#endif
                   
#endif
