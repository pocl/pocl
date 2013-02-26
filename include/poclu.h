/* OpenCL runtime library: poclu - useful utility functions for OpenCL programs

   Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
   
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

#ifndef POCLU_H
#define POCLU_H

#include <CL/opencl.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Byte swap functions for endianness swapping between the host
 * (current CPU) and a target device.
 *
 * Queries the target device using the OpenCL API for the endianness
 * and swaps if it differs from the host's.
 */
cl_int
poclu_bswap_cl_int(cl_device_id device, cl_int original);

cl_float
poclu_bswap_cl_float(cl_device_id device, cl_float original);

cl_float2
poclu_bswap_cl_float2(cl_device_id device, cl_float2 original);

/* In-place swapping of arrays. */
void
poclu_bswap_cl_int_array(cl_device_id device, cl_int* array, size_t num_elements);

void
poclu_bswap_cl_float_array(cl_device_id device, cl_float* array, size_t num_elements);

void
poclu_bswap_cl_float2_array(cl_device_id device, cl_float2* array, size_t num_elements);

/**
 * Misc. helper functions for streamlining OpenCL API usage. 
 */

/* Create a context in the first platform found. */
cl_context
poclu_create_any_context();

#ifdef __cplusplus
}
#endif

#endif
