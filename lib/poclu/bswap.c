/* poclu - byte swap functions

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

#include "pocl_opencl.h"

#define GENERIC_BYTESWAP(__DTYPE, __WORD)                         \
  do {                                                            \
    unsigned __i;                                                 \
    union _word_union                                             \
    {                                                             \
      __DTYPE full_word;                                          \
      unsigned char bytes[sizeof(__DTYPE)];                       \
    } __old, __neww;                                              \
                                                                  \
    __old.full_word = __WORD;                                     \
    for (__i = 0; __i < sizeof(__DTYPE); ++__i)                   \
      __neww.bytes[__i] = __old.bytes[sizeof(__DTYPE) - __i - 1]; \
    __WORD = __neww.full_word;                                    \
  } while(0)

static int
needs_swap(cl_device_id device) 
{
  cl_bool deviceLittle;
  clGetDeviceInfo 
    (device, CL_DEVICE_ENDIAN_LITTLE, sizeof(cl_bool), 
     &deviceLittle, NULL);

#if defined(WORDS_BIGENDIAN) && WORDS_BIGENDIAN == 1
  return deviceLittle;
#else
  return !deviceLittle;
#endif
}

cl_int
poclu_bswap_cl_int(cl_device_id device, cl_int original) 
{
  if (!needs_swap (device)) return original;
  GENERIC_BYTESWAP (cl_int, original);
  return original;
}

cl_half
poclu_bswap_cl_half(cl_device_id device, cl_half original)
{
  if (!needs_swap (device)) return original;
  GENERIC_BYTESWAP (cl_half, original);
  return original;
}

cl_float
poclu_bswap_cl_float(cl_device_id device, cl_float original)
{
  if (!needs_swap (device)) return original;
  GENERIC_BYTESWAP (cl_float, original);
  return original;
}

void
poclu_bswap_cl_int_array(cl_device_id device, cl_int* array, 
                         size_t num_elements)
{
  size_t i;
  if (!needs_swap (device)) return;
  for (i = 0; i < num_elements; ++i) 
    {
      GENERIC_BYTESWAP (cl_int, array[i]);
    }
}

void
poclu_bswap_cl_half_array(cl_device_id device, cl_half* array, 
                           size_t num_elements)
{
  size_t i;
  if (!needs_swap (device)) return;
  for (i = 0; i < num_elements; ++i) 
    {
      GENERIC_BYTESWAP (cl_half, array[i]);
    }
}

void
poclu_bswap_cl_float_array(cl_device_id device, cl_float* array, 
                           size_t num_elements)
{
  size_t i;
  if (!needs_swap (device)) return;
  for (i = 0; i < num_elements; ++i) 
    {
      GENERIC_BYTESWAP (cl_float, array[i]);
    }
}

void
poclu_bswap_cl_float2_array(cl_device_id device, cl_float2* array, 
                            size_t num_elements)
{
  size_t i;
  if (!needs_swap (device)) return;
  for (i = 0; i < num_elements; ++i) 
    {
      GENERIC_BYTESWAP (cl_float, array[i].s[0]);
      GENERIC_BYTESWAP (cl_float, array[i].s[1]);
    }
}
