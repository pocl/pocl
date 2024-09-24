/* OpenCL built-in library: Facilities for spawning work-group functions in
   the device side: _pocl_run_all_wgs()

   Copyright (c) 2018 Pekka Jääskeläinen

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include "pocl_device.h"

void
_pocl_memcpy_4 (__attribute__ ((address_space (0))) uint32_t *dst,
                __attribute__ ((address_space (1))) uint32_t *src,
                size_t bytes)
{
  for (size_t i = 0; i < (bytes >> 2); ++i)
    dst[i] = src[i];
}

void
_pocl_memcpy_1 (__attribute__ ((address_space (0))) char *dst,
                __attribute__ ((address_space (1))) char *src, size_t bytes)
{
  for (size_t i = 0; i < bytes; ++i)
    dst[i] = src[i];
}

/* Note: memcpy has to be in .ll because the last argument is actually "i1" not
"i8"
/*
void
__pocl_tce_memcpy_p1_p2_i32 (__attribute__ ((address_space (1))) char *dst,
                             __attribute__ ((address_space (2))) char *src,
                             uint32_t bytes, char unused) {
  for (uint32_t i = 0; i < bytes; ++i)
    dst[i] = src[i];
}
*/

/*
void
__pocl_tce_memcpy_p1_p2_i64 (__attribute__ ((address_space (1))) char *dst,
                             __attribute__ ((address_space (2))) char *src,
                             uint64_t bytes, char unused) {
  for (int32_t i = 0; i < (int32_t)bytes; ++i)
    dst[i] = src[i];
}
*/
