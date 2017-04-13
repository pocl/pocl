/* OpenCL built-in library: async_work_group_copy()

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

#include "templates.h"

/* The default implementation for "async copies" is a 
   blocking one which doesn't actually need events for 
   anything. 

   The devices (actually, platforms) should override these to
   implement proper block copies or similar. */

#define IMPLEMENT_ASYNC_COPY_FUNCS_SINGLE(GENTYPE)                      \
  __attribute__((overloadable))                                         \
  event_t async_work_group_copy(__local GENTYPE *dst,                   \
                                const __global GENTYPE *src,            \
                                size_t num_gentypes,                    \
                                event_t event)                          \
  {                                                                     \
    __SINGLE_WI {                                                       \
      for (size_t i = 0; i < num_gentypes; ++i) dst[i] = src[i];        \
    }                                                                   \
    return event;                                                       \
  }                                                                     \
                                                                        \
  __attribute__((overloadable))                                         \
  event_t async_work_group_copy(__global GENTYPE *dst,                  \
                                const __local GENTYPE *src,             \
                                size_t num_gentypes,                    \
                                event_t event)                          \
  {                                                                     \
    __SINGLE_WI {                                                       \
      for (size_t i = 0; i < num_gentypes; ++i) dst[i] = src[i];        \
    }                                                                   \
    return event;                                                       \
  }


#define IMPLEMENT_ASYNC_COPY_FUNCS(GENTYPE)             \
  IMPLEMENT_ASYNC_COPY_FUNCS_SINGLE(GENTYPE)            \
  IMPLEMENT_ASYNC_COPY_FUNCS_SINGLE(GENTYPE##2)         \
  IMPLEMENT_ASYNC_COPY_FUNCS_SINGLE(GENTYPE##3)         \
  IMPLEMENT_ASYNC_COPY_FUNCS_SINGLE(GENTYPE##4)         \
  IMPLEMENT_ASYNC_COPY_FUNCS_SINGLE(GENTYPE##8)         \
  IMPLEMENT_ASYNC_COPY_FUNCS_SINGLE(GENTYPE##16)

IMPLEMENT_ASYNC_COPY_FUNCS(char);
IMPLEMENT_ASYNC_COPY_FUNCS(uchar);
IMPLEMENT_ASYNC_COPY_FUNCS(short);
IMPLEMENT_ASYNC_COPY_FUNCS(ushort);
IMPLEMENT_ASYNC_COPY_FUNCS(int);
IMPLEMENT_ASYNC_COPY_FUNCS(uint);
__IF_INT64(IMPLEMENT_ASYNC_COPY_FUNCS(long));
__IF_INT64(IMPLEMENT_ASYNC_COPY_FUNCS(ulong));

IMPLEMENT_ASYNC_COPY_FUNCS(float);
__IF_FP64(IMPLEMENT_ASYNC_COPY_FUNCS(double));
__IF_FP16 (IMPLEMENT_ASYNC_COPY_FUNCS (half));
