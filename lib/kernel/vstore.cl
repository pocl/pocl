/* OpenCL built-in library: vstore()

   Copyright (c) 2011 Erik Schnetter <eschnetter@perimeterinstitute.ca>
                      Perimeter Institute for Theoretical Physics
   
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

#define IMPLEMENT_VSTORE(TYPE, MOD)                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore2 (TYPE##2 data, size_t offset, MOD TYPE *p)    \
  {                                                                           \
    __builtin_memcpy_inline (&p[offset * 2], &data, sizeof (TYPE##2));        \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore3 (TYPE##3 data, size_t offset, MOD TYPE *p)    \
  {                                                                           \
    __builtin_memcpy_inline (&p[offset * 3], &data, sizeof (TYPE) * 3);       \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore4 (TYPE##4 data, size_t offset, MOD TYPE *p)    \
  {                                                                           \
    __builtin_memcpy_inline (&p[offset * 4], &data, sizeof (TYPE##4));        \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore8 (TYPE##8 data, size_t offset, MOD TYPE *p)    \
  {                                                                           \
    __builtin_memcpy_inline (&p[offset * 8], &data, sizeof (TYPE##8));        \
  }                                                                           \
                                                                              \
  void _CL_OVERLOADABLE vstore16 (TYPE##16 data, size_t offset, MOD TYPE *p)  \
  {                                                                           \
    __builtin_memcpy_inline (&p[offset * 16], &data, sizeof (TYPE##16));      \
  }

IMPLEMENT_VSTORE(char  , __global)
IMPLEMENT_VSTORE(short , __global)
IMPLEMENT_VSTORE(int   , __global)
#if defined(cl_khr_int64)
IMPLEMENT_VSTORE(long  , __global)
IMPLEMENT_VSTORE(ulong , __global)
#endif
IMPLEMENT_VSTORE(uchar , __global)
IMPLEMENT_VSTORE(ushort, __global)
IMPLEMENT_VSTORE(uint  , __global)
IMPLEMENT_VSTORE(float , __global)
#if defined(cl_khr_fp64)
IMPLEMENT_VSTORE(double, __global)
#endif

IMPLEMENT_VSTORE(char  , __local)
IMPLEMENT_VSTORE(short , __local)
IMPLEMENT_VSTORE(int   , __local)
#if defined(cl_khr_int64)
IMPLEMENT_VSTORE(long  , __local)
IMPLEMENT_VSTORE(ulong , __local)
#endif
IMPLEMENT_VSTORE(uchar , __local)
IMPLEMENT_VSTORE(ushort, __local)
IMPLEMENT_VSTORE(uint  , __local)
IMPLEMENT_VSTORE(float , __local)
#if defined(cl_khr_fp64)
IMPLEMENT_VSTORE(double, __local)
#endif

IMPLEMENT_VSTORE(char  , __private)
IMPLEMENT_VSTORE(short , __private)
IMPLEMENT_VSTORE(int   , __private)
#if defined(cl_khr_int64)
IMPLEMENT_VSTORE(long  , __private)
#endif
IMPLEMENT_VSTORE(uchar , __private)
IMPLEMENT_VSTORE(ushort, __private)
IMPLEMENT_VSTORE(uint  , __private)
#if defined(cl_khr_int64)
IMPLEMENT_VSTORE(ulong , __private)
#endif
IMPLEMENT_VSTORE(float , __private)
#if defined(cl_khr_fp64)
IMPLEMENT_VSTORE(double, __private)
#endif


IF_GEN_AS(IMPLEMENT_VSTORE(char  , __generic))
IF_GEN_AS(IMPLEMENT_VSTORE(short , __generic))
IF_GEN_AS(IMPLEMENT_VSTORE(int   , __generic))
#if defined(cl_khr_int64)
IF_GEN_AS(IMPLEMENT_VSTORE(long  , __generic))
#endif
IF_GEN_AS(IMPLEMENT_VSTORE(uchar , __generic))
IF_GEN_AS(IMPLEMENT_VSTORE(ushort, __generic))
IF_GEN_AS(IMPLEMENT_VSTORE(uint  , __generic))
#if defined(cl_khr_int64)
IF_GEN_AS(IMPLEMENT_VSTORE(ulong , __generic))
#endif
IF_GEN_AS(IMPLEMENT_VSTORE(float , __generic))
#if defined(cl_khr_fp64)
IF_GEN_AS(IMPLEMENT_VSTORE(double, __generic))
#endif
#if defined(cl_khr_fp16)
IF_GEN_AS(IMPLEMENT_VSTORE(half, __generic))
#endif

