/* OpenCL built-in library: vload()

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



#define IMPLEMENT_VLOAD(TYPE, MOD)                                      \
                                                                        \
  TYPE##2 _CL_OVERLOADABLE                                              \
  vload2(size_t offset, const MOD TYPE *p)                              \
  {                                                                     \
    const MOD TYPE##2 *pp = (const MOD TYPE##2 *)(p + offset*2);        \
    return pp[0];                                                       \
  }                                                                     \
                                                                        \
  TYPE##3 _CL_OVERLOADABLE                                              \
  vload3(size_t offset, const MOD TYPE *p)                              \
  {                                                                     \
    return (TYPE##3)(vload2(0, &p[offset*3]), p[offset*3+2]);           \
  }                                                                     \
                                                                        \
  TYPE##4 _CL_OVERLOADABLE                                              \
  vload4(size_t offset, const MOD TYPE *p)                              \
  {                                                                     \
    const MOD TYPE##4 *pp = (const MOD TYPE##4 *)(p + offset*4);        \
    return pp[0];                                                       \
  }                                                                     \
                                                                        \
  TYPE##8 _CL_OVERLOADABLE                                              \
  vload8(size_t offset, const MOD TYPE *p)                              \
  {                                                                     \
    const MOD TYPE##8 *pp = (const MOD TYPE##8 *)(p + offset*8);        \
    return pp[0];                                                       \
  }                                                                     \
                                                                        \
  TYPE##16 _CL_OVERLOADABLE                                             \
  vload16(size_t offset, const MOD TYPE *p)                             \
  {                                                                     \
    const MOD TYPE##16 *pp = (const MOD TYPE##16 *)(p + offset*16);     \
    return pp[0];                                                       \
  }



IMPLEMENT_VLOAD(char  , __global)
IMPLEMENT_VLOAD(short , __global)
IMPLEMENT_VLOAD(int   , __global)
#if defined(cl_khr_int64)
IMPLEMENT_VLOAD(long  , __global)
IMPLEMENT_VLOAD(ulong , __global)
#endif
IMPLEMENT_VLOAD(uchar , __global)
IMPLEMENT_VLOAD(ushort, __global)
IMPLEMENT_VLOAD(uint  , __global)
IMPLEMENT_VLOAD(float , __global)
#if defined(cl_khr_fp64)
IMPLEMENT_VLOAD(double, __global)
#endif
#if defined(cl_khr_fp16)
IMPLEMENT_VLOAD(half, __global)
#endif

IMPLEMENT_VLOAD(char  , __local)
IMPLEMENT_VLOAD(short , __local)
IMPLEMENT_VLOAD(int   , __local)
#if defined(cl_khr_int64)
IMPLEMENT_VLOAD(long  , __local)
IMPLEMENT_VLOAD(ulong , __local)
#endif
IMPLEMENT_VLOAD(uchar , __local)
IMPLEMENT_VLOAD(ushort, __local)
IMPLEMENT_VLOAD(uint  , __local)
IMPLEMENT_VLOAD(float , __local)
#if defined(cl_khr_fp64)
IMPLEMENT_VLOAD(double, __local)
#endif
#if defined(cl_khr_fp16)
IMPLEMENT_VLOAD(half, __local)
#endif

IMPLEMENT_VLOAD(char  , __constant)
IMPLEMENT_VLOAD(short , __constant)
IMPLEMENT_VLOAD(int   , __constant)
#if defined(cl_khr_int64)
IMPLEMENT_VLOAD(long  , __constant)
IMPLEMENT_VLOAD(ulong , __constant)
#endif
IMPLEMENT_VLOAD(uchar , __constant)
IMPLEMENT_VLOAD(ushort, __constant)
IMPLEMENT_VLOAD(uint  , __constant)
IMPLEMENT_VLOAD(float , __constant)
#if defined(cl_khr_fp64)
IMPLEMENT_VLOAD(double, __constant)
#endif
#if defined(cl_khr_fp16)
IMPLEMENT_VLOAD(half, __constant)
#endif

IMPLEMENT_VLOAD(char  , __private)
IMPLEMENT_VLOAD(short , __private)
IMPLEMENT_VLOAD(int   , __private)
#if defined(cl_khr_int64)
IMPLEMENT_VLOAD(long  , __private)
#endif
IMPLEMENT_VLOAD(uchar , __private)
IMPLEMENT_VLOAD(ushort, __private)
IMPLEMENT_VLOAD(uint  , __private)
#if defined(cl_khr_int64)
IMPLEMENT_VLOAD(ulong , __private)
#endif
IMPLEMENT_VLOAD(float , __private)
#if defined(cl_khr_fp64)
IMPLEMENT_VLOAD(double, __private)
#endif
#if defined(cl_khr_fp16)
IMPLEMENT_VLOAD(half, __private)
#endif
