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



#define IMPLEMENT_VSTORE(TYPE, MOD)                     \
                                                        \
  void _CL_OVERLOADABLE                                 \
  vstore2(TYPE##2 data, size_t offset, MOD TYPE *p)     \
  {                                                     \
    p[offset*2] = data.lo;                              \
    p[offset*2+1] = data.hi;                            \
  }                                                     \
                                                        \
  void _CL_OVERLOADABLE                                 \
  vstore3(TYPE##3 data, size_t offset, MOD TYPE *p)     \
  {                                                     \
    vstore2(data.lo, 0, &p[offset*3]);                  \
    p[offset*3+2] = data.s2;                            \
  }                                                     \
                                                        \
  void _CL_OVERLOADABLE                                 \
  vstore4(TYPE##4 data, size_t offset, MOD TYPE *p)     \
  {                                                     \
    vstore2(data.lo, 0, &p[offset*4]);                  \
    vstore2(data.hi, 0, &p[offset*4+2]);                \
  }                                                     \
                                                        \
  void _CL_OVERLOADABLE                                 \
  vstore8(TYPE##8 data, size_t offset, MOD TYPE *p)     \
  {                                                     \
    vstore4(data.lo, 0, &p[offset*8]);                  \
    vstore4(data.hi, 0, &p[offset*8+4]);                \
  }                                                     \
                                                        \
  void _CL_OVERLOADABLE                                 \
  vstore16(TYPE##16 data, size_t offset, MOD TYPE *p)   \
  {                                                     \
    vstore8(data.lo, 0, &p[offset*16]);                 \
    vstore8(data.hi, 0, &p[offset*16+8]);               \
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

#ifdef __opencl_c_generic_address_space
IMPLEMENT_VSTORE(char  , __generic)
IMPLEMENT_VSTORE(short , __generic)
IMPLEMENT_VSTORE(int   , __generic)
#if defined(cl_khr_int64)
IMPLEMENT_VSTORE(long  , __generic)
#endif
IMPLEMENT_VSTORE(uchar , __generic)
IMPLEMENT_VSTORE(ushort, __generic)
IMPLEMENT_VSTORE(uint  , __generic)
#if defined(cl_khr_int64)
IMPLEMENT_VSTORE(ulong , __generic)
#endif
IMPLEMENT_VSTORE(float , __generic)
#if defined(cl_khr_fp64)
IMPLEMENT_VSTORE(double, __generic)
#endif
#endif /* __opencl_c_generic_address_space */
