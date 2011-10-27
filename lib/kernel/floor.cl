/* OpenCL built-in library: floor()

   Copyright (c) 2011 Universidad Rey Juan Carlos
   
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

#define _MM_FROUND_TO_NEAREST_INT 0x00
#define _MM_FROUND_TO_NEG_INF     0x01
#define _MM_FROUND_TO_POS_INF     0x02
#define _MM_FROUND_TO_ZERO        0x03
#define _MM_FROUND_CUR_DIRECTION  0x04

#define _MM_FROUND_RAISE_EXC 0x00
#define _MM_FROUND_NO_EXC    0x08

#define _MM_FROUND_NINT      (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_FLOOR     (_MM_FROUND_TO_NEG_INF | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_CEIL      (_MM_FROUND_TO_POS_INF | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_TRUNC     (_MM_FROUND_TO_ZERO | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_RINT      (_MM_FROUND_CUR_DIRECTION | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_NEARBYINT (_MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC)



float __attribute__ ((__overloadable__))
floor(float a)
{
#ifdef __SSE4_1__
  // LLVM does not optimise this on its own
  return ((float4)__builtin_ia32_roundss(*(float4*)&a, *(float4*)&a,
                                         _MM_FROUND_FLOOR)).s0;
#else
  return __builtin_floorf(a);
#endif
}

float2 __attribute__ ((__overloadable__))
floor(float2 a)
{
#ifdef __SSE4_1__
  return ((float4)floor(*(float4)&a)).s01;
#else
  return (float2)(floor(a.lo), floor(a.hi));
#endif
}

float3 __attribute__ ((__overloadable__))
floor(float3 a)
{
#ifdef __SSE4_1__
  return ((float4)floor(*(float4)&a)).s012;
#else
  return (float3)(floor(a.s01), floor(a.s2));
#endif
}

float4 __attribute__ ((__overloadable__))
floor(float4 a)
{
#ifdef __SSE4_1__
  return __builtin_ia32_roundps(a, _MM_FROUND_FLOOR);
#else
  return (float4)(floor(a.lo), floor(a.hi));
#endif
}

float8 __attribute__ ((__overloadable__))
floor(float8 a)
{
#ifdef __AVX__
  return __builtin_ia32_roundps256(a, _MM_FROUND_FLOOR);
#else
  return (float8)(floor(a.lo), floor(a.hi));
#endif
}

float16 __attribute__ ((__overloadable__))
floor(float16 a)
{
  return (float16)(floor(a.lo), floor(a.hi));
}

double __attribute__ ((__overloadable__))
floor(double a)
{
#ifdef __SSE4_1__
  // LLVM does not optimise this on its own
  return ((double2)__builtin_ia32_roundss(*(double2*)&a, *(double2*)&a,
                                          _MM_FROUND_FLOOR)).s0;
#else
  return __builtin_floor(a);
#endif
}

double2 __attribute__ ((__overloadable__))
floor(double2 a)
{
#ifdef __SSE4_1__
  return __builtin_ia32_roundpd(a, _MM_FROUND_FLOOR);
#else
  return (double2)(floor(a.lo), floor(a.hi));
#endif
}

double3 __attribute__ ((__overloadable__))
floor(double3 a)
{
#ifdef __AVX__
  return ((double4)floor(*(double4)&a)).s012;
#else
  return (double3)(floor(a.s01), floor(a.s2));
#endif
}

double4 __attribute__ ((__overloadable__))
floor(double4 a)
{
#ifdef __AVX__
  return __builtin_ia32_roundpd256(a, _MM_FROUND_FLOOR);
#else
  return (double4)(floor(a.lo), floor(a.hi));
#endif
}

double8 __attribute__ ((__overloadable__))
floor(double8 a)
{
  return (double8)(floor(a.lo), floor(a.hi));
}

double16 __attribute__ ((__overloadable__))
floor(double16 a)
{
  return (double16)(floor(a.lo), floor(a.hi));
}
