/* OpenCL built-in library: max()

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

#include "templates.h"

DEFINE_EXPR_G_GG(max, a>=b ? a : b)
DEFINE_EXPR_G_GS(max, max(a, (gtype)b))



// Note: max() has no special semantics for inf/nan, even if fmax does

DEFINE_EXPR_V_VS(max, max(a, (vtype)b))



float4 _cl_max_ensure_float4(float4 a)
{
  return a;
}

double2 _cl_max_ensure_double2(double2 a)
{
  return a;
}



float __attribute__ ((__overloadable__))
max(float a, float b)
{
#ifdef __SSE__
  // LLVM does not optimise this on its own
  // Can't convert to float4 (why?)
  // return ((float4)__builtin_ia32_maxss(*(float4*)&a, *(float4*)&b)).s0;
  return _cl_max_ensure_float4(__builtin_ia32_maxss(*(float4*)&a, *(float4*)&b)).s0;
#else
  return __builtin_fmaxf(a, b);
#endif
}

float2 __attribute__ ((__overloadable__))
max(float2 a, float2 b)
{
#ifdef __SSE__
  return ((float4)max(*(float4*)&a, *(float4*)&b)).s01;
#else
  return (float2)(max(a.lo, b.lo), max(a.hi, b.hi));
#endif
}

float3 __attribute__ ((__overloadable__))
max(float3 a, float3 b)
{
#ifdef __SSE__
  return ((float4)max(*(float4*)&a, *(float4*)&b)).s012;
#else
  return (float3)(max(a.s01, b.s01), max(a.s2, b.s2));
#endif
}

float4 __attribute__ ((__overloadable__))
max(float4 a, float4 b)
{
#ifdef __SSE__
  return __builtin_ia32_maxps(a, b);
#else
  return (float4)(max(a.lo, b.lo), max(a.hi, b.hi));
#endif
}

float8 __attribute__ ((__overloadable__))
max(float8 a, float8 b)
{
#ifdef __AVX__
  return __builtin_ia32_maxps256(a, b);
#else
  return (float8)(max(a.lo, b.lo), max(a.hi, b.hi));
#endif
}

float16 __attribute__ ((__overloadable__))
max(float16 a, float16 b)
{
  return (float16)(max(a.lo, b.lo), max(a.hi, b.hi));
}

double __attribute__ ((__overloadable__))
max(double a, double b)
{
#ifdef __SSE2__
  // LLVM does not optimise this on its own
  // Can't convert to double2 (why?)
  // return ((double2)__builtin_ia32_maxsd(*(double2*)&a, *(double2*)&b)).s0;
  return _cl_max_ensure_double2(__builtin_ia32_maxsd(*(double2*)&a, *(double2*)&b)).s0;
#else
  return __builtin_fmax(a, b);
#endif
}

double2 __attribute__ ((__overloadable__))
max(double2 a, double2 b)
{
#ifdef __SSE2__
  return __builtin_ia32_maxpd(a, b);
#else
  return (double2)(max(a.lo, b.lo), max(a.hi, b.hi));
#endif
}

double3 __attribute__ ((__overloadable__))
max(double3 a, double3 b)
{
#ifdef __AVX__
  return ((double4)max(*(double4*)&a, *(double4*)&b)).s012;
#else
  return (double3)(max(a.s01, b.s01), max(a.s2, b.s2));
#endif
}

double4 __attribute__ ((__overloadable__))
max(double4 a, double4 b)
{
#ifdef __AVX__
  return __builtin_ia32_maxpd256(a, b);
#else
  return (double4)(max(a.lo, b.lo), max(a.hi, b.hi));
#endif
}

double8 __attribute__ ((__overloadable__))
max(double8 a, double8 b)
{
  return (double8)(max(a.lo, b.lo), max(a.hi, b.hi));
}

double16 __attribute__ ((__overloadable__))
max(double16 a, double16 b)
{
  return (double16)(max(a.lo, b.lo), max(a.hi, b.hi));
}
