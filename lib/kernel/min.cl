/* OpenCL built-in library: min()

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

DEFINE_EXPR_G_GG(min, a<=b ? a : b)
DEFINE_EXPR_G_GS(min, min(a, (gtype)b))



// Note: min() has no special semantics for inf/nan, even if fmin does

DEFINE_EXPR_V_VS(min, min(a, (vtype)b))



float4 _cl_min_ensure_float4(float4 a)
{
  return a;
}

double2 _cl_min_ensure_double2(double2 a)
{
  return a;
}



float __attribute__ ((__overloadable__))
min(float a, float b)
{
#ifdef __SSE__
  // LLVM does not optimise this on its own
  // Can't convert to float4 (why?)
  // return ((float4)__builtin_ia32_minss(*(float4*)&a, *(float4*)&b)).s0;
  return _cl_min_ensure_float4(__builtin_ia32_minss(*(float4*)&a, *(float4*)&b)).s0;
#else
  return __builtin_fminf(a, b);
#endif
}

float2 __attribute__ ((__overloadable__))
min(float2 a, float2 b)
{
#ifdef __SSE__
  return ((float4)min(*(float4*)&a, *(float4*)&b)).s01;
#else
  return (float2)(min(a.lo, b.lo), min(a.hi, b.hi));
#endif
}

float3 __attribute__ ((__overloadable__))
min(float3 a, float3 b)
{
#ifdef __SSE__
  return ((float4)min(*(float4*)&a, *(float4*)&b)).s012;
#else
  return (float3)(min(a.s01, b.s01), min(a.s2, b.s2));
#endif
}

float4 __attribute__ ((__overloadable__))
min(float4 a, float4 b)
{
#ifdef __SSE__
  return __builtin_ia32_minps(a, b);
#else
  return (float4)(min(a.lo, b.lo), min(a.hi, b.hi));
#endif
}

float8 __attribute__ ((__overloadable__))
min(float8 a, float8 b)
{
#ifdef __AVX__
  return __builtin_ia32_minps256(a, b);
#else
  return (float8)(min(a.lo, b.lo), min(a.hi, b.hi));
#endif
}

float16 __attribute__ ((__overloadable__))
min(float16 a, float16 b)
{
  return (float16)(min(a.lo, b.lo), min(a.hi, b.hi));
}

double __attribute__ ((__overloadable__))
min(double a, double b)
{
#ifdef __SSE2__
  // LLVM does not optimise this on its own
  // Can't convert to double2 (why?)
  // return ((double2)__builtin_ia32_minsd(*(double2*)&a, *(double2*)&b)).s0;
  return _cl_min_ensure_double2(__builtin_ia32_minsd(*(double2*)&a, *(double2*)&b)).s0;
#else
  return __builtin_fmin(a, b);
#endif
}

double2 __attribute__ ((__overloadable__))
min(double2 a, double2 b)
{
#ifdef __SSE2__
  return __builtin_ia32_minpd(a, b);
#else
  return (double2)(min(a.lo, b.lo), min(a.hi, b.hi));
#endif
}

double3 __attribute__ ((__overloadable__))
min(double3 a, double3 b)
{
#ifdef __AVX__
  return ((double4)min(*(double4*)&a, *(double4*)&b)).s012;
#else
  return (double3)(min(a.s01, b.s01), min(a.s2, b.s2));
#endif
}

double4 __attribute__ ((__overloadable__))
min(double4 a, double4 b)
{
#ifdef __AVX__
  return __builtin_ia32_minpd256(a, b);
#else
  return (double4)(min(a.lo, b.lo), min(a.hi, b.hi));
#endif
}

double8 __attribute__ ((__overloadable__))
min(double8 a, double8 b)
{
  return (double8)(min(a.lo, b.lo), min(a.hi, b.hi));
}

double16 __attribute__ ((__overloadable__))
min(double16 a, double16 b)
{
  return (double16)(min(a.lo, b.lo), min(a.hi, b.hi));
}
