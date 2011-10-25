/* OpenCL built-in library: fmin()

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


float __attribute__ ((overloadable))
fmin(float a, float b)
{
#ifdef __SSE__
  // LLVM does not optimise this on its own
  return ((float4)__builtin_ia32_minss(*(float4*)&a, *(float4*)&b)).s0;
#else
  return __builtin_fminf(a, b);
#endif
}

float2 __attribute__ ((overloadable))
fmin(float2 a, float2 b)
{
#ifdef __SSE__
  return ((float4)fmin(*(float4*)&a, *(float4*)&b)).s01;
#else
  return (float2)(fmin(a.lo, b.lo), fmin(a.hi, b.hi));
#endif
}

float3 __attribute__ ((overloadable))
fmin(float3 a, float3 b)
{
#ifdef __SSE__
  return ((float4)fmin(*(float4*)&a, *(float4*)&b)).s012;
#else
  return (float3)(fmin(a.s01, b.s01), fmin(a.s2, b.s2));
#endif
}

float4 __attribute__ ((overloadable))
fmin(float4 a, float4 b)
{
#ifdef __SSE__
  return __builtin_ia32_minps(a, b);
#else
  return (float4)(fmin(a.lo, b.lo), fmin(a.hi, b.hi));
#endif
}

float8 __attribute__ ((overloadable))
fmin(float8 a, float8 b)
{
#ifdef __AVX__
  return __builtin_ia32_minps256(a, b);
#else
  return (float8)(fmin(a.lo, b.lo), fmin(a.hi, b.hi));
#endif
}

float16 __attribute__ ((overloadable))
fmin(float16 a, float16 b)
{
  return (float16)(fmin(a.lo, b.lo), fmin(a.hi, b.hi));
}

double __attribute__ ((overloadable))
fmin(double a, double b)
{
#ifdef __SSE2__
  // LLVM does not optimise this on its own
  return ((double2)__builtin_ia32_minsd(*(double2*)&a, *(double2*)&b)).s0;
#else
  return __builtin_fmin(a, b);
#endif
}

double2 __attribute__ ((overloadable))
fmin(double2 a, double2 b)
{
#ifdef __SSE2__
  return __builtin_ia32_minpd(a, b);
#else
  return (double2)(fmin(a.lo, b.lo), fmin(a.hi, b.hi));
#endif
}

double3 __attribute__ ((overloadable))
fmin(double3 a, double3 b)
{
#ifdef __AVX__
  return ((double4)fmin(*(double4*)&a, *(double4*)&b)).s012;
#else
  return (double3)(fmin(a.s01, b.s01), fmin(a.s2, b.s2));
#endif
}

double4 __attribute__ ((overloadable))
fmin(double4 a, double4 b)
{
#ifdef __AVX__
  return __builtin_ia32_minpd256(a, b);
#else
  return (double4)(fmin(a.lo, b.lo), fmin(a.hi, b.hi));
#endif
}

double8 __attribute__ ((overloadable))
fmin(double8 a, double8 b)
{
  return (double8)(fmin(a.lo, b.lo), fmin(a.hi, b.hi));
}

double16 __attribute__ ((overloadable))
fmin(double16 a, double16 b)
{
  return (double16)(fmin(a.lo, b.lo), fmin(a.hi, b.hi));
}
