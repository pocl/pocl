/* OpenCL built-in library: fmax()

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
fmax(float a, float b)
{
#ifdef __SSE__
  // LLVM does not optimise this on its own
  return ((float4)__builtin_ia32_maxss(*(float4*)&a, *(float4*)&b)).s0;
#else
  return __builtin_fmaxf(a, b);
#endif
}

float2 __attribute__ ((overloadable))
fmax(float2 a, float2 b)
{
#ifdef __SSE__
  return ((float4)fmax(*(float4*)&a, *(float4*)&b)).s01;
#else
  return (float2)(fmax(a.lo, b.lo), fmax(a.hi, b.hi));
#endif
}

float3 __attribute__ ((overloadable))
fmax(float3 a, float3 b)
{
#ifdef __SSE__
  return ((float4)fmax(*(float4*)&a, *(float4*)&b)).s012;
#else
  return (float3)(fmax(a.s01, b.s01), fmax(a.s2, b.s2));
#endif
}

float4 __attribute__ ((overloadable))
fmax(float4 a, float4 b)
{
#ifdef __SSE__
  return __builtin_ia32_maxps(a, b);
#else
  return (float4)(fmax(a.lo, b.lo), fmax(a.hi, b.hi));
#endif
}

float8 __attribute__ ((overloadable))
fmax(float8 a, float8 b)
{
#ifdef __AVX__
  return __builtin_ia32_maxps256(a, b);
#else
  return (float8)(fmax(a.lo, b.lo), fmax(a.hi, b.hi));
#endif
}

float16 __attribute__ ((overloadable))
fmax(float16 a, float16 b)
{
  return (float16)(fmax(a.lo, b.lo), fmax(a.hi, b.hi));
}

double __attribute__ ((overloadable))
fmax(double a, double b)
{
#ifdef __SSE2__
  // LLVM does not optimise this on its own
  return ((double2)__builtin_ia32_maxsd(*(double2*)&a, *(double2*)&b)).s0;
#else
  return __builtin_fmax(a, b);
#endif
}

double2 __attribute__ ((overloadable))
fmax(double2 a, double2 b)
{
#ifdef __SSE2__
  return __builtin_ia32_maxpd(a, b);
#else
  return (double2)(fmax(a.lo, b.lo), fmax(a.hi, b.hi));
#endif
}

double3 __attribute__ ((overloadable))
fmax(double3 a, double3 b)
{
#ifdef __AVX__
  return ((double4)fmax(*(double4*)&a, *(double4*)&b)).s012;
#else
  return (double3)(fmax(a.s01, b.s01), fmax(a.s2, b.s2));
#endif
}

double4 __attribute__ ((overloadable))
fmax(double4 a, double4 b)
{
#ifdef __AVX__
  return __builtin_ia32_maxpd256(a, b);
#else
  return (double4)(fmax(a.lo, b.lo), fmax(a.hi, b.hi));
#endif
}

double8 __attribute__ ((overloadable))
fmax(double8 a, double8 b)
{
  return (double8)(fmax(a.lo, b.lo), fmax(a.hi, b.hi));
}

double16 __attribute__ ((overloadable))
fmax(double16 a, double16 b)
{
  return (double16)(fmax(a.lo, b.lo), fmax(a.hi, b.hi));
}
