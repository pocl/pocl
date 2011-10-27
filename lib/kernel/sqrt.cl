/* OpenCL built-in library: sqrt()

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
sqrt(float a)
{
  return __builtin_sqrtf(a);
}

float2 __attribute__ ((overloadable))
sqrt(float2 a)
{
#ifdef __SSE__
  return ((float4)sqrt(*(float4*)&a)).s01;
#else
  return (float2)(sqrt(a.lo), sqrt(a.hi));
#endif
}

float3 __attribute__ ((overloadable))
sqrt(float3 a)
{
#ifdef __SSE__
  return ((float4)sqrt(*(float4*)&a)).s012;
#else
  return (float3)(sqrt(a.s01), sqrt(a.s2));
#endif
}

float4 __attribute__ ((overloadable))
sqrt(float4 a)
{
#ifdef __SSE__
  return __builtin_ia32_sqrtps(a);
#else
  return (float4)(sqrt(a.lo), sqrt(a.hi));
#endif
}

float8 __attribute__ ((overloadable))
sqrt(float8 a)
{
#ifdef __AVX__
  return __builtin_ia32_sqrtps256(a);
#else
  return (float8)(sqrt(a.lo), sqrt(a.hi));
#endif
}

float16 __attribute__ ((overloadable))
sqrt(float16 a)
{
  return (float16)(sqrt(a.lo), sqrt(a.hi));
}

double __attribute__ ((overloadable))
sqrt(double a)
{
  return __builtin_sqrt(a);
}

double2 __attribute__ ((overloadable))
sqrt(double2 a)
{
#ifdef __SSE2__
  return __builtin_ia32_sqrtpd(a);
#else
  return (double2)(sqrt(a.lo), sqrt(a.hi));
#endif
}

double3 __attribute__ ((overloadable))
sqrt(double3 a)
{
#ifdef __AVX__
  return ((double4)sqrt(*(double4*)&a)).s012;
#else
  return (double3)(sqrt(a.s01), sqrt(a.s2));
#endif
}

double4 __attribute__ ((overloadable))
sqrt(double4 a)
{
#ifdef __AVX__
  return __builtin_ia32_pd256(a);
#else
  return (double4)(sqrt(a.lo), sqrt(a.hi));
#endif
}

double8 __attribute__ ((overloadable))
sqrt(double8 a)
{
  return (double8)(sqrt(a.lo), sqrt(a.hi));
}

double16 __attribute__ ((overloadable))
sqrt(double16 a)
{
  return (double16)(sqrt(a.lo), sqrt(a.hi));
}
