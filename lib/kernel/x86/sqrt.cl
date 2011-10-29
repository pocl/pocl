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

#ifdef __SSE__
float4 _cl_sqrt_ensure_float4(float4 a)
{
  return a;
}

float __attribute__ ((__overloadable__))
sqrt(float a)
{
  return _cl_sqrt_ensure_float4(__builtin_ia32_sqrtss(*(float4*)&a)).s0;
}

float2 __attribute__ ((__overloadable__))
sqrt(float2 a)
{
  return ((float4)sqrt(*(float4*)&a)).s01;
}

float3 __attribute__ ((__overloadable__))
sqrt(float3 a)
{
  return ((float4)sqrt(*(float4*)&a)).s012;
}

float4 __attribute__ ((__overloadable__))
sqrt(float4 a)
{
  return __builtin_ia32_sqrtps(a);
}
#else
float __attribute__ ((__overloadable__))
sqrt(float a)
{
  return __builtin_fsqrt(a);
}

float2 __attribute__ ((__overloadable__))
sqrt(float2 a)
{
  return (float2)(sqrt(a.lo), sqrt(a.hi));
}

float3 __attribute__ ((__overloadable__))
sqrt(float3 a)
{
  return (float3)(sqrt(a.lo), sqrt(a.s2));
}

float4 __attribute__ ((__overloadable__))
sqrt(float4 a)
{
  return (float4)(sqrt(a.lo), sqrt(a.hi));
}
#endif

#ifdef __AVX__
float8 __attribute__ ((__overloadable__))
sqrt(float8 a)
{
  return __builtin_ia32_sqrtps256(a);
}
#else
float8 __attribute__ ((__overloadable__))
sqrt(float8 a)
{
  return (float8)(sqrt(a.lo), sqrt(a.hi));
}
#endif

float16 __attribute__ ((__overloadable__))
sqrt(float16 a)
{
  return (float16)(sqrt(a.lo), sqrt(a.hi));
}

#ifdef __SSE2_
double2 _cl_sqrt_ensure_double2(double2 a)
{
  return a;
}

double __attribute__ ((__overloadable__))
sqrt(double a)
{
  return _cl_sqrt_ensure_float4(__builtin_ia32_sqrtss(*(double2*)&a)).s0;
}

double2 __attribute__ ((__overloadable__))
sqrt(double2 a)
{
  return __builtin_ia32_sqrtpd(a);
}
#else
double __attribute__ ((__overloadable__))
sqrt(double a)
{
  return __builtin_sqrt(a);
}

double2 __attribute__ ((__overloadable__))
sqrt(double2 a)
{
  return (double2)(sqrt(a.lo), sqrt(a.hi));
}
#endif

#ifdef __AVX__
double3 __attribute__ ((__overloadable__))
sqrt(double3 a)
{
  return ((double4)sqrt(*(double4*)&a)).s012;
}

double4 __attribute__ ((__overloadable__))
sqrt(double4 a)
{
  return __builtin_ia32_pd256(a);
}
#else
double3 __attribute__ ((__overloadable__))
sqrt(double3 a)
{
  return (double3)(sqrt(a.lo), sqrt(a.s2));
}

double4 __attribute__ ((__overloadable__))
sqrt(double4 a)
{
  return (double4)(sqrt(a.lo), sqrt(a.hi));
}
#endif

double8 __attribute__ ((__overloadable__))
sqrt(double8 a)
{
  return (double8)(sqrt(a.lo), sqrt(a.hi));
}

double16 __attribute__ ((__overloadable__))
sqrt(double16 a)
{
  return (double16)(sqrt(a.lo), sqrt(a.hi));
}
