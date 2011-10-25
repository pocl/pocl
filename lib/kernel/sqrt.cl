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

#undef sqrt

float sqrtf(float a);
double sqrt(double a);



float __attribute__ ((overloadable))
cl_sqrt(float a)
{
#ifdef __SSE__
  return ((float4)__builtin_ia32_sqrtss(*(float4*)&a)).s0;
#else
  return sqrtf(a);
#endif
}

float2 __attribute__ ((overloadable))
cl_sqrt(float2 a)
{
#ifdef __SSE__
  return ((float4)cl_sqrt(*(float4*)&a)).s01;
#else
  return (float2)(cl_sqrt(a.lo), cl_sqrt(a.hi));
#endif
}

float3 __attribute__ ((overloadable))
cl_sqrt(float3 a)
{
#ifdef __SSE__
  return ((float4)cl_sqrt(*(float4*)&a)).s012;
#else
  return (float3)(cl_sqrt(a.s01), cl_sqrt(a.s2));
#endif
}

float4 __attribute__ ((overloadable))
cl_sqrt(float4 a)
{
#ifdef __SSE__
  return __builtin_ia32_sqrtps(a);
#else
  return (float4)(cl_sqrt(a.lo), cl_sqrt(a.hi));
#endif
}

float8 __attribute__ ((overloadable))
cl_sqrt(float8 a)
{
#ifdef __AVX__
  return __builtin_ia32_sqrtps256(a);
#else
  return (float8)(cl_sqrt(a.lo), cl_sqrt(a.hi));
#endif
}

float16 __attribute__ ((overloadable))
cl_sqrt(float16 a)
{
  return (float16)(cl_sqrt(a.lo), cl_sqrt(a.hi));
}

double __attribute__ ((overloadable))
cl_sqrt(double a)
{
#ifdef __SSE2__
  return ((double2)__builtin_ia32_sqrtss(*(double2*)&a)).s0;
#else
  return sqrt(a);
#endif
}

double2 __attribute__ ((overloadable))
cl_sqrt(double2 a)
{
#ifdef __SSE2__
  return __builtin_ia32_sqrtpd(a);
#else
  return (double2)(cl_sqrt(a.lo), cl_sqrt(a.hi));
#endif
}

double3 __attribute__ ((overloadable))
cl_sqrt(double3 a)
{
#ifdef __AVX__
  return ((double4)cl_sqrt(*(double4*)&a)).s012;
#else
  return (double3)(cl_sqrt(a.s01), cl_sqrt(a.s2));
#endif
}

double4 __attribute__ ((overloadable))
cl_sqrt(double4 a)
{
#ifdef __AVX__
  return __builtin_ia32_pd256(a);
#else
  return (double4)(cl_sqrt(a.lo), cl_sqrt(a.hi));
#endif
}

double8 __attribute__ ((overloadable))
cl_sqrt(double8 a)
{
  return (double8)(cl_sqrt(a.lo), cl_sqrt(a.hi));
}

double16 __attribute__ ((overloadable))
cl_sqrt(double16 a)
{
  return (double16)(cl_sqrt(a.lo), cl_sqrt(a.hi));
}
