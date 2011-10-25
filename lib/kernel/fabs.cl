/* OpenCL built-in library: fabs()

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

#undef fabs

float fabsf(float a);
double fabs(double a);



float __attribute__ ((overloadable))
cl_fabs(float a)
{
#ifdef __SSE__
  const uint sign_mask = 0x80000000U;
  return as_float(~sign_mask & as_uint(a));
#else
  return fabsf(a);
#endif
}

float2 __attribute__ ((overloadable))
cl_fabs(float2 a)
{
#ifdef __SSE__
  const uint2 sign_mask = {0x80000000U, 0x80000000U};
  return as_float2(~sign_mask & as_uint2(a));
#else
  return (float2)(cl_fabs(a.lo), cl_fabs(a.hi));
#endif
}

float3 __attribute__ ((overloadable))
cl_fabs(float3 a)
{
#ifdef __SSE__
  return cl_fabs(*(float4*)&a).s012;
#else
  return (float3)(cl_fabs(a.s01), cl_fabs(a.s2));
#endif
}

float4 __attribute__ ((overloadable))
cl_fabs(float4 a)
{
#ifdef __SSE__
  const uint4 sign_mask = {0x80000000U, 0x80000000U, 0x80000000U, 0x80000000U};
  return as_float4(~sign_mask & as_uint4(a));
#else
  return (float4)(cl_fabs(a.lo), cl_fabs(a.hi));
#endif
}

float8 __attribute__ ((overloadable))
cl_fabs(float8 a)
{
#ifdef __AVX__
  const uint8 sign_mask =
    {0x80000000U, 0x80000000U, 0x80000000U, 0x80000000U,
     0x80000000U, 0x80000000U, 0x80000000U, 0x80000000U};
  return as_float8(~sign_mask & as_uint8(a));
#else
  return (float8)(cl_fabs(a.lo), cl_fabs(a.hi));
#endif
}

float16 __attribute__ ((overloadable))
cl_fabs(float16 a)
{
  return (float16)(cl_fabs(a.lo), cl_fabs(a.hi));
}

double __attribute__ ((overloadable))
cl_fabs(double a)
{
#ifdef __SSE2__
  const ulong sign_mask = 0x8000000000000000UL;
  return as_double(~sign_mask & as_ulong(a));
#else
  return fabs(a);
#endif
}

double2 __attribute__ ((overloadable))
cl_fabs(double2 a)
{
#ifdef __SSE2__
  const ulong2 sign_mask = {0x8000000000000000UL, 0x8000000000000000UL};
  return as_double2(~sign_mask & as_ulong2(a));
#else
  return (double2)(cl_fabs(a.lo), cl_fabs(a.hi));
#endif
}

double3 __attribute__ ((overloadable))
cl_fabs(double3 a)
{
#ifdef __AVX__
  return cl_fabs(*(double4*)&a).s012;
#else
  return (double3)(cl_fabs(a.s01), cl_fabs(a.s2));
#endif
}

double4 __attribute__ ((overloadable))
cl_fabs(double4 a)
{
#ifdef __AVX__
  const ulong4 sign_mask =
    {0x8000000000000000UL, 0x8000000000000000UL,
     0x8000000000000000UL, 0x8000000000000000UL};
  return as_double4(~sign_mask & as_ulong4(a));
#else
  return (double4)(cl_fabs(a.lo), cl_fabs(a.hi));
#endif
}

double8 __attribute__ ((overloadable))
cl_fabs(double8 a)
{
  return (double8)(cl_fabs(a.lo), cl_fabs(a.hi));
}

double16 __attribute__ ((overloadable))
cl_fabs(double16 a)
{
  return (double16)(cl_fabs(a.lo), cl_fabs(a.hi));
}
