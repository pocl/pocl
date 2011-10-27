/* OpenCL built-in library: copysign()

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
copysign(float a, float b)
{
  return __builtin_copysignf(a, b);
}

float2 __attribute__ ((overloadable))
copysign(float2 a, float2 b)
{
#ifdef __SSE__
  return copysign(*(float4*)&a, *(float4*)&b).s01;
#else
  return (float2)(copysign(a.lo, b.lo), copysign(a.hi, b.hi));
#endif
}

float3 __attribute__ ((overloadable))
copysign(float3 a, float3 b)
{
#ifdef __SSE__
  return copysign(*(float4*)&a, *(float4*)&b).s012;
#else
  return (float3)(copysign(a.s01, b.s01), copysign(a.s2, b.s2));
#endif
}

float4 __attribute__ ((overloadable))
copysign(float4 a, float4 b)
{
#ifdef __SSE__
  const uint4 sign_mask = {0x80000000U, 0x80000000U, 0x80000000U, 0x80000000U};
  return as_float4((~sign_mask & as_uint4(a)) | (sign_mask & as_uint4(b)));
#else
  return (float4)(copysign(a.lo, b.lo), copysign(a.hi, b.hi));
#endif
}

float8 __attribute__ ((overloadable))
copysign(float8 a, float8 b)
{
#ifdef __AVX__
  const uint8 sign_mask =
    {0x80000000U, 0x80000000U, 0x80000000U, 0x80000000U,
     0x80000000U, 0x80000000U, 0x80000000U, 0x80000000U};
  return as_float8((~sign_mask & as_uint8(a)) | (sign_mask & as_uint8(b)));
#else
  return (float8)(copysign(a.lo, b.lo), copysign(a.hi, b.hi));
#endif
}

float16 __attribute__ ((overloadable))
copysign(float16 a, float16 b)
{
  return (float16)(copysign(a.lo, b.lo), copysign(a.hi, b.hi));
}

double __attribute__ ((overloadable))
copysign(double a, double b)
{
  return __builtin_copysign(a, b);
}

double2 __attribute__ ((overloadable))
copysign(double2 a, double2 b)
{
#ifdef __SSE2__
  const ulong2 sign_mask = {0x8000000000000000UL, 0x8000000000000000UL};
  return as_double2((~sign_mask & as_ulong2(a)) | (sign_mask & as_ulong2(b)));
#else
  return (double2)(copysign(a.lo, b.lo), copysign(a.hi, b.hi));
#endif
}

double3 __attribute__ ((overloadable))
copysign(double3 a, double3 b)
{
#ifdef __AVX__
  return copysign(*(double4*)&a, *(double4*)&b).s012;
#else
  return (double3)(copysign(a.s01, b.s01), copysign(a.s2, b.s2));
#endif
}

double4 __attribute__ ((overloadable))
copysign(double4 a, double4 b)
{
#ifdef __AVX__
  const ulong4 sign_mask =
    {0x8000000000000000UL, 0x8000000000000000UL,
     0x8000000000000000UL, 0x8000000000000000UL};
  return as_double4((~sign_mask & as_ulong4(a)) | (sign_mask & as_ulong4(b)));
#else
  return (double4)(copysign(a.lo, b.hi), copysign(a.lo, b.hi));
#endif
}

double8 __attribute__ ((overloadable))
copysign(double8 a, double8 b)
{
  return (double8)(copysign(a.lo, b.lo), copysign(a.hi, b.hi));
}

double16 __attribute__ ((overloadable))
copysign(double16 a, double16 b)
{
  return (double16)(copysign(a.lo, b.lo), copysign(a.hi, b.hi));
}
