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

#undef copysign

// Import Intel/AMD vector instructions
#ifdef __SSE__
#  define extern
#  define static
#  include <xmmintrin.h>
#endif

#ifdef __SSE2__
#  define extern
#  define static
#  include <emmintrin.h>
#endif

#ifdef __AVX__
#  define extern
#  define static
#  include <immintrin.h>
#endif

float copysignf(float a, float b);
double copysign(double a, double b);



float __attribute__ ((overloadable))
cl_copysign(float a, float b)
{
  return copysignf(a, b);
}

float2 __attribute__ ((overloadable))
cl_copysign(float2 a, float2 b)
{
#ifdef __SSE__
  return ((float4)cl_copysign((float4)(a, 0.0f, 0.0f),
                              (float4)(b, 0.0f, 0.0f))).s01;
#else
  return (float2)(cl_copysign(a.s0, b.s0),
                  cl_copysign(a.s1, b.s1));
#endif
}

float3 __attribute__ ((overloadable))
cl_copysign(float3 a, float3 b)
{
#ifdef __SSE__
  return ((float4)cl_copysign((float4)(a, 0.0f),
                              (float4)(b, 0.0f))).s012;
#else
  return (float3)(cl_copysign(a.s01, b.s01),
                  cl_copysign(a.s2, b.s2));
#endif
}

float4 __attribute__ ((overloadable))
cl_copysign(float4 a, float4 b)
{
#ifdef __SSE__
  const float4 sign_mask =
    as_float4((uint4)(0x80000000U, 0x80000000U, 0x80000000U, 0x80000000U));
  return _mm_or_ps(_mm_andnot_ps(sign_mask, a),
                   _mm_and_ps(sign_mask, b));
#else
  return (float4)(cl_copysign(a.s01, b.s01),
                  cl_copysign(a.s23, b.s23));
#endif
}

float8 __attribute__ ((overloadable))
cl_copysign(float8 a, float8 b)
{
#ifdef __AVX__
  const float8 sign_mask =
    as_float8((uint8)(0x80000000U, 0x80000000U, 0x80000000U, 0x80000000U,
                      0x80000000U, 0x80000000U, 0x80000000U, 0x80000000U));
  return _mm256_or_ps(_mm256_andnot_ps(sign_mask, a),
                      _mm256_and_ps(sign_mask, b));
#else
  return (float8)(cl_copysign(a.s0123, b.s0123),
                  cl_copysign(a.s4567, b.s4567));
#endif
}

float16 __attribute__ ((overloadable))
cl_copysign(float16 a, float16 b)
{
  return (float16)(cl_copysign(a.s01234567, b.s01234567),
                   cl_copysign(a.s89abcdef, b.s89abcdef));
}

double __attribute__ ((overloadable))
cl_copysign(double a, double b)
{
  return copysign(a, b);
}

double2 __attribute__ ((overloadable))
cl_copysign(double2 a, double2 b)
{
#ifdef __SSE2__
  const double2 sign_mask =
    as_double2((ulong2)(0x8000000000000000UL, 0x8000000000000000UL));
  return _mm_or_pd(_mm_andnot_pd(sign_mask, a),
                   _mm_and_pd(sign_mask, b));
#else
  return (double2)(cl_copysign(a.s0, b.s0),
                   cl_copysign(a.s1, b.s1));
#endif
}

double3 __attribute__ ((overloadable))
cl_copysign(double3 a, double3 b)
{
#ifdef __AVX__
  return ((double4)cl_copysign((double4)(a, 0.0),
                               (double4)(b, 0.0))).s012;
#else
  return (double3)(cl_copysign(a.s01, b.s01),
                   cl_copysign(a.s2, b.s2));
#endif
}

double4 __attribute__ ((overloadable))
cl_copysign(double4 a, double4 b)
{
#ifdef __AVX__
  const double4 sign_mask =
    as_double4((long4)(0x8000000000000000UL, 0x8000000000000000UL,
                       0x8000000000000000UL, 0x8000000000000000UL));
  return _mm256_or_pd(_mm256_andnot_pd(sign_mask, a),
                      _mm256_and_pd(sign_mask, b));
#else
  return (double4)(cl_copysign(a.s01, b.s01),
                   cl_copysign(a.s23, b.s23));
#endif
}

double8 __attribute__ ((overloadable))
cl_copysign(double8 a, double8 b)
{
  return (double8)(cl_copysign(a.s0123, b.s0123),
                   cl_copysign(a.s4567, b.s4567));
}

double16 __attribute__ ((overloadable))
cl_copysign(double16 a, double16 b)
{
  return (double16)(cl_copysign(a.s01234567, b.s01234567),
                    cl_copysign(a.s89abcdef, b.s89abcdef));
}
