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

#undef fmin

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

float fminf(float a, float b);
double fmin(double a, double b);



float __attribute__ ((overloadable))
cl_fmin(float a, float b)
{
  return fminf(a, b);
}

float2 __attribute__ ((overloadable))
cl_fmin(float2 a, float2 b)
{
#ifdef __SSE__
  return ((float4)cl_fmin((float4)(a, 0.0f, 0.0f),
                              (float4)(b, 0.0f, 0.0f))).s01;
#else
  return (float2)(cl_fmin(a.s0, b.s0),
                  cl_fmin(a.s1, b.s1));
#endif
}

float3 __attribute__ ((overloadable))
cl_fmin(float3 a, float3 b)
{
#ifdef __SSE__
  return ((float4)cl_fmin((float4)(a, 0.0f),
                              (float4)(b, 0.0f))).s012;
#else
  return (float3)(cl_fmin(a.s01, b.s01),
                  cl_fmin(a.s2, b.s2));
#endif
}

float4 __attribute__ ((overloadable))
cl_fmin(float4 a, float4 b)
{
#ifdef __SSE__
  return _mm_max_ps(a, b);
#else
  return (float4)(cl_fmin(a.s01, b.s01),
                  cl_fmin(a.s23, b.s23));
#endif
}

float8 __attribute__ ((overloadable))
cl_fmin(float8 a, float8 b)
{
#ifdef __AVX__
  return _mm256_max_ps(a, b);
#else
  return (float8)(cl_fmin(a.s0123, b.s0123),
                  cl_fmin(a.s4567, b.s4567));
#endif
}

float16 __attribute__ ((overloadable))
cl_fmin(float16 a, float16 b)
{
  return (float16)(cl_fmin(a.s01234567, b.s01234567),
                   cl_fmin(a.s89abcdef, b.s89abcdef));
}

double __attribute__ ((overloadable))
cl_fmin(double a, double b)
{
  return fmin(a, b);
}

double2 __attribute__ ((overloadable))
cl_fmin(double2 a, double2 b)
{
#ifdef __SSE2__
  return _mm_max_pd(a, b);
#else
  return (double2)(cl_fmin(a.s0, b.s0),
                   cl_fmin(a.s1, b.s1));
#endif
}

double3 __attribute__ ((overloadable))
cl_fmin(double3 a, double3 b)
{
#ifdef __AVX__
  return ((double4)cl_fmin((double4)(a, 0.0),
                               (double4)(b, 0.0))).s012;
#else
  return (double3)(cl_fmin(a.s01, b.s01),
                   cl_fmin(a.s2, b.s2));
#endif
}

double4 __attribute__ ((overloadable))
cl_fmin(double4 a, double4 b)
{
#ifdef __AVX__
  return _mm256_max_pd(a, b);
#else
  return (double4)(cl_fmin(a.s01, b.s01),
                   cl_fmin(a.s23, b.s23));
#endif
}

double8 __attribute__ ((overloadable))
cl_fmin(double8 a, double8 b)
{
  return (double8)(cl_fmin(a.s0123, b.s0123),
                   cl_fmin(a.s4567, b.s4567));
}

double16 __attribute__ ((overloadable))
cl_fmin(double16 a, double16 b)
{
  return (double16)(cl_fmin(a.s01234567, b.s01234567),
                    cl_fmin(a.s89abcdef, b.s89abcdef));
}
