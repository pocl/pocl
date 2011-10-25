/* OpenCL built-in library: implementation templates

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



#define DEFINE_SCALARISE_1(NAME)                \
                                                \
  float NAME##f(float a);                       \
  double NAME(double a);                        \
                                                \
  float __attribute__ ((overloadable))          \
  cl_##NAME(float a)                            \
  {                                             \
    return NAME##f(a);                          \
  }                                             \
                                                \
  float2 __attribute__ ((overloadable))         \
  cl_##NAME(float2 a)                           \
  {                                             \
    return (float2)(cl_##NAME(a.lo),            \
                    cl_##NAME(a.hi));           \
  }                                             \
                                                \
  float3 __attribute__ ((overloadable))         \
  cl_##NAME(float3 a)                           \
  {                                             \
    return (float3)(cl_##NAME(a.s01),           \
                    cl_##NAME(a.s2));           \
  }                                             \
                                                \
  float4 __attribute__ ((overloadable))         \
  cl_##NAME(float4 a)                           \
  {                                             \
    return (float4)(cl_##NAME(a.lo),            \
                    cl_##NAME(a.hi));           \
  }                                             \
                                                \
  float8 __attribute__ ((overloadable))         \
  cl_##NAME(float8 a)                           \
  {                                             \
    return (float8)(cl_##NAME(a.lo),            \
                    cl_##NAME(a.hi));           \
  }                                             \
                                                \
  float16 __attribute__ ((overloadable))        \
  cl_##NAME(float16 a)                          \
  {                                             \
    return (float16)(cl_##NAME(a.lo),           \
                     cl_##NAME(a.hi));          \
  }                                             \
                                                \
  double __attribute__ ((overloadable))         \
  cl_##NAME(double a)                           \
  {                                             \
    return NAME(a);                             \
  }                                             \
                                                \
  double2 __attribute__ ((overloadable))        \
  cl_##NAME(double2 a)                          \
  {                                             \
    return (double2)(cl_##NAME(a.lo),           \
                     cl_##NAME(a.hi));          \
  }                                             \
                                                \
  double3 __attribute__ ((overloadable))        \
  cl_##NAME(double3 a)                          \
  {                                             \
    return (double3)(cl_##NAME(a.s01),          \
                     cl_##NAME(a.s2));          \
  }                                             \
                                                \
  double4 __attribute__ ((overloadable))        \
  cl_##NAME(double4 a)                          \
  {                                             \
    return (double4)(cl_##NAME(a.lo),           \
                     cl_##NAME(a.hi));          \
  }                                             \
                                                \
  double8 __attribute__ ((overloadable))        \
  cl_##NAME(double8 a)                          \
  {                                             \
    return (double8)(cl_##NAME(a.lo),           \
                     cl_##NAME(a.hi));          \
  }                                             \
                                                \
  double16 __attribute__ ((overloadable))       \
  cl_##NAME(double16 a)                         \
  {                                             \
    return (double16)(cl_##NAME(a.lo),          \
                      cl_##NAME(a.hi));         \
  }



#define DEFINE_SCALARISE_2(NAME)                \
                                                \
  float NAME##f(float a, float b);              \
  double NAME(double a, double b);              \
                                                \
  float __attribute__ ((overloadable))          \
  cl_##NAME(float a, float b)                   \
  {                                             \
    return NAME##f(a, b);                       \
  }                                             \
                                                \
  float2 __attribute__ ((overloadable))         \
  cl_##NAME(float2 a, float2 b)                 \
  {                                             \
    return (float2)(cl_##NAME(a.lo, b.lo),      \
                    cl_##NAME(a.hi, b.hi));     \
  }                                             \
                                                \
  float3 __attribute__ ((overloadable))         \
  cl_##NAME(float3 a, float3 b)                 \
  {                                             \
    return (float3)(cl_##NAME(a.s01, b.s01),    \
                    cl_##NAME(a.s2, b.s2));     \
  }                                             \
                                                \
  float4 __attribute__ ((overloadable))         \
  cl_##NAME(float4 a, float4 b)                 \
  {                                             \
    return (float4)(cl_##NAME(a.lo, b.lo),      \
                    cl_##NAME(a.hi, b.hi));     \
  }                                             \
                                                \
  float8 __attribute__ ((overloadable))         \
  cl_##NAME(float8 a, float8 b)                 \
  {                                             \
    return (float8)(cl_##NAME(a.lo, b.lo),      \
                    cl_##NAME(a.hi, b.hi));     \
  }                                             \
                                                \
  float16 __attribute__ ((overloadable))        \
  cl_##NAME(float16 a, float16 b)               \
  {                                             \
    return (float16)(cl_##NAME(a.lo, b.lo),     \
                     cl_##NAME(a.hi, b.hi));    \
  }                                             \
                                                \
  double __attribute__ ((overloadable))         \
  cl_##NAME(double a, double b)                 \
  {                                             \
    return NAME(a, b);                          \
  }                                             \
                                                \
  double2 __attribute__ ((overloadable))        \
  cl_##NAME(double2 a, double2 b)               \
  {                                             \
    return (double2)(cl_##NAME(a.lo, b.lo),     \
                     cl_##NAME(a.hi, b.hi));    \
  }                                             \
                                                \
  double3 __attribute__ ((overloadable))        \
  cl_##NAME(double3 a, double3 b)               \
  {                                             \
    return (double3)(cl_##NAME(a.s01, b.s01),   \
                     cl_##NAME(a.s2, b.s2));    \
  }                                             \
                                                \
  double4 __attribute__ ((overloadable))        \
  cl_##NAME(double4 a, double4 b)               \
  {                                             \
    return (double4)(cl_##NAME(a.lo, b.lo),     \
                     cl_##NAME(a.hi, b.hi));    \
  }                                             \
                                                \
  double8 __attribute__ ((overloadable))        \
  cl_##NAME(double8 a, double8 b)               \
  {                                             \
    return (double8)(cl_##NAME(a.lo, b.lo),     \
                     cl_##NAME(a.hi, b.hi));    \
  }                                             \
                                                \
  double16 __attribute__ ((overloadable))       \
  cl_##NAME(double16 a, double16 b)             \
  {                                             \
    return (double16)(cl_##NAME(a.lo, b.lo),    \
                      cl_##NAME(a.hi, b.hi));   \
  }



#define DEFINE_SCALARISE_3(NAME)                        \
                                                        \
  float NAME##f(float a, float b, float c);             \
  double NAME(double a, double b, double c);            \
                                                        \
  float __attribute__ ((overloadable))                  \
  cl_##NAME(float a, float b, float c)                  \
  {                                                     \
    return NAME##f(a, b, c);                            \
  }                                                     \
                                                        \
  float2 __attribute__ ((overloadable))                 \
  cl_##NAME(float2 a, float2 b, float2 c)               \
  {                                                     \
    return (float2)(cl_##NAME(a.lo, b.lo, c.lo),        \
                    cl_##NAME(a.hi, b.hi, c.hi));       \
  }                                                     \
                                                        \
  float3 __attribute__ ((overloadable))                 \
  cl_##NAME(float3 a, float3 b, float3 c)               \
  {                                                     \
    return (float3)(cl_##NAME(a.s01, b.s01, c.s01),     \
                    cl_##NAME(a.s2, b.s2, c.s2));       \
  }                                                     \
                                                        \
  float4 __attribute__ ((overloadable))                 \
  cl_##NAME(float4 a, float4 b, float4 c)               \
  {                                                     \
    return (float4)(cl_##NAME(a.lo, b.lo, c.lo),        \
                    cl_##NAME(a.hi, b.hi, c.hi));       \
  }                                                     \
                                                        \
  float8 __attribute__ ((overloadable))                 \
  cl_##NAME(float8 a, float8 b, float8 c)               \
  {                                                     \
    return (float8)(cl_##NAME(a.lo, b.lo, c.lo),        \
                    cl_##NAME(a.hi, b.hi, c.hi));       \
  }                                                     \
                                                        \
  float16 __attribute__ ((overloadable))                \
  cl_##NAME(float16 a, float16 b, float16 c)            \
  {                                                     \
    return (float16)(cl_##NAME(a.lo, b.lo, c.lo),       \
                     cl_##NAME(a.hi, b.hi, c.hi));      \
  }                                                     \
                                                        \
  double __attribute__ ((overloadable))                 \
  cl_##NAME(double a, double b, double c)               \
  {                                                     \
    return NAME(a, b, c);                               \
  }                                                     \
                                                        \
  double2 __attribute__ ((overloadable))                \
  cl_##NAME(double2 a, double2 b, double2 c)            \
  {                                                     \
    return (double2)(cl_##NAME(a.lo, b.lo, c.lo),       \
                     cl_##NAME(a.hi, b.hi, c.hi));      \
  }                                                     \
                                                        \
  double3 __attribute__ ((overloadable))                \
  cl_##NAME(double3 a, double3 b, double3 c)            \
  {                                                     \
    return (double3)(cl_##NAME(a.s01, b.s01, c.s01),    \
                     cl_##NAME(a.s2, b.s2, c.s2));      \
  }                                                     \
                                                        \
  double4 __attribute__ ((overloadable))                \
  cl_##NAME(double4 a, double4 b, double4 c)            \
  {                                                     \
    return (double4)(cl_##NAME(a.lo, b.lo, c.lo),       \
                     cl_##NAME(a.hi, b.hi, c.hi));      \
  }                                                     \
                                                        \
  double8 __attribute__ ((overloadable))                \
  cl_##NAME(double8 a, double8 b, double8 c)            \
  {                                                     \
    return (double8)(cl_##NAME(a.lo, b.lo, c.lo),       \
                     cl_##NAME(a.hi, b.hi, c.hi));      \
  }                                                     \
                                                        \
  double16 __attribute__ ((overloadable))               \
  cl_##NAME(double16 a, double16 b, double16 c)         \
  {                                                     \
    return (double16)(cl_##NAME(a.lo, b.lo, c.lo),      \
                      cl_##NAME(a.hi, b.hi, c.hi));     \
  }



#define DEFINE_EXPR_1(NAME, EXPR)                               \
                                                                \
  float __attribute__ ((overloadable))                          \
  cl_##NAME(float a)                                            \
  {                                                             \
    typedef float stype;                                        \
    return EXPR;                                                \
  }                                                             \
                                                                \
  float2 __attribute__ ((overloadable))                         \
  cl_##NAME(float2 a)                                           \
  {                                                             \
    typedef float stype;                                        \
    return EXPR;                                                \
  }                                                             \
                                                                \
  float3 __attribute__ ((overloadable))                         \
  cl_##NAME(float3 a)                                           \
  {                                                             \
    typedef float stype;                                        \
    return EXPR;                                                \
  }                                                             \
                                                                \
  float4 __attribute__ ((overloadable))                         \
  cl_##NAME(float4 a)                                           \
  {                                                             \
    typedef float stype;                                        \
    return EXPR;                                                \
  }                                                             \
                                                                \
  float8 __attribute__ ((overloadable))                         \
  cl_##NAME(float8 a)                                           \
  {                                                             \
    typedef float stype;                                        \
    return EXPR;                                                \
  }                                                             \
                                                                \
  float16 __attribute__ ((overloadable))                        \
  cl_##NAME(float16 a)                                          \
  {                                                             \
    typedef float stype;                                        \
    return EXPR;                                                \
  }                                                             \
                                                                \
  double __attribute__ ((overloadable))                         \
  cl_##NAME(double a)                                           \
  {                                                             \
    typedef double stype;                                       \
    return EXPR;                                                \
  }                                                             \
                                                                \
  double2 __attribute__ ((overloadable))                        \
  cl_##NAME(double2 a)                                          \
  {                                                             \
    typedef double stype;                                       \
    return EXPR;                                                \
  }                                                             \
                                                                \
  double3 __attribute__ ((overloadable))                        \
  cl_##NAME(double3 a)                                          \
  {                                                             \
    typedef double stype;                                       \
    return EXPR;                                                \
  }                                                             \
                                                                \
  double4 __attribute__ ((overloadable))                        \
  cl_##NAME(double4 a)                                          \
  {                                                             \
    typedef double stype;                                       \
    return EXPR;                                                \
  }                                                             \
                                                                \
  double8 __attribute__ ((overloadable))                        \
  cl_##NAME(double8 a)                                          \
  {                                                             \
    typedef double stype;                                       \
    return EXPR;                                                \
  }                                                             \
                                                                \
  double16 __attribute__ ((overloadable))                       \
  cl_##NAME(double16 a)                                         \
  {                                                             \
    typedef double stype;                                       \
    return EXPR;                                                \
  }



#define DEFINE_EXPR_2(NAME, EXPR)                                       \
                                                                        \
  float __attribute__ ((overloadable))                                  \
  cl_##NAME(float a, float b)                                           \
  {                                                                     \
    typedef float stype;                                                \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  float2 __attribute__ ((overloadable))                                 \
  cl_##NAME(float2 a, float2 b)                                         \
  {                                                                     \
    typedef float stype;                                                \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  float3 __attribute__ ((overloadable))                                 \
  cl_##NAME(float3 a, float3 b)                                         \
  {                                                                     \
    typedef float stype;                                                \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  float4 __attribute__ ((overloadable))                                 \
  cl_##NAME(float4 a, float4 b)                                         \
  {                                                                     \
    typedef float stype;                                                \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  float8 __attribute__ ((overloadable))                                 \
  cl_##NAME(float8 a, float8 b)                                         \
  {                                                                     \
    typedef float stype;                                                \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  float16 __attribute__ ((overloadable))                                \
  cl_##NAME(float16 a, float16 b)                                       \
  {                                                                     \
    typedef float stype;                                                \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  double __attribute__ ((overloadable))                                 \
  cl_##NAME(double a, double b)                                         \
  {                                                                     \
    typedef double stype;                                               \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  double2 __attribute__ ((overloadable))                                \
  cl_##NAME(double2 a, double2 b)                                       \
  {                                                                     \
    typedef double stype;                                               \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  double3 __attribute__ ((overloadable))                                \
  cl_##NAME(double3 a, double3 b)                                       \
  {                                                                     \
    typedef double stype;                                               \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  double4 __attribute__ ((overloadable))                                \
  cl_##NAME(double4 a, double4 b)                                       \
  {                                                                     \
    typedef double stype;                                               \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  double8 __attribute__ ((overloadable))                                \
  cl_##NAME(double8 a, double8 b)                                       \
  {                                                                     \
    typedef double stype;                                               \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  double16 __attribute__ ((overloadable))                               \
  cl_##NAME(double16 a, double16 b)                                     \
  {                                                                     \
    typedef double stype;                                               \
    return EXPR;                                                        \
  }



#define DEFINE_EXPR_3(NAME, EXPR)                                       \
                                                                        \
  float __attribute__ ((overloadable))                                  \
  cl_##NAME(float a, float b, float c)                                  \
  {                                                                     \
    typedef float stype;                                                \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  float2 __attribute__ ((overloadable))                                 \
  cl_##NAME(float2 a, float2 b, float2 c)                               \
  {                                                                     \
    typedef float stype;                                                \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  float3 __attribute__ ((overloadable))                                 \
  cl_##NAME(float3 a, float3 b, float3 c)                               \
  {                                                                     \
    typedef float stype;                                                \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  float4 __attribute__ ((overloadable))                                 \
  cl_##NAME(float4 a, float4 b, float4 c)                               \
  {                                                                     \
    typedef float stype;                                                \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  float8 __attribute__ ((overloadable))                                 \
  cl_##NAME(float8 a, float8 b, float8 c)                               \
  {                                                                     \
    typedef float stype;                                                \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  float16 __attribute__ ((overloadable))                                \
  cl_##NAME(float16 a, float16 b, float16 c)                            \
  {                                                                     \
    typedef float stype;                                                \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  double __attribute__ ((overloadable))                                 \
  cl_##NAME(double a, double b, double c)                               \
  {                                                                     \
    typedef double stype;                                               \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  double2 __attribute__ ((overloadable))                                \
  cl_##NAME(double2 a, double2 b, double2 c)                            \
  {                                                                     \
    typedef double stype;                                               \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  double3 __attribute__ ((overloadable))                                \
  cl_##NAME(double3 a, double3 b, double3 c)                            \
  {                                                                     \
    typedef double stype;                                               \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  double4 __attribute__ ((overloadable))                                \
  cl_##NAME(double4 a, double4 b, double4 c)                            \
  {                                                                     \
    typedef double stype;                                               \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  double8 __attribute__ ((overloadable))                                \
  cl_##NAME(double8 a, double8 b, double8 c)                            \
  {                                                                     \
    typedef double stype;                                               \
    return EXPR;                                                        \
  }                                                                     \
                                                                        \
  double16 __attribute__ ((overloadable))                               \
  cl_##NAME(double16 a, double16 b, double16 c)                         \
  {                                                                     \
    typedef double stype;                                               \
    return EXPR;                                                        \
  }
