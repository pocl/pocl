/* pocl/_kernel.h - OpenCL types and runtime library
   functions declarations.

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

/* Enable double precision. This should really only be done when
   building the run-time library; when building application code, we
   should instead check a macro to see whether the application has
   enabled this. At the moment, always enable this seems fine, since
   all our target devices will support double precision anyway.
*/
#pragma OPENCL EXTENSION cl_khr_fp64: enable

//#define __kernel __attribute__ ((noinline))
#define __global __attribute__ ((address_space(3)))
#define __local __attribute__ ((address_space(4)))
#define __constant __attribute__ ((address_space(5)))

#define global __attribute__ ((address_space(3)))
#define local __attribute__ ((address_space(4)))
#define constant __attribute__ ((address_space(5)))

/* Data types */

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

/* Ensure the data types have the right sizes */
#define cl_static_assert(_t, _x) typedef int ai##_t[(_x) ? 1 : -1]
cl_static_assert(char, sizeof(char) == 1);
cl_static_assert(uchar, sizeof(uchar) == 1);
cl_static_assert(short, sizeof(short) == 2);
cl_static_assert(ushort, sizeof(ushort) == 2);
cl_static_assert(int, sizeof(int) == 4);
cl_static_assert(uint, sizeof(uint) == 4);
cl_static_assert(long, sizeof(long) == 8);
cl_static_assert(ulong, sizeof(ulong) == 8);

typedef char char2 __attribute__((ext_vector_type(2)));
typedef char char3 __attribute__((ext_vector_type(3)));
typedef char char4 __attribute__((ext_vector_type(4)));
typedef char char8 __attribute__((ext_vector_type(8)));
typedef char char16 __attribute__((ext_vector_type(16)));

typedef uchar uchar2 __attribute__((ext_vector_type(2)));
typedef uchar uchar3 __attribute__((ext_vector_type(3)));
typedef uchar uchar4 __attribute__((ext_vector_type(4)));
typedef uchar uchar8 __attribute__((ext_vector_type(8)));
typedef uchar uchar16 __attribute__((ext_vector_type(16)));

typedef short short2 __attribute__((ext_vector_type(2)));
typedef short short3 __attribute__((ext_vector_type(3)));
typedef short short4 __attribute__((ext_vector_type(4)));
typedef short short8 __attribute__((ext_vector_type(8)));
typedef short short16 __attribute__((ext_vector_type(16)));

typedef ushort ushort2 __attribute__((ext_vector_type(2)));
typedef ushort ushort3 __attribute__((ext_vector_type(3)));
typedef ushort ushort4 __attribute__((ext_vector_type(4)));
typedef ushort ushort8 __attribute__((ext_vector_type(8)));
typedef ushort ushort16 __attribute__((ext_vector_type(16)));

typedef int int2 __attribute__((ext_vector_type(2)));
typedef int int3 __attribute__((ext_vector_type(3)));
typedef int int4 __attribute__((ext_vector_type(4)));
typedef int int8 __attribute__((ext_vector_type(8)));
typedef int int16 __attribute__((ext_vector_type(16)));

typedef uint uint2 __attribute__((ext_vector_type(2)));
typedef uint uint3 __attribute__((ext_vector_type(3)));
typedef uint uint4 __attribute__((ext_vector_type(4)));
typedef uint uint8 __attribute__((ext_vector_type(8)));
typedef uint uint16 __attribute__((ext_vector_type(16)));

typedef long long2 __attribute__((ext_vector_type(2)));
typedef long long3 __attribute__((ext_vector_type(3)));
typedef long long4 __attribute__((ext_vector_type(4)));
typedef long long8 __attribute__((ext_vector_type(8)));
typedef long long16 __attribute__((ext_vector_type(16)));

typedef ulong ulong2 __attribute__((ext_vector_type(2)));
typedef ulong ulong3 __attribute__((ext_vector_type(3)));
typedef ulong ulong4 __attribute__((ext_vector_type(4)));
typedef ulong ulong8 __attribute__((ext_vector_type(8)));
typedef ulong ulong16 __attribute__((ext_vector_type(16)));

typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef float float16 __attribute__((ext_vector_type(16)));

typedef double double2 __attribute__((ext_vector_type(2)));
typedef double double3 __attribute__((ext_vector_type(3)));
typedef double double4 __attribute__((ext_vector_type(4)));
typedef double double8 __attribute__((ext_vector_type(8)));
typedef double double16 __attribute__((ext_vector_type(16)));

typedef enum {
  CLK_LOCAL_MEM_FENCE = 0x1,
  CLK_GLOBAL_MEM_FENCE = 0x2
} cl_mem_fence_flags;

uint get_global_id(uint);
uint get_group_id(uint);
uint get_local_id(uint);

int mad24(int x, int y, int z);

/* Constants */

#define FLT_DIG          6
#define FLT_MANT_DIG     24
#define FLT_MAX_10_EXP   +38
#define FLT_MAX_EXP      +128
#define FLT_MIN_10_EXP   -37
#define FLT_MIN_EXP      -125
#define FLT_RADIX        2
#define FLT_MAX          0x1.fffffep127f
#define FLT_MIN          0x1.0p-126f
#define FLT_EPSILON      0x1.0p-23f

#define HUGE_VALF __builtin_huge_valf()

#define M_E_F        2.71828182845904523536028747135f
#define M_LOG2E_F    1.44269504088896340735992468100f
#define M_LOG10E_F   0.434294481903251827651128918917f
#define M_LN2_F      0.693147180559945309417232121458f
#define M_LN10_F     2.30258509299404568401799145468f
#define M_PI_F       3.14159265358979323846264338328f
#define M_PI_2_F     1.57079632679489661923132169164f
#define M_PI_4_F     0.785398163397448309615660845820f
#define M_1_PI_F     0.318309886183790671537767526745f
#define M_2_PI_F     0.636619772367581343075535053490f
#define M_2_SQRTPI_F 1.12837916709551257389615890312f
#define M_SQRT2_F    1.41421356237309504880168872421f
#define M_SQRT1_2_F  0.707106781186547524400844362105f

#define DBL_DIG        15
#define DBL_MANT_DIG   53
#define DBL_MAX_10_EXP +308
#define DBL_MAX_EXP    +1024
#define DBL_MIN_10_EXP -307
#define DBL_MIN_EXP    -1021
#define DBL_MAX        0x1.fffffffffffffp1023
#define DBL_MIN        0x1.0p-1022
#define DBL_EPSILON    0x1.0p-52

#define HUGE_VAL __builtin_huge_val()

#define M_E        2.71828182845904523536028747135
#define M_LOG2E    1.44269504088896340735992468100
#define M_LOG10E   0.434294481903251827651128918917
#define M_LN2      0.693147180559945309417232121458
#define M_LN10     2.30258509299404568401799145468
#define M_PI       3.14159265358979323846264338328
#define M_PI_2     1.57079632679489661923132169164
#define M_PI_4     0.785398163397448309615660845820
#define M_1_PI     0.318309886183790671537767526745
#define M_2_PI     0.636619772367581343075535053490
#define M_2_SQRTPI 1.12837916709551257389615890312
#define M_SQRT2    1.41421356237309504880168872421
#define M_SQRT1_2  0.707106781186547524400844362105

/* Conversion functions */

#define CL_DECLARE_AS_TYPE(SRC, DST)                    \
  DST __attribute__ ((overloadable)) as_##DST(SRC a);

/* 1 byte */
#define CL_DECLARE_AS_TYPE_1(DST)                  \
  CL_DECLARE_AS_TYPE(DST, char)                    \
  CL_DECLARE_AS_TYPE(DST, uchar)
CL_DECLARE_AS_TYPE_1(char)
CL_DECLARE_AS_TYPE_1(uchar)

/* 2 bytes */
#define CL_DECLARE_AS_TYPE_2(DST)                  \
  CL_DECLARE_AS_TYPE(DST, char2)                   \
  CL_DECLARE_AS_TYPE(DST, uchar2)                  \
  CL_DECLARE_AS_TYPE(DST, short)                   \
  CL_DECLARE_AS_TYPE(DST, ushort)
CL_DECLARE_AS_TYPE_2(char2)
CL_DECLARE_AS_TYPE_2(uchar2)
CL_DECLARE_AS_TYPE_2(short)
CL_DECLARE_AS_TYPE_2(ushort)

/* 4 bytes */
#define CL_DECLARE_AS_TYPE_4(DST)                  \
  CL_DECLARE_AS_TYPE(DST, char4)                   \
  CL_DECLARE_AS_TYPE(DST, uchar4)                  \
  CL_DECLARE_AS_TYPE(DST, short2)                  \
  CL_DECLARE_AS_TYPE(DST, ushort2)                 \
  CL_DECLARE_AS_TYPE(DST, int)                     \
  CL_DECLARE_AS_TYPE(DST, uint)                    \
  CL_DECLARE_AS_TYPE(DST, float)
CL_DECLARE_AS_TYPE_4(char4)
CL_DECLARE_AS_TYPE_4(uchar4)
CL_DECLARE_AS_TYPE_4(short2)
CL_DECLARE_AS_TYPE_4(ushort2)
CL_DECLARE_AS_TYPE_4(int)
CL_DECLARE_AS_TYPE_4(uint)
CL_DECLARE_AS_TYPE_4(float)

/* 8 bytes */
#define CL_DECLARE_AS_TYPE_8(DST)                  \
  CL_DECLARE_AS_TYPE(DST, char8)                   \
  CL_DECLARE_AS_TYPE(DST, uchar8)                  \
  CL_DECLARE_AS_TYPE(DST, short4)                  \
  CL_DECLARE_AS_TYPE(DST, ushort4)                 \
  CL_DECLARE_AS_TYPE(DST, int2)                    \
  CL_DECLARE_AS_TYPE(DST, uint2)                   \
  CL_DECLARE_AS_TYPE(DST, long)                    \
  CL_DECLARE_AS_TYPE(DST, ulong)                   \
  CL_DECLARE_AS_TYPE(DST, float2)                  \
  CL_DECLARE_AS_TYPE(DST, double)
CL_DECLARE_AS_TYPE_8(char8)
CL_DECLARE_AS_TYPE_8(uchar8)
CL_DECLARE_AS_TYPE_8(short4)
CL_DECLARE_AS_TYPE_8(ushort4)
CL_DECLARE_AS_TYPE_8(int2)
CL_DECLARE_AS_TYPE_8(uint2)
CL_DECLARE_AS_TYPE_8(long)
CL_DECLARE_AS_TYPE_8(ulong)
CL_DECLARE_AS_TYPE_8(float2)
CL_DECLARE_AS_TYPE_8(double)

/* 16 bytes */
#define CL_DECLARE_AS_TYPE_16(DST)                 \
  CL_DECLARE_AS_TYPE(DST, char16)                  \
  CL_DECLARE_AS_TYPE(DST, uchar16)                 \
  CL_DECLARE_AS_TYPE(DST, short8)                  \
  CL_DECLARE_AS_TYPE(DST, ushort8)                 \
  CL_DECLARE_AS_TYPE(DST, int4)                    \
  CL_DECLARE_AS_TYPE(DST, uint4)                   \
  CL_DECLARE_AS_TYPE(DST, long2)                   \
  CL_DECLARE_AS_TYPE(DST, ulong2)                  \
  CL_DECLARE_AS_TYPE(DST, float4)                  \
  CL_DECLARE_AS_TYPE(DST, double2)
CL_DECLARE_AS_TYPE_16(char16)
CL_DECLARE_AS_TYPE_16(uchar16)
CL_DECLARE_AS_TYPE_16(short8)
CL_DECLARE_AS_TYPE_16(ushort8)
CL_DECLARE_AS_TYPE_16(int4)
CL_DECLARE_AS_TYPE_16(uint4)
CL_DECLARE_AS_TYPE_16(long2)
CL_DECLARE_AS_TYPE_16(ulong2)
CL_DECLARE_AS_TYPE_16(float4)
CL_DECLARE_AS_TYPE_16(double2)

/* 32 bytes */
#define CL_DECLARE_AS_TYPE_32(DST)                 \
  CL_DECLARE_AS_TYPE(DST, short16)                 \
  CL_DECLARE_AS_TYPE(DST, ushort16)                \
  CL_DECLARE_AS_TYPE(DST, int8)                    \
  CL_DECLARE_AS_TYPE(DST, uint8)                   \
  CL_DECLARE_AS_TYPE(DST, long4)                   \
  CL_DECLARE_AS_TYPE(DST, ulong4)                  \
  CL_DECLARE_AS_TYPE(DST, float8)                  \
  CL_DECLARE_AS_TYPE(DST, double4)
CL_DECLARE_AS_TYPE_32(short16)
CL_DECLARE_AS_TYPE_32(ushort16)
CL_DECLARE_AS_TYPE_32(int8)
CL_DECLARE_AS_TYPE_32(uint8)
CL_DECLARE_AS_TYPE_32(long4)
CL_DECLARE_AS_TYPE_32(ulong4)
CL_DECLARE_AS_TYPE_32(float8)
CL_DECLARE_AS_TYPE_32(double4)

/* 64 bytes */
#define CL_DECLARE_AS_TYPE_64(DST)                 \
  CL_DECLARE_AS_TYPE(DST, int16)                   \
  CL_DECLARE_AS_TYPE(DST, uint16)                  \
  CL_DECLARE_AS_TYPE(DST, long8)                   \
  CL_DECLARE_AS_TYPE(DST, ulong8)                  \
  CL_DECLARE_AS_TYPE(DST, float16)                 \
  CL_DECLARE_AS_TYPE(DST, double8)
CL_DECLARE_AS_TYPE_64(int16)
CL_DECLARE_AS_TYPE_64(uint16)
CL_DECLARE_AS_TYPE_64(long8)
CL_DECLARE_AS_TYPE_64(ulong8)
CL_DECLARE_AS_TYPE_64(float16)
CL_DECLARE_AS_TYPE_64(double8)

/* 128 bytes */
#define CL_DECLARE_AS_TYPE_128(DST)                \
  CL_DECLARE_AS_TYPE(DST, long16)                  \
  CL_DECLARE_AS_TYPE(DST, ulong16)                 \
  CL_DECLARE_AS_TYPE(DST, double16)
CL_DECLARE_AS_TYPE_128(long16)
CL_DECLARE_AS_TYPE_128(ulong16)
CL_DECLARE_AS_TYPE_128(double16)

/* Trigonometric and other functions */

#define CL_DECLARE_FUNC1(NAME)                                          \
  float __attribute__ ((overloadable)) cl_##NAME(float a);              \
  float2 __attribute__ ((overloadable)) cl_##NAME(float2 a);            \
  float3 __attribute__ ((overloadable)) cl_##NAME(float3 a);            \
  float4 __attribute__ ((overloadable)) cl_##NAME(float4 a);            \
  float8 __attribute__ ((overloadable)) cl_##NAME(float8 a);            \
  float16 __attribute__ ((overloadable)) cl_##NAME(float16 a);          \
  double __attribute__ ((overloadable)) cl_##NAME(double a);            \
  double2 __attribute__ ((overloadable)) cl_##NAME(double2 a);          \
  double3 __attribute__ ((overloadable)) cl_##NAME(double3 a);          \
  double4 __attribute__ ((overloadable)) cl_##NAME(double4 a);          \
  double8 __attribute__ ((overloadable)) cl_##NAME(double8 a);          \
  double16 __attribute__ ((overloadable)) cl_##NAME(double16 a);
#define CL_DECLARE_FUNC2(NAME)                                          \
  float __attribute__ ((overloadable)) cl_##NAME(float a, float b);     \
  float2 __attribute__ ((overloadable)) cl_##NAME(float2 a, float2 b);  \
  float3 __attribute__ ((overloadable)) cl_##NAME(float3 a, float3 b);  \
  float4 __attribute__ ((overloadable)) cl_##NAME(float4 a, float4 b);  \
  float8 __attribute__ ((overloadable)) cl_##NAME(float8 a, float8 b);  \
  float16 __attribute__ ((overloadable)) cl_##NAME(float16 a, float16 b); \
  double __attribute__ ((overloadable)) cl_##NAME(double a, double b);  \
  double2 __attribute__ ((overloadable)) cl_##NAME(double2 a, double2 b); \
  double3 __attribute__ ((overloadable)) cl_##NAME(double3 a, double3 b); \
  double4 __attribute__ ((overloadable)) cl_##NAME(double4 a, double4 b); \
  double8 __attribute__ ((overloadable)) cl_##NAME(double8 a, double8 b); \
  double16 __attribute__ ((overloadable)) cl_##NAME(double16 a, double16 b);
#define CL_DECLARE_SFUNC2(NAME)                                         \
  float __attribute__ ((overloadable)) cl_##NAME(float a, float b);     \
  float __attribute__ ((overloadable)) cl_##NAME(float2 a, float2 b);   \
  float __attribute__ ((overloadable)) cl_##NAME(float3 a, float3 b);   \
  float __attribute__ ((overloadable)) cl_##NAME(float4 a, float4 b);   \
  float __attribute__ ((overloadable)) cl_##NAME(float8 a, float8 b);   \
  float __attribute__ ((overloadable)) cl_##NAME(float16 a, float16 b); \
  double __attribute__ ((overloadable)) cl_##NAME(double a, double b);  \
  double __attribute__ ((overloadable)) cl_##NAME(double2 a, double2 b); \
  double __attribute__ ((overloadable)) cl_##NAME(double3 a, double3 b); \
  double __attribute__ ((overloadable)) cl_##NAME(double4 a, double4 b); \
  double __attribute__ ((overloadable)) cl_##NAME(double8 a, double8 b); \
  double __attribute__ ((overloadable)) cl_##NAME(double16 a, double16 b);
#define CL_DECLARE_FUNC3(NAME)                          \
  float __attribute__ ((overloadable))                  \
    cl_##NAME(float a, float b, float c);               \
  float2 __attribute__ ((overloadable))                 \
    cl_##NAME(float2 a, float2 b, float2 c);            \
  float3 __attribute__ ((overloadable))                 \
    cl_##NAME(float3 a, float3 b, float3 c);            \
  float4 __attribute__ ((overloadable))                 \
    cl_##NAME(float4 a, float4 b, float4 c);            \
  float8 __attribute__ ((overloadable))                 \
    cl_##NAME(float8 a, float8 b, float8 c);            \
  float16 __attribute__ ((overloadable))                \
    cl_##NAME(float16 a, float16 b, float16 c);         \
  double __attribute__ ((overloadable))                 \
    cl_##NAME(double a, double b, double c);            \
  double2 __attribute__ ((overloadable))                \
    cl_##NAME(double2 a, double2 b, double2 c);         \
  double3 __attribute__ ((overloadable))                \
    cl_##NAME(double3 a, double3 b, double3 c);         \
  double4 __attribute__ ((overloadable))                \
    cl_##NAME(double4 a, double4 b, double4 c);         \
  double8 __attribute__ ((overloadable))                \
    cl_##NAME(double8 a, double8 b, double8 c);         \
  double16 __attribute__ ((overloadable))               \
    cl_##NAME(double16 a, double16 b, double16 c);

#define acos      cl_acos
#define acosh     cl_acosh
#define acospi    cl_acospi
#define asin      cl_asin
#define asinh     cl_asinh
#define asinpi    cl_asinpi
#define atan      cl_atan
#define atan2     cl_atan2
#define atan2pi   cl_atan2pi
#define atanh     cl_atanh
#define atanpi    cl_atanpi
#define cbrt      cl_cbrt
#define copysign  cl_copysign
#define cos       cl_cos
#define cosh      cl_cosh
#define cospi     cl_cospi
#define dot       cl_dot
#define erfc      cl_erfc
#define erf       cl_erf
#define exp       cl_exp
#define exp2      cl_exp2
#define exp10     cl_exp10
#define expm1     cl_expm1
#define fabs      cl_fabs
#define fdim      cl_fdim
#define fma       cl_fma
#define fmax      cl_fmax
#define fmin      cl_fmin
#define fmod      cl_fmod
#define gamma     cl_gamma
#define hypot     cl_hypot
#define lgamma    cl_lgamma
#define log       cl_log
#define log2      cl_log2
#define log10     cl_log10
#define log1p     cl_log1p
#define logb      cl_logb
#define mad       cl_mad
#define maxmag    cl_maxmag
#define minmag    cl_minmag
#define nextafter cl_nextafter
#define pow       cl_pow
#define remainder cl_remainder
#define rint      cl_rint
#define round     cl_round
#define rsqrt     cl_rsqrt
#define sin       cl_sin
#define sinh      cl_sinh
#define sinpi     cl_sinpi
#define sqrt      cl_sqrt
#define tan       cl_tan
#define tanh      cl_tanh
#define tanpi     cl_tanpi
#define tgamma    cl_tgamma
#define trunc     cl_trunc

CL_DECLARE_FUNC1(acos)
CL_DECLARE_FUNC1(acosh)
CL_DECLARE_FUNC1(acospi)
CL_DECLARE_FUNC1(asin)
CL_DECLARE_FUNC1(asinh)
CL_DECLARE_FUNC1(asinpi)
CL_DECLARE_FUNC1(atan)
CL_DECLARE_FUNC2(atan2)
CL_DECLARE_FUNC2(atan2pi)
CL_DECLARE_FUNC1(atanh)
CL_DECLARE_FUNC1(atanpi)
CL_DECLARE_FUNC1(cbrt)
CL_DECLARE_FUNC2(copysign)
CL_DECLARE_FUNC1(cos)
CL_DECLARE_FUNC1(cosh)
CL_DECLARE_FUNC1(cospi)
CL_DECLARE_SFUNC2(dot)
CL_DECLARE_FUNC1(erfc)
CL_DECLARE_FUNC1(erf)
CL_DECLARE_FUNC1(exp)
CL_DECLARE_FUNC1(exp2)
CL_DECLARE_FUNC1(exp10)
CL_DECLARE_FUNC1(expm1)
CL_DECLARE_FUNC1(fabs)
CL_DECLARE_FUNC2(fdim)
CL_DECLARE_FUNC3(fma)
CL_DECLARE_FUNC2(fmax)
CL_DECLARE_FUNC2(fmin)
CL_DECLARE_FUNC2(fmod)
CL_DECLARE_FUNC1(gamma)
CL_DECLARE_FUNC2(hypot)
CL_DECLARE_FUNC1(lgamma)
CL_DECLARE_FUNC1(log)
CL_DECLARE_FUNC1(log2)
CL_DECLARE_FUNC1(log10)
CL_DECLARE_FUNC1(log1p)
CL_DECLARE_FUNC1(logb)
CL_DECLARE_FUNC3(mad)
CL_DECLARE_FUNC2(maxmag)
CL_DECLARE_FUNC2(minmag)
CL_DECLARE_FUNC2(nextafter)
CL_DECLARE_FUNC1(pow)
CL_DECLARE_FUNC2(remainder)
CL_DECLARE_FUNC1(rint)
CL_DECLARE_FUNC1(round)
CL_DECLARE_FUNC1(rsqrt)
CL_DECLARE_FUNC1(sin)
CL_DECLARE_FUNC1(sinh)
CL_DECLARE_FUNC1(sinpi)
CL_DECLARE_FUNC1(sqrt)
CL_DECLARE_FUNC1(tan)
CL_DECLARE_FUNC1(tanh)
CL_DECLARE_FUNC1(tanpi)
CL_DECLARE_FUNC1(tgamma)
CL_DECLARE_FUNC1(trunc)

__attribute__ ((noinline)) void barrier (cl_mem_fence_flags flags);
