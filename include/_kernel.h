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
#define _cl_static_assert(_t, _x) typedef int ai##_t[(_x) ? 1 : -1]
_cl_static_assert(char  , sizeof(char  ) == 1);
_cl_static_assert(uchar , sizeof(uchar ) == 1);
_cl_static_assert(short , sizeof(short ) == 2);
_cl_static_assert(ushort, sizeof(ushort) == 2);
_cl_static_assert(int   , sizeof(int   ) == 4);
_cl_static_assert(uint  , sizeof(uint  ) == 4);
_cl_static_assert(long  , sizeof(long  ) == 8);
_cl_static_assert(ulong , sizeof(ulong ) == 8);
_cl_static_assert(float , sizeof(float ) == 4);
_cl_static_assert(double, sizeof(double) == 8);

typedef char char2  __attribute__((ext_vector_type(2)));
typedef char char3  __attribute__((ext_vector_type(3)));
typedef char char4  __attribute__((ext_vector_type(4)));
typedef char char8  __attribute__((ext_vector_type(8)));
typedef char char16 __attribute__((ext_vector_type(16)));

typedef uchar uchar2  __attribute__((ext_vector_type(2)));
typedef uchar uchar3  __attribute__((ext_vector_type(3)));
typedef uchar uchar4  __attribute__((ext_vector_type(4)));
typedef uchar uchar8  __attribute__((ext_vector_type(8)));
typedef uchar uchar16 __attribute__((ext_vector_type(16)));

typedef short short2  __attribute__((ext_vector_type(2)));
typedef short short3  __attribute__((ext_vector_type(3)));
typedef short short4  __attribute__((ext_vector_type(4)));
typedef short short8  __attribute__((ext_vector_type(8)));
typedef short short16 __attribute__((ext_vector_type(16)));

typedef ushort ushort2  __attribute__((ext_vector_type(2)));
typedef ushort ushort3  __attribute__((ext_vector_type(3)));
typedef ushort ushort4  __attribute__((ext_vector_type(4)));
typedef ushort ushort8  __attribute__((ext_vector_type(8)));
typedef ushort ushort16 __attribute__((ext_vector_type(16)));

typedef int int2  __attribute__((ext_vector_type(2)));
typedef int int3  __attribute__((ext_vector_type(3)));
typedef int int4  __attribute__((ext_vector_type(4)));
typedef int int8  __attribute__((ext_vector_type(8)));
typedef int int16 __attribute__((ext_vector_type(16)));

typedef uint uint2  __attribute__((ext_vector_type(2)));
typedef uint uint3  __attribute__((ext_vector_type(3)));
typedef uint uint4  __attribute__((ext_vector_type(4)));
typedef uint uint8  __attribute__((ext_vector_type(8)));
typedef uint uint16 __attribute__((ext_vector_type(16)));

typedef long long2  __attribute__((ext_vector_type(2)));
typedef long long3  __attribute__((ext_vector_type(3)));
typedef long long4  __attribute__((ext_vector_type(4)));
typedef long long8  __attribute__((ext_vector_type(8)));
typedef long long16 __attribute__((ext_vector_type(16)));

typedef ulong ulong2  __attribute__((ext_vector_type(2)));
typedef ulong ulong3  __attribute__((ext_vector_type(3)));
typedef ulong ulong4  __attribute__((ext_vector_type(4)));
typedef ulong ulong8  __attribute__((ext_vector_type(8)));
typedef ulong ulong16 __attribute__((ext_vector_type(16)));

typedef float float2  __attribute__((ext_vector_type(2)));
typedef float float3  __attribute__((ext_vector_type(3)));
typedef float float4  __attribute__((ext_vector_type(4)));
typedef float float8  __attribute__((ext_vector_type(8)));
typedef float float16 __attribute__((ext_vector_type(16)));

typedef double double2  __attribute__((ext_vector_type(2)));
typedef double double3  __attribute__((ext_vector_type(3)));
typedef double double4  __attribute__((ext_vector_type(4)));
typedef double double8  __attribute__((ext_vector_type(8)));
typedef double double16 __attribute__((ext_vector_type(16)));

typedef enum {
  CLK_LOCAL_MEM_FENCE = 0x1,
  CLK_GLOBAL_MEM_FENCE = 0x2
} cl_mem_fence_flags;

uint get_global_size(uint);
uint get_global_id(uint);
uint get_local_id(uint);
uint get_num_groups(uint);
uint get_group_id(uint);

int mad24(int x, int y, int z);

/* Constants */

#define FLT_DIG        6
#define FLT_MANT_DIG   24
#define FLT_MAX_10_EXP +38
#define FLT_MAX_EXP    +128
#define FLT_MIN_10_EXP -37
#define FLT_MIN_EXP    -125
#define FLT_RADIX      2
#define FLT_MAX        0x1.fffffep127f
#define FLT_MIN        0x1.0p-126f
#define FLT_EPSILON    0x1.0p-23f

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

#define _CL_DECLARE_AS_TYPE(SRC, DST)                    \
  DST __attribute__ ((overloadable)) as_##DST(SRC a);

/* 1 byte */
#define _CL_DECLARE_AS_TYPE_1(DST)                  \
  _CL_DECLARE_AS_TYPE(DST, char)                    \
  _CL_DECLARE_AS_TYPE(DST, uchar)
_CL_DECLARE_AS_TYPE_1(char)
_CL_DECLARE_AS_TYPE_1(uchar)

/* 2 bytes */
#define _CL_DECLARE_AS_TYPE_2(DST)                  \
  _CL_DECLARE_AS_TYPE(DST, char2)                   \
  _CL_DECLARE_AS_TYPE(DST, uchar2)                  \
  _CL_DECLARE_AS_TYPE(DST, short)                   \
  _CL_DECLARE_AS_TYPE(DST, ushort)
_CL_DECLARE_AS_TYPE_2(char2)
_CL_DECLARE_AS_TYPE_2(uchar2)
_CL_DECLARE_AS_TYPE_2(short)
_CL_DECLARE_AS_TYPE_2(ushort)

/* 4 bytes */
#define _CL_DECLARE_AS_TYPE_4(DST)                  \
  _CL_DECLARE_AS_TYPE(DST, char4)                   \
  _CL_DECLARE_AS_TYPE(DST, uchar4)                  \
  _CL_DECLARE_AS_TYPE(DST, short2)                  \
  _CL_DECLARE_AS_TYPE(DST, ushort2)                 \
  _CL_DECLARE_AS_TYPE(DST, int)                     \
  _CL_DECLARE_AS_TYPE(DST, uint)                    \
  _CL_DECLARE_AS_TYPE(DST, float)
_CL_DECLARE_AS_TYPE_4(char4)
_CL_DECLARE_AS_TYPE_4(uchar4)
_CL_DECLARE_AS_TYPE_4(short2)
_CL_DECLARE_AS_TYPE_4(ushort2)
_CL_DECLARE_AS_TYPE_4(int)
_CL_DECLARE_AS_TYPE_4(uint)
_CL_DECLARE_AS_TYPE_4(float)

/* 8 bytes */
#define _CL_DECLARE_AS_TYPE_8(DST)                  \
  _CL_DECLARE_AS_TYPE(DST, char8)                   \
  _CL_DECLARE_AS_TYPE(DST, uchar8)                  \
  _CL_DECLARE_AS_TYPE(DST, short4)                  \
  _CL_DECLARE_AS_TYPE(DST, ushort4)                 \
  _CL_DECLARE_AS_TYPE(DST, int2)                    \
  _CL_DECLARE_AS_TYPE(DST, uint2)                   \
  _CL_DECLARE_AS_TYPE(DST, long)                    \
  _CL_DECLARE_AS_TYPE(DST, ulong)                   \
  _CL_DECLARE_AS_TYPE(DST, float2)                  \
  _CL_DECLARE_AS_TYPE(DST, double)
_CL_DECLARE_AS_TYPE_8(char8)
_CL_DECLARE_AS_TYPE_8(uchar8)
_CL_DECLARE_AS_TYPE_8(short4)
_CL_DECLARE_AS_TYPE_8(ushort4)
_CL_DECLARE_AS_TYPE_8(int2)
_CL_DECLARE_AS_TYPE_8(uint2)
_CL_DECLARE_AS_TYPE_8(long)
_CL_DECLARE_AS_TYPE_8(ulong)
_CL_DECLARE_AS_TYPE_8(float2)
_CL_DECLARE_AS_TYPE_8(double)

/* 16 bytes */
#define _CL_DECLARE_AS_TYPE_16(DST)                 \
  _CL_DECLARE_AS_TYPE(DST, char16)                  \
  _CL_DECLARE_AS_TYPE(DST, uchar16)                 \
  _CL_DECLARE_AS_TYPE(DST, short8)                  \
  _CL_DECLARE_AS_TYPE(DST, ushort8)                 \
  _CL_DECLARE_AS_TYPE(DST, int4)                    \
  _CL_DECLARE_AS_TYPE(DST, uint4)                   \
  _CL_DECLARE_AS_TYPE(DST, long2)                   \
  _CL_DECLARE_AS_TYPE(DST, ulong2)                  \
  _CL_DECLARE_AS_TYPE(DST, float4)                  \
  _CL_DECLARE_AS_TYPE(DST, double2)
_CL_DECLARE_AS_TYPE_16(char16)
_CL_DECLARE_AS_TYPE_16(uchar16)
_CL_DECLARE_AS_TYPE_16(short8)
_CL_DECLARE_AS_TYPE_16(ushort8)
_CL_DECLARE_AS_TYPE_16(int4)
_CL_DECLARE_AS_TYPE_16(uint4)
_CL_DECLARE_AS_TYPE_16(long2)
_CL_DECLARE_AS_TYPE_16(ulong2)
_CL_DECLARE_AS_TYPE_16(float4)
_CL_DECLARE_AS_TYPE_16(double2)

/* 32 bytes */
#define _CL_DECLARE_AS_TYPE_32(DST)                 \
  _CL_DECLARE_AS_TYPE(DST, short16)                 \
  _CL_DECLARE_AS_TYPE(DST, ushort16)                \
  _CL_DECLARE_AS_TYPE(DST, int8)                    \
  _CL_DECLARE_AS_TYPE(DST, uint8)                   \
  _CL_DECLARE_AS_TYPE(DST, long4)                   \
  _CL_DECLARE_AS_TYPE(DST, ulong4)                  \
  _CL_DECLARE_AS_TYPE(DST, float8)                  \
  _CL_DECLARE_AS_TYPE(DST, double4)
_CL_DECLARE_AS_TYPE_32(short16)
_CL_DECLARE_AS_TYPE_32(ushort16)
_CL_DECLARE_AS_TYPE_32(int8)
_CL_DECLARE_AS_TYPE_32(uint8)
_CL_DECLARE_AS_TYPE_32(long4)
_CL_DECLARE_AS_TYPE_32(ulong4)
_CL_DECLARE_AS_TYPE_32(float8)
_CL_DECLARE_AS_TYPE_32(double4)

/* 64 bytes */
#define _CL_DECLARE_AS_TYPE_64(DST)                 \
  _CL_DECLARE_AS_TYPE(DST, int16)                   \
  _CL_DECLARE_AS_TYPE(DST, uint16)                  \
  _CL_DECLARE_AS_TYPE(DST, long8)                   \
  _CL_DECLARE_AS_TYPE(DST, ulong8)                  \
  _CL_DECLARE_AS_TYPE(DST, float16)                 \
  _CL_DECLARE_AS_TYPE(DST, double8)
_CL_DECLARE_AS_TYPE_64(int16)
_CL_DECLARE_AS_TYPE_64(uint16)
_CL_DECLARE_AS_TYPE_64(long8)
_CL_DECLARE_AS_TYPE_64(ulong8)
_CL_DECLARE_AS_TYPE_64(float16)
_CL_DECLARE_AS_TYPE_64(double8)

/* 128 bytes */
#define _CL_DECLARE_AS_TYPE_128(DST)                \
  _CL_DECLARE_AS_TYPE(DST, long16)                  \
  _CL_DECLARE_AS_TYPE(DST, ulong16)                 \
  _CL_DECLARE_AS_TYPE(DST, double16)
_CL_DECLARE_AS_TYPE_128(long16)
_CL_DECLARE_AS_TYPE_128(ulong16)
_CL_DECLARE_AS_TYPE_128(double16)

/* Trigonometric and other functions */

#define _CL_DECLARE_FUNC_1(NAME)                                \
  float    __attribute__ ((overloadable)) NAME(float   );       \
  float2   __attribute__ ((overloadable)) NAME(float2  );       \
  float3   __attribute__ ((overloadable)) NAME(float3  );       \
  float4   __attribute__ ((overloadable)) NAME(float4  );       \
  float8   __attribute__ ((overloadable)) NAME(float8  );       \
  float16  __attribute__ ((overloadable)) NAME(float16 );       \
  double   __attribute__ ((overloadable)) NAME(double  );       \
  double2  __attribute__ ((overloadable)) NAME(double2 );       \
  double3  __attribute__ ((overloadable)) NAME(double3 );       \
  double4  __attribute__ ((overloadable)) NAME(double4 );       \
  double8  __attribute__ ((overloadable)) NAME(double8 );       \
  double16 __attribute__ ((overloadable)) NAME(double16);
#define _CL_DECLARE_FUNC_2(NAME)                                        \
  float    __attribute__ ((overloadable)) NAME(float   , float   );     \
  float2   __attribute__ ((overloadable)) NAME(float2  , float2  );     \
  float3   __attribute__ ((overloadable)) NAME(float3  , float3  );     \
  float4   __attribute__ ((overloadable)) NAME(float4  , float4  );     \
  float8   __attribute__ ((overloadable)) NAME(float8  , float8  );     \
  float16  __attribute__ ((overloadable)) NAME(float16 , float16 );     \
  double   __attribute__ ((overloadable)) NAME(double  , double  );     \
  double2  __attribute__ ((overloadable)) NAME(double2 , double2 );     \
  double3  __attribute__ ((overloadable)) NAME(double3 , double3 );     \
  double4  __attribute__ ((overloadable)) NAME(double4 , double4 );     \
  double8  __attribute__ ((overloadable)) NAME(double8 , double8 );     \
  double16 __attribute__ ((overloadable)) NAME(double16, double16);
#define _CL_DECLARE_FUNC_3(NAME)                                        \
  float    __attribute__ ((overloadable)) NAME(float   , float   , float   ); \
  float2   __attribute__ ((overloadable)) NAME(float2  , float2  , float2  ); \
  float3   __attribute__ ((overloadable)) NAME(float3  , float3  , float3  ); \
  float4   __attribute__ ((overloadable)) NAME(float4  , float4  , float4  ); \
  float8   __attribute__ ((overloadable)) NAME(float8  , float8  , float8  ); \
  float16  __attribute__ ((overloadable)) NAME(float16 , float16 , float16 ); \
  double   __attribute__ ((overloadable)) NAME(double  , double  , double  ); \
  double2  __attribute__ ((overloadable)) NAME(double2 , double2 , double2 ); \
  double3  __attribute__ ((overloadable)) NAME(double3 , double3 , double3 ); \
  double4  __attribute__ ((overloadable)) NAME(double4 , double4 , double4 ); \
  double8  __attribute__ ((overloadable)) NAME(double8 , double8 , double8 ); \
  double16 __attribute__ ((overloadable)) NAME(double16, double16, double16);
#define _CL_DECLARE_SCALAR_FUNC_2(NAME)                                 \
  float  __attribute__ ((overloadable)) NAME(float   , float   );       \
  float  __attribute__ ((overloadable)) NAME(float2  , float2  );       \
  float  __attribute__ ((overloadable)) NAME(float3  , float3  );       \
  float  __attribute__ ((overloadable)) NAME(float4  , float4  );       \
  float  __attribute__ ((overloadable)) NAME(float8  , float8  );       \
  float  __attribute__ ((overloadable)) NAME(float16 , float16 );       \
  double __attribute__ ((overloadable)) NAME(double  , double  );       \
  double __attribute__ ((overloadable)) NAME(double2 , double2 );       \
  double __attribute__ ((overloadable)) NAME(double3 , double3 );       \
  double __attribute__ ((overloadable)) NAME(double4 , double4 );       \
  double __attribute__ ((overloadable)) NAME(double8 , double8 );       \
  double __attribute__ ((overloadable)) NAME(double16, double16);

/* Move built-in declarations out of the way. (There should be a
   better way of doing so.) These five functions are built-in math
   functions for all Clang languages; see Clang's "Builtin.def".
   */
#define cos _cl_cos
#define fma _cl_fma
#define pow _cl_pow
#define sin _cl_sin
#define sqrt _cl_sqrt

_CL_DECLARE_FUNC_1(acos)
_CL_DECLARE_FUNC_1(acosh)
_CL_DECLARE_FUNC_1(acospi)
_CL_DECLARE_FUNC_1(asin)
_CL_DECLARE_FUNC_1(asinh)
_CL_DECLARE_FUNC_1(asinpi)
_CL_DECLARE_FUNC_1(atan)
_CL_DECLARE_FUNC_2(atan2)
_CL_DECLARE_FUNC_2(atan2pi)
_CL_DECLARE_FUNC_1(atanh)
_CL_DECLARE_FUNC_1(atanpi)
_CL_DECLARE_FUNC_1(cbrt)
_CL_DECLARE_FUNC_1(ceil)
_CL_DECLARE_FUNC_2(copysign)
_CL_DECLARE_FUNC_1(cos)
_CL_DECLARE_FUNC_1(cosh)
_CL_DECLARE_FUNC_1(cospi)
_CL_DECLARE_SCALAR_FUNC_2(dot)
_CL_DECLARE_FUNC_1(erfc)
_CL_DECLARE_FUNC_1(erf)
_CL_DECLARE_FUNC_1(exp)
_CL_DECLARE_FUNC_1(exp2)
_CL_DECLARE_FUNC_1(exp10)
_CL_DECLARE_FUNC_1(expm1)
_CL_DECLARE_FUNC_1(fabs)
_CL_DECLARE_FUNC_2(fdim)
_CL_DECLARE_FUNC_1(floor)
_CL_DECLARE_FUNC_3(fma)
_CL_DECLARE_FUNC_2(fmax)
_CL_DECLARE_FUNC_2(fmin)
_CL_DECLARE_FUNC_2(fmod)
_CL_DECLARE_FUNC_2(hypot)
_CL_DECLARE_FUNC_1(lgamma)
_CL_DECLARE_FUNC_1(log)
_CL_DECLARE_FUNC_1(log2)
_CL_DECLARE_FUNC_1(log10)
_CL_DECLARE_FUNC_1(log1p)
_CL_DECLARE_FUNC_1(logb)
_CL_DECLARE_FUNC_3(mad)
_CL_DECLARE_FUNC_2(maxmag)
_CL_DECLARE_FUNC_2(minmag)
_CL_DECLARE_FUNC_2(nextafter)
_CL_DECLARE_FUNC_1(pow)
_CL_DECLARE_FUNC_2(remainder)
_CL_DECLARE_FUNC_1(rint)
_CL_DECLARE_FUNC_1(round)
_CL_DECLARE_FUNC_1(rsqrt)
_CL_DECLARE_FUNC_1(sin)
_CL_DECLARE_FUNC_1(sinh)
_CL_DECLARE_FUNC_1(sinpi)
_CL_DECLARE_FUNC_1(sqrt)
_CL_DECLARE_FUNC_1(tan)
_CL_DECLARE_FUNC_1(tanh)
_CL_DECLARE_FUNC_1(tanpi)
_CL_DECLARE_FUNC_1(tgamma)
_CL_DECLARE_FUNC_1(trunc)

__attribute__ ((noinline)) void barrier (cl_mem_fence_flags flags);
