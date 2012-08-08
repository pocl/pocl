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

   FIX: this is not really true. TCE target is 32-bit scalars
   only. Seems the pragma does not add the macro, so we have the target
   define the macro and the pragma is conditionally enabled.
*/
#ifdef cl_khr_fp16
#  pragma OPENCL EXTENSION cl_khr_fp16: enable
#endif
#ifdef cl_khr_fp64
#  pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

/* Define some feature macros to help write generic code */
#ifdef cles_khr_int64
#  define __IF_INT64(x) x
#else
#  define __IF_INT64(x)
#endif
#ifdef cl_khr_fp16
#  define __IF_FP16(x) x
#else
#  define __IF_FP16(x)
#endif
#ifdef cl_khr_fp64
#  define __IF_FP64(x) x
#else
#  define __IF_FP64(x)
#endif

#if defined(cl_khr_fp64) && !defined(cles_khr_int64)
#  error "cl_khr_fp64 requires cles_khr_int64"
#endif



/* A static assert statement to catch inconsistencies at build time */
#define _cl_static_assert(_t, _x) typedef int ai##_t[(_x) ? 1 : -1]

/* Let's try to make things easier for post-preprocessing pass. */
#define kernel __kernel
#define local __local

/* #define __global __attribute__ ((address_space(3))) */
/* #define __local __attribute__ ((address_space(4))) */
/* #define __constant __attribute__ ((address_space(5))) */

/* #define global __attribute__ ((address_space(3))) */
/* #define local __attribute__ ((address_space(4))) */
/* #define constant __attribute__ ((address_space(5))) */

typedef enum {
  CLK_LOCAL_MEM_FENCE = 0x1,
  CLK_GLOBAL_MEM_FENCE = 0x2
} cl_mem_fence_flags;


/* Data types */

/* Disable undefined datatypes */
#ifndef cles_khr_int64
typedef struct error_undefined_type_long error_undefined_type_long;
#  define long error_undefined_type_long
typedef struct error_undefined_type_ulong error_undefined_type_ulong;
#  define ulong error_undefined_type_ulong
#endif
#ifndef cl_khr_fp16
typedef struct error_undefined_type_half error_undefined_type_half;
#  define half error_undefined_type_half
#endif
#ifndef cl_khr_fp64
typedef struct error_undefined_type_double error_undefined_type_double;
#  define double error_undefined_type_double
#endif

// We align the 3-vectors, so that their sizeof is correct. Is there a
// better way? Should we also align the other vectors?

typedef char char2  __attribute__((__ext_vector_type__(2)));
typedef char char3  __attribute__((__ext_vector_type__(3), __aligned__(4)));
typedef char char4  __attribute__((__ext_vector_type__(4)));
typedef char char8  __attribute__((__ext_vector_type__(8)));
typedef char char16 __attribute__((__ext_vector_type__(16)));

typedef uchar uchar2  __attribute__((__ext_vector_type__(2)));
typedef uchar uchar3  __attribute__((__ext_vector_type__(3), __aligned__(4)));
typedef uchar uchar4  __attribute__((__ext_vector_type__(4)));
typedef uchar uchar8  __attribute__((__ext_vector_type__(8)));
typedef uchar uchar16 __attribute__((__ext_vector_type__(16)));

typedef short short2  __attribute__((__ext_vector_type__(2)));
typedef short short3  __attribute__((__ext_vector_type__(3), __aligned__(8)));
typedef short short4  __attribute__((__ext_vector_type__(4)));
typedef short short8  __attribute__((__ext_vector_type__(8)));
typedef short short16 __attribute__((__ext_vector_type__(16)));

typedef ushort ushort2  __attribute__((__ext_vector_type__(2)));
typedef ushort ushort3  __attribute__((__ext_vector_type__(3), __aligned__(8)));
typedef ushort ushort4  __attribute__((__ext_vector_type__(4)));
typedef ushort ushort8  __attribute__((__ext_vector_type__(8)));
typedef ushort ushort16 __attribute__((__ext_vector_type__(16)));

typedef int int2  __attribute__((__ext_vector_type__(2)));
typedef int int3  __attribute__((__ext_vector_type__(3), __aligned__(16)));
typedef int int4  __attribute__((__ext_vector_type__(4)));
typedef int int8  __attribute__((__ext_vector_type__(8)));
typedef int int16 __attribute__((__ext_vector_type__(16)));

typedef uint uint2  __attribute__((__ext_vector_type__(2)));
typedef uint uint3  __attribute__((__ext_vector_type__(3), __aligned__(16)));
typedef uint uint4  __attribute__((__ext_vector_type__(4)));
typedef uint uint8  __attribute__((__ext_vector_type__(8)));
typedef uint uint16 __attribute__((__ext_vector_type__(16)));

#ifdef cles_khr_int64
typedef long long2  __attribute__((__ext_vector_type__(2)));
typedef long long3  __attribute__((__ext_vector_type__(3), __aligned__(32)));
typedef long long4  __attribute__((__ext_vector_type__(4)));
typedef long long8  __attribute__((__ext_vector_type__(8)));
typedef long long16 __attribute__((__ext_vector_type__(16)));

typedef ulong ulong2  __attribute__((__ext_vector_type__(2)));
typedef ulong ulong3  __attribute__((__ext_vector_type__(3), __aligned__(32)));
typedef ulong ulong4  __attribute__((__ext_vector_type__(4)));
typedef ulong ulong8  __attribute__((__ext_vector_type__(8)));
typedef ulong ulong16 __attribute__((__ext_vector_type__(16)));
#endif

typedef float float2  __attribute__((__ext_vector_type__(2)));
typedef float float3  __attribute__((__ext_vector_type__(3), __aligned__(16)));
typedef float float4  __attribute__((__ext_vector_type__(4)));
typedef float float8  __attribute__((__ext_vector_type__(8)));
typedef float float16 __attribute__((__ext_vector_type__(16)));

#ifdef cl_khr_fp64
typedef double double2  __attribute__((__ext_vector_type__(2)));
typedef double double3  __attribute__((__ext_vector_type__(3), __aligned__(32)));
typedef double double4  __attribute__((__ext_vector_type__(4)));
typedef double double8  __attribute__((__ext_vector_type__(8)));
typedef double double16 __attribute__((__ext_vector_type__(16)));
#endif

/* Ensure the data types have the right sizes */
_cl_static_assert(char  , sizeof(char  ) == 1);
_cl_static_assert(char2 , sizeof(char2 ) == 2 *sizeof(char));
_cl_static_assert(char3 , sizeof(char3 ) == 4 *sizeof(char));
_cl_static_assert(char4 , sizeof(char4 ) == 4 *sizeof(char));
_cl_static_assert(char8 , sizeof(char8 ) == 8 *sizeof(char));
_cl_static_assert(char16, sizeof(char16) == 16*sizeof(char));

_cl_static_assert(uchar , sizeof(uchar ) == 1);
_cl_static_assert(uchar2 , sizeof(uchar2 ) == 2 *sizeof(uchar));
_cl_static_assert(uchar3 , sizeof(uchar3 ) == 4 *sizeof(uchar));
_cl_static_assert(uchar4 , sizeof(uchar4 ) == 4 *sizeof(uchar));
_cl_static_assert(uchar8 , sizeof(uchar8 ) == 8 *sizeof(uchar));
_cl_static_assert(uchar16, sizeof(uchar16) == 16*sizeof(uchar));

_cl_static_assert(short , sizeof(short ) == 2);
_cl_static_assert(short2 , sizeof(short2 ) == 2 *sizeof(short));
_cl_static_assert(short3 , sizeof(short3 ) == 4 *sizeof(short));
_cl_static_assert(short4 , sizeof(short4 ) == 4 *sizeof(short));
_cl_static_assert(short8 , sizeof(short8 ) == 8 *sizeof(short));
_cl_static_assert(short16, sizeof(short16) == 16*sizeof(short));

_cl_static_assert(ushort, sizeof(ushort) == 2);
_cl_static_assert(ushort2 , sizeof(ushort2 ) == 2 *sizeof(ushort));
_cl_static_assert(ushort3 , sizeof(ushort3 ) == 4 *sizeof(ushort));
_cl_static_assert(ushort4 , sizeof(ushort4 ) == 4 *sizeof(ushort));
_cl_static_assert(ushort8 , sizeof(ushort8 ) == 8 *sizeof(ushort));
_cl_static_assert(ushort16, sizeof(ushort16) == 16*sizeof(ushort));

_cl_static_assert(int   , sizeof(int   ) == 4);
_cl_static_assert(int2 , sizeof(int2 ) == 2 *sizeof(int));
_cl_static_assert(int3 , sizeof(int3 ) == 4 *sizeof(int));
_cl_static_assert(int4 , sizeof(int4 ) == 4 *sizeof(int));
_cl_static_assert(int8 , sizeof(int8 ) == 8 *sizeof(int));
_cl_static_assert(int16, sizeof(int16) == 16*sizeof(int));

_cl_static_assert(uint  , sizeof(uint  ) == 4);
_cl_static_assert(uint2 , sizeof(uint2 ) == 2 *sizeof(uint));
_cl_static_assert(uint3 , sizeof(uint3 ) == 4 *sizeof(uint));
_cl_static_assert(uint4 , sizeof(uint4 ) == 4 *sizeof(uint));
_cl_static_assert(uint8 , sizeof(uint8 ) == 8 *sizeof(uint));
_cl_static_assert(uint16, sizeof(uint16) == 16*sizeof(uint));

#ifdef cles_khr_int64 
_cl_static_assert(long  , sizeof(long  ) == 8);
_cl_static_assert(long2 , sizeof(long2 ) == 2 *sizeof(long));
_cl_static_assert(long3 , sizeof(long3 ) == 4 *sizeof(long));
_cl_static_assert(long4 , sizeof(long4 ) == 4 *sizeof(long));
_cl_static_assert(long8 , sizeof(long8 ) == 8 *sizeof(long));
_cl_static_assert(long16, sizeof(long16) == 16*sizeof(long));

_cl_static_assert(ulong  , sizeof(ulong  ) == 8);
_cl_static_assert(ulong2 , sizeof(ulong2 ) == 2 *sizeof(ulong));
_cl_static_assert(ulong3 , sizeof(ulong3 ) == 4 *sizeof(ulong));
_cl_static_assert(ulong4 , sizeof(ulong4 ) == 4 *sizeof(ulong));
_cl_static_assert(ulong8 , sizeof(ulong8 ) == 8 *sizeof(ulong));
_cl_static_assert(ulong16, sizeof(ulong16) == 16*sizeof(ulong));
#endif

#ifdef cl_khr_fp16
_cl_static_assert(half, sizeof(half) == 2);
/* There are no vectors of type half */
#endif

_cl_static_assert(float , sizeof(float ) == 4);
_cl_static_assert(float2 , sizeof(float2 ) == 2 *sizeof(float));
_cl_static_assert(float3 , sizeof(float3 ) == 4 *sizeof(float));
_cl_static_assert(float4 , sizeof(float4 ) == 4 *sizeof(float));
_cl_static_assert(float8 , sizeof(float8 ) == 8 *sizeof(float));
_cl_static_assert(float16, sizeof(float16) == 16*sizeof(float));

#ifdef cl_khr_fp64
_cl_static_assert(double, sizeof(double) == 8);
_cl_static_assert(double2 , sizeof(double2 ) == 2 *sizeof(double));
_cl_static_assert(double3 , sizeof(double3 ) == 4 *sizeof(double));
_cl_static_assert(double4 , sizeof(double4 ) == 4 *sizeof(double));
_cl_static_assert(double8 , sizeof(double8 ) == 8 *sizeof(double));
_cl_static_assert(double16, sizeof(double16) == 16*sizeof(double));
#endif

_cl_static_assert(size_t, sizeof(size_t) == sizeof(void*));
_cl_static_assert(ptrdiff_t, sizeof(ptrdiff_t) == sizeof(void*));
_cl_static_assert(intptr_t, sizeof(intptr_t) == sizeof(void*));
_cl_static_assert(uintptr_t, sizeof(uintptr_t) == sizeof(void*));


#define _cl_overloadable __attribute__ ((__overloadable__))


/* Conversion functions */

#define _CL_DECLARE_AS_TYPE(SRC, DST)           \
  DST _cl_overloadable as_##DST(SRC a);

/* 1 byte */
#define _CL_DECLARE_AS_TYPE_1(SRC)              \
  _CL_DECLARE_AS_TYPE(SRC, char)                \
  _CL_DECLARE_AS_TYPE(SRC, uchar)
_CL_DECLARE_AS_TYPE_1(char)
_CL_DECLARE_AS_TYPE_1(uchar)

/* 2 bytes */
#define _CL_DECLARE_AS_TYPE_2(SRC)              \
  _CL_DECLARE_AS_TYPE(SRC, char2)               \
  _CL_DECLARE_AS_TYPE(SRC, uchar2)              \
  _CL_DECLARE_AS_TYPE(SRC, short)               \
  _CL_DECLARE_AS_TYPE(SRC, ushort)
_CL_DECLARE_AS_TYPE_2(char2)
_CL_DECLARE_AS_TYPE_2(uchar2)
_CL_DECLARE_AS_TYPE_2(short)
_CL_DECLARE_AS_TYPE_2(ushort)

/* 4 bytes */
#define _CL_DECLARE_AS_TYPE_4(SRC)              \
  _CL_DECLARE_AS_TYPE(SRC, char4)               \
  _CL_DECLARE_AS_TYPE(SRC, uchar4)              \
  _CL_DECLARE_AS_TYPE(SRC, short2)              \
  _CL_DECLARE_AS_TYPE(SRC, ushort2)             \
  _CL_DECLARE_AS_TYPE(SRC, int)                 \
  _CL_DECLARE_AS_TYPE(SRC, uint)                \
  _CL_DECLARE_AS_TYPE(SRC, float)
_CL_DECLARE_AS_TYPE_4(char4)
_CL_DECLARE_AS_TYPE_4(uchar4)
_CL_DECLARE_AS_TYPE_4(short2)
_CL_DECLARE_AS_TYPE_4(ushort2)
_CL_DECLARE_AS_TYPE_4(int)
_CL_DECLARE_AS_TYPE_4(uint)
_CL_DECLARE_AS_TYPE_4(float)

/* 8 bytes */
#define _CL_DECLARE_AS_TYPE_8(SRC)              \
  _CL_DECLARE_AS_TYPE(SRC, char8)               \
  _CL_DECLARE_AS_TYPE(SRC, uchar8)              \
  _CL_DECLARE_AS_TYPE(SRC, short4)              \
  _CL_DECLARE_AS_TYPE(SRC, ushort4)             \
  _CL_DECLARE_AS_TYPE(SRC, int2)                \
  _CL_DECLARE_AS_TYPE(SRC, uint2)               \
  __IF_INT64(_CL_DECLARE_AS_TYPE(SRC, long))    \
  __IF_INT64(_CL_DECLARE_AS_TYPE(SRC, ulong))   \
  _CL_DECLARE_AS_TYPE(SRC, float2)              \
  __IF_FP64(_CL_DECLARE_AS_TYPE(SRC, double))
_CL_DECLARE_AS_TYPE_8(char8)
_CL_DECLARE_AS_TYPE_8(uchar8)
_CL_DECLARE_AS_TYPE_8(short4)
_CL_DECLARE_AS_TYPE_8(ushort4)
_CL_DECLARE_AS_TYPE_8(int2)
_CL_DECLARE_AS_TYPE_8(uint2)
__IF_INT64(_CL_DECLARE_AS_TYPE_8(long))
__IF_INT64(_CL_DECLARE_AS_TYPE_8(ulong))
_CL_DECLARE_AS_TYPE_8(float2)
__IF_FP64(_CL_DECLARE_AS_TYPE_8(double))

/* 16 bytes */
#define _CL_DECLARE_AS_TYPE_16(SRC)             \
  _CL_DECLARE_AS_TYPE(SRC, char16)              \
  _CL_DECLARE_AS_TYPE(SRC, uchar16)             \
  _CL_DECLARE_AS_TYPE(SRC, short8)              \
  _CL_DECLARE_AS_TYPE(SRC, ushort8)             \
  _CL_DECLARE_AS_TYPE(SRC, int4)                \
  _CL_DECLARE_AS_TYPE(SRC, uint4)               \
  __IF_INT64(_CL_DECLARE_AS_TYPE(SRC, long2))   \
  __IF_INT64(_CL_DECLARE_AS_TYPE(SRC, ulong2))  \
  _CL_DECLARE_AS_TYPE(SRC, float4)              \
  __IF_FP64(_CL_DECLARE_AS_TYPE(SRC, double2))
_CL_DECLARE_AS_TYPE_16(char16)
_CL_DECLARE_AS_TYPE_16(uchar16)
_CL_DECLARE_AS_TYPE_16(short8)
_CL_DECLARE_AS_TYPE_16(ushort8)
_CL_DECLARE_AS_TYPE_16(int4)
_CL_DECLARE_AS_TYPE_16(uint4)
__IF_INT64(_CL_DECLARE_AS_TYPE_16(long2))
__IF_INT64(_CL_DECLARE_AS_TYPE_16(ulong2))
_CL_DECLARE_AS_TYPE_16(float4)
__IF_FP64(_CL_DECLARE_AS_TYPE_16(double2))

/* 32 bytes */
#define _CL_DECLARE_AS_TYPE_32(SRC)             \
  _CL_DECLARE_AS_TYPE(SRC, short16)             \
  _CL_DECLARE_AS_TYPE(SRC, ushort16)            \
  _CL_DECLARE_AS_TYPE(SRC, int8)                \
  _CL_DECLARE_AS_TYPE(SRC, uint8)               \
  __IF_INT64(_CL_DECLARE_AS_TYPE(SRC, long4))   \
  __IF_INT64(_CL_DECLARE_AS_TYPE(SRC, ulong4))  \
  _CL_DECLARE_AS_TYPE(SRC, float8)              \
  __IF_FP64(_CL_DECLARE_AS_TYPE(SRC, double4))
_CL_DECLARE_AS_TYPE_32(short16)
_CL_DECLARE_AS_TYPE_32(ushort16)
_CL_DECLARE_AS_TYPE_32(int8)
_CL_DECLARE_AS_TYPE_32(uint8)
__IF_INT64(_CL_DECLARE_AS_TYPE_32(long4))
__IF_INT64(_CL_DECLARE_AS_TYPE_32(ulong4))
_CL_DECLARE_AS_TYPE_32(float8)
__IF_FP64(_CL_DECLARE_AS_TYPE_32(double4))

/* 64 bytes */
#define _CL_DECLARE_AS_TYPE_64(SRC)             \
  _CL_DECLARE_AS_TYPE(SRC, int16)               \
  _CL_DECLARE_AS_TYPE(SRC, uint16)              \
  __IF_INT64(_CL_DECLARE_AS_TYPE(SRC, long8))   \
  __IF_INT64(_CL_DECLARE_AS_TYPE(SRC, ulong8))  \
  _CL_DECLARE_AS_TYPE(SRC, float16)             \
  __IF_FP64(_CL_DECLARE_AS_TYPE(SRC, double8))
_CL_DECLARE_AS_TYPE_64(int16)
_CL_DECLARE_AS_TYPE_64(uint16)
__IF_INT64(_CL_DECLARE_AS_TYPE_64(long8))
__IF_INT64(_CL_DECLARE_AS_TYPE_64(ulong8))
_CL_DECLARE_AS_TYPE_64(float16)
__IF_FP64(_CL_DECLARE_AS_TYPE_64(double8))

/* 128 bytes */
#define _CL_DECLARE_AS_TYPE_128(SRC)            \
  __IF_INT64(_CL_DECLARE_AS_TYPE(SRC, long16))  \
  __IF_INT64(_CL_DECLARE_AS_TYPE(SRC, ulong16)) \
  __IF_FP64(_CL_DECLARE_AS_TYPE(SRC, double16))
__IF_INT64(_CL_DECLARE_AS_TYPE_128(long16))
__IF_INT64(_CL_DECLARE_AS_TYPE_128(ulong16))
__IF_FP64(_CL_DECLARE_AS_TYPE_128(double16))

#define _CL_DECLARE_CONVERT_TYPE(SRC, DST, SIZE, INTSUFFIX, FLOATSUFFIX) \
  DST##SIZE _cl_overloadable                                            \
    convert_##DST##SIZE##INTSUFFIX##FLOATSUFFIX(SRC##SIZE a);

/* conversions to int may have a suffix: _sat */
#define _CL_DECLARE_CONVERT_TYPE_DST(SRC, SIZE, FLOATSUFFIX)            \
  _CL_DECLARE_CONVERT_TYPE(SRC, char  , SIZE,     , FLOATSUFFIX)        \
  _CL_DECLARE_CONVERT_TYPE(SRC, char  , SIZE, _sat, FLOATSUFFIX)        \
  _CL_DECLARE_CONVERT_TYPE(SRC, uchar , SIZE,     , FLOATSUFFIX)        \
  _CL_DECLARE_CONVERT_TYPE(SRC, uchar , SIZE, _sat, FLOATSUFFIX)        \
  _CL_DECLARE_CONVERT_TYPE(SRC, short , SIZE,     , FLOATSUFFIX)        \
  _CL_DECLARE_CONVERT_TYPE(SRC, short , SIZE, _sat, FLOATSUFFIX)        \
  _CL_DECLARE_CONVERT_TYPE(SRC, ushort, SIZE,     , FLOATSUFFIX)        \
  _CL_DECLARE_CONVERT_TYPE(SRC, ushort, SIZE, _sat, FLOATSUFFIX)        \
  _CL_DECLARE_CONVERT_TYPE(SRC, int   , SIZE,     , FLOATSUFFIX)        \
  _CL_DECLARE_CONVERT_TYPE(SRC, int   , SIZE, _sat, FLOATSUFFIX)        \
  _CL_DECLARE_CONVERT_TYPE(SRC, uint  , SIZE,     , FLOATSUFFIX)        \
  _CL_DECLARE_CONVERT_TYPE(SRC, uint  , SIZE, _sat, FLOATSUFFIX)        \
  __IF_INT64(                                                           \
  _CL_DECLARE_CONVERT_TYPE(SRC, long  , SIZE,     , FLOATSUFFIX)        \
  _CL_DECLARE_CONVERT_TYPE(SRC, long  , SIZE, _sat, FLOATSUFFIX)        \
  _CL_DECLARE_CONVERT_TYPE(SRC, ulong , SIZE,     , FLOATSUFFIX)        \
  _CL_DECLARE_CONVERT_TYPE(SRC, ulong , SIZE, _sat, FLOATSUFFIX))       \
  _CL_DECLARE_CONVERT_TYPE(SRC, float , SIZE,     , FLOATSUFFIX)        \
  __IF_FP64(                                                            \
  _CL_DECLARE_CONVERT_TYPE(SRC, double, SIZE,     , FLOATSUFFIX))

/* conversions from float may have a suffix: _rte _rtz _rtp _rtn */
#define _CL_DECLARE_CONVERT_TYPE_SRC_DST(SIZE)          \
  _CL_DECLARE_CONVERT_TYPE_DST(char  , SIZE,     )      \
  _CL_DECLARE_CONVERT_TYPE_DST(uchar , SIZE,     )      \
  _CL_DECLARE_CONVERT_TYPE_DST(short , SIZE,     )      \
  _CL_DECLARE_CONVERT_TYPE_DST(ushort, SIZE,     )      \
  _CL_DECLARE_CONVERT_TYPE_DST(int   , SIZE,     )      \
  _CL_DECLARE_CONVERT_TYPE_DST(uint  , SIZE,     )      \
  __IF_INT64(                                           \
  _CL_DECLARE_CONVERT_TYPE_DST(long  , SIZE,     )      \
  _CL_DECLARE_CONVERT_TYPE_DST(ulong , SIZE,     ))     \
  _CL_DECLARE_CONVERT_TYPE_DST(float , SIZE,     )      \
  _CL_DECLARE_CONVERT_TYPE_DST(float , SIZE, _rte)      \
  _CL_DECLARE_CONVERT_TYPE_DST(float , SIZE, _rtz)      \
  _CL_DECLARE_CONVERT_TYPE_DST(float , SIZE, _rtp)      \
  _CL_DECLARE_CONVERT_TYPE_DST(float , SIZE, _rtn)      \
  __IF_FP64(                                            \
  _CL_DECLARE_CONVERT_TYPE_DST(double, SIZE,     )      \
  _CL_DECLARE_CONVERT_TYPE_DST(double, SIZE, _rte)      \
  _CL_DECLARE_CONVERT_TYPE_DST(double, SIZE, _rtz)      \
  _CL_DECLARE_CONVERT_TYPE_DST(double, SIZE, _rtp)      \
  _CL_DECLARE_CONVERT_TYPE_DST(double, SIZE, _rtn))

_CL_DECLARE_CONVERT_TYPE_SRC_DST(  )
_CL_DECLARE_CONVERT_TYPE_SRC_DST( 2)
_CL_DECLARE_CONVERT_TYPE_SRC_DST( 3)
_CL_DECLARE_CONVERT_TYPE_SRC_DST( 4)
_CL_DECLARE_CONVERT_TYPE_SRC_DST( 8)
_CL_DECLARE_CONVERT_TYPE_SRC_DST(16)


/* Work-Item Functions */

uint get_work_dim();
size_t get_global_size(uint);
size_t get_global_id(uint);
size_t get_local_size(uint);
size_t get_local_id(uint);
size_t get_num_groups(uint);
size_t get_group_id(uint);
size_t get_global_offset(uint);

void barrier (cl_mem_fence_flags flags);


/* Math Constants */

#define MAXFLOAT  FLT_MAX
#define HUGE_VALF __builtin_huge_valf()
#define INFINITY  (1.0f / 0.0f)
#define NAN       (0.0f / 0.0f)

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

#ifdef cl_khr_fp64
#define HUGE_VAL __builtin_huge_val()

#define DBL_DIG        15
#define DBL_MANT_DIG   53
#define DBL_MAX_10_EXP +308
#define DBL_MAX_EXP    +1024
#define DBL_MIN_10_EXP -307
#define DBL_MIN_EXP    -1021
#define DBL_MAX        0x1.fffffffffffffp1023
#define DBL_MIN        0x1.0p-1022
#define DBL_EPSILON    0x1.0p-52

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
#endif


/* Math Functions */

/* Naming scheme:
 *    [NAME]_[R]_[A]*
 * where [R] is the return type, and [A] are the argument types:
 *    I: int
 *    J: vector of int
 *    U: vector of uint or ulong
 *    S: scalar (float or double)
 *    F: vector of float
 *    V: vector of float or double
 */

#define _CL_DECLARE_FUNC_V_V(NAME)              \
  float    _cl_overloadable NAME(float   );     \
  float2   _cl_overloadable NAME(float2  );     \
  float3   _cl_overloadable NAME(float3  );     \
  float4   _cl_overloadable NAME(float4  );     \
  float8   _cl_overloadable NAME(float8  );     \
  float16  _cl_overloadable NAME(float16 );     \
  __IF_FP64(                                    \
  double   _cl_overloadable NAME(double  );     \
  double2  _cl_overloadable NAME(double2 );     \
  double3  _cl_overloadable NAME(double3 );     \
  double4  _cl_overloadable NAME(double4 );     \
  double8  _cl_overloadable NAME(double8 );     \
  double16 _cl_overloadable NAME(double16);)
#define _CL_DECLARE_FUNC_V_VV(NAME)                     \
  float    _cl_overloadable NAME(float   , float   );   \
  float2   _cl_overloadable NAME(float2  , float2  );   \
  float3   _cl_overloadable NAME(float3  , float3  );   \
  float4   _cl_overloadable NAME(float4  , float4  );   \
  float8   _cl_overloadable NAME(float8  , float8  );   \
  float16  _cl_overloadable NAME(float16 , float16 );   \
  __IF_FP64(                                            \
  double   _cl_overloadable NAME(double  , double  );   \
  double2  _cl_overloadable NAME(double2 , double2 );   \
  double3  _cl_overloadable NAME(double3 , double3 );   \
  double4  _cl_overloadable NAME(double4 , double4 );   \
  double8  _cl_overloadable NAME(double8 , double8 );   \
  double16 _cl_overloadable NAME(double16, double16);)
#define _CL_DECLARE_FUNC_V_VVV(NAME)                                    \
  float    _cl_overloadable NAME(float   , float   , float   );         \
  float2   _cl_overloadable NAME(float2  , float2  , float2  );         \
  float3   _cl_overloadable NAME(float3  , float3  , float3  );         \
  float4   _cl_overloadable NAME(float4  , float4  , float4  );         \
  float8   _cl_overloadable NAME(float8  , float8  , float8  );         \
  float16  _cl_overloadable NAME(float16 , float16 , float16 );         \
  __IF_FP64(                                                            \
  double   _cl_overloadable NAME(double  , double  , double  );         \
  double2  _cl_overloadable NAME(double2 , double2 , double2 );         \
  double3  _cl_overloadable NAME(double3 , double3 , double3 );         \
  double4  _cl_overloadable NAME(double4 , double4 , double4 );         \
  double8  _cl_overloadable NAME(double8 , double8 , double8 );         \
  double16 _cl_overloadable NAME(double16, double16, double16);)
#define _CL_DECLARE_FUNC_V_VVS(NAME)                            \
  float2   _cl_overloadable NAME(float2  , float2  , float );   \
  float3   _cl_overloadable NAME(float3  , float3  , float );   \
  float4   _cl_overloadable NAME(float4  , float4  , float );   \
  float8   _cl_overloadable NAME(float8  , float8  , float );   \
  float16  _cl_overloadable NAME(float16 , float16 , float );   \
  __IF_FP64(                                                    \
  double2  _cl_overloadable NAME(double2 , double2 , double);   \
  double3  _cl_overloadable NAME(double3 , double3 , double);   \
  double4  _cl_overloadable NAME(double4 , double4 , double);   \
  double8  _cl_overloadable NAME(double8 , double8 , double);   \
  double16 _cl_overloadable NAME(double16, double16, double);)
#define _CL_DECLARE_FUNC_V_VSS(NAME)                            \
  float2   _cl_overloadable NAME(float2  , float , float );     \
  float3   _cl_overloadable NAME(float3  , float , float );     \
  float4   _cl_overloadable NAME(float4  , float , float );     \
  float8   _cl_overloadable NAME(float8  , float , float );     \
  float16  _cl_overloadable NAME(float16 , float , float );     \
  __IF_FP64(                                                    \
  double2  _cl_overloadable NAME(double2 , double, double);     \
  double3  _cl_overloadable NAME(double3 , double, double);     \
  double4  _cl_overloadable NAME(double4 , double, double);     \
  double8  _cl_overloadable NAME(double8 , double, double);     \
  double16 _cl_overloadable NAME(double16, double, double);)
#define _CL_DECLARE_FUNC_V_SSV(NAME)                            \
  float2   _cl_overloadable NAME(float , float , float2  );     \
  float3   _cl_overloadable NAME(float , float , float3  );     \
  float4   _cl_overloadable NAME(float , float , float4  );     \
  float8   _cl_overloadable NAME(float , float , float8  );     \
  float16  _cl_overloadable NAME(float , float , float16 );     \
  __IF_FP64(                                                    \
  double2  _cl_overloadable NAME(double, double, double2 );     \
  double3  _cl_overloadable NAME(double, double, double3 );     \
  double4  _cl_overloadable NAME(double, double, double4 );     \
  double8  _cl_overloadable NAME(double, double, double8 );     \
  double16 _cl_overloadable NAME(double, double, double16);)
#define _CL_DECLARE_FUNC_V_VVJ(NAME)                            \
  float    _cl_overloadable NAME(float   , float   , int   );   \
  float2   _cl_overloadable NAME(float2  , float2  , int2  );   \
  float3   _cl_overloadable NAME(float3  , float3  , int3  );   \
  float4   _cl_overloadable NAME(float4  , float4  , int4  );   \
  float8   _cl_overloadable NAME(float8  , float8  , int8  );   \
  float16  _cl_overloadable NAME(float16 , float16 , int16 );   \
  __IF_FP64(                                                    \
  double   _cl_overloadable NAME(double  , double  , long  );   \
  double2  _cl_overloadable NAME(double2 , double2 , long2 );   \
  double3  _cl_overloadable NAME(double3 , double3 , long3 );   \
  double4  _cl_overloadable NAME(double4 , double4 , long4 );   \
  double8  _cl_overloadable NAME(double8 , double8 , long8 );   \
  double16 _cl_overloadable NAME(double16, double16, long16);)
#define _CL_DECLARE_FUNC_V_U(NAME)              \
  float    _cl_overloadable NAME(uint   );      \
  float2   _cl_overloadable NAME(uint2  );      \
  float3   _cl_overloadable NAME(uint3  );      \
  float4   _cl_overloadable NAME(uint4  );      \
  float8   _cl_overloadable NAME(uint8  );      \
  float16  _cl_overloadable NAME(uint16 );      \
  __IF_FP64(                                    \
  double   _cl_overloadable NAME(ulong  );      \
  double2  _cl_overloadable NAME(ulong2 );      \
  double3  _cl_overloadable NAME(ulong3 );      \
  double4  _cl_overloadable NAME(ulong4 );      \
  double8  _cl_overloadable NAME(ulong8 );      \
  double16 _cl_overloadable NAME(ulong16);)
#define _CL_DECLARE_FUNC_V_VS(NAME)                     \
  float2   _cl_overloadable NAME(float2  , float );     \
  float3   _cl_overloadable NAME(float3  , float );     \
  float4   _cl_overloadable NAME(float4  , float );     \
  float8   _cl_overloadable NAME(float8  , float );     \
  float16  _cl_overloadable NAME(float16 , float );     \
  __IF_FP64(                                            \
  double2  _cl_overloadable NAME(double2 , double);     \
  double3  _cl_overloadable NAME(double3 , double);     \
  double4  _cl_overloadable NAME(double4 , double);     \
  double8  _cl_overloadable NAME(double8 , double);     \
  double16 _cl_overloadable NAME(double16, double);)
#define _CL_DECLARE_FUNC_V_VJ(NAME)                     \
  float    _cl_overloadable NAME(float   , int  );      \
  float2   _cl_overloadable NAME(float2  , int2 );      \
  float3   _cl_overloadable NAME(float3  , int3 );      \
  float4   _cl_overloadable NAME(float4  , int4 );      \
  float8   _cl_overloadable NAME(float8  , int8 );      \
  float16  _cl_overloadable NAME(float16 , int16);      \
  __IF_FP64(                                            \
  double   _cl_overloadable NAME(double  , int  );      \
  double2  _cl_overloadable NAME(double2 , int2 );      \
  double3  _cl_overloadable NAME(double3 , int3 );      \
  double4  _cl_overloadable NAME(double4 , int4 );      \
  double8  _cl_overloadable NAME(double8 , int8 );      \
  double16 _cl_overloadable NAME(double16, int16);)
#define _CL_DECLARE_FUNC_J_VV(NAME)                     \
  int    _cl_overloadable NAME(float   , float   );     \
  int2   _cl_overloadable NAME(float2  , float2  );     \
  int3   _cl_overloadable NAME(float3  , float3  );     \
  int4   _cl_overloadable NAME(float4  , float4  );     \
  int8   _cl_overloadable NAME(float8  , float8  );     \
  int16  _cl_overloadable NAME(float16 , float16 );     \
  __IF_FP64(                                            \
  int    _cl_overloadable NAME(double  , double  );     \
  long2  _cl_overloadable NAME(double2 , double2 );     \
  long3  _cl_overloadable NAME(double3 , double3 );     \
  long4  _cl_overloadable NAME(double4 , double4 );     \
  long8  _cl_overloadable NAME(double8 , double8 );     \
  long16 _cl_overloadable NAME(double16, double16);)
#define _CL_DECLARE_FUNC_V_VI(NAME)                     \
  float2   _cl_overloadable NAME(float2  , int);        \
  float3   _cl_overloadable NAME(float3  , int);        \
  float4   _cl_overloadable NAME(float4  , int);        \
  float8   _cl_overloadable NAME(float8  , int);        \
  float16  _cl_overloadable NAME(float16 , int);        \
  __IF_FP64(                                            \
  double2  _cl_overloadable NAME(double2 , int);        \
  double3  _cl_overloadable NAME(double3 , int);        \
  double4  _cl_overloadable NAME(double4 , int);        \
  double8  _cl_overloadable NAME(double8 , int);        \
  double16 _cl_overloadable NAME(double16, int);)
#define _CL_DECLARE_FUNC_V_VPV(NAME)                                    \
  float    _cl_overloadable NAME(float   , __global  float   *);        \
  float2   _cl_overloadable NAME(float2  , __global  float2  *);        \
  float3   _cl_overloadable NAME(float3  , __global  float3  *);        \
  float4   _cl_overloadable NAME(float4  , __global  float4  *);        \
  float8   _cl_overloadable NAME(float8  , __global  float8  *);        \
  float16  _cl_overloadable NAME(float16 , __global  float16 *);        \
  __IF_FP64(                                                            \
  double   _cl_overloadable NAME(double  , __global  double  *);        \
  double2  _cl_overloadable NAME(double2 , __global  double2 *);        \
  double3  _cl_overloadable NAME(double3 , __global  double3 *);        \
  double4  _cl_overloadable NAME(double4 , __global  double4 *);        \
  double8  _cl_overloadable NAME(double8 , __global  double8 *);        \
  double16 _cl_overloadable NAME(double16, __global  double16*);)       \
  float    _cl_overloadable NAME(float   , __local   float   *);        \
  float2   _cl_overloadable NAME(float2  , __local   float2  *);        \
  float3   _cl_overloadable NAME(float3  , __local   float3  *);        \
  float4   _cl_overloadable NAME(float4  , __local   float4  *);        \
  float8   _cl_overloadable NAME(float8  , __local   float8  *);        \
  float16  _cl_overloadable NAME(float16 , __local   float16 *);        \
  __IF_FP64(                                                            \
  double   _cl_overloadable NAME(double  , __local   double  *);        \
  double2  _cl_overloadable NAME(double2 , __local   double2 *);        \
  double3  _cl_overloadable NAME(double3 , __local   double3 *);        \
  double4  _cl_overloadable NAME(double4 , __local   double4 *);        \
  double8  _cl_overloadable NAME(double8 , __local   double8 *);        \
  double16 _cl_overloadable NAME(double16, __local   double16*);)       \
  float    _cl_overloadable NAME(float   , __private float   *);        \
  float2   _cl_overloadable NAME(float2  , __private float2  *);        \
  float3   _cl_overloadable NAME(float3  , __private float3  *);        \
  float4   _cl_overloadable NAME(float4  , __private float4  *);        \
  float8   _cl_overloadable NAME(float8  , __private float8  *);        \
  float16  _cl_overloadable NAME(float16 , __private float16 *);        \
  __IF_FP64(                                                            \
  double   _cl_overloadable NAME(double  , __private double  *);        \
  double2  _cl_overloadable NAME(double2 , __private double2 *);        \
  double3  _cl_overloadable NAME(double3 , __private double3 *);        \
  double4  _cl_overloadable NAME(double4 , __private double4 *);        \
  double8  _cl_overloadable NAME(double8 , __private double8 *);        \
  double16 _cl_overloadable NAME(double16, __private double16*);)
#define _CL_DECLARE_FUNC_V_SV(NAME)                     \
  float2   _cl_overloadable NAME(float , float2  );     \
  float3   _cl_overloadable NAME(float , float3  );     \
  float4   _cl_overloadable NAME(float , float4  );     \
  float8   _cl_overloadable NAME(float , float8  );     \
  float16  _cl_overloadable NAME(float , float16 );     \
  __IF_FP64(                                            \
  double2  _cl_overloadable NAME(double, double2 );     \
  double3  _cl_overloadable NAME(double, double3 );     \
  double4  _cl_overloadable NAME(double, double4 );     \
  double8  _cl_overloadable NAME(double, double8 );     \
  double16 _cl_overloadable NAME(double, double16);)
#define _CL_DECLARE_FUNC_J_V(NAME)              \
  int   _cl_overloadable NAME(float   );        \
  int2  _cl_overloadable NAME(float2  );        \
  int3  _cl_overloadable NAME(float3  );        \
  int4  _cl_overloadable NAME(float4  );        \
  int8  _cl_overloadable NAME(float8  );        \
  int16 _cl_overloadable NAME(float16 );        \
  __IF_FP64(                                    \
  int   _cl_overloadable NAME(double  );        \
  int2  _cl_overloadable NAME(double2 );        \
  int3  _cl_overloadable NAME(double3 );        \
  int4  _cl_overloadable NAME(double4 );        \
  int8  _cl_overloadable NAME(double8 );        \
  int16 _cl_overloadable NAME(double16);)
#define _CL_DECLARE_FUNC_K_V(NAME)              \
  int    _cl_overloadable NAME(float   );       \
  int2   _cl_overloadable NAME(float2  );       \
  int3   _cl_overloadable NAME(float3  );       \
  int4   _cl_overloadable NAME(float4  );       \
  int8   _cl_overloadable NAME(float8  );       \
  int16  _cl_overloadable NAME(float16 );       \
  __IF_FP64(                                    \
  int    _cl_overloadable NAME(double  );       \
  long2  _cl_overloadable NAME(double2 );       \
  long3  _cl_overloadable NAME(double3 );       \
  long4  _cl_overloadable NAME(double4 );       \
  long8  _cl_overloadable NAME(double8 );       \
  long16 _cl_overloadable NAME(double16);)
#define _CL_DECLARE_FUNC_S_V(NAME)              \
  float  _cl_overloadable NAME(float   );       \
  float  _cl_overloadable NAME(float2  );       \
  float  _cl_overloadable NAME(float3  );       \
  float  _cl_overloadable NAME(float4  );       \
  float  _cl_overloadable NAME(float8  );       \
  float  _cl_overloadable NAME(float16 );       \
  __IF_FP64(                                    \
  double _cl_overloadable NAME(double  );       \
  double _cl_overloadable NAME(double2 );       \
  double _cl_overloadable NAME(double3 );       \
  double _cl_overloadable NAME(double4 );       \
  double _cl_overloadable NAME(double8 );       \
  double _cl_overloadable NAME(double16);)
#define _CL_DECLARE_FUNC_S_VV(NAME)                     \
  float  _cl_overloadable NAME(float   , float   );     \
  float  _cl_overloadable NAME(float2  , float2  );     \
  float  _cl_overloadable NAME(float3  , float3  );     \
  float  _cl_overloadable NAME(float4  , float4  );     \
  float  _cl_overloadable NAME(float8  , float8  );     \
  float  _cl_overloadable NAME(float16 , float16 );     \
  __IF_FP64(                                            \
  double _cl_overloadable NAME(double  , double  );     \
  double _cl_overloadable NAME(double2 , double2 );     \
  double _cl_overloadable NAME(double3 , double3 );     \
  double _cl_overloadable NAME(double4 , double4 );     \
  double _cl_overloadable NAME(double8 , double8 );     \
  double _cl_overloadable NAME(double16, double16);)
#define _CL_DECLARE_FUNC_F_F(NAME)              \
  float    _cl_overloadable NAME(float   );     \
  float2   _cl_overloadable NAME(float2  );     \
  float3   _cl_overloadable NAME(float3  );     \
  float4   _cl_overloadable NAME(float4  );     \
  float8   _cl_overloadable NAME(float8  );     \
  float16  _cl_overloadable NAME(float16 );
#define _CL_DECLARE_FUNC_F_FF(NAME)                     \
  float    _cl_overloadable NAME(float   , float   );   \
  float2   _cl_overloadable NAME(float2  , float2  );   \
  float3   _cl_overloadable NAME(float3  , float3  );   \
  float4   _cl_overloadable NAME(float4  , float4  );   \
  float8   _cl_overloadable NAME(float8  , float8  );   \
  float16  _cl_overloadable NAME(float16 , float16 );

/* Move built-in declarations out of the way. (There should be a
   better way of doing so.) These five functions are built-in math
   functions for all Clang languages; see Clang's "Builtins.def".
   */
#define cos _cl_cos
#define fma _cl_fma
#define pow _cl_pow
#define sin _cl_sin
#define sqrt _cl_sqrt

#if __clang_major__ >= 3 && __clang_minor__ >=2
#define acos _cl_acos
#define asin _cl_asin
#define atan _cl_atan
#define atan2 _cl_atan2
#define ceil _cl_ceil
#define exp _cl_exp
#define floor _cl_floor
#define fabs _cl_fabs
#define log _cl_log
#define round _cl_round
#define tan _cl_tan
#endif

_CL_DECLARE_FUNC_V_V(acos)
_CL_DECLARE_FUNC_V_V(acosh)
_CL_DECLARE_FUNC_V_V(acospi)
_CL_DECLARE_FUNC_V_V(asin)
_CL_DECLARE_FUNC_V_V(asinh)
_CL_DECLARE_FUNC_V_V(asinpi)
_CL_DECLARE_FUNC_V_V(atan)
_CL_DECLARE_FUNC_V_VV(atan2)
_CL_DECLARE_FUNC_V_VV(atan2pi)
_CL_DECLARE_FUNC_V_V(atanh)
_CL_DECLARE_FUNC_V_V(atanpi)
_CL_DECLARE_FUNC_V_V(cbrt)
_CL_DECLARE_FUNC_V_V(ceil)
_CL_DECLARE_FUNC_V_VV(copysign)
_CL_DECLARE_FUNC_V_V(cos)
_CL_DECLARE_FUNC_V_V(cosh)
_CL_DECLARE_FUNC_V_V(cospi)
_CL_DECLARE_FUNC_S_VV(dot)
_CL_DECLARE_FUNC_V_V(erfc)
_CL_DECLARE_FUNC_V_V(erf)
_CL_DECLARE_FUNC_V_V(exp)
_CL_DECLARE_FUNC_V_V(exp2)
_CL_DECLARE_FUNC_V_V(exp10)
_CL_DECLARE_FUNC_V_V(expm1)
_CL_DECLARE_FUNC_V_V(fabs)
_CL_DECLARE_FUNC_V_VV(fdim)
_CL_DECLARE_FUNC_V_V(floor)
#if __FAST__RELAXED__MATH__
#  define _cl_fma _cl_fast_fma
#else
#  define _cl_fma _cl_std_fma
#endif
#define _cl_fast_fma mad
_CL_DECLARE_FUNC_V_VVV(_cl_std_fma)
#if __FAST__RELAXED__MATH__
#  define fmax _cl_fast_fmax
#  define fmin _cl_fast_fmin
#else
#  define fmax _cl_std_fmax
#  define fmin _cl_std_fmin
#endif
#define _cl_fast_fmax max
#define _cl_fast_fmin min
_CL_DECLARE_FUNC_V_VV(_cl_std_fmax)
_CL_DECLARE_FUNC_V_VS(_cl_std_fmax)
_CL_DECLARE_FUNC_V_VV(_cl_std_fmin)
_CL_DECLARE_FUNC_V_VS(_cl_std_fmin)
_CL_DECLARE_FUNC_V_VV(fmod)
_CL_DECLARE_FUNC_V_VPV(fract)
// frexp
_CL_DECLARE_FUNC_V_VV(hypot)
_CL_DECLARE_FUNC_J_V(ilogb)
_CL_DECLARE_FUNC_V_VJ(ldexp)
_CL_DECLARE_FUNC_V_VI(ldexp)
_CL_DECLARE_FUNC_V_V(lgamma)
// lgamma_r
_CL_DECLARE_FUNC_V_V(log)
_CL_DECLARE_FUNC_V_V(log2)
_CL_DECLARE_FUNC_V_V(log10)
_CL_DECLARE_FUNC_V_V(log1p)
_CL_DECLARE_FUNC_V_V(logb)
_CL_DECLARE_FUNC_V_VVV(mad)
_CL_DECLARE_FUNC_V_VV(maxmag)
_CL_DECLARE_FUNC_V_VV(minmag)
// modf
_CL_DECLARE_FUNC_V_U(nan)
_CL_DECLARE_FUNC_V_VV(nextafter)
_CL_DECLARE_FUNC_V_VV(pow)
_CL_DECLARE_FUNC_V_VJ(pown)
_CL_DECLARE_FUNC_V_VI(pown)
_CL_DECLARE_FUNC_V_VV(powr)
_CL_DECLARE_FUNC_V_VV(remainder)
// remquo
_CL_DECLARE_FUNC_V_V(rint)
_CL_DECLARE_FUNC_V_VJ(rootn)
_CL_DECLARE_FUNC_V_VI(rootn)
_CL_DECLARE_FUNC_V_V(round)
_CL_DECLARE_FUNC_V_V(rsqrt)
_CL_DECLARE_FUNC_V_V(sin)
_CL_DECLARE_FUNC_V_VPV(sincos)
_CL_DECLARE_FUNC_V_V(sinh)
_CL_DECLARE_FUNC_V_V(sinpi)
_CL_DECLARE_FUNC_V_V(sqrt)
_CL_DECLARE_FUNC_V_V(tan)
_CL_DECLARE_FUNC_V_V(tanh)
_CL_DECLARE_FUNC_V_V(tanpi)
_CL_DECLARE_FUNC_V_V(tgamma)
_CL_DECLARE_FUNC_V_V(trunc)

_CL_DECLARE_FUNC_F_F(half_cos)
_CL_DECLARE_FUNC_F_FF(half_divide)
_CL_DECLARE_FUNC_F_F(half_exp)
_CL_DECLARE_FUNC_F_F(half_exp2)
_CL_DECLARE_FUNC_F_F(half_exp10)
_CL_DECLARE_FUNC_F_F(half_log)
_CL_DECLARE_FUNC_F_F(half_log2)
_CL_DECLARE_FUNC_F_F(half_log10)
_CL_DECLARE_FUNC_F_FF(half_powr)
_CL_DECLARE_FUNC_F_F(half_recip)
_CL_DECLARE_FUNC_F_F(half_rsqrt)
_CL_DECLARE_FUNC_F_F(half_sin)
_CL_DECLARE_FUNC_F_F(half_sqrt)
_CL_DECLARE_FUNC_F_F(half_tan)
_CL_DECLARE_FUNC_F_F(native_cos)
_CL_DECLARE_FUNC_F_FF(native_divide)
_CL_DECLARE_FUNC_F_F(native_exp)
_CL_DECLARE_FUNC_F_F(native_exp2)
_CL_DECLARE_FUNC_F_F(native_exp10)
_CL_DECLARE_FUNC_F_F(native_log)
_CL_DECLARE_FUNC_F_F(native_log2)
_CL_DECLARE_FUNC_F_F(native_log10)
_CL_DECLARE_FUNC_F_FF(native_powr)
_CL_DECLARE_FUNC_F_F(native_recip)
_CL_DECLARE_FUNC_F_F(native_rsqrt)
_CL_DECLARE_FUNC_F_F(native_sin)
_CL_DECLARE_FUNC_F_F(native_sqrt)
_CL_DECLARE_FUNC_F_F(native_tan)


/* Integer Constants */

#define CHAR_BIT  8
#define CHAR_MAX  SCHAR_MAX
#define CHAR_MIN  SCHAR_MIN
#define INT_MAX   2147483647
#define INT_MIN   (-2147483647 - 1)
#ifdef cles_khr_int64
#define LONG_MAX  0x7fffffffffffffffL
#define LONG_MIN  (-0x7fffffffffffffffL - 1)
#endif
#define SCHAR_MAX 127
#define SCHAR_MIN (-127 - 1)
#define SHRT_MAX  32767
#define SHRT_MIN  (-32767 - 1)
#define UCHAR_MAX 255
#define USHRT_MAX 65535
#define UINT_MAX  0xffffffff
#ifdef cles_khr_int64
#define ULONG_MAX 0xffffffffffffffffUL
#endif


/* Integer Functions */
#define _CL_DECLARE_FUNC_G_G(NAME)              \
  char     _cl_overloadable NAME(char    );     \
  char2    _cl_overloadable NAME(char2   );     \
  char3    _cl_overloadable NAME(char3   );     \
  char4    _cl_overloadable NAME(char4   );     \
  char8    _cl_overloadable NAME(char8   );     \
  char16   _cl_overloadable NAME(char16  );     \
  uchar    _cl_overloadable NAME(uchar   );     \
  uchar2   _cl_overloadable NAME(uchar2  );     \
  uchar3   _cl_overloadable NAME(uchar3  );     \
  uchar4   _cl_overloadable NAME(uchar4  );     \
  uchar8   _cl_overloadable NAME(uchar8  );     \
  uchar16  _cl_overloadable NAME(uchar16 );     \
  short    _cl_overloadable NAME(short   );     \
  short2   _cl_overloadable NAME(short2  );     \
  short3   _cl_overloadable NAME(short3  );     \
  short4   _cl_overloadable NAME(short4  );     \
  short8   _cl_overloadable NAME(short8  );     \
  short16  _cl_overloadable NAME(short16 );     \
  ushort   _cl_overloadable NAME(ushort  );     \
  ushort2  _cl_overloadable NAME(ushort2 );     \
  ushort3  _cl_overloadable NAME(ushort3 );     \
  ushort4  _cl_overloadable NAME(ushort4 );     \
  ushort8  _cl_overloadable NAME(ushort8 );     \
  ushort16 _cl_overloadable NAME(ushort16);     \
  int      _cl_overloadable NAME(int     );     \
  int2     _cl_overloadable NAME(int2    );     \
  int3     _cl_overloadable NAME(int3    );     \
  int4     _cl_overloadable NAME(int4    );     \
  int8     _cl_overloadable NAME(int8    );     \
  int16    _cl_overloadable NAME(int16   );     \
  uint     _cl_overloadable NAME(uint    );     \
  uint2    _cl_overloadable NAME(uint2   );     \
  uint3    _cl_overloadable NAME(uint3   );     \
  uint4    _cl_overloadable NAME(uint4   );     \
  uint8    _cl_overloadable NAME(uint8   );     \
  uint16   _cl_overloadable NAME(uint16  );     \
  __IF_INT64(                                   \
  long     _cl_overloadable NAME(long    );     \
  long2    _cl_overloadable NAME(long2   );     \
  long3    _cl_overloadable NAME(long3   );     \
  long4    _cl_overloadable NAME(long4   );     \
  long8    _cl_overloadable NAME(long8   );     \
  long16   _cl_overloadable NAME(long16  );     \
  ulong    _cl_overloadable NAME(ulong   );     \
  ulong2   _cl_overloadable NAME(ulong2  );     \
  ulong3   _cl_overloadable NAME(ulong3  );     \
  ulong4   _cl_overloadable NAME(ulong4  );     \
  ulong8   _cl_overloadable NAME(ulong8  );     \
  ulong16  _cl_overloadable NAME(ulong16 );)
#define _CL_DECLARE_FUNC_G_GG(NAME)                     \
  char     _cl_overloadable NAME(char    , char    );   \
  char2    _cl_overloadable NAME(char2   , char2   );   \
  char3    _cl_overloadable NAME(char3   , char3   );   \
  char4    _cl_overloadable NAME(char4   , char4   );   \
  char8    _cl_overloadable NAME(char8   , char8   );   \
  char16   _cl_overloadable NAME(char16  , char16  );   \
  uchar    _cl_overloadable NAME(uchar   , uchar   );   \
  uchar2   _cl_overloadable NAME(uchar2  , uchar2  );   \
  uchar3   _cl_overloadable NAME(uchar3  , uchar3  );   \
  uchar4   _cl_overloadable NAME(uchar4  , uchar4  );   \
  uchar8   _cl_overloadable NAME(uchar8  , uchar8  );   \
  uchar16  _cl_overloadable NAME(uchar16 , uchar16 );   \
  short    _cl_overloadable NAME(short   , short   );   \
  short2   _cl_overloadable NAME(short2  , short2  );   \
  short3   _cl_overloadable NAME(short3  , short3  );   \
  short4   _cl_overloadable NAME(short4  , short4  );   \
  short8   _cl_overloadable NAME(short8  , short8  );   \
  short16  _cl_overloadable NAME(short16 , short16 );   \
  ushort   _cl_overloadable NAME(ushort  , ushort  );   \
  ushort2  _cl_overloadable NAME(ushort2 , ushort2 );   \
  ushort3  _cl_overloadable NAME(ushort3 , ushort3 );   \
  ushort4  _cl_overloadable NAME(ushort4 , ushort4 );   \
  ushort8  _cl_overloadable NAME(ushort8 , ushort8 );   \
  ushort16 _cl_overloadable NAME(ushort16, ushort16);   \
  int      _cl_overloadable NAME(int     , int     );   \
  int2     _cl_overloadable NAME(int2    , int2    );   \
  int3     _cl_overloadable NAME(int3    , int3    );   \
  int4     _cl_overloadable NAME(int4    , int4    );   \
  int8     _cl_overloadable NAME(int8    , int8    );   \
  int16    _cl_overloadable NAME(int16   , int16   );   \
  uint     _cl_overloadable NAME(uint    , uint    );   \
  uint2    _cl_overloadable NAME(uint2   , uint2   );   \
  uint3    _cl_overloadable NAME(uint3   , uint3   );   \
  uint4    _cl_overloadable NAME(uint4   , uint4   );   \
  uint8    _cl_overloadable NAME(uint8   , uint8   );   \
  uint16   _cl_overloadable NAME(uint16  , uint16  );   \
  __IF_INT64(                                           \
  long     _cl_overloadable NAME(long    , long    );   \
  long2    _cl_overloadable NAME(long2   , long2   );   \
  long3    _cl_overloadable NAME(long3   , long3   );   \
  long4    _cl_overloadable NAME(long4   , long4   );   \
  long8    _cl_overloadable NAME(long8   , long8   );   \
  long16   _cl_overloadable NAME(long16  , long16  );   \
  ulong    _cl_overloadable NAME(ulong   , ulong   );   \
  ulong2   _cl_overloadable NAME(ulong2  , ulong2  );   \
  ulong3   _cl_overloadable NAME(ulong3  , ulong3  );   \
  ulong4   _cl_overloadable NAME(ulong4  , ulong4  );   \
  ulong8   _cl_overloadable NAME(ulong8  , ulong8  );   \
  ulong16  _cl_overloadable NAME(ulong16 , ulong16 );)
#define _CL_DECLARE_FUNC_G_GGG(NAME)                            \
  char     _cl_overloadable NAME(char    , char    , char    ); \
  char2    _cl_overloadable NAME(char2   , char2   , char2   ); \
  char3    _cl_overloadable NAME(char3   , char3   , char3   ); \
  char4    _cl_overloadable NAME(char4   , char4   , char4   ); \
  char8    _cl_overloadable NAME(char8   , char8   , char8   ); \
  char16   _cl_overloadable NAME(char16  , char16  , char16  ); \
  uchar    _cl_overloadable NAME(uchar   , uchar   , uchar   ); \
  uchar2   _cl_overloadable NAME(uchar2  , uchar2  , uchar2  ); \
  uchar3   _cl_overloadable NAME(uchar3  , uchar3  , uchar3  ); \
  uchar4   _cl_overloadable NAME(uchar4  , uchar4  , uchar4  ); \
  uchar8   _cl_overloadable NAME(uchar8  , uchar8  , uchar8  ); \
  uchar16  _cl_overloadable NAME(uchar16 , uchar16 , uchar16 ); \
  short    _cl_overloadable NAME(short   , short   , short   ); \
  short2   _cl_overloadable NAME(short2  , short2  , short2  ); \
  short3   _cl_overloadable NAME(short3  , short3  , short3  ); \
  short4   _cl_overloadable NAME(short4  , short4  , short4  ); \
  short8   _cl_overloadable NAME(short8  , short8  , short8  ); \
  short16  _cl_overloadable NAME(short16 , short16 , short16 ); \
  ushort   _cl_overloadable NAME(ushort  , ushort  , ushort  ); \
  ushort2  _cl_overloadable NAME(ushort2 , ushort2 , ushort2 ); \
  ushort3  _cl_overloadable NAME(ushort3 , ushort3 , ushort3 ); \
  ushort4  _cl_overloadable NAME(ushort4 , ushort4 , ushort4 ); \
  ushort8  _cl_overloadable NAME(ushort8 , ushort8 , ushort8 ); \
  ushort16 _cl_overloadable NAME(ushort16, ushort16, ushort16); \
  int      _cl_overloadable NAME(int     , int     , int     ); \
  int2     _cl_overloadable NAME(int2    , int2    , int2    ); \
  int3     _cl_overloadable NAME(int3    , int3    , int3    ); \
  int4     _cl_overloadable NAME(int4    , int4    , int4    ); \
  int8     _cl_overloadable NAME(int8    , int8    , int8    ); \
  int16    _cl_overloadable NAME(int16   , int16   , int16   ); \
  uint     _cl_overloadable NAME(uint    , uint    , uint    ); \
  uint2    _cl_overloadable NAME(uint2   , uint2   , uint2   ); \
  uint3    _cl_overloadable NAME(uint3   , uint3   , uint3   ); \
  uint4    _cl_overloadable NAME(uint4   , uint4   , uint4   ); \
  uint8    _cl_overloadable NAME(uint8   , uint8   , uint8   ); \
  uint16   _cl_overloadable NAME(uint16  , uint16  , uint16  ); \
  __IF_INT64(                                                   \
  long     _cl_overloadable NAME(long    , long    , long    ); \
  long2    _cl_overloadable NAME(long2   , long2   , long2   ); \
  long3    _cl_overloadable NAME(long3   , long3   , long3   ); \
  long4    _cl_overloadable NAME(long4   , long4   , long4   ); \
  long8    _cl_overloadable NAME(long8   , long8   , long8   ); \
  long16   _cl_overloadable NAME(long16  , long16  , long16  ); \
  ulong    _cl_overloadable NAME(ulong   , ulong   , ulong   ); \
  ulong2   _cl_overloadable NAME(ulong2  , ulong2  , ulong2  ); \
  ulong3   _cl_overloadable NAME(ulong3  , ulong3  , ulong3  ); \
  ulong4   _cl_overloadable NAME(ulong4  , ulong4  , ulong4  ); \
  ulong8   _cl_overloadable NAME(ulong8  , ulong8  , ulong8  ); \
  ulong16  _cl_overloadable NAME(ulong16 , ulong16 , ulong16 );)
#define _CL_DECLARE_FUNC_G_GS(NAME)                     \
  char2    _cl_overloadable NAME(char2   , char  );     \
  char3    _cl_overloadable NAME(char3   , char  );     \
  char4    _cl_overloadable NAME(char4   , char  );     \
  char8    _cl_overloadable NAME(char8   , char  );     \
  char16   _cl_overloadable NAME(char16  , char  );     \
  uchar2   _cl_overloadable NAME(uchar2  , uchar );     \
  uchar3   _cl_overloadable NAME(uchar3  , uchar );     \
  uchar4   _cl_overloadable NAME(uchar4  , uchar );     \
  uchar8   _cl_overloadable NAME(uchar8  , uchar );     \
  uchar16  _cl_overloadable NAME(uchar16 , uchar );     \
  short2   _cl_overloadable NAME(short2  , short );     \
  short3   _cl_overloadable NAME(short3  , short );     \
  short4   _cl_overloadable NAME(short4  , short );     \
  short8   _cl_overloadable NAME(short8  , short );     \
  short16  _cl_overloadable NAME(short16 , short );     \
  ushort2  _cl_overloadable NAME(ushort2 , ushort);     \
  ushort3  _cl_overloadable NAME(ushort3 , ushort);     \
  ushort4  _cl_overloadable NAME(ushort4 , ushort);     \
  ushort8  _cl_overloadable NAME(ushort8 , ushort);     \
  ushort16 _cl_overloadable NAME(ushort16, ushort);     \
  int2     _cl_overloadable NAME(int2    , int   );     \
  int3     _cl_overloadable NAME(int3    , int   );     \
  int4     _cl_overloadable NAME(int4    , int   );     \
  int8     _cl_overloadable NAME(int8    , int   );     \
  int16    _cl_overloadable NAME(int16   , int   );     \
  uint2    _cl_overloadable NAME(uint2   , uint  );     \
  uint3    _cl_overloadable NAME(uint3   , uint  );     \
  uint4    _cl_overloadable NAME(uint4   , uint  );     \
  uint8    _cl_overloadable NAME(uint8   , uint  );     \
  uint16   _cl_overloadable NAME(uint16  , uint  );     \
  __IF_INT64(                                           \
  long2    _cl_overloadable NAME(long2   , long  );     \
  long3    _cl_overloadable NAME(long3   , long  );     \
  long4    _cl_overloadable NAME(long4   , long  );     \
  long8    _cl_overloadable NAME(long8   , long  );     \
  long16   _cl_overloadable NAME(long16  , long  );     \
  ulong2   _cl_overloadable NAME(ulong2  , ulong );     \
  ulong3   _cl_overloadable NAME(ulong3  , ulong );     \
  ulong4   _cl_overloadable NAME(ulong4  , ulong );     \
  ulong8   _cl_overloadable NAME(ulong8  , ulong );     \
  ulong16  _cl_overloadable NAME(ulong16 , ulong );)
#define _CL_DECLARE_FUNC_UG_G(NAME)             \
  uchar    _cl_overloadable NAME(char    );     \
  uchar2   _cl_overloadable NAME(char2   );     \
  uchar3   _cl_overloadable NAME(char3   );     \
  uchar4   _cl_overloadable NAME(char4   );     \
  uchar8   _cl_overloadable NAME(char8   );     \
  uchar16  _cl_overloadable NAME(char16  );     \
  ushort   _cl_overloadable NAME(short   );     \
  ushort2  _cl_overloadable NAME(short2  );     \
  ushort3  _cl_overloadable NAME(short3  );     \
  ushort4  _cl_overloadable NAME(short4  );     \
  ushort8  _cl_overloadable NAME(short8  );     \
  ushort16 _cl_overloadable NAME(short16 );     \
  uint     _cl_overloadable NAME(int     );     \
  uint2    _cl_overloadable NAME(int2    );     \
  uint3    _cl_overloadable NAME(int3    );     \
  uint4    _cl_overloadable NAME(int4    );     \
  uint8    _cl_overloadable NAME(int8    );     \
  uint16   _cl_overloadable NAME(int16   );     \
  __IF_INT64(                                   \
  ulong    _cl_overloadable NAME(long    );     \
  ulong2   _cl_overloadable NAME(long2   );     \
  ulong3   _cl_overloadable NAME(long3   );     \
  ulong4   _cl_overloadable NAME(long4   );     \
  ulong8   _cl_overloadable NAME(long8   );     \
  ulong16  _cl_overloadable NAME(long16  );)    \
  uchar    _cl_overloadable NAME(uchar   );     \
  uchar2   _cl_overloadable NAME(uchar2  );     \
  uchar3   _cl_overloadable NAME(uchar3  );     \
  uchar4   _cl_overloadable NAME(uchar4  );     \
  uchar8   _cl_overloadable NAME(uchar8  );     \
  uchar16  _cl_overloadable NAME(uchar16 );     \
  ushort   _cl_overloadable NAME(ushort  );     \
  ushort2  _cl_overloadable NAME(ushort2 );     \
  ushort3  _cl_overloadable NAME(ushort3 );     \
  ushort4  _cl_overloadable NAME(ushort4 );     \
  ushort8  _cl_overloadable NAME(ushort8 );     \
  ushort16 _cl_overloadable NAME(ushort16);     \
  uint     _cl_overloadable NAME(uint    );     \
  uint2    _cl_overloadable NAME(uint2   );     \
  uint3    _cl_overloadable NAME(uint3   );     \
  uint4    _cl_overloadable NAME(uint4   );     \
  uint8    _cl_overloadable NAME(uint8   );     \
  uint16   _cl_overloadable NAME(uint16  );     \
  __IF_INT64(                                   \
  ulong    _cl_overloadable NAME(ulong   );     \
  ulong2   _cl_overloadable NAME(ulong2  );     \
  ulong3   _cl_overloadable NAME(ulong3  );     \
  ulong4   _cl_overloadable NAME(ulong4  );     \
  ulong8   _cl_overloadable NAME(ulong8  );     \
  ulong16  _cl_overloadable NAME(ulong16 );)
#define _CL_DECLARE_FUNC_UG_GG(NAME)                    \
  uchar    _cl_overloadable NAME(char    , char    );   \
  uchar2   _cl_overloadable NAME(char2   , char2   );   \
  uchar3   _cl_overloadable NAME(char3   , char3   );   \
  uchar4   _cl_overloadable NAME(char4   , char4   );   \
  uchar8   _cl_overloadable NAME(char8   , char8   );   \
  uchar16  _cl_overloadable NAME(char16  , char16  );   \
  ushort   _cl_overloadable NAME(short   , short   );   \
  ushort2  _cl_overloadable NAME(short2  , short2  );   \
  ushort3  _cl_overloadable NAME(short3  , short3  );   \
  ushort4  _cl_overloadable NAME(short4  , short4  );   \
  ushort8  _cl_overloadable NAME(short8  , short8  );   \
  ushort16 _cl_overloadable NAME(short16 , short16 );   \
  uint     _cl_overloadable NAME(int     , int     );   \
  uint2    _cl_overloadable NAME(int2    , int2    );   \
  uint3    _cl_overloadable NAME(int3    , int3    );   \
  uint4    _cl_overloadable NAME(int4    , int4    );   \
  uint8    _cl_overloadable NAME(int8    , int8    );   \
  uint16   _cl_overloadable NAME(int16   , int16   );   \
  __IF_INT64(                                           \
  ulong    _cl_overloadable NAME(long    , long    );   \
  ulong2   _cl_overloadable NAME(long2   , long2   );   \
  ulong3   _cl_overloadable NAME(long3   , long3   );   \
  ulong4   _cl_overloadable NAME(long4   , long4   );   \
  ulong8   _cl_overloadable NAME(long8   , long8   );   \
  ulong16  _cl_overloadable NAME(long16  , long16  );)  \
  uchar    _cl_overloadable NAME(uchar   , uchar   );   \
  uchar2   _cl_overloadable NAME(uchar2  , uchar2  );   \
  uchar3   _cl_overloadable NAME(uchar3  , uchar3  );   \
  uchar4   _cl_overloadable NAME(uchar4  , uchar4  );   \
  uchar8   _cl_overloadable NAME(uchar8  , uchar8  );   \
  uchar16  _cl_overloadable NAME(uchar16 , uchar16 );   \
  ushort   _cl_overloadable NAME(ushort  , ushort  );   \
  ushort2  _cl_overloadable NAME(ushort2 , ushort2 );   \
  ushort3  _cl_overloadable NAME(ushort3 , ushort3 );   \
  ushort4  _cl_overloadable NAME(ushort4 , ushort4 );   \
  ushort8  _cl_overloadable NAME(ushort8 , ushort8 );   \
  ushort16 _cl_overloadable NAME(ushort16, ushort16);   \
  uint     _cl_overloadable NAME(uint    , uint    );   \
  uint2    _cl_overloadable NAME(uint2   , uint2   );   \
  uint3    _cl_overloadable NAME(uint3   , uint3   );   \
  uint4    _cl_overloadable NAME(uint4   , uint4   );   \
  uint8    _cl_overloadable NAME(uint8   , uint8   );   \
  uint16   _cl_overloadable NAME(uint16  , uint16  );   \
  __IF_INT64(                                           \
  ulong    _cl_overloadable NAME(ulong   , ulong   );   \
  ulong2   _cl_overloadable NAME(ulong2  , ulong2  );   \
  ulong3   _cl_overloadable NAME(ulong3  , ulong3  );   \
  ulong4   _cl_overloadable NAME(ulong4  , ulong4  );   \
  ulong8   _cl_overloadable NAME(ulong8  , ulong8  );   \
  ulong16  _cl_overloadable NAME(ulong16 , ulong16 );)
#define _CL_DECLARE_FUNC_LG_GUG(NAME)                   \
  short    _cl_overloadable NAME(char    , uchar   );   \
  short2   _cl_overloadable NAME(char2   , uchar2  );   \
  short3   _cl_overloadable NAME(char3   , uchar3  );   \
  short4   _cl_overloadable NAME(char4   , uchar4  );   \
  short8   _cl_overloadable NAME(char8   , uchar8  );   \
  short16  _cl_overloadable NAME(char16  , uchar16 );   \
  ushort   _cl_overloadable NAME(uchar   , uchar   );   \
  ushort2  _cl_overloadable NAME(uchar2  , uchar2  );   \
  ushort3  _cl_overloadable NAME(uchar3  , uchar3  );   \
  ushort4  _cl_overloadable NAME(uchar4  , uchar4  );   \
  ushort8  _cl_overloadable NAME(uchar8  , uchar8  );   \
  ushort16 _cl_overloadable NAME(uchar16 , uchar16 );   \
  uint     _cl_overloadable NAME(ushort  , ushort  );   \
  uint2    _cl_overloadable NAME(ushort2 , ushort2 );   \
  uint3    _cl_overloadable NAME(ushort3 , ushort3 );   \
  uint4    _cl_overloadable NAME(ushort4 , ushort4 );   \
  uint8    _cl_overloadable NAME(ushort8 , ushort8 );   \
  uint16   _cl_overloadable NAME(ushort16, ushort16);   \
  int      _cl_overloadable NAME(short   , ushort  );   \
  int2     _cl_overloadable NAME(short2  , ushort2 );   \
  int3     _cl_overloadable NAME(short3  , ushort3 );   \
  int4     _cl_overloadable NAME(short4  , ushort4 );   \
  int8     _cl_overloadable NAME(short8  , ushort8 );   \
  int16    _cl_overloadable NAME(short16 , ushort16);   \
  __IF_INT64(                                           \
  long     _cl_overloadable NAME(int     , uint    );   \
  long2    _cl_overloadable NAME(int2    , uint2   );   \
  long3    _cl_overloadable NAME(int3    , uint3   );   \
  long4    _cl_overloadable NAME(int4    , uint4   );   \
  long8    _cl_overloadable NAME(int8    , uint8   );   \
  long16   _cl_overloadable NAME(int16   , uint16  );   \
  ulong    _cl_overloadable NAME(uint    , uint    );   \
  ulong2   _cl_overloadable NAME(uint2   , uint2   );   \
  ulong3   _cl_overloadable NAME(uint3   , uint3   );   \
  ulong4   _cl_overloadable NAME(uint4   , uint4   );   \
  ulong8   _cl_overloadable NAME(uint8   , uint8   );   \
  ulong16  _cl_overloadable NAME(uint16  , uint16  );)
#define _CL_DECLARE_FUNC_I_IG(NAME)             \
  int _cl_overloadable NAME(char   );           \
  int _cl_overloadable NAME(char2  );           \
  int _cl_overloadable NAME(char3  );           \
  int _cl_overloadable NAME(char4  );           \
  int _cl_overloadable NAME(char8  );           \
  int _cl_overloadable NAME(char16 );           \
  int _cl_overloadable NAME(short  );           \
  int _cl_overloadable NAME(short2 );           \
  int _cl_overloadable NAME(short3 );           \
  int _cl_overloadable NAME(short4 );           \
  int _cl_overloadable NAME(short8 );           \
  int _cl_overloadable NAME(short16);           \
  int _cl_overloadable NAME(int    );           \
  int _cl_overloadable NAME(int2   );           \
  int _cl_overloadable NAME(int3   );           \
  int _cl_overloadable NAME(int4   );           \
  int _cl_overloadable NAME(int8   );           \
  int _cl_overloadable NAME(int16  );           \
  __IF_INT64(                                   \
  int _cl_overloadable NAME(long   );           \
  int _cl_overloadable NAME(long2  );           \
  int _cl_overloadable NAME(long3  );           \
  int _cl_overloadable NAME(long4  );           \
  int _cl_overloadable NAME(long8  );           \
  int _cl_overloadable NAME(long16 );)
#define _CL_DECLARE_FUNC_J_JJ(NAME)                     \
  int      _cl_overloadable NAME(int     , int     );   \
  int2     _cl_overloadable NAME(int2    , int2    );   \
  int3     _cl_overloadable NAME(int3    , int3    );   \
  int4     _cl_overloadable NAME(int4    , int4    );   \
  int8     _cl_overloadable NAME(int8    , int8    );   \
  int16    _cl_overloadable NAME(int16   , int16   );   \
  uint     _cl_overloadable NAME(uint    , uint    );   \
  uint2    _cl_overloadable NAME(uint2   , uint2   );   \
  uint3    _cl_overloadable NAME(uint3   , uint3   );   \
  uint4    _cl_overloadable NAME(uint4   , uint4   );   \
  uint8    _cl_overloadable NAME(uint8   , uint8   );   \
  uint16   _cl_overloadable NAME(uint16  , uint16  );
#define _CL_DECLARE_FUNC_J_JJJ(NAME)                            \
  int      _cl_overloadable NAME(int     , int     , int     ); \
  int2     _cl_overloadable NAME(int2    , int2    , int2    ); \
  int3     _cl_overloadable NAME(int3    , int3    , int3    ); \
  int4     _cl_overloadable NAME(int4    , int4    , int4    ); \
  int8     _cl_overloadable NAME(int8    , int8    , int8    ); \
  int16    _cl_overloadable NAME(int16   , int16   , int16   ); \
  uint     _cl_overloadable NAME(uint    , uint    , uint    ); \
  uint2    _cl_overloadable NAME(uint2   , uint2   , uint2   ); \
  uint3    _cl_overloadable NAME(uint3   , uint3   , uint3   ); \
  uint4    _cl_overloadable NAME(uint4   , uint4   , uint4   ); \
  uint8    _cl_overloadable NAME(uint8   , uint8   , uint8   ); \
  uint16   _cl_overloadable NAME(uint16  , uint16  , uint16  );

_CL_DECLARE_FUNC_UG_G(abs)
_CL_DECLARE_FUNC_UG_GG(abs_diff)
_CL_DECLARE_FUNC_G_GG(add_sat)
_CL_DECLARE_FUNC_G_GG(hadd)
_CL_DECLARE_FUNC_G_GG(rhadd)
_CL_DECLARE_FUNC_G_GGG(clamp)
_CL_DECLARE_FUNC_G_G(clz)
_CL_DECLARE_FUNC_G_GGG(mad_hi)
_CL_DECLARE_FUNC_G_GGG(mad_sat)
_CL_DECLARE_FUNC_G_GG(max)
_CL_DECLARE_FUNC_G_GS(max)
_CL_DECLARE_FUNC_G_GG(min)
_CL_DECLARE_FUNC_G_GS(min)
_CL_DECLARE_FUNC_G_GG(mul_hi)
_CL_DECLARE_FUNC_G_GG(rotate)
_CL_DECLARE_FUNC_G_GG(sub_sat)
_CL_DECLARE_FUNC_LG_GUG(upsample)
_CL_DECLARE_FUNC_G_G(popcount)
_CL_DECLARE_FUNC_J_JJJ(mad24)
_CL_DECLARE_FUNC_J_JJ(mul24)


/* Common Functions */

_CL_DECLARE_FUNC_V_VVV(clamp)
_CL_DECLARE_FUNC_V_VSS(clamp)
_CL_DECLARE_FUNC_V_V(degrees)
_CL_DECLARE_FUNC_V_VV(max)
_CL_DECLARE_FUNC_V_VS(max)
_CL_DECLARE_FUNC_V_VV(min)
_CL_DECLARE_FUNC_V_VS(min)
_CL_DECLARE_FUNC_V_VVV(mix)
_CL_DECLARE_FUNC_V_VVS(mix)
_CL_DECLARE_FUNC_V_V(radians)
_CL_DECLARE_FUNC_V_VV(step)
_CL_DECLARE_FUNC_V_SV(step)
_CL_DECLARE_FUNC_V_VVV(smoothstep)
_CL_DECLARE_FUNC_V_SSV(smoothstep)
_CL_DECLARE_FUNC_V_V(sign)


/* Geometric Functions */

float4 _cl_overloadable cross(float4, float4);
float3 _cl_overloadable cross(float3, float3);
#ifdef cl_khr_fp64
double4 _cl_overloadable cross(double4, double4);
double3 _cl_overloadable cross(double3, double3);
#endif
_CL_DECLARE_FUNC_S_VV(dot)
_CL_DECLARE_FUNC_S_VV(distance)
_CL_DECLARE_FUNC_S_V(length)
_CL_DECLARE_FUNC_V_V(normalize)
// TODO: no double version of these
_CL_DECLARE_FUNC_S_VV(fast_distance)
_CL_DECLARE_FUNC_S_V(fast_length)
_CL_DECLARE_FUNC_V_V(fast_normalize)


/* Relational Functions */

_CL_DECLARE_FUNC_J_VV(isequal)
_CL_DECLARE_FUNC_J_VV(isnotequal)
_CL_DECLARE_FUNC_J_VV(isgreater)
_CL_DECLARE_FUNC_J_VV(isgreaterequal)
_CL_DECLARE_FUNC_J_VV(isless)
_CL_DECLARE_FUNC_J_VV(islessequal)
_CL_DECLARE_FUNC_J_VV(islessgreater)
_CL_DECLARE_FUNC_J_VV(isfinite)
_CL_DECLARE_FUNC_J_VV(isinf)
_CL_DECLARE_FUNC_J_VV(isnan)
_CL_DECLARE_FUNC_J_VV(isnormal)
_CL_DECLARE_FUNC_J_VV(isordered)
_CL_DECLARE_FUNC_J_VV(isunordered)
_CL_DECLARE_FUNC_K_V(signbit)
_CL_DECLARE_FUNC_I_IG(any)
_CL_DECLARE_FUNC_I_IG(all)
_CL_DECLARE_FUNC_G_GGG(bitselect)
_CL_DECLARE_FUNC_V_VVV(bitselect)
_CL_DECLARE_FUNC_G_GGG(select)
_CL_DECLARE_FUNC_V_VVJ(select)


/* Vector Functions */

#define _CL_DECLARE_VLOAD(TYPE, MOD)                                    \
  TYPE##2  _cl_overloadable vload2 (size_t offset, const MOD TYPE *p);  \
  TYPE##3  _cl_overloadable vload3 (size_t offset, const MOD TYPE *p);  \
  TYPE##4  _cl_overloadable vload4 (size_t offset, const MOD TYPE *p);  \
  TYPE##8  _cl_overloadable vload8 (size_t offset, const MOD TYPE *p);  \
  TYPE##16 _cl_overloadable vload16(size_t offset, const MOD TYPE *p);

_CL_DECLARE_VLOAD(char  , __global)
_CL_DECLARE_VLOAD(uchar , __global)
_CL_DECLARE_VLOAD(short , __global)
_CL_DECLARE_VLOAD(ushort, __global)
_CL_DECLARE_VLOAD(int   , __global)
_CL_DECLARE_VLOAD(uint  , __global)
#ifdef cles_khr_int64
_CL_DECLARE_VLOAD(long  , __global)
_CL_DECLARE_VLOAD(ulong , __global)
#endif
_CL_DECLARE_VLOAD(float , __global)
#ifdef cl_khr_fp64
_CL_DECLARE_VLOAD(double, __global)
#endif

_CL_DECLARE_VLOAD(char  , __local)
_CL_DECLARE_VLOAD(uchar , __local)
_CL_DECLARE_VLOAD(short , __local)
_CL_DECLARE_VLOAD(ushort, __local)
_CL_DECLARE_VLOAD(int   , __local)
_CL_DECLARE_VLOAD(uint  , __local)
#ifdef cles_khr_int64
_CL_DECLARE_VLOAD(long  , __local)
_CL_DECLARE_VLOAD(ulong , __local)
#endif
_CL_DECLARE_VLOAD(float , __local)
#ifdef cl_khr_fp64
_CL_DECLARE_VLOAD(double, __local)
#endif

_CL_DECLARE_VLOAD(char  , __constant)
_CL_DECLARE_VLOAD(uchar , __constant)
_CL_DECLARE_VLOAD(short , __constant)
_CL_DECLARE_VLOAD(ushort, __constant)
_CL_DECLARE_VLOAD(int   , __constant)
_CL_DECLARE_VLOAD(uint  , __constant)
#ifdef cles_khr_int64
_CL_DECLARE_VLOAD(long  , __constant)
_CL_DECLARE_VLOAD(ulong , __constant)
#endif
_CL_DECLARE_VLOAD(float , __constant)
#ifdef cl_khr_fp64
_CL_DECLARE_VLOAD(double, __constant)
#endif

_CL_DECLARE_VLOAD(char  , __private)
_CL_DECLARE_VLOAD(uchar , __private)
_CL_DECLARE_VLOAD(short , __private)
_CL_DECLARE_VLOAD(ushort, __private)
_CL_DECLARE_VLOAD(int   , __private)
_CL_DECLARE_VLOAD(uint  , __private)
#ifdef cles_khr_int64
_CL_DECLARE_VLOAD(long  , __private)
_CL_DECLARE_VLOAD(ulong , __private)
#endif
_CL_DECLARE_VLOAD(float , __private)
#ifdef cl_khr_fp64
_CL_DECLARE_VLOAD(double, __private)
#endif

#define _CL_DECLARE_VSTORE(TYPE, MOD)                                   \
  void _cl_overloadable vstore2 (TYPE##2  data, size_t offset, MOD TYPE *p); \
  void _cl_overloadable vstore3 (TYPE##3  data, size_t offset, MOD TYPE *p); \
  void _cl_overloadable vstore4 (TYPE##4  data, size_t offset, MOD TYPE *p); \
  void _cl_overloadable vstore8 (TYPE##8  data, size_t offset, MOD TYPE *p); \
  void _cl_overloadable vstore16(TYPE##16 data, size_t offset, MOD TYPE *p);

_CL_DECLARE_VSTORE(char  , __global)
_CL_DECLARE_VSTORE(uchar , __global)
_CL_DECLARE_VSTORE(short , __global)
_CL_DECLARE_VSTORE(ushort, __global)
_CL_DECLARE_VSTORE(int   , __global)
_CL_DECLARE_VSTORE(uint  , __global)
#ifdef cles_khr_int64
_CL_DECLARE_VSTORE(long  , __global)
_CL_DECLARE_VSTORE(ulong , __global)
#endif
_CL_DECLARE_VSTORE(float , __global)
#ifdef cl_khr_fp64
_CL_DECLARE_VSTORE(double, __global)
#endif

_CL_DECLARE_VSTORE(char  , __local)
_CL_DECLARE_VSTORE(uchar , __local)
_CL_DECLARE_VSTORE(short , __local)
_CL_DECLARE_VSTORE(ushort, __local)
_CL_DECLARE_VSTORE(int   , __local)
_CL_DECLARE_VSTORE(uint  , __local)
#ifdef cles_khr_int64
_CL_DECLARE_VSTORE(long  , __local)
_CL_DECLARE_VSTORE(ulong , __local)
#endif
_CL_DECLARE_VSTORE(float , __local)
#ifdef cl_khr_fp64
_CL_DECLARE_VSTORE(double, __local)
#endif

_CL_DECLARE_VSTORE(char  , __private)
_CL_DECLARE_VSTORE(uchar , __private)
_CL_DECLARE_VSTORE(short , __private)
_CL_DECLARE_VSTORE(ushort, __private)
_CL_DECLARE_VSTORE(int   , __private)
_CL_DECLARE_VSTORE(uint  , __private)
#ifdef cles_khr_int64
_CL_DECLARE_VSTORE(long  , __private)
_CL_DECLARE_VSTORE(ulong , __private)
#endif
_CL_DECLARE_VSTORE(float , __private)
#ifdef cl_khr_fp64
_CL_DECLARE_VSTORE(double, __private)
#endif

#ifdef cl_khr_fp16

#define _CL_DECLARE_VLOAD_HALF(MOD)                                     \
  float   _cl_overloadable vload_half   (size_t offset, const MOD half *p); \
  float2  _cl_overloadable vload_half2  (size_t offset, const MOD half *p); \
  float3  _cl_overloadable vload_half3  (size_t offset, const MOD half *p); \
  float4  _cl_overloadable vload_half4  (size_t offset, const MOD half *p); \
  float8  _cl_overloadable vload_half8  (size_t offset, const MOD half *p); \
  float16 _cl_overloadable vload_half16 (size_t offset, const MOD half *p); \
  float2  _cl_overloadable vloada_half2 (size_t offset, const MOD half *p); \
  float3  _cl_overloadable vloada_half3 (size_t offset, const MOD half *p); \
  float4  _cl_overloadable vloada_half4 (size_t offset, const MOD half *p); \
  float8  _cl_overloadable vloada_half8 (size_t offset, const MOD half *p); \
  float16 _cl_overloadable vloada_half16(size_t offset, const MOD half *p);

_CL_DECLARE_VLOAD_HALF(__global)
_CL_DECLARE_VLOAD_HALF(__local)
_CL_DECLARE_VLOAD_HALF(__constant)
_CL_DECLARE_VLOAD_HALF(__private)

/* stores to half may have a suffix: _rte _rtz _rtp _rtn */
#define _CL_DECLARE_VSTORE_HALF(MOD, SUFFIX)                            \
  void _cl_overloadable vstore_half##SUFFIX   (float   data, size_t offset, MOD half *p); \
  void _cl_overloadable vstore_half2##SUFFIX  (float2  data, size_t offset, MOD half *p); \
  void _cl_overloadable vstore_half3##SUFFIX  (float3  data, size_t offset, MOD half *p); \
  void _cl_overloadable vstore_half4##SUFFIX  (float4  data, size_t offset, MOD half *p); \
  void _cl_overloadable vstore_half8##SUFFIX  (float8  data, size_t offset, MOD half *p); \
  void _cl_overloadable vstore_half16##SUFFIX (float16 data, size_t offset, MOD half *p); \
  void _cl_overloadable vstorea_half2##SUFFIX (float2  data, size_t offset, MOD half *p); \
  void _cl_overloadable vstorea_half3##SUFFIX (float3  data, size_t offset, MOD half *p); \
  void _cl_overloadable vstorea_half4##SUFFIX (float4  data, size_t offset, MOD half *p); \
  void _cl_overloadable vstorea_half8##SUFFIX (float8  data, size_t offset, MOD half *p); \
  void _cl_overloadable vstorea_half16##SUFFIX(float16 data, size_t offset, MOD half *p);

_CL_DECLARE_VSTORE_HALF(__global  ,     )
_CL_DECLARE_VSTORE_HALF(__global  , _rte)
_CL_DECLARE_VSTORE_HALF(__global  , _rtz)
_CL_DECLARE_VSTORE_HALF(__global  , _rtp)
_CL_DECLARE_VSTORE_HALF(__global  , _rtn)
_CL_DECLARE_VSTORE_HALF(__local   ,     )
_CL_DECLARE_VSTORE_HALF(__local   , _rte)
_CL_DECLARE_VSTORE_HALF(__local   , _rtz)
_CL_DECLARE_VSTORE_HALF(__local   , _rtp)
_CL_DECLARE_VSTORE_HALF(__local   , _rtn)
_CL_DECLARE_VSTORE_HALF(__private ,     )
_CL_DECLARE_VSTORE_HALF(__private , _rte)
_CL_DECLARE_VSTORE_HALF(__private , _rtz)
_CL_DECLARE_VSTORE_HALF(__private , _rtp)
_CL_DECLARE_VSTORE_HALF(__private , _rtn)

#endif


/* Miscellaneous Vector Functions */

// This code leads to an ICE in Clang

// #define _CL_DECLARE_SHUFFLE_2(GTYPE, UGTYPE, STYPE, M)                  \
//   GTYPE##2 _cl_overloadable shuffle(GTYPE##M x, UGTYPE##2 mask)         \
//   {                                                                     \
//     UGTYPE bits = (UGTYPE)1 << (UGTYPE)M;                               \
//     UGTYPE bmask = bits - (UGTYPE)1;                                    \
//     return __builtin_shufflevector(x, x,                                \
//                                    mask.s0 & bmask, mask.s1 & bmask);   \
//   }
// #define _CL_DECLARE_SHUFFLE_3(GTYPE, UGTYPE, STYPE, M)                  \
//   GTYPE##3 _cl_overloadable shuffle(GTYPE##M x, UGTYPE##3 mask)         \
//   {                                                                     \
//     UGTYPE bits = (UGTYPE)1 << (UGTYPE)M;                               \
//     UGTYPE bmask = bits - (UGTYPE)1;                                    \
//     return __builtin_shufflevector(x, x,                                \
//                                    mask.s0 & bmask, mask.s1 & bmask,    \
//                                    mask.s2 & bmask);                    \
//   }
// #define _CL_DECLARE_SHUFFLE_4(GTYPE, UGTYPE, STYPE, M)                  \
//   GTYPE##4 _cl_overloadable shuffle(GTYPE##M x, UGTYPE##4 mask)         \
//   {                                                                     \
//     UGTYPE bits = (UGTYPE)1 << (UGTYPE)M;                               \
//     UGTYPE bmask = bits - (UGTYPE)1;                                    \
//     return __builtin_shufflevector(x, x,                                \
//                                    mask.s0 & bmask, mask.s1 & bmask,    \
//                                    mask.s2 & bmask, mask.s3 & bmask);   \
//   }
// #define _CL_DECLARE_SHUFFLE_8(GTYPE, UGTYPE, STYPE, M)                  \
//   GTYPE##8 _cl_overloadable shuffle(GTYPE##M x, UGTYPE##8 mask)         \
//   {                                                                     \
//     UGTYPE bits = (UGTYPE)1 << (UGTYPE)M;                               \
//     UGTYPE bmask = bits - (UGTYPE)1;                                    \
//     return __builtin_shufflevector(x, x,                                \
//                                    mask.s0 & bmask, mask.s1 & bmask,    \
//                                    mask.s2 & bmask, mask.s3 & bmask,    \
//                                    mask.s4 & bmask, mask.s5 & bmask,    \
//                                    mask.s6 & bmask, mask.s7 & bmask);   \
//   }
// #define _CL_DECLARE_SHUFFLE_16(GTYPE, UGTYPE, STYPE, M)                 \
//   GTYPE##16 _cl_overloadable shuffle(GTYPE##M x, UGTYPE##16 mask)       \
//   {                                                                     \
//     UGTYPE bits = (UGTYPE)1 << (UGTYPE)M;                               \
//     UGTYPE bmask = bits - (UGTYPE)1;                                    \
//     return __builtin_shufflevector(x, x,                                \
//                                    mask.s0 & bmask, mask.s1 & bmask,    \
//                                    mask.s2 & bmask, mask.s3 & bmask,    \
//                                    mask.s4 & bmask, mask.s5 & bmask,    \
//                                    mask.s6 & bmask, mask.s7 & bmask,    \
//                                    mask.s8 & bmask, mask.s9 & bmask,    \
//                                    mask.sa & bmask, mask.sb & bmask,    \
//                                    mask.sc & bmask, mask.sd & bmask,    \
//                                    mask.se & bmask, mask.sf & bmask);   \
//   }
// 
// #define _CL_DECLARE_SHUFFLE(GTYPE, UGTYPE, STYPE, M)    \
//   _CL_DECLARE_SHUFFLE_2 (GTYPE, UGTYPE, STYPE, M)       \
//   _CL_DECLARE_SHUFFLE_3 (GTYPE, UGTYPE, STYPE, M)       \
//   _CL_DECLARE_SHUFFLE_4 (GTYPE, UGTYPE, STYPE, M)       \
//   _CL_DECLARE_SHUFFLE_8 (GTYPE, UGTYPE, STYPE, M)       \
//   _CL_DECLARE_SHUFFLE_16(GTYPE, UGTYPE, STYPE, M)
// 
// _CL_DECLARE_SHUFFLE(char  , uchar , char  , 2 )
// _CL_DECLARE_SHUFFLE(char  , uchar , char  , 3 )
// _CL_DECLARE_SHUFFLE(char  , uchar , char  , 4 )
// _CL_DECLARE_SHUFFLE(char  , uchar , char  , 8 )
// _CL_DECLARE_SHUFFLE(char  , uchar , char  , 16)
// _CL_DECLARE_SHUFFLE(uchar , uchar , char  , 2 )
// _CL_DECLARE_SHUFFLE(uchar , uchar , char  , 3 )
// _CL_DECLARE_SHUFFLE(uchar , uchar , char  , 4 )
// _CL_DECLARE_SHUFFLE(uchar , uchar , char  , 8 )
// _CL_DECLARE_SHUFFLE(uchar , uchar , char  , 16)
// _CL_DECLARE_SHUFFLE(short , ushort, short , 2 )
// _CL_DECLARE_SHUFFLE(short , ushort, short , 3 )
// _CL_DECLARE_SHUFFLE(short , ushort, short , 4 )
// _CL_DECLARE_SHUFFLE(short , ushort, short , 8 )
// _CL_DECLARE_SHUFFLE(short , ushort, short , 16)
// _CL_DECLARE_SHUFFLE(ushort, ushort, short , 2 )
// _CL_DECLARE_SHUFFLE(ushort, ushort, short , 3 )
// _CL_DECLARE_SHUFFLE(ushort, ushort, short , 4 )
// _CL_DECLARE_SHUFFLE(ushort, ushort, short , 8 )
// _CL_DECLARE_SHUFFLE(ushort, ushort, short , 16)
// _CL_DECLARE_SHUFFLE(int   , uint  , int   , 2 )
// _CL_DECLARE_SHUFFLE(int   , uint  , int   , 3 )
// _CL_DECLARE_SHUFFLE(int   , uint  , int   , 4 )
// _CL_DECLARE_SHUFFLE(int   , uint  , int   , 8 )
// _CL_DECLARE_SHUFFLE(int   , uint  , int   , 16)
// _CL_DECLARE_SHUFFLE(uint  , uint  , int   , 2 )
// _CL_DECLARE_SHUFFLE(uint  , uint  , int   , 3 )
// _CL_DECLARE_SHUFFLE(uint  , uint  , int   , 4 )
// _CL_DECLARE_SHUFFLE(uint  , uint  , int   , 8 )
// _CL_DECLARE_SHUFFLE(uint  , uint  , int   , 16)
// _CL_DECLARE_SHUFFLE(long  , ulong , long  , 2 )
// _CL_DECLARE_SHUFFLE(long  , ulong , long  , 3 )
// _CL_DECLARE_SHUFFLE(long  , ulong , long  , 4 )
// _CL_DECLARE_SHUFFLE(long  , ulong , long  , 8 )
// _CL_DECLARE_SHUFFLE(long  , ulong , long  , 16)
// _CL_DECLARE_SHUFFLE(ulong , ulong , long  , 2 )
// _CL_DECLARE_SHUFFLE(ulong , ulong , long  , 3 )
// _CL_DECLARE_SHUFFLE(ulong , ulong , long  , 4 )
// _CL_DECLARE_SHUFFLE(ulong , ulong , long  , 8 )
// _CL_DECLARE_SHUFFLE(ulong , ulong , long  , 16)
// _CL_DECLARE_SHUFFLE(float , uint  , float , 2 )
// _CL_DECLARE_SHUFFLE(float , uint  , float , 3 )
// _CL_DECLARE_SHUFFLE(float , uint  , float , 4 )
// _CL_DECLARE_SHUFFLE(float , uint  , float , 8 )
// _CL_DECLARE_SHUFFLE(float , uint  , float , 16)
// _CL_DECLARE_SHUFFLE(double, ulong , double, 2 )
// _CL_DECLARE_SHUFFLE(double, ulong , double, 3 )
// _CL_DECLARE_SHUFFLE(double, ulong , double, 4 )
// _CL_DECLARE_SHUFFLE(double, ulong , double, 8 )
// _CL_DECLARE_SHUFFLE(double, ulong , double, 16)

// shuffle2


int printf(const /*constant*/ char * restrict format, ...)
  __attribute__((format(printf, 1, 2)));


/* Async Copies from Global to Local Memory, Local to
   Global Memory, and Prefetch */

typedef uint event_t;

#define _CL_DECLARE_ASYNC_COPY_FUNCS_SINGLE(GENTYPE)            \
  _cl_overloadable                                              \
  event_t async_work_group_copy (__local GENTYPE *dst,          \
                                 const __global GENTYPE *src,   \
                                 size_t num_gentypes,           \
                                 event_t event);                \
                                                                \
  _cl_overloadable                                              \
  event_t async_work_group_copy (__global GENTYPE *dst,         \
                                 const __local GENTYPE *src,    \
                                 size_t num_gentypes,           \
                                 event_t event);                \
                                                                
void wait_group_events (int num_events,                      
                        event_t *event_list);                 

#define _CL_DECLARE_ASYNC_COPY_FUNCS(GENTYPE)      \
  _CL_DECLARE_ASYNC_COPY_FUNCS_SINGLE(GENTYPE)     \
  _CL_DECLARE_ASYNC_COPY_FUNCS_SINGLE(GENTYPE##2)   \
  _CL_DECLARE_ASYNC_COPY_FUNCS_SINGLE(GENTYPE##3)   \
  _CL_DECLARE_ASYNC_COPY_FUNCS_SINGLE(GENTYPE##4)   \
  _CL_DECLARE_ASYNC_COPY_FUNCS_SINGLE(GENTYPE##8)   \
  _CL_DECLARE_ASYNC_COPY_FUNCS_SINGLE(GENTYPE##16)  \

_CL_DECLARE_ASYNC_COPY_FUNCS(char);
_CL_DECLARE_ASYNC_COPY_FUNCS(uchar);
_CL_DECLARE_ASYNC_COPY_FUNCS(short);
_CL_DECLARE_ASYNC_COPY_FUNCS(ushort);
_CL_DECLARE_ASYNC_COPY_FUNCS(int);
_CL_DECLARE_ASYNC_COPY_FUNCS(uint);
__IF_INT64(_CL_DECLARE_ASYNC_COPY_FUNCS(long));
__IF_INT64(_CL_DECLARE_ASYNC_COPY_FUNCS(ulong));

__IF_FP16(_CL_DECLARE_ASYNC_COPY_FUNCS_SINGLE(half));
_CL_DECLARE_ASYNC_COPY_FUNCS(float);
__IF_FP64(_CL_DECLARE_ASYNC_COPY_FUNCS(double));

// Image support

typedef int sampler_t;

#define CLK_ADDRESS_NONE                0x00
#define CLK_ADDRESS_MIRRORED_REPEAT     0x01
#define CLK_ADDRESS_REPEAT              0x02
#define CLK_ADDRESS_CLAMP_TO_EDGE       0x03
#define CLK_ADDRESS_CLAMP               0x04

#define CLK_NORMALIZED_COORDS_FALSE     0x00
#define CLK_NORMALIZED_COORDS_TRUE      0x08

#define CLK_FILTER_NEAREST              0x00
#define CLK_FILTER_LINEAR               0x10

typedef struct image2d_t_* image2d_t;

float4 _cl_overloadable read_imagef( image2d_t image,
        sampler_t sampler,
        int2 coord);

float4 _cl_overloadable read_imagef( image2d_t image,
        sampler_t sampler,
        float2 coord);

void _cl_overloadable write_imagef( image2d_t image,
        int2 coord,
        float4 color);

void _cl_overloadable write_imagei( image2d_t image,
        int2 coord,
        int4 color);

int get_image_width (image2d_t image);
int get_image_height (image2d_t image);
