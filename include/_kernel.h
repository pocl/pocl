/* pocl/_kernel.h - OpenCL types and runtime library
   functions declarations.

   Copyright (c) 2011 Universidad Rey Juan Carlos
   Copyright (c) 2011-2013 Pekka Jääskeläinen / TUT
   Copyright (c) 2011-2013 Erik Schnetter <eschnetter@perimeterinstitute.ca>
                           Perimeter Institute for Theoretical Physics
   
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

/* Language feature detection */
#if (__clang_major__ == 3) && (__clang_minor__ >= 3)
#  define _CL_HAS_EVENT_T
#  define _CL_HAS_IMAGE_ACCESS
#endif
#include "_kernel_c.h"

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
#ifdef cl_khr_int64
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


/* A static assert statement to catch inconsistencies at build time */
#if __has_extension(__c_static_assert__)
#  define _CL_STATIC_ASSERT(_t, _x) _Static_assert(_x, #_t)
#else
#  define _CL_STATIC_ASSERT(_t, _x) typedef int __cl_ai##_t[(x) ? 1 : -1];
#endif

typedef enum {
  CLK_LOCAL_MEM_FENCE = 0x1,
  CLK_GLOBAL_MEM_FENCE = 0x2
} cl_mem_fence_flags;


/* Data types */



#ifdef cl_khr_int64
typedef long long2  __attribute__((__ext_vector_type__(2)));
typedef long long3  __attribute__((__ext_vector_type__(3)));
typedef long long4  __attribute__((__ext_vector_type__(4)));
typedef long long8  __attribute__((__ext_vector_type__(8)));
typedef long long16 __attribute__((__ext_vector_type__(16)));

typedef ulong ulong2  __attribute__((__ext_vector_type__(2)));
typedef ulong ulong3  __attribute__((__ext_vector_type__(3)));
typedef ulong ulong4  __attribute__((__ext_vector_type__(4)));
typedef ulong ulong8  __attribute__((__ext_vector_type__(8)));
typedef ulong ulong16 __attribute__((__ext_vector_type__(16)));
#endif

#ifdef cl_khr_fp16
typedef half half2  __attribute__((__ext_vector_type__(2)));
typedef half half3  __attribute__((__ext_vector_type__(3)));
typedef half half4  __attribute__((__ext_vector_type__(4)));
typedef half half8  __attribute__((__ext_vector_type__(8)));
typedef half half16 __attribute__((__ext_vector_type__(16)));
#endif



#ifdef cl_khr_fp64
typedef double double2  __attribute__((__ext_vector_type__(2)));
typedef double double3  __attribute__((__ext_vector_type__(3)));
typedef double double4  __attribute__((__ext_vector_type__(4)));
typedef double double8  __attribute__((__ext_vector_type__(8)));
typedef double double16 __attribute__((__ext_vector_type__(16)));
#endif

/* Ensure the data types have the right sizes */
_CL_STATIC_ASSERT(char  , sizeof(char  ) == 1);
_CL_STATIC_ASSERT(char2 , sizeof(char2 ) == 2 *sizeof(char));
_CL_STATIC_ASSERT(char3 , sizeof(char3 ) == 4 *sizeof(char));
_CL_STATIC_ASSERT(char4 , sizeof(char4 ) == 4 *sizeof(char));
_CL_STATIC_ASSERT(char8 , sizeof(char8 ) == 8 *sizeof(char));
_CL_STATIC_ASSERT(char16, sizeof(char16) == 16*sizeof(char));

_CL_STATIC_ASSERT(uchar , sizeof(uchar ) == 1);
_CL_STATIC_ASSERT(uchar2 , sizeof(uchar2 ) == 2 *sizeof(uchar));
_CL_STATIC_ASSERT(uchar3 , sizeof(uchar3 ) == 4 *sizeof(uchar));
_CL_STATIC_ASSERT(uchar4 , sizeof(uchar4 ) == 4 *sizeof(uchar));
_CL_STATIC_ASSERT(uchar8 , sizeof(uchar8 ) == 8 *sizeof(uchar));
_CL_STATIC_ASSERT(uchar16, sizeof(uchar16) == 16*sizeof(uchar));

_CL_STATIC_ASSERT(short , sizeof(short ) == 2);
_CL_STATIC_ASSERT(short2 , sizeof(short2 ) == 2 *sizeof(short));
_CL_STATIC_ASSERT(short3 , sizeof(short3 ) == 4 *sizeof(short));
_CL_STATIC_ASSERT(short4 , sizeof(short4 ) == 4 *sizeof(short));
_CL_STATIC_ASSERT(short8 , sizeof(short8 ) == 8 *sizeof(short));
_CL_STATIC_ASSERT(short16, sizeof(short16) == 16*sizeof(short));

_CL_STATIC_ASSERT(ushort, sizeof(ushort) == 2);
_CL_STATIC_ASSERT(ushort2 , sizeof(ushort2 ) == 2 *sizeof(ushort));
_CL_STATIC_ASSERT(ushort3 , sizeof(ushort3 ) == 4 *sizeof(ushort));
_CL_STATIC_ASSERT(ushort4 , sizeof(ushort4 ) == 4 *sizeof(ushort));
_CL_STATIC_ASSERT(ushort8 , sizeof(ushort8 ) == 8 *sizeof(ushort));
_CL_STATIC_ASSERT(ushort16, sizeof(ushort16) == 16*sizeof(ushort));

_CL_STATIC_ASSERT(int   , sizeof(int   ) == 4);
_CL_STATIC_ASSERT(int2 , sizeof(int2 ) == 2 *sizeof(int));
_CL_STATIC_ASSERT(int3 , sizeof(int3 ) == 4 *sizeof(int));
_CL_STATIC_ASSERT(int4 , sizeof(int4 ) == 4 *sizeof(int));
_CL_STATIC_ASSERT(int8 , sizeof(int8 ) == 8 *sizeof(int));
_CL_STATIC_ASSERT(int16, sizeof(int16) == 16*sizeof(int));

_CL_STATIC_ASSERT(uint  , sizeof(uint  ) == 4);
_CL_STATIC_ASSERT(uint2 , sizeof(uint2 ) == 2 *sizeof(uint));
_CL_STATIC_ASSERT(uint3 , sizeof(uint3 ) == 4 *sizeof(uint));
_CL_STATIC_ASSERT(uint4 , sizeof(uint4 ) == 4 *sizeof(uint));
_CL_STATIC_ASSERT(uint8 , sizeof(uint8 ) == 8 *sizeof(uint));
_CL_STATIC_ASSERT(uint16, sizeof(uint16) == 16*sizeof(uint));

#ifdef cl_khr_int64 
_CL_STATIC_ASSERT(long  , sizeof(long  ) == 8);
_CL_STATIC_ASSERT(long2 , sizeof(long2 ) == 2 *sizeof(long));
_CL_STATIC_ASSERT(long3 , sizeof(long3 ) == 4 *sizeof(long));
_CL_STATIC_ASSERT(long4 , sizeof(long4 ) == 4 *sizeof(long));
_CL_STATIC_ASSERT(long8 , sizeof(long8 ) == 8 *sizeof(long));
_CL_STATIC_ASSERT(long16, sizeof(long16) == 16*sizeof(long));

_CL_STATIC_ASSERT(ulong  , sizeof(ulong  ) == 8);
_CL_STATIC_ASSERT(ulong2 , sizeof(ulong2 ) == 2 *sizeof(ulong));
_CL_STATIC_ASSERT(ulong3 , sizeof(ulong3 ) == 4 *sizeof(ulong));
_CL_STATIC_ASSERT(ulong4 , sizeof(ulong4 ) == 4 *sizeof(ulong));
_CL_STATIC_ASSERT(ulong8 , sizeof(ulong8 ) == 8 *sizeof(ulong));
_CL_STATIC_ASSERT(ulong16, sizeof(ulong16) == 16*sizeof(ulong));
#endif

#ifdef cl_khr_fp16
_CL_STATIC_ASSERT(half, sizeof(half) == 2);
_CL_STATIC_ASSERT(half2 , sizeof(half2 ) == 2 *sizeof(half));
_CL_STATIC_ASSERT(half3 , sizeof(half3 ) == 4 *sizeof(half));
_CL_STATIC_ASSERT(half4 , sizeof(half4 ) == 4 *sizeof(half));
_CL_STATIC_ASSERT(half8 , sizeof(half8 ) == 8 *sizeof(half));
_CL_STATIC_ASSERT(half16, sizeof(half16) == 16*sizeof(half));
#endif

_CL_STATIC_ASSERT(float , sizeof(float ) == 4);
_CL_STATIC_ASSERT(float2 , sizeof(float2 ) == 2 *sizeof(float));
_CL_STATIC_ASSERT(float3 , sizeof(float3 ) == 4 *sizeof(float));
_CL_STATIC_ASSERT(float4 , sizeof(float4 ) == 4 *sizeof(float));
_CL_STATIC_ASSERT(float8 , sizeof(float8 ) == 8 *sizeof(float));
_CL_STATIC_ASSERT(float16, sizeof(float16) == 16*sizeof(float));

#ifdef cl_khr_fp64
_CL_STATIC_ASSERT(double, sizeof(double) == 8);
_CL_STATIC_ASSERT(double2 , sizeof(double2 ) == 2 *sizeof(double));
_CL_STATIC_ASSERT(double3 , sizeof(double3 ) == 4 *sizeof(double));
_CL_STATIC_ASSERT(double4 , sizeof(double4 ) == 4 *sizeof(double));
_CL_STATIC_ASSERT(double8 , sizeof(double8 ) == 8 *sizeof(double));
_CL_STATIC_ASSERT(double16, sizeof(double16) == 16*sizeof(double));
#endif

_CL_STATIC_ASSERT(size_t, sizeof(size_t) == sizeof(void*));
_CL_STATIC_ASSERT(ptrdiff_t, sizeof(ptrdiff_t) == sizeof(void*));
_CL_STATIC_ASSERT(intptr_t, sizeof(intptr_t) == sizeof(void*));
_CL_STATIC_ASSERT(uintptr_t, sizeof(uintptr_t) == sizeof(void*));


/* Conversion functions */

#define _CL_DECLARE_AS_TYPE(SRC, DST)           \
  DST _CL_OVERLOADABLE as_##DST(SRC a);

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
  _CL_DECLARE_AS_TYPE(SRC, char3)               \
  _CL_DECLARE_AS_TYPE(SRC, uchar3)              \
  _CL_DECLARE_AS_TYPE(SRC, short2)              \
  _CL_DECLARE_AS_TYPE(SRC, ushort2)             \
  _CL_DECLARE_AS_TYPE(SRC, int)                 \
  _CL_DECLARE_AS_TYPE(SRC, uint)                \
  _CL_DECLARE_AS_TYPE(SRC, float)
_CL_DECLARE_AS_TYPE_4(char4)
_CL_DECLARE_AS_TYPE_4(uchar4)
_CL_DECLARE_AS_TYPE_4(char3)
_CL_DECLARE_AS_TYPE_4(uchar3)
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
  _CL_DECLARE_AS_TYPE(SRC, short3)              \
  _CL_DECLARE_AS_TYPE(SRC, ushort3)             \
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
_CL_DECLARE_AS_TYPE_8(short3)
_CL_DECLARE_AS_TYPE_8(ushort3)
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
  _CL_DECLARE_AS_TYPE(SRC, int3)                \
  _CL_DECLARE_AS_TYPE(SRC, uint3)               \
  __IF_INT64(_CL_DECLARE_AS_TYPE(SRC, long2))   \
  __IF_INT64(_CL_DECLARE_AS_TYPE(SRC, ulong2))  \
  _CL_DECLARE_AS_TYPE(SRC, float4)              \
  _CL_DECLARE_AS_TYPE(SRC, float3)              \
  __IF_FP64(_CL_DECLARE_AS_TYPE(SRC, double2))
_CL_DECLARE_AS_TYPE_16(char16)
_CL_DECLARE_AS_TYPE_16(uchar16)
_CL_DECLARE_AS_TYPE_16(short8)
_CL_DECLARE_AS_TYPE_16(ushort8)
_CL_DECLARE_AS_TYPE_16(int4)
_CL_DECLARE_AS_TYPE_16(uint4)
_CL_DECLARE_AS_TYPE_16(int3)
_CL_DECLARE_AS_TYPE_16(uint3)
__IF_INT64(_CL_DECLARE_AS_TYPE_16(long2))
__IF_INT64(_CL_DECLARE_AS_TYPE_16(ulong2))
_CL_DECLARE_AS_TYPE_16(float4)
_CL_DECLARE_AS_TYPE_16(float3)
__IF_FP64(_CL_DECLARE_AS_TYPE_16(double2))

/* 32 bytes */
#define _CL_DECLARE_AS_TYPE_32(SRC)             \
  _CL_DECLARE_AS_TYPE(SRC, short16)             \
  _CL_DECLARE_AS_TYPE(SRC, ushort16)            \
  _CL_DECLARE_AS_TYPE(SRC, int8)                \
  _CL_DECLARE_AS_TYPE(SRC, uint8)               \
  __IF_INT64(_CL_DECLARE_AS_TYPE(SRC, long4))   \
  __IF_INT64(_CL_DECLARE_AS_TYPE(SRC, ulong4))  \
  __IF_INT64(_CL_DECLARE_AS_TYPE(SRC, long3))   \
  __IF_INT64(_CL_DECLARE_AS_TYPE(SRC, ulong3))  \
  _CL_DECLARE_AS_TYPE(SRC, float8)              \
  __IF_FP64(_CL_DECLARE_AS_TYPE(SRC, double4))  \
  __IF_FP64(_CL_DECLARE_AS_TYPE(SRC, double3))
_CL_DECLARE_AS_TYPE_32(short16)
_CL_DECLARE_AS_TYPE_32(ushort16)
_CL_DECLARE_AS_TYPE_32(int8)
_CL_DECLARE_AS_TYPE_32(uint8)
__IF_INT64(_CL_DECLARE_AS_TYPE_32(long4))
__IF_INT64(_CL_DECLARE_AS_TYPE_32(ulong4))
__IF_INT64(_CL_DECLARE_AS_TYPE_32(long3))
__IF_INT64(_CL_DECLARE_AS_TYPE_32(ulong3))
_CL_DECLARE_AS_TYPE_32(float8)
__IF_FP64(_CL_DECLARE_AS_TYPE_32(double4))
__IF_FP64(_CL_DECLARE_AS_TYPE_32(double3))

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

/* Conversions between builtin types.
 * 
 * Even though the OpenCL specification isn't entirely clear on this
 * matter, we implement all rounding mode combinations even for
 * integer-to-integer conversions. The rounding mode is essentially
 * redundant and thus ignored.
 *
 * Other OpenCL implementations seem to allow this in user code, and some
 * of the test suites/benchmarks out in the wild expect these functions
 * are available.
 *
 * Saturating conversions are only allowed when the destination type
 * is an integer.
 */

#define _CL_DECLARE_CONVERT_TYPE(SRC, DST, SIZE, INTSUFFIX, FLOATSUFFIX) \
  DST##SIZE _CL_OVERLOADABLE                                             \
  convert_##DST##SIZE##INTSUFFIX##FLOATSUFFIX(SRC##SIZE a);

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

#define _CL_DECLARE_CONVERT_TYPE_SRC_DST(SIZE, FLOATSUFFIX) \
  _CL_DECLARE_CONVERT_TYPE_DST(char  , SIZE, FLOATSUFFIX)   \
  _CL_DECLARE_CONVERT_TYPE_DST(uchar , SIZE, FLOATSUFFIX)   \
  _CL_DECLARE_CONVERT_TYPE_DST(short , SIZE, FLOATSUFFIX)   \
  _CL_DECLARE_CONVERT_TYPE_DST(ushort, SIZE, FLOATSUFFIX)   \
  _CL_DECLARE_CONVERT_TYPE_DST(int   , SIZE, FLOATSUFFIX)   \
  _CL_DECLARE_CONVERT_TYPE_DST(uint  , SIZE, FLOATSUFFIX)   \
  __IF_INT64(                                               \
  _CL_DECLARE_CONVERT_TYPE_DST(long  , SIZE, FLOATSUFFIX)   \
  _CL_DECLARE_CONVERT_TYPE_DST(ulong , SIZE, FLOATSUFFIX))  \
  _CL_DECLARE_CONVERT_TYPE_DST(float , SIZE, FLOATSUFFIX)   \
  __IF_FP64(                                                \
  _CL_DECLARE_CONVERT_TYPE_DST(double, SIZE, FLOATSUFFIX))

#define _CL_DECLARE_CONVERT_TYPE_SRC_DST_SIZE(FLOATSUFFIX) \
  _CL_DECLARE_CONVERT_TYPE_SRC_DST(  , FLOATSUFFIX)   \
  _CL_DECLARE_CONVERT_TYPE_SRC_DST( 2, FLOATSUFFIX)   \
  _CL_DECLARE_CONVERT_TYPE_SRC_DST( 3, FLOATSUFFIX)   \
  _CL_DECLARE_CONVERT_TYPE_SRC_DST( 4, FLOATSUFFIX)   \
  _CL_DECLARE_CONVERT_TYPE_SRC_DST( 8, FLOATSUFFIX)   \
  _CL_DECLARE_CONVERT_TYPE_SRC_DST(16, FLOATSUFFIX)

_CL_DECLARE_CONVERT_TYPE_SRC_DST_SIZE(    )
_CL_DECLARE_CONVERT_TYPE_SRC_DST_SIZE(_rtz)
_CL_DECLARE_CONVERT_TYPE_SRC_DST_SIZE(_rte)
_CL_DECLARE_CONVERT_TYPE_SRC_DST_SIZE(_rtp)
_CL_DECLARE_CONVERT_TYPE_SRC_DST_SIZE(_rtn)


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

#define FP_ILOGB0   INT_MIN
#define FP_ILOGBNAN INT_MAX

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
  float    _CL_OVERLOADABLE NAME(float   );     \
  float2   _CL_OVERLOADABLE NAME(float2  );     \
  float3   _CL_OVERLOADABLE NAME(float3  );     \
  float4   _CL_OVERLOADABLE NAME(float4  );     \
  float8   _CL_OVERLOADABLE NAME(float8  );     \
  float16  _CL_OVERLOADABLE NAME(float16 );     \
  __IF_FP64(                                    \
  double   _CL_OVERLOADABLE NAME(double  );     \
  double2  _CL_OVERLOADABLE NAME(double2 );     \
  double3  _CL_OVERLOADABLE NAME(double3 );     \
  double4  _CL_OVERLOADABLE NAME(double4 );     \
  double8  _CL_OVERLOADABLE NAME(double8 );     \
  double16 _CL_OVERLOADABLE NAME(double16);)
#define _CL_DECLARE_FUNC_V_VV(NAME)                     \
  float    _CL_OVERLOADABLE NAME(float   , float   );   \
  float2   _CL_OVERLOADABLE NAME(float2  , float2  );   \
  float3   _CL_OVERLOADABLE NAME(float3  , float3  );   \
  float4   _CL_OVERLOADABLE NAME(float4  , float4  );   \
  float8   _CL_OVERLOADABLE NAME(float8  , float8  );   \
  float16  _CL_OVERLOADABLE NAME(float16 , float16 );   \
  __IF_FP64(                                            \
  double   _CL_OVERLOADABLE NAME(double  , double  );   \
  double2  _CL_OVERLOADABLE NAME(double2 , double2 );   \
  double3  _CL_OVERLOADABLE NAME(double3 , double3 );   \
  double4  _CL_OVERLOADABLE NAME(double4 , double4 );   \
  double8  _CL_OVERLOADABLE NAME(double8 , double8 );   \
  double16 _CL_OVERLOADABLE NAME(double16, double16);)
#define _CL_DECLARE_FUNC_V_VVV(NAME)                                    \
  float    _CL_OVERLOADABLE NAME(float   , float   , float   );         \
  float2   _CL_OVERLOADABLE NAME(float2  , float2  , float2  );         \
  float3   _CL_OVERLOADABLE NAME(float3  , float3  , float3  );         \
  float4   _CL_OVERLOADABLE NAME(float4  , float4  , float4  );         \
  float8   _CL_OVERLOADABLE NAME(float8  , float8  , float8  );         \
  float16  _CL_OVERLOADABLE NAME(float16 , float16 , float16 );         \
  __IF_FP64(                                                            \
  double   _CL_OVERLOADABLE NAME(double  , double  , double  );         \
  double2  _CL_OVERLOADABLE NAME(double2 , double2 , double2 );         \
  double3  _CL_OVERLOADABLE NAME(double3 , double3 , double3 );         \
  double4  _CL_OVERLOADABLE NAME(double4 , double4 , double4 );         \
  double8  _CL_OVERLOADABLE NAME(double8 , double8 , double8 );         \
  double16 _CL_OVERLOADABLE NAME(double16, double16, double16);)
#define _CL_DECLARE_FUNC_V_VVS(NAME)                            \
  float2   _CL_OVERLOADABLE NAME(float2  , float2  , float );   \
  float3   _CL_OVERLOADABLE NAME(float3  , float3  , float );   \
  float4   _CL_OVERLOADABLE NAME(float4  , float4  , float );   \
  float8   _CL_OVERLOADABLE NAME(float8  , float8  , float );   \
  float16  _CL_OVERLOADABLE NAME(float16 , float16 , float );   \
  __IF_FP64(                                                    \
  double2  _CL_OVERLOADABLE NAME(double2 , double2 , double);   \
  double3  _CL_OVERLOADABLE NAME(double3 , double3 , double);   \
  double4  _CL_OVERLOADABLE NAME(double4 , double4 , double);   \
  double8  _CL_OVERLOADABLE NAME(double8 , double8 , double);   \
  double16 _CL_OVERLOADABLE NAME(double16, double16, double);)
#define _CL_DECLARE_FUNC_V_VSS(NAME)                            \
  float2   _CL_OVERLOADABLE NAME(float2  , float , float );     \
  float3   _CL_OVERLOADABLE NAME(float3  , float , float );     \
  float4   _CL_OVERLOADABLE NAME(float4  , float , float );     \
  float8   _CL_OVERLOADABLE NAME(float8  , float , float );     \
  float16  _CL_OVERLOADABLE NAME(float16 , float , float );     \
  __IF_FP64(                                                    \
  double2  _CL_OVERLOADABLE NAME(double2 , double, double);     \
  double3  _CL_OVERLOADABLE NAME(double3 , double, double);     \
  double4  _CL_OVERLOADABLE NAME(double4 , double, double);     \
  double8  _CL_OVERLOADABLE NAME(double8 , double, double);     \
  double16 _CL_OVERLOADABLE NAME(double16, double, double);)
#define _CL_DECLARE_FUNC_V_SSV(NAME)                            \
  float2   _CL_OVERLOADABLE NAME(float , float , float2  );     \
  float3   _CL_OVERLOADABLE NAME(float , float , float3  );     \
  float4   _CL_OVERLOADABLE NAME(float , float , float4  );     \
  float8   _CL_OVERLOADABLE NAME(float , float , float8  );     \
  float16  _CL_OVERLOADABLE NAME(float , float , float16 );     \
  __IF_FP64(                                                    \
  double2  _CL_OVERLOADABLE NAME(double, double, double2 );     \
  double3  _CL_OVERLOADABLE NAME(double, double, double3 );     \
  double4  _CL_OVERLOADABLE NAME(double, double, double4 );     \
  double8  _CL_OVERLOADABLE NAME(double, double, double8 );     \
  double16 _CL_OVERLOADABLE NAME(double, double, double16);)
#define _CL_DECLARE_FUNC_V_VVJ(NAME)                                    \
  __IF_FP16(                                                            \
  /*half     _CL_OVERLOADABLE NAME(half    , half    , short  );*/      \
  half2    _CL_OVERLOADABLE NAME(half2   , half2   , short2 );          \
  half3    _CL_OVERLOADABLE NAME(half3   , half3   , short3 );          \
  half4    _CL_OVERLOADABLE NAME(half4   , half4   , short4 );          \
  half8    _CL_OVERLOADABLE NAME(half8   , half8   , short8 );          \
  half16   _CL_OVERLOADABLE NAME(half16  , half16  , short16);)         \
  float    _CL_OVERLOADABLE NAME(float   , float   , int    );          \
  float2   _CL_OVERLOADABLE NAME(float2  , float2  , int2   );          \
  float3   _CL_OVERLOADABLE NAME(float3  , float3  , int3   );          \
  float4   _CL_OVERLOADABLE NAME(float4  , float4  , int4   );          \
  float8   _CL_OVERLOADABLE NAME(float8  , float8  , int8   );          \
  float16  _CL_OVERLOADABLE NAME(float16 , float16 , int16  );          \
  __IF_FP64(                                                            \
  double   _CL_OVERLOADABLE NAME(double  , double  , long   );          \
  double2  _CL_OVERLOADABLE NAME(double2 , double2 , long2  );          \
  double3  _CL_OVERLOADABLE NAME(double3 , double3 , long3  );          \
  double4  _CL_OVERLOADABLE NAME(double4 , double4 , long4  );          \
  double8  _CL_OVERLOADABLE NAME(double8 , double8 , long8  );          \
  double16 _CL_OVERLOADABLE NAME(double16, double16, long16 );)
#define _CL_DECLARE_FUNC_V_VVU(NAME)                                    \
  __IF_FP16(                                                            \
  /*half     _CL_OVERLOADABLE NAME(half    , half    , ushort  );*/     \
  half2    _CL_OVERLOADABLE NAME(half2   , half2   , ushort2 );         \
  half3    _CL_OVERLOADABLE NAME(half3   , half3   , ushort3 );         \
  half4    _CL_OVERLOADABLE NAME(half4   , half4   , ushort4 );         \
  half8    _CL_OVERLOADABLE NAME(half8   , half8   , ushort8 );         \
  half16   _CL_OVERLOADABLE NAME(half16  , half16  , ushort16);)        \
  float    _CL_OVERLOADABLE NAME(float   , float   , uint    );         \
  float2   _CL_OVERLOADABLE NAME(float2  , float2  , uint2   );         \
  float3   _CL_OVERLOADABLE NAME(float3  , float3  , uint3   );         \
  float4   _CL_OVERLOADABLE NAME(float4  , float4  , uint4   );         \
  float8   _CL_OVERLOADABLE NAME(float8  , float8  , uint8   );         \
  float16  _CL_OVERLOADABLE NAME(float16 , float16 , uint16  );         \
  __IF_FP64(                                                            \
  double   _CL_OVERLOADABLE NAME(double  , double  , ulong   );         \
  double2  _CL_OVERLOADABLE NAME(double2 , double2 , ulong2  );         \
  double3  _CL_OVERLOADABLE NAME(double3 , double3 , ulong3  );         \
  double4  _CL_OVERLOADABLE NAME(double4 , double4 , ulong4  );         \
  double8  _CL_OVERLOADABLE NAME(double8 , double8 , ulong8  );         \
  double16 _CL_OVERLOADABLE NAME(double16, double16, ulong16 );)
#define _CL_DECLARE_FUNC_V_U(NAME)              \
  float    _CL_OVERLOADABLE NAME(uint   );      \
  float2   _CL_OVERLOADABLE NAME(uint2  );      \
  float3   _CL_OVERLOADABLE NAME(uint3  );      \
  float4   _CL_OVERLOADABLE NAME(uint4  );      \
  float8   _CL_OVERLOADABLE NAME(uint8  );      \
  float16  _CL_OVERLOADABLE NAME(uint16 );      \
  __IF_FP64(                                    \
  double   _CL_OVERLOADABLE NAME(ulong  );      \
  double2  _CL_OVERLOADABLE NAME(ulong2 );      \
  double3  _CL_OVERLOADABLE NAME(ulong3 );      \
  double4  _CL_OVERLOADABLE NAME(ulong4 );      \
  double8  _CL_OVERLOADABLE NAME(ulong8 );      \
  double16 _CL_OVERLOADABLE NAME(ulong16);)
#define _CL_DECLARE_FUNC_V_VS(NAME)                     \
  float2   _CL_OVERLOADABLE NAME(float2  , float );     \
  float3   _CL_OVERLOADABLE NAME(float3  , float );     \
  float4   _CL_OVERLOADABLE NAME(float4  , float );     \
  float8   _CL_OVERLOADABLE NAME(float8  , float );     \
  float16  _CL_OVERLOADABLE NAME(float16 , float );     \
  __IF_FP64(                                            \
  double2  _CL_OVERLOADABLE NAME(double2 , double);     \
  double3  _CL_OVERLOADABLE NAME(double3 , double);     \
  double4  _CL_OVERLOADABLE NAME(double4 , double);     \
  double8  _CL_OVERLOADABLE NAME(double8 , double);     \
  double16 _CL_OVERLOADABLE NAME(double16, double);)
#define _CL_DECLARE_FUNC_V_VJ(NAME)                     \
  float    _CL_OVERLOADABLE NAME(float   , int  );      \
  float2   _CL_OVERLOADABLE NAME(float2  , int2 );      \
  float3   _CL_OVERLOADABLE NAME(float3  , int3 );      \
  float4   _CL_OVERLOADABLE NAME(float4  , int4 );      \
  float8   _CL_OVERLOADABLE NAME(float8  , int8 );      \
  float16  _CL_OVERLOADABLE NAME(float16 , int16);      \
  __IF_FP64(                                            \
  double   _CL_OVERLOADABLE NAME(double  , int  );      \
  double2  _CL_OVERLOADABLE NAME(double2 , int2 );      \
  double3  _CL_OVERLOADABLE NAME(double3 , int3 );      \
  double4  _CL_OVERLOADABLE NAME(double4 , int4 );      \
  double8  _CL_OVERLOADABLE NAME(double8 , int8 );      \
  double16 _CL_OVERLOADABLE NAME(double16, int16);)
#define _CL_DECLARE_FUNC_J_VV(NAME)                     \
  int    _CL_OVERLOADABLE NAME(float   , float   );     \
  int2   _CL_OVERLOADABLE NAME(float2  , float2  );     \
  int3   _CL_OVERLOADABLE NAME(float3  , float3  );     \
  int4   _CL_OVERLOADABLE NAME(float4  , float4  );     \
  int8   _CL_OVERLOADABLE NAME(float8  , float8  );     \
  int16  _CL_OVERLOADABLE NAME(float16 , float16 );     \
  __IF_FP64(                                            \
  int    _CL_OVERLOADABLE NAME(double  , double  );     \
  long2  _CL_OVERLOADABLE NAME(double2 , double2 );     \
  long3  _CL_OVERLOADABLE NAME(double3 , double3 );     \
  long4  _CL_OVERLOADABLE NAME(double4 , double4 );     \
  long8  _CL_OVERLOADABLE NAME(double8 , double8 );     \
  long16 _CL_OVERLOADABLE NAME(double16, double16);)
#define _CL_DECLARE_FUNC_V_VI(NAME)                     \
  float2   _CL_OVERLOADABLE NAME(float2  , int);        \
  float3   _CL_OVERLOADABLE NAME(float3  , int);        \
  float4   _CL_OVERLOADABLE NAME(float4  , int);        \
  float8   _CL_OVERLOADABLE NAME(float8  , int);        \
  float16  _CL_OVERLOADABLE NAME(float16 , int);        \
  __IF_FP64(                                            \
  double2  _CL_OVERLOADABLE NAME(double2 , int);        \
  double3  _CL_OVERLOADABLE NAME(double3 , int);        \
  double4  _CL_OVERLOADABLE NAME(double4 , int);        \
  double8  _CL_OVERLOADABLE NAME(double8 , int);        \
  double16 _CL_OVERLOADABLE NAME(double16, int);)
#define _CL_DECLARE_FUNC_V_VPV(NAME)                                    \
  float    _CL_OVERLOADABLE NAME(float   , __global  float   *);        \
  float2   _CL_OVERLOADABLE NAME(float2  , __global  float2  *);        \
  float3   _CL_OVERLOADABLE NAME(float3  , __global  float3  *);        \
  float4   _CL_OVERLOADABLE NAME(float4  , __global  float4  *);        \
  float8   _CL_OVERLOADABLE NAME(float8  , __global  float8  *);        \
  float16  _CL_OVERLOADABLE NAME(float16 , __global  float16 *);        \
  __IF_FP64(                                                            \
  double   _CL_OVERLOADABLE NAME(double  , __global  double  *);        \
  double2  _CL_OVERLOADABLE NAME(double2 , __global  double2 *);        \
  double3  _CL_OVERLOADABLE NAME(double3 , __global  double3 *);        \
  double4  _CL_OVERLOADABLE NAME(double4 , __global  double4 *);        \
  double8  _CL_OVERLOADABLE NAME(double8 , __global  double8 *);        \
  double16 _CL_OVERLOADABLE NAME(double16, __global  double16*);)       \
  float    _CL_OVERLOADABLE NAME(float   , __local   float   *);        \
  float2   _CL_OVERLOADABLE NAME(float2  , __local   float2  *);        \
  float3   _CL_OVERLOADABLE NAME(float3  , __local   float3  *);        \
  float4   _CL_OVERLOADABLE NAME(float4  , __local   float4  *);        \
  float8   _CL_OVERLOADABLE NAME(float8  , __local   float8  *);        \
  float16  _CL_OVERLOADABLE NAME(float16 , __local   float16 *);        \
  __IF_FP64(                                                            \
  double   _CL_OVERLOADABLE NAME(double  , __local   double  *);        \
  double2  _CL_OVERLOADABLE NAME(double2 , __local   double2 *);        \
  double3  _CL_OVERLOADABLE NAME(double3 , __local   double3 *);        \
  double4  _CL_OVERLOADABLE NAME(double4 , __local   double4 *);        \
  double8  _CL_OVERLOADABLE NAME(double8 , __local   double8 *);        \
  double16 _CL_OVERLOADABLE NAME(double16, __local   double16*);)       \
  float    _CL_OVERLOADABLE NAME(float   , __private float   *);        \
  float2   _CL_OVERLOADABLE NAME(float2  , __private float2  *);        \
  float3   _CL_OVERLOADABLE NAME(float3  , __private float3  *);        \
  float4   _CL_OVERLOADABLE NAME(float4  , __private float4  *);        \
  float8   _CL_OVERLOADABLE NAME(float8  , __private float8  *);        \
  float16  _CL_OVERLOADABLE NAME(float16 , __private float16 *);        \
  __IF_FP64(                                                            \
  double   _CL_OVERLOADABLE NAME(double  , __private double  *);        \
  double2  _CL_OVERLOADABLE NAME(double2 , __private double2 *);        \
  double3  _CL_OVERLOADABLE NAME(double3 , __private double3 *);        \
  double4  _CL_OVERLOADABLE NAME(double4 , __private double4 *);        \
  double8  _CL_OVERLOADABLE NAME(double8 , __private double8 *);        \
  double16 _CL_OVERLOADABLE NAME(double16, __private double16*);)
#define _CL_DECLARE_FUNC_V_SV(NAME)                     \
  float2   _CL_OVERLOADABLE NAME(float , float2  );     \
  float3   _CL_OVERLOADABLE NAME(float , float3  );     \
  float4   _CL_OVERLOADABLE NAME(float , float4  );     \
  float8   _CL_OVERLOADABLE NAME(float , float8  );     \
  float16  _CL_OVERLOADABLE NAME(float , float16 );     \
  __IF_FP64(                                            \
  double2  _CL_OVERLOADABLE NAME(double, double2 );     \
  double3  _CL_OVERLOADABLE NAME(double, double3 );     \
  double4  _CL_OVERLOADABLE NAME(double, double4 );     \
  double8  _CL_OVERLOADABLE NAME(double, double8 );     \
  double16 _CL_OVERLOADABLE NAME(double, double16);)
#define _CL_DECLARE_FUNC_J_V(NAME)              \
  int   _CL_OVERLOADABLE NAME(float   );        \
  int2  _CL_OVERLOADABLE NAME(float2  );        \
  int3  _CL_OVERLOADABLE NAME(float3  );        \
  int4  _CL_OVERLOADABLE NAME(float4  );        \
  int8  _CL_OVERLOADABLE NAME(float8  );        \
  int16 _CL_OVERLOADABLE NAME(float16 );        \
  __IF_FP64(                                    \
  int    _CL_OVERLOADABLE NAME(double  );       \
  long2  _CL_OVERLOADABLE NAME(double2 );       \
  long3  _CL_OVERLOADABLE NAME(double3 );       \
  long4  _CL_OVERLOADABLE NAME(double4 );       \
  long8  _CL_OVERLOADABLE NAME(double8 );       \
  long16 _CL_OVERLOADABLE NAME(double16);)
#define _CL_DECLARE_FUNC_K_V(NAME)              \
  int   _CL_OVERLOADABLE NAME(float   );        \
  int2  _CL_OVERLOADABLE NAME(float2  );        \
  int3  _CL_OVERLOADABLE NAME(float3  );        \
  int4  _CL_OVERLOADABLE NAME(float4  );        \
  int8  _CL_OVERLOADABLE NAME(float8  );        \
  int16 _CL_OVERLOADABLE NAME(float16 );        \
  __IF_FP64(                                    \
  int   _CL_OVERLOADABLE NAME(double  );        \
  int2  _CL_OVERLOADABLE NAME(double2 );        \
  int3  _CL_OVERLOADABLE NAME(double3 );        \
  int4  _CL_OVERLOADABLE NAME(double4 );        \
  int8  _CL_OVERLOADABLE NAME(double8 );        \
  int16 _CL_OVERLOADABLE NAME(double16);)
#define _CL_DECLARE_FUNC_S_V(NAME)              \
  float  _CL_OVERLOADABLE NAME(float   );       \
  float  _CL_OVERLOADABLE NAME(float2  );       \
  float  _CL_OVERLOADABLE NAME(float3  );       \
  float  _CL_OVERLOADABLE NAME(float4  );       \
  float  _CL_OVERLOADABLE NAME(float8  );       \
  float  _CL_OVERLOADABLE NAME(float16 );       \
  __IF_FP64(                                    \
  double _CL_OVERLOADABLE NAME(double  );       \
  double _CL_OVERLOADABLE NAME(double2 );       \
  double _CL_OVERLOADABLE NAME(double3 );       \
  double _CL_OVERLOADABLE NAME(double4 );       \
  double _CL_OVERLOADABLE NAME(double8 );       \
  double _CL_OVERLOADABLE NAME(double16);)
#define _CL_DECLARE_FUNC_S_VV(NAME)                     \
  float  _CL_OVERLOADABLE NAME(float   , float   );     \
  float  _CL_OVERLOADABLE NAME(float2  , float2  );     \
  float  _CL_OVERLOADABLE NAME(float3  , float3  );     \
  float  _CL_OVERLOADABLE NAME(float4  , float4  );     \
  float  _CL_OVERLOADABLE NAME(float8  , float8  );     \
  float  _CL_OVERLOADABLE NAME(float16 , float16 );     \
  __IF_FP64(                                            \
  double _CL_OVERLOADABLE NAME(double  , double  );     \
  double _CL_OVERLOADABLE NAME(double2 , double2 );     \
  double _CL_OVERLOADABLE NAME(double3 , double3 );     \
  double _CL_OVERLOADABLE NAME(double4 , double4 );     \
  double _CL_OVERLOADABLE NAME(double8 , double8 );     \
  double _CL_OVERLOADABLE NAME(double16, double16);)
#define _CL_DECLARE_FUNC_F_F(NAME)              \
  float    _CL_OVERLOADABLE NAME(float   );     \
  float2   _CL_OVERLOADABLE NAME(float2  );     \
  float3   _CL_OVERLOADABLE NAME(float3  );     \
  float4   _CL_OVERLOADABLE NAME(float4  );     \
  float8   _CL_OVERLOADABLE NAME(float8  );     \
  float16  _CL_OVERLOADABLE NAME(float16 );
#define _CL_DECLARE_FUNC_F_FF(NAME)                     \
  float    _CL_OVERLOADABLE NAME(float   , float   );   \
  float2   _CL_OVERLOADABLE NAME(float2  , float2  );   \
  float3   _CL_OVERLOADABLE NAME(float3  , float3  );   \
  float4   _CL_OVERLOADABLE NAME(float4  , float4  );   \
  float8   _CL_OVERLOADABLE NAME(float8  , float8  );   \
  float16  _CL_OVERLOADABLE NAME(float16 , float16 );

/* Move built-in declarations and libm functions out of the way.
  (There should be a better way of doing so. These functions are
  either built-in math functions for OpenCL (see Clang's
  "Builtins.def"), although the either should not be, or should have
  the correct prototype. Functions defined in libc or libm may also
  interfere with OpenCL's functions, since their prototypes will be
  wrong. */
#define abs            _cl_abs
#define abs_diff       _cl_abs_diff
#define acos           _cl_acos
#define acosh          _cl_acosh
#define acospi         _cl_acospi
#define add_sat        _cl_add_sat
#define all            _cl_all
#define any            _cl_any
#define asin           _cl_asin
#define asinh          _cl_asinh
#define asinpi         _cl_asinpi
#define atan           _cl_atan
#define atan2          _cl_atan2
#define atan2pi        _cl_atan2pi
#define atanh          _cl_atanh
#define atanpi         _cl_atanpi
#define bitselect      _cl_bitselect
#define cbrt           _cl_cbrt
#define ceil           _cl_ceil
#define clamp          _cl_clamp
#define clz            _cl_clz
#define copysign       _cl_copysign
#define cos            _cl_cos
#define cosh           _cl_cosh
#define cospi          _cl_cospi
#define cross          _cl_cross
#define degrees        _cl_degrees
#define distance       _cl_distance
#define dot            _cl_dot
#define erf            _cl_erf
#define erfc           _cl_erfc
#define exp            _cl_exp
#define exp10          _cl_exp10
#define exp2           _cl_exp2
#define expm1          _cl_expm1
#define fabs           _cl_fabs
#define fast_distance  _cl_fast_distance
#define fast_length    _cl_fast_length
#define fast_normalize _cl_fast_normalize
#define fdim           _cl_fdim
#define floor          _cl_floor
#define fma            _cl_fma
#define fmax           _cl_fmax
#define fmin           _cl_fmin
#define fmod           _cl_fmod
#define fract          _cl_fract
#define frexp          _cl_frexp
#define hadd           _cl_hadd
#define half_cos       _cl_half_cos
#define half_divide    _cl_half_divide
#define half_exp       _cl_half_exp
#define half_exp10     _cl_half_exp10
#define half_exp2      _cl_half_exp2
#define half_log       _cl_half_log
#define half_log10     _cl_half_log10
#define half_log2      _cl_half_log2
#define half_powr      _cl_half_powr
#define half_recip     _cl_half_recip
#define half_rsqrt     _cl_half_rsqrt
#define half_sin       _cl_half_sin
#define half_sqrt      _cl_half_sqrt
#define half_tan       _cl_half_tan
#define hypot          _cl_hypot
#define ilogb          _cl_ilogb
#define isequal        _cl_isequal
#define isfinite       _cl_isfinite
#define isgreater      _cl_isgreater
#define isgreaterequal _cl_isgreaterequal
#define isinf          _cl_isinf
#define isless         _cl_isless
#define islessequal    _cl_islessequal
#define islessgreater  _cl_islessgreater
#define isnan          _cl_isnan
#define isnormal       _cl_isnormal
#define isnotequal     _cl_isnotequal
#define isordered      _cl_isordered
#define isunordered    _cl_isunordered
#define ldexp          _cl_ldexp
#define ldexp          _cl_ldexp
#define length         _cl_length
#define lgamma         _cl_lgamma
#define lgamma_r       _cl_lgamma_r
#define log            _cl_log
#define log10          _cl_log10
#define log1p          _cl_log1p
#define log2           _cl_log2
#define logb           _cl_logb
#define mad            _cl_mad
#define mad24          _cl_mad24
#define mad_hi         _cl_mad_hi
#define mad_sat        _cl_mad_sat
#define max            _cl_max
#define maxmag         _cl_maxmag
#define min            _cl_min
#define minmag         _cl_minmag
#define mix            _cl_mix
#define modf           _cl_modf
#define mul24          _cl_mul24
#define mul_hi         _cl_mul_hi
#define nan            _cl_nan
#define native_cos     _cl_native_cos
#define native_divide  _cl_native_divide
#define native_exp     _cl_native_exp
#define native_exp10   _cl_native_exp10
#define native_exp2    _cl_native_exp2
#define native_log     _cl_native_log
#define native_log10   _cl_native_log10
#define native_log2    _cl_native_log2
#define native_powr    _cl_native_powr
#define native_recip   _cl_native_recip
#define native_rsqrt   _cl_native_rsqrt
#define native_sin     _cl_native_sin
#define native_sqrt    _cl_native_sqrt
#define native_tan     _cl_native_tan
#define nextafter      _cl_nextafter
#define normalize      _cl_normalize
#define popcount       _cl_popcount
#define pow            _cl_pow
#define pown           _cl_pown
#define pown           _cl_pown
#define powr           _cl_powr
#define radians        _cl_radians
#define remainder      _cl_remainder
#define remquo         _cl_remquo
#define rhadd          _cl_rhadd
#define rint           _cl_rint
#define rootn          _cl_rootn
#define rootn          _cl_rootn
#define rotate         _cl_rotate
#define round          _cl_round
#define rsqrt          _cl_rsqrt
#define select         _cl_select
#define sign           _cl_sign
#define signbit        _cl_signbit
#define sin            _cl_sin
#define sincos         _cl_sincos
#define sinh           _cl_sinh
#define sinpi          _cl_sinpi
#define smoothstep     _cl_smoothstep
#define sqrt           _cl_sqrt
#define step           _cl_step
#define sub_sat        _cl_sub_sat
#define tan            _cl_tan
#define tanh           _cl_tanh
#define tanpi          _cl_tanpi
#define tgamma         _cl_tgamma
#define trunc          _cl_trunc
#define upsample       _cl_upsample

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
_CL_DECLARE_FUNC_V_VVV(fma)
#if __FAST__RELAXED__MATH__
#  undef fma
#  define fma mad
#endif
_CL_DECLARE_FUNC_V_VV(fmax)
_CL_DECLARE_FUNC_V_VS(fmax)
#if __FAST__RELAXED__MATH__
#  undef fmax
#  define fmax max
#endif
_CL_DECLARE_FUNC_V_VV(fmin)
_CL_DECLARE_FUNC_V_VS(fmin)
#if __FAST__RELAXED__MATH__
#  undef fmin
#  define fmin min
#endif
_CL_DECLARE_FUNC_V_VV(fmod)
_CL_DECLARE_FUNC_V_VPV(fract)
// frexp
_CL_DECLARE_FUNC_V_VV(hypot)
_CL_DECLARE_FUNC_K_V(ilogb)
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
#ifdef cl_khr_int64
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
#ifdef cl_khr_int64
#define ULONG_MAX 0xffffffffffffffffUL
#endif


/* Integer Functions */
#define _CL_DECLARE_FUNC_G_G(NAME)              \
  char     _CL_OVERLOADABLE NAME(char    );     \
  char2    _CL_OVERLOADABLE NAME(char2   );     \
  char3    _CL_OVERLOADABLE NAME(char3   );     \
  char4    _CL_OVERLOADABLE NAME(char4   );     \
  char8    _CL_OVERLOADABLE NAME(char8   );     \
  char16   _CL_OVERLOADABLE NAME(char16  );     \
  uchar    _CL_OVERLOADABLE NAME(uchar   );     \
  uchar2   _CL_OVERLOADABLE NAME(uchar2  );     \
  uchar3   _CL_OVERLOADABLE NAME(uchar3  );     \
  uchar4   _CL_OVERLOADABLE NAME(uchar4  );     \
  uchar8   _CL_OVERLOADABLE NAME(uchar8  );     \
  uchar16  _CL_OVERLOADABLE NAME(uchar16 );     \
  short    _CL_OVERLOADABLE NAME(short   );     \
  short2   _CL_OVERLOADABLE NAME(short2  );     \
  short3   _CL_OVERLOADABLE NAME(short3  );     \
  short4   _CL_OVERLOADABLE NAME(short4  );     \
  short8   _CL_OVERLOADABLE NAME(short8  );     \
  short16  _CL_OVERLOADABLE NAME(short16 );     \
  ushort   _CL_OVERLOADABLE NAME(ushort  );     \
  ushort2  _CL_OVERLOADABLE NAME(ushort2 );     \
  ushort3  _CL_OVERLOADABLE NAME(ushort3 );     \
  ushort4  _CL_OVERLOADABLE NAME(ushort4 );     \
  ushort8  _CL_OVERLOADABLE NAME(ushort8 );     \
  ushort16 _CL_OVERLOADABLE NAME(ushort16);     \
  int      _CL_OVERLOADABLE NAME(int     );     \
  int2     _CL_OVERLOADABLE NAME(int2    );     \
  int3     _CL_OVERLOADABLE NAME(int3    );     \
  int4     _CL_OVERLOADABLE NAME(int4    );     \
  int8     _CL_OVERLOADABLE NAME(int8    );     \
  int16    _CL_OVERLOADABLE NAME(int16   );     \
  uint     _CL_OVERLOADABLE NAME(uint    );     \
  uint2    _CL_OVERLOADABLE NAME(uint2   );     \
  uint3    _CL_OVERLOADABLE NAME(uint3   );     \
  uint4    _CL_OVERLOADABLE NAME(uint4   );     \
  uint8    _CL_OVERLOADABLE NAME(uint8   );     \
  uint16   _CL_OVERLOADABLE NAME(uint16  );     \
  __IF_INT64(                                   \
  long     _CL_OVERLOADABLE NAME(long    );     \
  long2    _CL_OVERLOADABLE NAME(long2   );     \
  long3    _CL_OVERLOADABLE NAME(long3   );     \
  long4    _CL_OVERLOADABLE NAME(long4   );     \
  long8    _CL_OVERLOADABLE NAME(long8   );     \
  long16   _CL_OVERLOADABLE NAME(long16  );     \
  ulong    _CL_OVERLOADABLE NAME(ulong   );     \
  ulong2   _CL_OVERLOADABLE NAME(ulong2  );     \
  ulong3   _CL_OVERLOADABLE NAME(ulong3  );     \
  ulong4   _CL_OVERLOADABLE NAME(ulong4  );     \
  ulong8   _CL_OVERLOADABLE NAME(ulong8  );     \
  ulong16  _CL_OVERLOADABLE NAME(ulong16 );)
#define _CL_DECLARE_FUNC_G_GG(NAME)                     \
  char     _CL_OVERLOADABLE NAME(char    , char    );   \
  char2    _CL_OVERLOADABLE NAME(char2   , char2   );   \
  char3    _CL_OVERLOADABLE NAME(char3   , char3   );   \
  char4    _CL_OVERLOADABLE NAME(char4   , char4   );   \
  char8    _CL_OVERLOADABLE NAME(char8   , char8   );   \
  char16   _CL_OVERLOADABLE NAME(char16  , char16  );   \
  uchar    _CL_OVERLOADABLE NAME(uchar   , uchar   );   \
  uchar2   _CL_OVERLOADABLE NAME(uchar2  , uchar2  );   \
  uchar3   _CL_OVERLOADABLE NAME(uchar3  , uchar3  );   \
  uchar4   _CL_OVERLOADABLE NAME(uchar4  , uchar4  );   \
  uchar8   _CL_OVERLOADABLE NAME(uchar8  , uchar8  );   \
  uchar16  _CL_OVERLOADABLE NAME(uchar16 , uchar16 );   \
  short    _CL_OVERLOADABLE NAME(short   , short   );   \
  short2   _CL_OVERLOADABLE NAME(short2  , short2  );   \
  short3   _CL_OVERLOADABLE NAME(short3  , short3  );   \
  short4   _CL_OVERLOADABLE NAME(short4  , short4  );   \
  short8   _CL_OVERLOADABLE NAME(short8  , short8  );   \
  short16  _CL_OVERLOADABLE NAME(short16 , short16 );   \
  ushort   _CL_OVERLOADABLE NAME(ushort  , ushort  );   \
  ushort2  _CL_OVERLOADABLE NAME(ushort2 , ushort2 );   \
  ushort3  _CL_OVERLOADABLE NAME(ushort3 , ushort3 );   \
  ushort4  _CL_OVERLOADABLE NAME(ushort4 , ushort4 );   \
  ushort8  _CL_OVERLOADABLE NAME(ushort8 , ushort8 );   \
  ushort16 _CL_OVERLOADABLE NAME(ushort16, ushort16);   \
  int      _CL_OVERLOADABLE NAME(int     , int     );   \
  int2     _CL_OVERLOADABLE NAME(int2    , int2    );   \
  int3     _CL_OVERLOADABLE NAME(int3    , int3    );   \
  int4     _CL_OVERLOADABLE NAME(int4    , int4    );   \
  int8     _CL_OVERLOADABLE NAME(int8    , int8    );   \
  int16    _CL_OVERLOADABLE NAME(int16   , int16   );   \
  uint     _CL_OVERLOADABLE NAME(uint    , uint    );   \
  uint2    _CL_OVERLOADABLE NAME(uint2   , uint2   );   \
  uint3    _CL_OVERLOADABLE NAME(uint3   , uint3   );   \
  uint4    _CL_OVERLOADABLE NAME(uint4   , uint4   );   \
  uint8    _CL_OVERLOADABLE NAME(uint8   , uint8   );   \
  uint16   _CL_OVERLOADABLE NAME(uint16  , uint16  );   \
  __IF_INT64(                                           \
  long     _CL_OVERLOADABLE NAME(long    , long    );   \
  long2    _CL_OVERLOADABLE NAME(long2   , long2   );   \
  long3    _CL_OVERLOADABLE NAME(long3   , long3   );   \
  long4    _CL_OVERLOADABLE NAME(long4   , long4   );   \
  long8    _CL_OVERLOADABLE NAME(long8   , long8   );   \
  long16   _CL_OVERLOADABLE NAME(long16  , long16  );   \
  ulong    _CL_OVERLOADABLE NAME(ulong   , ulong   );   \
  ulong2   _CL_OVERLOADABLE NAME(ulong2  , ulong2  );   \
  ulong3   _CL_OVERLOADABLE NAME(ulong3  , ulong3  );   \
  ulong4   _CL_OVERLOADABLE NAME(ulong4  , ulong4  );   \
  ulong8   _CL_OVERLOADABLE NAME(ulong8  , ulong8  );   \
  ulong16  _CL_OVERLOADABLE NAME(ulong16 , ulong16 );)
#define _CL_DECLARE_FUNC_G_GGG(NAME)                            \
  char     _CL_OVERLOADABLE NAME(char    , char    , char    ); \
  char2    _CL_OVERLOADABLE NAME(char2   , char2   , char2   ); \
  char3    _CL_OVERLOADABLE NAME(char3   , char3   , char3   ); \
  char4    _CL_OVERLOADABLE NAME(char4   , char4   , char4   ); \
  char8    _CL_OVERLOADABLE NAME(char8   , char8   , char8   ); \
  char16   _CL_OVERLOADABLE NAME(char16  , char16  , char16  ); \
  uchar    _CL_OVERLOADABLE NAME(uchar   , uchar   , uchar   ); \
  uchar2   _CL_OVERLOADABLE NAME(uchar2  , uchar2  , uchar2  ); \
  uchar3   _CL_OVERLOADABLE NAME(uchar3  , uchar3  , uchar3  ); \
  uchar4   _CL_OVERLOADABLE NAME(uchar4  , uchar4  , uchar4  ); \
  uchar8   _CL_OVERLOADABLE NAME(uchar8  , uchar8  , uchar8  ); \
  uchar16  _CL_OVERLOADABLE NAME(uchar16 , uchar16 , uchar16 ); \
  short    _CL_OVERLOADABLE NAME(short   , short   , short   ); \
  short2   _CL_OVERLOADABLE NAME(short2  , short2  , short2  ); \
  short3   _CL_OVERLOADABLE NAME(short3  , short3  , short3  ); \
  short4   _CL_OVERLOADABLE NAME(short4  , short4  , short4  ); \
  short8   _CL_OVERLOADABLE NAME(short8  , short8  , short8  ); \
  short16  _CL_OVERLOADABLE NAME(short16 , short16 , short16 ); \
  ushort   _CL_OVERLOADABLE NAME(ushort  , ushort  , ushort  ); \
  ushort2  _CL_OVERLOADABLE NAME(ushort2 , ushort2 , ushort2 ); \
  ushort3  _CL_OVERLOADABLE NAME(ushort3 , ushort3 , ushort3 ); \
  ushort4  _CL_OVERLOADABLE NAME(ushort4 , ushort4 , ushort4 ); \
  ushort8  _CL_OVERLOADABLE NAME(ushort8 , ushort8 , ushort8 ); \
  ushort16 _CL_OVERLOADABLE NAME(ushort16, ushort16, ushort16); \
  int      _CL_OVERLOADABLE NAME(int     , int     , int     ); \
  int2     _CL_OVERLOADABLE NAME(int2    , int2    , int2    ); \
  int3     _CL_OVERLOADABLE NAME(int3    , int3    , int3    ); \
  int4     _CL_OVERLOADABLE NAME(int4    , int4    , int4    ); \
  int8     _CL_OVERLOADABLE NAME(int8    , int8    , int8    ); \
  int16    _CL_OVERLOADABLE NAME(int16   , int16   , int16   ); \
  uint     _CL_OVERLOADABLE NAME(uint    , uint    , uint    ); \
  uint2    _CL_OVERLOADABLE NAME(uint2   , uint2   , uint2   ); \
  uint3    _CL_OVERLOADABLE NAME(uint3   , uint3   , uint3   ); \
  uint4    _CL_OVERLOADABLE NAME(uint4   , uint4   , uint4   ); \
  uint8    _CL_OVERLOADABLE NAME(uint8   , uint8   , uint8   ); \
  uint16   _CL_OVERLOADABLE NAME(uint16  , uint16  , uint16  ); \
  __IF_INT64(                                                   \
  long     _CL_OVERLOADABLE NAME(long    , long    , long    ); \
  long2    _CL_OVERLOADABLE NAME(long2   , long2   , long2   ); \
  long3    _CL_OVERLOADABLE NAME(long3   , long3   , long3   ); \
  long4    _CL_OVERLOADABLE NAME(long4   , long4   , long4   ); \
  long8    _CL_OVERLOADABLE NAME(long8   , long8   , long8   ); \
  long16   _CL_OVERLOADABLE NAME(long16  , long16  , long16  ); \
  ulong    _CL_OVERLOADABLE NAME(ulong   , ulong   , ulong   ); \
  ulong2   _CL_OVERLOADABLE NAME(ulong2  , ulong2  , ulong2  ); \
  ulong3   _CL_OVERLOADABLE NAME(ulong3  , ulong3  , ulong3  ); \
  ulong4   _CL_OVERLOADABLE NAME(ulong4  , ulong4  , ulong4  ); \
  ulong8   _CL_OVERLOADABLE NAME(ulong8  , ulong8  , ulong8  ); \
  ulong16  _CL_OVERLOADABLE NAME(ulong16 , ulong16 , ulong16 );)
#define _CL_DECLARE_FUNC_G_GGIG(NAME)                                   \
  char     _CL_OVERLOADABLE NAME(char    , char    , char    );         \
  char2    _CL_OVERLOADABLE NAME(char2   , char2   , char2   );         \
  char3    _CL_OVERLOADABLE NAME(char3   , char3   , char3   );         \
  char4    _CL_OVERLOADABLE NAME(char4   , char4   , char4   );         \
  char8    _CL_OVERLOADABLE NAME(char8   , char8   , char8   );         \
  char16   _CL_OVERLOADABLE NAME(char16  , char16  , char16  );         \
  uchar    _CL_OVERLOADABLE NAME(uchar   , uchar   , char    );         \
  uchar2   _CL_OVERLOADABLE NAME(uchar2  , uchar2  , char2   );         \
  uchar3   _CL_OVERLOADABLE NAME(uchar3  , uchar3  , char3   );         \
  uchar4   _CL_OVERLOADABLE NAME(uchar4  , uchar4  , char4   );         \
  uchar8   _CL_OVERLOADABLE NAME(uchar8  , uchar8  , char8   );         \
  uchar16  _CL_OVERLOADABLE NAME(uchar16 , uchar16 , char16  );         \
  short    _CL_OVERLOADABLE NAME(short   , short   , short   );         \
  short2   _CL_OVERLOADABLE NAME(short2  , short2  , short2  );         \
  short3   _CL_OVERLOADABLE NAME(short3  , short3  , short3  );         \
  short4   _CL_OVERLOADABLE NAME(short4  , short4  , short4  );         \
  short8   _CL_OVERLOADABLE NAME(short8  , short8  , short8  );         \
  short16  _CL_OVERLOADABLE NAME(short16 , short16 , short16 );         \
  ushort   _CL_OVERLOADABLE NAME(ushort  , ushort  , short   );         \
  ushort2  _CL_OVERLOADABLE NAME(ushort2 , ushort2 , short2  );         \
  ushort3  _CL_OVERLOADABLE NAME(ushort3 , ushort3 , short3  );         \
  ushort4  _CL_OVERLOADABLE NAME(ushort4 , ushort4 , short4  );         \
  ushort8  _CL_OVERLOADABLE NAME(ushort8 , ushort8 , short8  );         \
  ushort16 _CL_OVERLOADABLE NAME(ushort16, ushort16, short16 );         \
  int      _CL_OVERLOADABLE NAME(int     , int     , int     );         \
  int2     _CL_OVERLOADABLE NAME(int2    , int2    , int2    );         \
  int3     _CL_OVERLOADABLE NAME(int3    , int3    , int3    );         \
  int4     _CL_OVERLOADABLE NAME(int4    , int4    , int4    );         \
  int8     _CL_OVERLOADABLE NAME(int8    , int8    , int8    );         \
  int16    _CL_OVERLOADABLE NAME(int16   , int16   , int16   );         \
  uint     _CL_OVERLOADABLE NAME(uint    , uint    , int     );         \
  uint2    _CL_OVERLOADABLE NAME(uint2   , uint2   , int2    );         \
  uint3    _CL_OVERLOADABLE NAME(uint3   , uint3   , int3    );         \
  uint4    _CL_OVERLOADABLE NAME(uint4   , uint4   , int4    );         \
  uint8    _CL_OVERLOADABLE NAME(uint8   , uint8   , int8    );         \
  uint16   _CL_OVERLOADABLE NAME(uint16  , uint16  , int16   );         \
  __IF_INT64(                                                           \
  long     _CL_OVERLOADABLE NAME(long    , long    , long    );         \
  long2    _CL_OVERLOADABLE NAME(long2   , long2   , long2   );         \
  long3    _CL_OVERLOADABLE NAME(long3   , long3   , long3   );         \
  long4    _CL_OVERLOADABLE NAME(long4   , long4   , long4   );         \
  long8    _CL_OVERLOADABLE NAME(long8   , long8   , long8   );         \
  long16   _CL_OVERLOADABLE NAME(long16  , long16  , long16  );         \
  ulong    _CL_OVERLOADABLE NAME(ulong   , ulong   , long    );         \
  ulong2   _CL_OVERLOADABLE NAME(ulong2  , ulong2  , long2   );         \
  ulong3   _CL_OVERLOADABLE NAME(ulong3  , ulong3  , long3   );         \
  ulong4   _CL_OVERLOADABLE NAME(ulong4  , ulong4  , long4   );         \
  ulong8   _CL_OVERLOADABLE NAME(ulong8  , ulong8  , long8   );         \
  ulong16  _CL_OVERLOADABLE NAME(ulong16 , ulong16 , long16  );)
#define _CL_DECLARE_FUNC_G_GGUG(NAME)                                   \
  char     _CL_OVERLOADABLE NAME(char    , char    , uchar    );        \
  char2    _CL_OVERLOADABLE NAME(char2   , char2   , uchar2   );        \
  char3    _CL_OVERLOADABLE NAME(char3   , char3   , uchar3   );        \
  char4    _CL_OVERLOADABLE NAME(char4   , char4   , uchar4   );        \
  char8    _CL_OVERLOADABLE NAME(char8   , char8   , uchar8   );        \
  char16   _CL_OVERLOADABLE NAME(char16  , char16  , uchar16  );        \
  uchar    _CL_OVERLOADABLE NAME(uchar   , uchar   , uchar    );        \
  uchar2   _CL_OVERLOADABLE NAME(uchar2  , uchar2  , uchar2   );        \
  uchar3   _CL_OVERLOADABLE NAME(uchar3  , uchar3  , uchar3   );        \
  uchar4   _CL_OVERLOADABLE NAME(uchar4  , uchar4  , uchar4   );        \
  uchar8   _CL_OVERLOADABLE NAME(uchar8  , uchar8  , uchar8   );        \
  uchar16  _CL_OVERLOADABLE NAME(uchar16 , uchar16 , uchar16  );        \
  short    _CL_OVERLOADABLE NAME(short   , short   , ushort   );        \
  short2   _CL_OVERLOADABLE NAME(short2  , short2  , ushort2  );        \
  short3   _CL_OVERLOADABLE NAME(short3  , short3  , ushort3  );        \
  short4   _CL_OVERLOADABLE NAME(short4  , short4  , ushort4  );        \
  short8   _CL_OVERLOADABLE NAME(short8  , short8  , ushort8  );        \
  short16  _CL_OVERLOADABLE NAME(short16 , short16 , ushort16 );        \
  ushort   _CL_OVERLOADABLE NAME(ushort  , ushort  , ushort   );        \
  ushort2  _CL_OVERLOADABLE NAME(ushort2 , ushort2 , ushort2  );        \
  ushort3  _CL_OVERLOADABLE NAME(ushort3 , ushort3 , ushort3  );        \
  ushort4  _CL_OVERLOADABLE NAME(ushort4 , ushort4 , ushort4  );        \
  ushort8  _CL_OVERLOADABLE NAME(ushort8 , ushort8 , ushort8  );        \
  ushort16 _CL_OVERLOADABLE NAME(ushort16, ushort16, ushort16 );        \
  int      _CL_OVERLOADABLE NAME(int     , int     , uint     );        \
  int2     _CL_OVERLOADABLE NAME(int2    , int2    , uint2    );        \
  int3     _CL_OVERLOADABLE NAME(int3    , int3    , uint3    );        \
  int4     _CL_OVERLOADABLE NAME(int4    , int4    , uint4    );        \
  int8     _CL_OVERLOADABLE NAME(int8    , int8    , uint8    );        \
  int16    _CL_OVERLOADABLE NAME(int16   , int16   , uint16   );        \
  uint     _CL_OVERLOADABLE NAME(uint    , uint    , uint     );        \
  uint2    _CL_OVERLOADABLE NAME(uint2   , uint2   , uint2    );        \
  uint3    _CL_OVERLOADABLE NAME(uint3   , uint3   , uint3    );        \
  uint4    _CL_OVERLOADABLE NAME(uint4   , uint4   , uint4    );        \
  uint8    _CL_OVERLOADABLE NAME(uint8   , uint8   , uint8    );        \
  uint16   _CL_OVERLOADABLE NAME(uint16  , uint16  , uint16   );        \
  __IF_INT64(                                                           \
  long     _CL_OVERLOADABLE NAME(long    , long    , ulong    );        \
  long2    _CL_OVERLOADABLE NAME(long2   , long2   , ulong2   );        \
  long3    _CL_OVERLOADABLE NAME(long3   , long3   , ulong3   );        \
  long4    _CL_OVERLOADABLE NAME(long4   , long4   , ulong4   );        \
  long8    _CL_OVERLOADABLE NAME(long8   , long8   , ulong8   );        \
  long16   _CL_OVERLOADABLE NAME(long16  , long16  , ulong16  );        \
  ulong    _CL_OVERLOADABLE NAME(ulong   , ulong   , ulong    );        \
  ulong2   _CL_OVERLOADABLE NAME(ulong2  , ulong2  , ulong2   );        \
  ulong3   _CL_OVERLOADABLE NAME(ulong3  , ulong3  , ulong3   );        \
  ulong4   _CL_OVERLOADABLE NAME(ulong4  , ulong4  , ulong4   );        \
  ulong8   _CL_OVERLOADABLE NAME(ulong8  , ulong8  , ulong8   );        \
  ulong16  _CL_OVERLOADABLE NAME(ulong16 , ulong16 , ulong16  );)
#define _CL_DECLARE_FUNC_G_GS(NAME)                     \
  char2    _CL_OVERLOADABLE NAME(char2   , char  );     \
  char3    _CL_OVERLOADABLE NAME(char3   , char  );     \
  char4    _CL_OVERLOADABLE NAME(char4   , char  );     \
  char8    _CL_OVERLOADABLE NAME(char8   , char  );     \
  char16   _CL_OVERLOADABLE NAME(char16  , char  );     \
  uchar2   _CL_OVERLOADABLE NAME(uchar2  , uchar );     \
  uchar3   _CL_OVERLOADABLE NAME(uchar3  , uchar );     \
  uchar4   _CL_OVERLOADABLE NAME(uchar4  , uchar );     \
  uchar8   _CL_OVERLOADABLE NAME(uchar8  , uchar );     \
  uchar16  _CL_OVERLOADABLE NAME(uchar16 , uchar );     \
  short2   _CL_OVERLOADABLE NAME(short2  , short );     \
  short3   _CL_OVERLOADABLE NAME(short3  , short );     \
  short4   _CL_OVERLOADABLE NAME(short4  , short );     \
  short8   _CL_OVERLOADABLE NAME(short8  , short );     \
  short16  _CL_OVERLOADABLE NAME(short16 , short );     \
  ushort2  _CL_OVERLOADABLE NAME(ushort2 , ushort);     \
  ushort3  _CL_OVERLOADABLE NAME(ushort3 , ushort);     \
  ushort4  _CL_OVERLOADABLE NAME(ushort4 , ushort);     \
  ushort8  _CL_OVERLOADABLE NAME(ushort8 , ushort);     \
  ushort16 _CL_OVERLOADABLE NAME(ushort16, ushort);     \
  int2     _CL_OVERLOADABLE NAME(int2    , int   );     \
  int3     _CL_OVERLOADABLE NAME(int3    , int   );     \
  int4     _CL_OVERLOADABLE NAME(int4    , int   );     \
  int8     _CL_OVERLOADABLE NAME(int8    , int   );     \
  int16    _CL_OVERLOADABLE NAME(int16   , int   );     \
  uint2    _CL_OVERLOADABLE NAME(uint2   , uint  );     \
  uint3    _CL_OVERLOADABLE NAME(uint3   , uint  );     \
  uint4    _CL_OVERLOADABLE NAME(uint4   , uint  );     \
  uint8    _CL_OVERLOADABLE NAME(uint8   , uint  );     \
  uint16   _CL_OVERLOADABLE NAME(uint16  , uint  );     \
  __IF_INT64(                                           \
  long2    _CL_OVERLOADABLE NAME(long2   , long  );     \
  long3    _CL_OVERLOADABLE NAME(long3   , long  );     \
  long4    _CL_OVERLOADABLE NAME(long4   , long  );     \
  long8    _CL_OVERLOADABLE NAME(long8   , long  );     \
  long16   _CL_OVERLOADABLE NAME(long16  , long  );     \
  ulong2   _CL_OVERLOADABLE NAME(ulong2  , ulong );     \
  ulong3   _CL_OVERLOADABLE NAME(ulong3  , ulong );     \
  ulong4   _CL_OVERLOADABLE NAME(ulong4  , ulong );     \
  ulong8   _CL_OVERLOADABLE NAME(ulong8  , ulong );     \
  ulong16  _CL_OVERLOADABLE NAME(ulong16 , ulong );)
#define _CL_DECLARE_FUNC_UG_G(NAME)             \
  uchar    _CL_OVERLOADABLE NAME(char    );     \
  uchar2   _CL_OVERLOADABLE NAME(char2   );     \
  uchar3   _CL_OVERLOADABLE NAME(char3   );     \
  uchar4   _CL_OVERLOADABLE NAME(char4   );     \
  uchar8   _CL_OVERLOADABLE NAME(char8   );     \
  uchar16  _CL_OVERLOADABLE NAME(char16  );     \
  ushort   _CL_OVERLOADABLE NAME(short   );     \
  ushort2  _CL_OVERLOADABLE NAME(short2  );     \
  ushort3  _CL_OVERLOADABLE NAME(short3  );     \
  ushort4  _CL_OVERLOADABLE NAME(short4  );     \
  ushort8  _CL_OVERLOADABLE NAME(short8  );     \
  ushort16 _CL_OVERLOADABLE NAME(short16 );     \
  uint     _CL_OVERLOADABLE NAME(int     );     \
  uint2    _CL_OVERLOADABLE NAME(int2    );     \
  uint3    _CL_OVERLOADABLE NAME(int3    );     \
  uint4    _CL_OVERLOADABLE NAME(int4    );     \
  uint8    _CL_OVERLOADABLE NAME(int8    );     \
  uint16   _CL_OVERLOADABLE NAME(int16   );     \
  __IF_INT64(                                   \
  ulong    _CL_OVERLOADABLE NAME(long    );     \
  ulong2   _CL_OVERLOADABLE NAME(long2   );     \
  ulong3   _CL_OVERLOADABLE NAME(long3   );     \
  ulong4   _CL_OVERLOADABLE NAME(long4   );     \
  ulong8   _CL_OVERLOADABLE NAME(long8   );     \
  ulong16  _CL_OVERLOADABLE NAME(long16  );)    \
  uchar    _CL_OVERLOADABLE NAME(uchar   );     \
  uchar2   _CL_OVERLOADABLE NAME(uchar2  );     \
  uchar3   _CL_OVERLOADABLE NAME(uchar3  );     \
  uchar4   _CL_OVERLOADABLE NAME(uchar4  );     \
  uchar8   _CL_OVERLOADABLE NAME(uchar8  );     \
  uchar16  _CL_OVERLOADABLE NAME(uchar16 );     \
  ushort   _CL_OVERLOADABLE NAME(ushort  );     \
  ushort2  _CL_OVERLOADABLE NAME(ushort2 );     \
  ushort3  _CL_OVERLOADABLE NAME(ushort3 );     \
  ushort4  _CL_OVERLOADABLE NAME(ushort4 );     \
  ushort8  _CL_OVERLOADABLE NAME(ushort8 );     \
  ushort16 _CL_OVERLOADABLE NAME(ushort16);     \
  uint     _CL_OVERLOADABLE NAME(uint    );     \
  uint2    _CL_OVERLOADABLE NAME(uint2   );     \
  uint3    _CL_OVERLOADABLE NAME(uint3   );     \
  uint4    _CL_OVERLOADABLE NAME(uint4   );     \
  uint8    _CL_OVERLOADABLE NAME(uint8   );     \
  uint16   _CL_OVERLOADABLE NAME(uint16  );     \
  __IF_INT64(                                   \
  ulong    _CL_OVERLOADABLE NAME(ulong   );     \
  ulong2   _CL_OVERLOADABLE NAME(ulong2  );     \
  ulong3   _CL_OVERLOADABLE NAME(ulong3  );     \
  ulong4   _CL_OVERLOADABLE NAME(ulong4  );     \
  ulong8   _CL_OVERLOADABLE NAME(ulong8  );     \
  ulong16  _CL_OVERLOADABLE NAME(ulong16 );)
#define _CL_DECLARE_FUNC_UG_GG(NAME)                    \
  uchar    _CL_OVERLOADABLE NAME(char    , char    );   \
  uchar2   _CL_OVERLOADABLE NAME(char2   , char2   );   \
  uchar3   _CL_OVERLOADABLE NAME(char3   , char3   );   \
  uchar4   _CL_OVERLOADABLE NAME(char4   , char4   );   \
  uchar8   _CL_OVERLOADABLE NAME(char8   , char8   );   \
  uchar16  _CL_OVERLOADABLE NAME(char16  , char16  );   \
  ushort   _CL_OVERLOADABLE NAME(short   , short   );   \
  ushort2  _CL_OVERLOADABLE NAME(short2  , short2  );   \
  ushort3  _CL_OVERLOADABLE NAME(short3  , short3  );   \
  ushort4  _CL_OVERLOADABLE NAME(short4  , short4  );   \
  ushort8  _CL_OVERLOADABLE NAME(short8  , short8  );   \
  ushort16 _CL_OVERLOADABLE NAME(short16 , short16 );   \
  uint     _CL_OVERLOADABLE NAME(int     , int     );   \
  uint2    _CL_OVERLOADABLE NAME(int2    , int2    );   \
  uint3    _CL_OVERLOADABLE NAME(int3    , int3    );   \
  uint4    _CL_OVERLOADABLE NAME(int4    , int4    );   \
  uint8    _CL_OVERLOADABLE NAME(int8    , int8    );   \
  uint16   _CL_OVERLOADABLE NAME(int16   , int16   );   \
  __IF_INT64(                                           \
  ulong    _CL_OVERLOADABLE NAME(long    , long    );   \
  ulong2   _CL_OVERLOADABLE NAME(long2   , long2   );   \
  ulong3   _CL_OVERLOADABLE NAME(long3   , long3   );   \
  ulong4   _CL_OVERLOADABLE NAME(long4   , long4   );   \
  ulong8   _CL_OVERLOADABLE NAME(long8   , long8   );   \
  ulong16  _CL_OVERLOADABLE NAME(long16  , long16  );)  \
  uchar    _CL_OVERLOADABLE NAME(uchar   , uchar   );   \
  uchar2   _CL_OVERLOADABLE NAME(uchar2  , uchar2  );   \
  uchar3   _CL_OVERLOADABLE NAME(uchar3  , uchar3  );   \
  uchar4   _CL_OVERLOADABLE NAME(uchar4  , uchar4  );   \
  uchar8   _CL_OVERLOADABLE NAME(uchar8  , uchar8  );   \
  uchar16  _CL_OVERLOADABLE NAME(uchar16 , uchar16 );   \
  ushort   _CL_OVERLOADABLE NAME(ushort  , ushort  );   \
  ushort2  _CL_OVERLOADABLE NAME(ushort2 , ushort2 );   \
  ushort3  _CL_OVERLOADABLE NAME(ushort3 , ushort3 );   \
  ushort4  _CL_OVERLOADABLE NAME(ushort4 , ushort4 );   \
  ushort8  _CL_OVERLOADABLE NAME(ushort8 , ushort8 );   \
  ushort16 _CL_OVERLOADABLE NAME(ushort16, ushort16);   \
  uint     _CL_OVERLOADABLE NAME(uint    , uint    );   \
  uint2    _CL_OVERLOADABLE NAME(uint2   , uint2   );   \
  uint3    _CL_OVERLOADABLE NAME(uint3   , uint3   );   \
  uint4    _CL_OVERLOADABLE NAME(uint4   , uint4   );   \
  uint8    _CL_OVERLOADABLE NAME(uint8   , uint8   );   \
  uint16   _CL_OVERLOADABLE NAME(uint16  , uint16  );   \
  __IF_INT64(                                           \
  ulong    _CL_OVERLOADABLE NAME(ulong   , ulong   );   \
  ulong2   _CL_OVERLOADABLE NAME(ulong2  , ulong2  );   \
  ulong3   _CL_OVERLOADABLE NAME(ulong3  , ulong3  );   \
  ulong4   _CL_OVERLOADABLE NAME(ulong4  , ulong4  );   \
  ulong8   _CL_OVERLOADABLE NAME(ulong8  , ulong8  );   \
  ulong16  _CL_OVERLOADABLE NAME(ulong16 , ulong16 );)
#define _CL_DECLARE_FUNC_LG_GUG(NAME)                   \
  short    _CL_OVERLOADABLE NAME(char    , uchar   );   \
  short2   _CL_OVERLOADABLE NAME(char2   , uchar2  );   \
  short3   _CL_OVERLOADABLE NAME(char3   , uchar3  );   \
  short4   _CL_OVERLOADABLE NAME(char4   , uchar4  );   \
  short8   _CL_OVERLOADABLE NAME(char8   , uchar8  );   \
  short16  _CL_OVERLOADABLE NAME(char16  , uchar16 );   \
  ushort   _CL_OVERLOADABLE NAME(uchar   , uchar   );   \
  ushort2  _CL_OVERLOADABLE NAME(uchar2  , uchar2  );   \
  ushort3  _CL_OVERLOADABLE NAME(uchar3  , uchar3  );   \
  ushort4  _CL_OVERLOADABLE NAME(uchar4  , uchar4  );   \
  ushort8  _CL_OVERLOADABLE NAME(uchar8  , uchar8  );   \
  ushort16 _CL_OVERLOADABLE NAME(uchar16 , uchar16 );   \
  uint     _CL_OVERLOADABLE NAME(ushort  , ushort  );   \
  uint2    _CL_OVERLOADABLE NAME(ushort2 , ushort2 );   \
  uint3    _CL_OVERLOADABLE NAME(ushort3 , ushort3 );   \
  uint4    _CL_OVERLOADABLE NAME(ushort4 , ushort4 );   \
  uint8    _CL_OVERLOADABLE NAME(ushort8 , ushort8 );   \
  uint16   _CL_OVERLOADABLE NAME(ushort16, ushort16);   \
  int      _CL_OVERLOADABLE NAME(short   , ushort  );   \
  int2     _CL_OVERLOADABLE NAME(short2  , ushort2 );   \
  int3     _CL_OVERLOADABLE NAME(short3  , ushort3 );   \
  int4     _CL_OVERLOADABLE NAME(short4  , ushort4 );   \
  int8     _CL_OVERLOADABLE NAME(short8  , ushort8 );   \
  int16    _CL_OVERLOADABLE NAME(short16 , ushort16);   \
  __IF_INT64(                                           \
  long     _CL_OVERLOADABLE NAME(int     , uint    );   \
  long2    _CL_OVERLOADABLE NAME(int2    , uint2   );   \
  long3    _CL_OVERLOADABLE NAME(int3    , uint3   );   \
  long4    _CL_OVERLOADABLE NAME(int4    , uint4   );   \
  long8    _CL_OVERLOADABLE NAME(int8    , uint8   );   \
  long16   _CL_OVERLOADABLE NAME(int16   , uint16  );   \
  ulong    _CL_OVERLOADABLE NAME(uint    , uint    );   \
  ulong2   _CL_OVERLOADABLE NAME(uint2   , uint2   );   \
  ulong3   _CL_OVERLOADABLE NAME(uint3   , uint3   );   \
  ulong4   _CL_OVERLOADABLE NAME(uint4   , uint4   );   \
  ulong8   _CL_OVERLOADABLE NAME(uint8   , uint8   );   \
  ulong16  _CL_OVERLOADABLE NAME(uint16  , uint16  );)
#define _CL_DECLARE_FUNC_I_IG(NAME)             \
  int _CL_OVERLOADABLE NAME(char   );           \
  int _CL_OVERLOADABLE NAME(char2  );           \
  int _CL_OVERLOADABLE NAME(char3  );           \
  int _CL_OVERLOADABLE NAME(char4  );           \
  int _CL_OVERLOADABLE NAME(char8  );           \
  int _CL_OVERLOADABLE NAME(char16 );           \
  int _CL_OVERLOADABLE NAME(short  );           \
  int _CL_OVERLOADABLE NAME(short2 );           \
  int _CL_OVERLOADABLE NAME(short3 );           \
  int _CL_OVERLOADABLE NAME(short4 );           \
  int _CL_OVERLOADABLE NAME(short8 );           \
  int _CL_OVERLOADABLE NAME(short16);           \
  int _CL_OVERLOADABLE NAME(int    );           \
  int _CL_OVERLOADABLE NAME(int2   );           \
  int _CL_OVERLOADABLE NAME(int3   );           \
  int _CL_OVERLOADABLE NAME(int4   );           \
  int _CL_OVERLOADABLE NAME(int8   );           \
  int _CL_OVERLOADABLE NAME(int16  );           \
  __IF_INT64(                                   \
  int _CL_OVERLOADABLE NAME(long   );           \
  int _CL_OVERLOADABLE NAME(long2  );           \
  int _CL_OVERLOADABLE NAME(long3  );           \
  int _CL_OVERLOADABLE NAME(long4  );           \
  int _CL_OVERLOADABLE NAME(long8  );           \
  int _CL_OVERLOADABLE NAME(long16 );)
#define _CL_DECLARE_FUNC_J_JJ(NAME)                     \
  int      _CL_OVERLOADABLE NAME(int     , int     );   \
  int2     _CL_OVERLOADABLE NAME(int2    , int2    );   \
  int3     _CL_OVERLOADABLE NAME(int3    , int3    );   \
  int4     _CL_OVERLOADABLE NAME(int4    , int4    );   \
  int8     _CL_OVERLOADABLE NAME(int8    , int8    );   \
  int16    _CL_OVERLOADABLE NAME(int16   , int16   );   \
  uint     _CL_OVERLOADABLE NAME(uint    , uint    );   \
  uint2    _CL_OVERLOADABLE NAME(uint2   , uint2   );   \
  uint3    _CL_OVERLOADABLE NAME(uint3   , uint3   );   \
  uint4    _CL_OVERLOADABLE NAME(uint4   , uint4   );   \
  uint8    _CL_OVERLOADABLE NAME(uint8   , uint8   );   \
  uint16   _CL_OVERLOADABLE NAME(uint16  , uint16  );
#define _CL_DECLARE_FUNC_J_JJJ(NAME)                            \
  int      _CL_OVERLOADABLE NAME(int     , int     , int     ); \
  int2     _CL_OVERLOADABLE NAME(int2    , int2    , int2    ); \
  int3     _CL_OVERLOADABLE NAME(int3    , int3    , int3    ); \
  int4     _CL_OVERLOADABLE NAME(int4    , int4    , int4    ); \
  int8     _CL_OVERLOADABLE NAME(int8    , int8    , int8    ); \
  int16    _CL_OVERLOADABLE NAME(int16   , int16   , int16   ); \
  uint     _CL_OVERLOADABLE NAME(uint    , uint    , uint    ); \
  uint2    _CL_OVERLOADABLE NAME(uint2   , uint2   , uint2   ); \
  uint3    _CL_OVERLOADABLE NAME(uint3   , uint3   , uint3   ); \
  uint4    _CL_OVERLOADABLE NAME(uint4   , uint4   , uint4   ); \
  uint8    _CL_OVERLOADABLE NAME(uint8   , uint8   , uint8   ); \
  uint16   _CL_OVERLOADABLE NAME(uint16  , uint16  , uint16  );

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

float4 _CL_OVERLOADABLE cross(float4, float4);
float3 _CL_OVERLOADABLE cross(float3, float3);
#ifdef cl_khr_fp64
double4 _CL_OVERLOADABLE cross(double4, double4);
double3 _CL_OVERLOADABLE cross(double3, double3);
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
_CL_DECLARE_FUNC_J_V(isfinite)
_CL_DECLARE_FUNC_J_V(isinf)
_CL_DECLARE_FUNC_J_V(isnan)
_CL_DECLARE_FUNC_J_V(isnormal)
_CL_DECLARE_FUNC_J_VV(isordered)
_CL_DECLARE_FUNC_J_VV(isunordered)
_CL_DECLARE_FUNC_J_V(signbit)
_CL_DECLARE_FUNC_I_IG(any)
_CL_DECLARE_FUNC_I_IG(all)
_CL_DECLARE_FUNC_G_GGG(bitselect)
_CL_DECLARE_FUNC_V_VVV(bitselect)
_CL_DECLARE_FUNC_G_GGIG(select)
_CL_DECLARE_FUNC_G_GGUG(select)
_CL_DECLARE_FUNC_V_VVJ(select)
_CL_DECLARE_FUNC_V_VVU(select)


/* Vector Functions */

#define _CL_DECLARE_VLOAD(TYPE, MOD)                                    \
  TYPE##2  _CL_OVERLOADABLE vload2 (size_t offset, const MOD TYPE *p);  \
  TYPE##3  _CL_OVERLOADABLE vload3 (size_t offset, const MOD TYPE *p);  \
  TYPE##4  _CL_OVERLOADABLE vload4 (size_t offset, const MOD TYPE *p);  \
  TYPE##8  _CL_OVERLOADABLE vload8 (size_t offset, const MOD TYPE *p);  \
  TYPE##16 _CL_OVERLOADABLE vload16(size_t offset, const MOD TYPE *p);

_CL_DECLARE_VLOAD(char  , __global)
_CL_DECLARE_VLOAD(uchar , __global)
_CL_DECLARE_VLOAD(short , __global)
_CL_DECLARE_VLOAD(ushort, __global)
_CL_DECLARE_VLOAD(int   , __global)
_CL_DECLARE_VLOAD(uint  , __global)
#ifdef cl_khr_int64
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
#ifdef cl_khr_int64
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
#ifdef cl_khr_int64
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
#ifdef cl_khr_int64
_CL_DECLARE_VLOAD(long  , __private)
_CL_DECLARE_VLOAD(ulong , __private)
#endif
_CL_DECLARE_VLOAD(float , __private)
#ifdef cl_khr_fp64
_CL_DECLARE_VLOAD(double, __private)
#endif

#define _CL_DECLARE_VSTORE(TYPE, MOD)                                   \
  void _CL_OVERLOADABLE vstore2 (TYPE##2  data, size_t offset, MOD TYPE *p); \
  void _CL_OVERLOADABLE vstore3 (TYPE##3  data, size_t offset, MOD TYPE *p); \
  void _CL_OVERLOADABLE vstore4 (TYPE##4  data, size_t offset, MOD TYPE *p); \
  void _CL_OVERLOADABLE vstore8 (TYPE##8  data, size_t offset, MOD TYPE *p); \
  void _CL_OVERLOADABLE vstore16(TYPE##16 data, size_t offset, MOD TYPE *p);

_CL_DECLARE_VSTORE(char  , __global)
_CL_DECLARE_VSTORE(uchar , __global)
_CL_DECLARE_VSTORE(short , __global)
_CL_DECLARE_VSTORE(ushort, __global)
_CL_DECLARE_VSTORE(int   , __global)
_CL_DECLARE_VSTORE(uint  , __global)
#ifdef cl_khr_int64
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
#ifdef cl_khr_int64
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
#ifdef cl_khr_int64
_CL_DECLARE_VSTORE(long  , __private)
_CL_DECLARE_VSTORE(ulong , __private)
#endif
_CL_DECLARE_VSTORE(float , __private)
#ifdef cl_khr_fp64
_CL_DECLARE_VSTORE(double, __private)
#endif

#ifdef cl_khr_fp16

#define _CL_DECLARE_VLOAD_HALF(MOD)                                     \
  float   _CL_OVERLOADABLE vload_half   (size_t offset, const MOD half *p); \
  float2  _CL_OVERLOADABLE vload_half2  (size_t offset, const MOD half *p); \
  float3  _CL_OVERLOADABLE vload_half3  (size_t offset, const MOD half *p); \
  float4  _CL_OVERLOADABLE vload_half4  (size_t offset, const MOD half *p); \
  float8  _CL_OVERLOADABLE vload_half8  (size_t offset, const MOD half *p); \
  float16 _CL_OVERLOADABLE vload_half16 (size_t offset, const MOD half *p); \
  float2  _CL_OVERLOADABLE vloada_half2 (size_t offset, const MOD half *p); \
  float3  _CL_OVERLOADABLE vloada_half3 (size_t offset, const MOD half *p); \
  float4  _CL_OVERLOADABLE vloada_half4 (size_t offset, const MOD half *p); \
  float8  _CL_OVERLOADABLE vloada_half8 (size_t offset, const MOD half *p); \
  float16 _CL_OVERLOADABLE vloada_half16(size_t offset, const MOD half *p);

_CL_DECLARE_VLOAD_HALF(__global)
_CL_DECLARE_VLOAD_HALF(__local)
_CL_DECLARE_VLOAD_HALF(__constant)
_CL_DECLARE_VLOAD_HALF(__private)

/* stores to half may have a suffix: _rte _rtz _rtp _rtn */
#define _CL_DECLARE_VSTORE_HALF(MOD, SUFFIX)                            \
  void _CL_OVERLOADABLE vstore_half##SUFFIX   (float   data, size_t offset, MOD half *p); \
  void _CL_OVERLOADABLE vstore_half2##SUFFIX  (float2  data, size_t offset, MOD half *p); \
  void _CL_OVERLOADABLE vstore_half3##SUFFIX  (float3  data, size_t offset, MOD half *p); \
  void _CL_OVERLOADABLE vstore_half4##SUFFIX  (float4  data, size_t offset, MOD half *p); \
  void _CL_OVERLOADABLE vstore_half8##SUFFIX  (float8  data, size_t offset, MOD half *p); \
  void _CL_OVERLOADABLE vstore_half16##SUFFIX (float16 data, size_t offset, MOD half *p); \
  void _CL_OVERLOADABLE vstorea_half2##SUFFIX (float2  data, size_t offset, MOD half *p); \
  void _CL_OVERLOADABLE vstorea_half3##SUFFIX (float3  data, size_t offset, MOD half *p); \
  void _CL_OVERLOADABLE vstorea_half4##SUFFIX (float4  data, size_t offset, MOD half *p); \
  void _CL_OVERLOADABLE vstorea_half8##SUFFIX (float8  data, size_t offset, MOD half *p); \
  void _CL_OVERLOADABLE vstorea_half16##SUFFIX(float16 data, size_t offset, MOD half *p);

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


/* Atomic operations */

#define _CL_DECLARE_ATOMICS(MOD, TYPE)                                  \
  _CL_OVERLOADABLE TYPE atomic_add    (volatile MOD TYPE *p, TYPE val); \
  _CL_OVERLOADABLE TYPE atomic_sub    (volatile MOD TYPE *p, TYPE val); \
  _CL_OVERLOADABLE TYPE atomic_xchg   (volatile MOD TYPE *p, TYPE val); \
  _CL_OVERLOADABLE TYPE atomic_inc    (volatile MOD TYPE *p);           \
  _CL_OVERLOADABLE TYPE atomic_dec    (volatile MOD TYPE *p);           \
  _CL_OVERLOADABLE TYPE atomic_cmpxchg(volatile MOD TYPE *p, TYPE cmp, TYPE val); \
  _CL_OVERLOADABLE TYPE atomic_min    (volatile MOD TYPE *p, TYPE val); \
  _CL_OVERLOADABLE TYPE atomic_max    (volatile MOD TYPE *p, TYPE val); \
  _CL_OVERLOADABLE TYPE atomic_and    (volatile MOD TYPE *p, TYPE val); \
  _CL_OVERLOADABLE TYPE atomic_or     (volatile MOD TYPE *p, TYPE val); \
  _CL_OVERLOADABLE TYPE atomic_xor    (volatile MOD TYPE *p, TYPE val);
_CL_DECLARE_ATOMICS(__global, int )
_CL_DECLARE_ATOMICS(__global, uint)
_CL_DECLARE_ATOMICS(__local , int )
_CL_DECLARE_ATOMICS(__local , uint)

_CL_OVERLOADABLE float atomic_xchg(volatile __global float *p, float val);
_CL_OVERLOADABLE float atomic_xchg(volatile __local  float *p, float val);

#define atom_add     atomic_add
#define atom_sub     atomic_sub
#define atom_xchg    atomic_xchg
#define atom_inc     atomic_inc
#define atom_dec     atomic_dec
#define atom_cmpxchg atomic_cmpxchg
#define atom_min     atomic_min
#define atom_max     atomic_max
#define atom_and     atomic_and
#define atom_or      atomic_or
#define atom_xor     atomic_xor


/* Miscellaneous Vector Functions */

// This code leads to an ICE in Clang

// #define _CL_DECLARE_SHUFFLE_2(GTYPE, UGTYPE, STYPE, M)                  \
//   GTYPE##2 _CL_OVERLOADABLE shuffle(GTYPE##M x, UGTYPE##2 mask)         \
//   {                                                                     \
//     UGTYPE bits = (UGTYPE)1 << (UGTYPE)M;                               \
//     UGTYPE bmask = bits - (UGTYPE)1;                                    \
//     return __builtin_shufflevector(x, x,                                \
//                                    mask.s0 & bmask, mask.s1 & bmask);   \
//   }
// #define _CL_DECLARE_SHUFFLE_3(GTYPE, UGTYPE, STYPE, M)                  \
//   GTYPE##3 _CL_OVERLOADABLE shuffle(GTYPE##M x, UGTYPE##3 mask)         \
//   {                                                                     \
//     UGTYPE bits = (UGTYPE)1 << (UGTYPE)M;                               \
//     UGTYPE bmask = bits - (UGTYPE)1;                                    \
//     return __builtin_shufflevector(x, x,                                \
//                                    mask.s0 & bmask, mask.s1 & bmask,    \
//                                    mask.s2 & bmask);                    \
//   }
// #define _CL_DECLARE_SHUFFLE_4(GTYPE, UGTYPE, STYPE, M)                  \
//   GTYPE##4 _CL_OVERLOADABLE shuffle(GTYPE##M x, UGTYPE##4 mask)         \
//   {                                                                     \
//     UGTYPE bits = (UGTYPE)1 << (UGTYPE)M;                               \
//     UGTYPE bmask = bits - (UGTYPE)1;                                    \
//     return __builtin_shufflevector(x, x,                                \
//                                    mask.s0 & bmask, mask.s1 & bmask,    \
//                                    mask.s2 & bmask, mask.s3 & bmask);   \
//   }
// #define _CL_DECLARE_SHUFFLE_8(GTYPE, UGTYPE, STYPE, M)                  \
//   GTYPE##8 _CL_OVERLOADABLE shuffle(GTYPE##M x, UGTYPE##8 mask)         \
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
//   GTYPE##16 _CL_OVERLOADABLE shuffle(GTYPE##M x, UGTYPE##16 mask)       \
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


// Note: Using "const" instead of "constant", since string literals
// are currently "const" instead of "constant".
int _cl_printf(/*constant*/ const char* restrict format, ...)
  __attribute__((format(printf, 1, 2)));

#if ((__clang_major__ == 3) && (__clang_minor__ > 3)) && !defined (__TCE__)
// this will be overridden in printf.cl by an OpenCL-compliant printf
#define printf _cl_printf
#else
// The new printf requires Clang 3.4. Fall back to the libc one
// if using an older version.
int printf(const char* restrict fmt, ...);
#endif


/* Async Copies from Global to Local Memory, Local to
   Global Memory, and Prefetch */

#ifndef _CL_HAS_EVENT_T
typedef uint event_t;
#endif

#define _CL_DECLARE_ASYNC_COPY_FUNCS_SINGLE(GENTYPE)            \
  _CL_OVERLOADABLE                                              \
  event_t async_work_group_copy (__local GENTYPE *dst,          \
                                 const __global GENTYPE *src,   \
                                 size_t num_gentypes,           \
                                 event_t event);                \
                                                                \
  _CL_OVERLOADABLE                                              \
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

