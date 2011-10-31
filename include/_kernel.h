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

#define __SSE4_1__


#ifndef __TCE__
//#define __kernel __attribute__ ((noinline))
#define __global __attribute__ ((address_space(3)))
#define __local __attribute__ ((address_space(4)))
#define __constant __attribute__ ((address_space(5)))

#define global __attribute__ ((address_space(3)))
#define local __attribute__ ((address_space(4)))
#define constant __attribute__ ((address_space(5)))
#endif


typedef enum {
  CLK_LOCAL_MEM_FENCE = 0x1,
  CLK_GLOBAL_MEM_FENCE = 0x2
} cl_mem_fence_flags;



/* Data types */

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

#if 0
/* 32 bit systems */
typedef uint size_t;
typedef int ptrdiff_t;
typedef int intptr_t;
typedef uint uintptr_t;
#else
/* 64 bit systems */
typedef ulong size_t;
typedef long ptrdiff_t;
typedef long intptr_t;
typedef ulong uintptr_t;
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

typedef float float2  __attribute__((__ext_vector_type__(2)));
typedef float float3  __attribute__((__ext_vector_type__(3), __aligned__(16)));
typedef float float4  __attribute__((__ext_vector_type__(4)));
typedef float float8  __attribute__((__ext_vector_type__(8)));
typedef float float16 __attribute__((__ext_vector_type__(16)));

typedef double double2  __attribute__((__ext_vector_type__(2)));
typedef double double3  __attribute__((__ext_vector_type__(3), __aligned__(32)));
typedef double double4  __attribute__((__ext_vector_type__(4)));
typedef double double8  __attribute__((__ext_vector_type__(8)));
typedef double double16 __attribute__((__ext_vector_type__(16)));

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
_cl_static_assert(size_t, sizeof(size_t) == sizeof(void*));

_cl_static_assert(char2 , sizeof(char2 ) == 2 *sizeof(char));
_cl_static_assert(char3 , sizeof(char3 ) == 4 *sizeof(char));
_cl_static_assert(char4 , sizeof(char4 ) == 4 *sizeof(char));
_cl_static_assert(char8 , sizeof(char8 ) == 8 *sizeof(char));
_cl_static_assert(char16, sizeof(char16) == 16*sizeof(char));

_cl_static_assert(uchar2 , sizeof(uchar2 ) == 2 *sizeof(uchar));
_cl_static_assert(uchar3 , sizeof(uchar3 ) == 4 *sizeof(uchar));
_cl_static_assert(uchar4 , sizeof(uchar4 ) == 4 *sizeof(uchar));
_cl_static_assert(uchar8 , sizeof(uchar8 ) == 8 *sizeof(uchar));
_cl_static_assert(uchar16, sizeof(uchar16) == 16*sizeof(uchar));

_cl_static_assert(short2 , sizeof(short2 ) == 2 *sizeof(short));
_cl_static_assert(short3 , sizeof(short3 ) == 4 *sizeof(short));
_cl_static_assert(short4 , sizeof(short4 ) == 4 *sizeof(short));
_cl_static_assert(short8 , sizeof(short8 ) == 8 *sizeof(short));
_cl_static_assert(short16, sizeof(short16) == 16*sizeof(short));

_cl_static_assert(ushort2 , sizeof(ushort2 ) == 2 *sizeof(ushort));
_cl_static_assert(ushort3 , sizeof(ushort3 ) == 4 *sizeof(ushort));
_cl_static_assert(ushort4 , sizeof(ushort4 ) == 4 *sizeof(ushort));
_cl_static_assert(ushort8 , sizeof(ushort8 ) == 8 *sizeof(ushort));
_cl_static_assert(ushort16, sizeof(ushort16) == 16*sizeof(ushort));

_cl_static_assert(int2 , sizeof(int2 ) == 2 *sizeof(int));
_cl_static_assert(int3 , sizeof(int3 ) == 4 *sizeof(int));
_cl_static_assert(int4 , sizeof(int4 ) == 4 *sizeof(int));
_cl_static_assert(int8 , sizeof(int8 ) == 8 *sizeof(int));
_cl_static_assert(int16, sizeof(int16) == 16*sizeof(int));

_cl_static_assert(uint2 , sizeof(uint2 ) == 2 *sizeof(uint));
_cl_static_assert(uint3 , sizeof(uint3 ) == 4 *sizeof(uint));
_cl_static_assert(uint4 , sizeof(uint4 ) == 4 *sizeof(uint));
_cl_static_assert(uint8 , sizeof(uint8 ) == 8 *sizeof(uint));
_cl_static_assert(uint16, sizeof(uint16) == 16*sizeof(uint));

_cl_static_assert(float2 , sizeof(float2 ) == 2 *sizeof(float));
_cl_static_assert(float3 , sizeof(float3 ) == 4 *sizeof(float));
_cl_static_assert(float4 , sizeof(float4 ) == 4 *sizeof(float));
_cl_static_assert(float8 , sizeof(float8 ) == 8 *sizeof(float));
_cl_static_assert(float16, sizeof(float16) == 16*sizeof(float));

_cl_static_assert(double2 , sizeof(double2 ) == 2 *sizeof(double));
_cl_static_assert(double3 , sizeof(double3 ) == 4 *sizeof(double));
_cl_static_assert(double4 , sizeof(double4 ) == 4 *sizeof(double));
_cl_static_assert(double8 , sizeof(double8 ) == 8 *sizeof(double));
_cl_static_assert(double16, sizeof(double16) == 16*sizeof(double));



/* Conversion functions */

#define _cl_overloadable __attribute__ ((__overloadable__))

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
  _CL_DECLARE_AS_TYPE(SRC, long)                \
  _CL_DECLARE_AS_TYPE(SRC, ulong)               \
  _CL_DECLARE_AS_TYPE(SRC, float2)              \
  _CL_DECLARE_AS_TYPE(SRC, double)
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
#define _CL_DECLARE_AS_TYPE_16(SRC)             \
  _CL_DECLARE_AS_TYPE(SRC, char16)              \
  _CL_DECLARE_AS_TYPE(SRC, uchar16)             \
  _CL_DECLARE_AS_TYPE(SRC, short8)              \
  _CL_DECLARE_AS_TYPE(SRC, ushort8)             \
  _CL_DECLARE_AS_TYPE(SRC, int4)                \
  _CL_DECLARE_AS_TYPE(SRC, uint4)               \
  _CL_DECLARE_AS_TYPE(SRC, long2)               \
  _CL_DECLARE_AS_TYPE(SRC, ulong2)              \
  _CL_DECLARE_AS_TYPE(SRC, float4)              \
  _CL_DECLARE_AS_TYPE(SRC, double2)
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
#define _CL_DECLARE_AS_TYPE_32(SRC)             \
  _CL_DECLARE_AS_TYPE(SRC, short16)             \
  _CL_DECLARE_AS_TYPE(SRC, ushort16)            \
  _CL_DECLARE_AS_TYPE(SRC, int8)                \
  _CL_DECLARE_AS_TYPE(SRC, uint8)               \
  _CL_DECLARE_AS_TYPE(SRC, long4)               \
  _CL_DECLARE_AS_TYPE(SRC, ulong4)              \
  _CL_DECLARE_AS_TYPE(SRC, float8)              \
  _CL_DECLARE_AS_TYPE(SRC, double4)
_CL_DECLARE_AS_TYPE_32(short16)
_CL_DECLARE_AS_TYPE_32(ushort16)
_CL_DECLARE_AS_TYPE_32(int8)
_CL_DECLARE_AS_TYPE_32(uint8)
_CL_DECLARE_AS_TYPE_32(long4)
_CL_DECLARE_AS_TYPE_32(ulong4)
_CL_DECLARE_AS_TYPE_32(float8)
_CL_DECLARE_AS_TYPE_32(double4)

/* 64 bytes */
#define _CL_DECLARE_AS_TYPE_64(SRC)             \
  _CL_DECLARE_AS_TYPE(SRC, int16)               \
  _CL_DECLARE_AS_TYPE(SRC, uint16)              \
  _CL_DECLARE_AS_TYPE(SRC, long8)               \
  _CL_DECLARE_AS_TYPE(SRC, ulong8)              \
  _CL_DECLARE_AS_TYPE(SRC, float16)             \
  _CL_DECLARE_AS_TYPE(SRC, double8)
_CL_DECLARE_AS_TYPE_64(int16)
_CL_DECLARE_AS_TYPE_64(uint16)
_CL_DECLARE_AS_TYPE_64(long8)
_CL_DECLARE_AS_TYPE_64(ulong8)
_CL_DECLARE_AS_TYPE_64(float16)
_CL_DECLARE_AS_TYPE_64(double8)

/* 128 bytes */
#define _CL_DECLARE_AS_TYPE_128(SRC)            \
  _CL_DECLARE_AS_TYPE(SRC, long16)              \
  _CL_DECLARE_AS_TYPE(SRC, ulong16)             \
  _CL_DECLARE_AS_TYPE(SRC, double16)
_CL_DECLARE_AS_TYPE_128(long16)
_CL_DECLARE_AS_TYPE_128(ulong16)
_CL_DECLARE_AS_TYPE_128(double16)

#define _CL_DECLARE_CONVERT_TYPE(SRC, DST)      \
  DST _cl_overloadable convert_##DST(SRC a);

/* 1 element */
#define _CL_DECLARE_CONVERT_TYPE_1(SRC)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, char)           \
  _CL_DECLARE_CONVERT_TYPE(SRC, short)          \
  _CL_DECLARE_CONVERT_TYPE(SRC, int)            \
  _CL_DECLARE_CONVERT_TYPE(SRC, long)           \
  _CL_DECLARE_CONVERT_TYPE(SRC, uchar)          \
  _CL_DECLARE_CONVERT_TYPE(SRC, ushort)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, uint)           \
  _CL_DECLARE_CONVERT_TYPE(SRC, ulong)          \
  _CL_DECLARE_CONVERT_TYPE(SRC, float)          \
  _CL_DECLARE_CONVERT_TYPE(SRC, double)
_CL_DECLARE_CONVERT_TYPE_1(char)
_CL_DECLARE_CONVERT_TYPE_1(short)
_CL_DECLARE_CONVERT_TYPE_1(int)
_CL_DECLARE_CONVERT_TYPE_1(long)
_CL_DECLARE_CONVERT_TYPE_1(uchar)
_CL_DECLARE_CONVERT_TYPE_1(ushort)
_CL_DECLARE_CONVERT_TYPE_1(uint)
_CL_DECLARE_CONVERT_TYPE_1(ulong)
_CL_DECLARE_CONVERT_TYPE_1(float)
_CL_DECLARE_CONVERT_TYPE_1(double)

/* 2 elements */
#define _CL_DECLARE_CONVERT_TYPE_2(SRC)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, char2)          \
  _CL_DECLARE_CONVERT_TYPE(SRC, short2)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, int2)           \
  _CL_DECLARE_CONVERT_TYPE(SRC, long2)          \
  _CL_DECLARE_CONVERT_TYPE(SRC, uchar2)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, ushort2)        \
  _CL_DECLARE_CONVERT_TYPE(SRC, uint2)          \
  _CL_DECLARE_CONVERT_TYPE(SRC, ulong2)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, float2)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, double2)
_CL_DECLARE_CONVERT_TYPE_2(char2)
_CL_DECLARE_CONVERT_TYPE_2(short2)
_CL_DECLARE_CONVERT_TYPE_2(int2)
_CL_DECLARE_CONVERT_TYPE_2(long2)
_CL_DECLARE_CONVERT_TYPE_2(uchar2)
_CL_DECLARE_CONVERT_TYPE_2(ushort2)
_CL_DECLARE_CONVERT_TYPE_2(uint2)
_CL_DECLARE_CONVERT_TYPE_2(ulong2)
_CL_DECLARE_CONVERT_TYPE_2(float2)
_CL_DECLARE_CONVERT_TYPE_2(double2)

/* 3 elements */
#define _CL_DECLARE_CONVERT_TYPE_3(SRC)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, char3)          \
  _CL_DECLARE_CONVERT_TYPE(SRC, short3)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, int3)           \
  _CL_DECLARE_CONVERT_TYPE(SRC, long3)          \
  _CL_DECLARE_CONVERT_TYPE(SRC, uchar3)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, ushort3)        \
  _CL_DECLARE_CONVERT_TYPE(SRC, uint3)          \
  _CL_DECLARE_CONVERT_TYPE(SRC, ulong3)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, float3)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, double3)
_CL_DECLARE_CONVERT_TYPE_3(char3)
_CL_DECLARE_CONVERT_TYPE_3(short3)
_CL_DECLARE_CONVERT_TYPE_3(int3)
_CL_DECLARE_CONVERT_TYPE_3(long3)
_CL_DECLARE_CONVERT_TYPE_3(uchar3)
_CL_DECLARE_CONVERT_TYPE_3(ushort3)
_CL_DECLARE_CONVERT_TYPE_3(uint3)
_CL_DECLARE_CONVERT_TYPE_3(ulong3)
_CL_DECLARE_CONVERT_TYPE_3(float3)
_CL_DECLARE_CONVERT_TYPE_3(double3)

/* 4 elements */
#define _CL_DECLARE_CONVERT_TYPE_4(SRC)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, char4)          \
  _CL_DECLARE_CONVERT_TYPE(SRC, short4)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, int4)           \
  _CL_DECLARE_CONVERT_TYPE(SRC, long4)          \
  _CL_DECLARE_CONVERT_TYPE(SRC, uchar4)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, ushort4)        \
  _CL_DECLARE_CONVERT_TYPE(SRC, uint4)          \
  _CL_DECLARE_CONVERT_TYPE(SRC, ulong4)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, float4)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, double4)
_CL_DECLARE_CONVERT_TYPE_4(char4)
_CL_DECLARE_CONVERT_TYPE_4(short4)
_CL_DECLARE_CONVERT_TYPE_4(int4)
_CL_DECLARE_CONVERT_TYPE_4(long4)
_CL_DECLARE_CONVERT_TYPE_4(uchar4)
_CL_DECLARE_CONVERT_TYPE_4(ushort4)
_CL_DECLARE_CONVERT_TYPE_4(uint4)
_CL_DECLARE_CONVERT_TYPE_4(ulong4)
_CL_DECLARE_CONVERT_TYPE_4(float4)
_CL_DECLARE_CONVERT_TYPE_4(double4)

/* 8 elements */
#define _CL_DECLARE_CONVERT_TYPE_8(SRC)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, char8)          \
  _CL_DECLARE_CONVERT_TYPE(SRC, short8)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, int8)           \
  _CL_DECLARE_CONVERT_TYPE(SRC, long8)          \
  _CL_DECLARE_CONVERT_TYPE(SRC, uchar8)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, ushort8)        \
  _CL_DECLARE_CONVERT_TYPE(SRC, uint8)          \
  _CL_DECLARE_CONVERT_TYPE(SRC, ulong8)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, float8)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, double8)
_CL_DECLARE_CONVERT_TYPE_8(char8)
_CL_DECLARE_CONVERT_TYPE_8(short8)
_CL_DECLARE_CONVERT_TYPE_8(int8)
_CL_DECLARE_CONVERT_TYPE_8(long8)
_CL_DECLARE_CONVERT_TYPE_8(uchar8)
_CL_DECLARE_CONVERT_TYPE_8(ushort8)
_CL_DECLARE_CONVERT_TYPE_8(uint8)
_CL_DECLARE_CONVERT_TYPE_8(ulong8)
_CL_DECLARE_CONVERT_TYPE_8(float8)
_CL_DECLARE_CONVERT_TYPE_8(double8)

/* 16 elements */
#define _CL_DECLARE_CONVERT_TYPE_16(SRC)        \
  _CL_DECLARE_CONVERT_TYPE(SRC, char16)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, short16)        \
  _CL_DECLARE_CONVERT_TYPE(SRC, int16)          \
  _CL_DECLARE_CONVERT_TYPE(SRC, long16)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, uchar16)        \
  _CL_DECLARE_CONVERT_TYPE(SRC, ushort16)       \
  _CL_DECLARE_CONVERT_TYPE(SRC, uint16)         \
  _CL_DECLARE_CONVERT_TYPE(SRC, ulong16)        \
  _CL_DECLARE_CONVERT_TYPE(SRC, float16)        \
  _CL_DECLARE_CONVERT_TYPE(SRC, double16)
_CL_DECLARE_CONVERT_TYPE_16(char16)
_CL_DECLARE_CONVERT_TYPE_16(short16)
_CL_DECLARE_CONVERT_TYPE_16(int16)
_CL_DECLARE_CONVERT_TYPE_16(long16)
_CL_DECLARE_CONVERT_TYPE_16(uchar16)
_CL_DECLARE_CONVERT_TYPE_16(ushort16)
_CL_DECLARE_CONVERT_TYPE_16(uint16)
_CL_DECLARE_CONVERT_TYPE_16(ulong16)
_CL_DECLARE_CONVERT_TYPE_16(float16)
_CL_DECLARE_CONVERT_TYPE_16(double16)



/* Work-Item Functions */

// uint get_work_dim();
uint get_global_size(uint);     // should return size_t
uint get_global_id(uint);       // should return size_t
// size_t get_local_size(uint);
uint get_local_id(uint);        // should return size_t
uint get_num_groups(uint);      // should return size_t
uint get_group_id(uint);        // should return size_t
// size_t get_global_offset(uint);

__attribute__ ((noinline)) void barrier (cl_mem_fence_flags flags);



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



/* Math Functions */

/* Naming scheme:
 *    [NAME]_[R]_[A]*
 * where [R] is the return type, and [A] are the argument types:
 *    I: int
 *    J: vector of int
 *    U: vector of uint or ulong
 *    S: scalar (float or double)
 *    V: vector of float or double
 */

#define _CL_DECLARE_FUNC_V_V(NAME)              \
  float    _cl_overloadable NAME(float   );     \
  float2   _cl_overloadable NAME(float2  );     \
  float3   _cl_overloadable NAME(float3  );     \
  float4   _cl_overloadable NAME(float4  );     \
  float8   _cl_overloadable NAME(float8  );     \
  float16  _cl_overloadable NAME(float16 );     \
  double   _cl_overloadable NAME(double  );     \
  double2  _cl_overloadable NAME(double2 );     \
  double3  _cl_overloadable NAME(double3 );     \
  double4  _cl_overloadable NAME(double4 );     \
  double8  _cl_overloadable NAME(double8 );     \
  double16 _cl_overloadable NAME(double16);
#define _CL_DECLARE_FUNC_V_VV(NAME)                     \
  float    _cl_overloadable NAME(float   , float   );   \
  float2   _cl_overloadable NAME(float2  , float2  );   \
  float3   _cl_overloadable NAME(float3  , float3  );   \
  float4   _cl_overloadable NAME(float4  , float4  );   \
  float8   _cl_overloadable NAME(float8  , float8  );   \
  float16  _cl_overloadable NAME(float16 , float16 );   \
  double   _cl_overloadable NAME(double  , double  );   \
  double2  _cl_overloadable NAME(double2 , double2 );   \
  double3  _cl_overloadable NAME(double3 , double3 );   \
  double4  _cl_overloadable NAME(double4 , double4 );   \
  double8  _cl_overloadable NAME(double8 , double8 );   \
  double16 _cl_overloadable NAME(double16, double16);
#define _CL_DECLARE_FUNC_V_VVV(NAME)                            \
  float    _cl_overloadable NAME(float   , float   , float   ); \
  float2   _cl_overloadable NAME(float2  , float2  , float2  ); \
  float3   _cl_overloadable NAME(float3  , float3  , float3  ); \
  float4   _cl_overloadable NAME(float4  , float4  , float4  ); \
  float8   _cl_overloadable NAME(float8  , float8  , float8  ); \
  float16  _cl_overloadable NAME(float16 , float16 , float16 ); \
  double   _cl_overloadable NAME(double  , double  , double  ); \
  double2  _cl_overloadable NAME(double2 , double2 , double2 ); \
  double3  _cl_overloadable NAME(double3 , double3 , double3 ); \
  double4  _cl_overloadable NAME(double4 , double4 , double4 ); \
  double8  _cl_overloadable NAME(double8 , double8 , double8 ); \
  double16 _cl_overloadable NAME(double16, double16, double16);
#define _CL_DECLARE_FUNC_V_VVS(NAME)                            \
  float2   _cl_overloadable NAME(float2  , float2  , float );   \
  float3   _cl_overloadable NAME(float3  , float3  , float );   \
  float4   _cl_overloadable NAME(float4  , float4  , float );   \
  float8   _cl_overloadable NAME(float8  , float8  , float );   \
  float16  _cl_overloadable NAME(float16 , float16 , float );   \
  double2  _cl_overloadable NAME(double2 , double2 , double);   \
  double3  _cl_overloadable NAME(double3 , double3 , double);   \
  double4  _cl_overloadable NAME(double4 , double4 , double);   \
  double8  _cl_overloadable NAME(double8 , double8 , double);   \
  double16 _cl_overloadable NAME(double16, double16, double);
#define _CL_DECLARE_FUNC_V_VSS(NAME)                            \
  float2   _cl_overloadable NAME(float2  , float , float );     \
  float3   _cl_overloadable NAME(float3  , float , float );     \
  float4   _cl_overloadable NAME(float4  , float , float );     \
  float8   _cl_overloadable NAME(float8  , float , float );     \
  float16  _cl_overloadable NAME(float16 , float , float );     \
  double2  _cl_overloadable NAME(double2 , double, double);     \
  double3  _cl_overloadable NAME(double3 , double, double);     \
  double4  _cl_overloadable NAME(double4 , double, double);     \
  double8  _cl_overloadable NAME(double8 , double, double);     \
  double16 _cl_overloadable NAME(double16, double, double);
#define _CL_DECLARE_FUNC_V_SSV(NAME)                            \
  float2   _cl_overloadable NAME(float , float , float2  );     \
  float3   _cl_overloadable NAME(float , float , float3  );     \
  float4   _cl_overloadable NAME(float , float , float4  );     \
  float8   _cl_overloadable NAME(float , float , float8  );     \
  float16  _cl_overloadable NAME(float , float , float16 );     \
  double2  _cl_overloadable NAME(double, double, double2 );     \
  double3  _cl_overloadable NAME(double, double, double3 );     \
  double4  _cl_overloadable NAME(double, double, double4 );     \
  double8  _cl_overloadable NAME(double, double, double8 );     \
  double16 _cl_overloadable NAME(double, double, double16);
#define _CL_DECLARE_FUNC_V_VVJ(NAME)                            \
  float    _cl_overloadable NAME(float   , float   , int   );   \
  float2   _cl_overloadable NAME(float2  , float2  , int2  );   \
  float3   _cl_overloadable NAME(float3  , float3  , int3  );   \
  float4   _cl_overloadable NAME(float4  , float4  , int4  );   \
  float8   _cl_overloadable NAME(float8  , float8  , int8  );   \
  float16  _cl_overloadable NAME(float16 , float16 , int16 );   \
  double   _cl_overloadable NAME(double  , double  , long  );   \
  double2  _cl_overloadable NAME(double2 , double2 , long2 );   \
  double3  _cl_overloadable NAME(double3 , double3 , long3 );   \
  double4  _cl_overloadable NAME(double4 , double4 , long4 );   \
  double8  _cl_overloadable NAME(double8 , double8 , long8 );   \
  double16 _cl_overloadable NAME(double16, double16, long16);
#define _CL_DECLARE_FUNC_V_U(NAME)              \
  float    _cl_overloadable NAME(uint   );      \
  float2   _cl_overloadable NAME(uint2  );      \
  float3   _cl_overloadable NAME(uint3  );      \
  float4   _cl_overloadable NAME(uint4  );      \
  float8   _cl_overloadable NAME(uint8  );      \
  float16  _cl_overloadable NAME(uint16 );      \
  double   _cl_overloadable NAME(ulong  );      \
  double2  _cl_overloadable NAME(ulong2 );      \
  double3  _cl_overloadable NAME(ulong3 );      \
  double4  _cl_overloadable NAME(ulong4 );      \
  double8  _cl_overloadable NAME(ulong8 );      \
  double16 _cl_overloadable NAME(ulong16);
#define _CL_DECLARE_FUNC_V_VS(NAME)                     \
  float2   _cl_overloadable NAME(float2  , float );     \
  float3   _cl_overloadable NAME(float3  , float );     \
  float4   _cl_overloadable NAME(float4  , float );     \
  float8   _cl_overloadable NAME(float8  , float );     \
  float16  _cl_overloadable NAME(float16 , float );     \
  double2  _cl_overloadable NAME(double2 , double);     \
  double3  _cl_overloadable NAME(double3 , double);     \
  double4  _cl_overloadable NAME(double4 , double);     \
  double8  _cl_overloadable NAME(double8 , double);     \
  double16 _cl_overloadable NAME(double16, double);
#define _CL_DECLARE_FUNC_V_VJ(NAME)                     \
  float    _cl_overloadable NAME(float   , int  );      \
  float2   _cl_overloadable NAME(float2  , int2 );      \
  float3   _cl_overloadable NAME(float3  , int3 );      \
  float4   _cl_overloadable NAME(float4  , int4 );      \
  float8   _cl_overloadable NAME(float8  , int8 );      \
  float16  _cl_overloadable NAME(float16 , int16);      \
  double   _cl_overloadable NAME(double  , int  );      \
  double2  _cl_overloadable NAME(double2 , int2 );      \
  double3  _cl_overloadable NAME(double3 , int3 );      \
  double4  _cl_overloadable NAME(double4 , int4 );      \
  double8  _cl_overloadable NAME(double8 , int8 );      \
  double16 _cl_overloadable NAME(double16, int16);
#define _CL_DECLARE_FUNC_J_VV(NAME)                     \
  int    _cl_overloadable NAME(float   , float   );     \
  int2   _cl_overloadable NAME(float2  , float2  );     \
  int3   _cl_overloadable NAME(float3  , float3  );     \
  int4   _cl_overloadable NAME(float4  , float4  );     \
  int8   _cl_overloadable NAME(float8  , float8  );     \
  int16  _cl_overloadable NAME(float16 , float16 );     \
  int    _cl_overloadable NAME(double  , double  );     \
  long2  _cl_overloadable NAME(double2 , double2 );     \
  long3  _cl_overloadable NAME(double3 , double3 );     \
  long4  _cl_overloadable NAME(double4 , double4 );     \
  long8  _cl_overloadable NAME(double8 , double8 );     \
  long16 _cl_overloadable NAME(double16, double16);
#define _CL_DECLARE_FUNC_V_VI(NAME)                     \
  float2   _cl_overloadable NAME(float2  , int);        \
  float3   _cl_overloadable NAME(float3  , int);        \
  float4   _cl_overloadable NAME(float4  , int);        \
  float8   _cl_overloadable NAME(float8  , int);        \
  float16  _cl_overloadable NAME(float16 , int);        \
  double2  _cl_overloadable NAME(double2 , int);        \
  double3  _cl_overloadable NAME(double3 , int);        \
  double4  _cl_overloadable NAME(double4 , int);        \
  double8  _cl_overloadable NAME(double8 , int);        \
  double16 _cl_overloadable NAME(double16, int);
#define _CL_DECLARE_FUNC_V_VPV(NAME)                                    \
  float    _cl_overloadable NAME(float   , __global  float   *);        \
  float2   _cl_overloadable NAME(float2  , __global  float2  *);        \
  float3   _cl_overloadable NAME(float3  , __global  float3  *);        \
  float4   _cl_overloadable NAME(float4  , __global  float4  *);        \
  float8   _cl_overloadable NAME(float8  , __global  float8  *);        \
  float16  _cl_overloadable NAME(float16 , __global  float16 *);        \
  double   _cl_overloadable NAME(double  , __global  double  *);        \
  double2  _cl_overloadable NAME(double2 , __global  double2 *);        \
  double3  _cl_overloadable NAME(double3 , __global  double3 *);        \
  double4  _cl_overloadable NAME(double4 , __global  double4 *);        \
  double8  _cl_overloadable NAME(double8 , __global  double8 *);        \
  double16 _cl_overloadable NAME(double16, __global  double16*);        \
  float    _cl_overloadable NAME(float   , __local   float   *);        \
  float2   _cl_overloadable NAME(float2  , __local   float2  *);        \
  float3   _cl_overloadable NAME(float3  , __local   float3  *);        \
  float4   _cl_overloadable NAME(float4  , __local   float4  *);        \
  float8   _cl_overloadable NAME(float8  , __local   float8  *);        \
  float16  _cl_overloadable NAME(float16 , __local   float16 *);        \
  double   _cl_overloadable NAME(double  , __local   double  *);        \
  double2  _cl_overloadable NAME(double2 , __local   double2 *);        \
  double3  _cl_overloadable NAME(double3 , __local   double3 *);        \
  double4  _cl_overloadable NAME(double4 , __local   double4 *);        \
  double8  _cl_overloadable NAME(double8 , __local   double8 *);        \
  double16 _cl_overloadable NAME(double16, __local   double16*);        \
  /* __private is not supported yet                                     \
  float    _cl_overloadable NAME(float   , __private float   *);        \
  float2   _cl_overloadable NAME(float2  , __private float2  *);        \
  float3   _cl_overloadable NAME(float3  , __private float3  *);        \
  float4   _cl_overloadable NAME(float4  , __private float4  *);        \
  float8   _cl_overloadable NAME(float8  , __private float8  *);        \
  float16  _cl_overloadable NAME(float16 , __private float16 *);        \
  double   _cl_overloadable NAME(double  , __private double  *);        \
  double2  _cl_overloadable NAME(double2 , __private double2 *);        \
  double3  _cl_overloadable NAME(double3 , __private double3 *);        \
  double4  _cl_overloadable NAME(double4 , __private double4 *);        \
  double8  _cl_overloadable NAME(double8 , __private double8 *);        \
  double16 _cl_overloadable NAME(double16, __private double16*);        \
  */
#define _CL_DECLARE_FUNC_V_SV(NAME)                     \
  float2   _cl_overloadable NAME(float , float2  );     \
  float3   _cl_overloadable NAME(float , float3  );     \
  float4   _cl_overloadable NAME(float , float4  );     \
  float8   _cl_overloadable NAME(float , float8  );     \
  float16  _cl_overloadable NAME(float , float16 );     \
  double2  _cl_overloadable NAME(double, double2 );     \
  double3  _cl_overloadable NAME(double, double3 );     \
  double4  _cl_overloadable NAME(double, double4 );     \
  double8  _cl_overloadable NAME(double, double8 );     \
  double16 _cl_overloadable NAME(double, double16);
#define _CL_DECLARE_FUNC_J_V(NAME)              \
  int   _cl_overloadable NAME(float   );        \
  int2  _cl_overloadable NAME(float2  );        \
  int3  _cl_overloadable NAME(float3  );        \
  int4  _cl_overloadable NAME(float4  );        \
  int8  _cl_overloadable NAME(float8  );        \
  int16 _cl_overloadable NAME(float16 );        \
  int   _cl_overloadable NAME(double  );        \
  int2  _cl_overloadable NAME(double2 );        \
  int3  _cl_overloadable NAME(double3 );        \
  int4  _cl_overloadable NAME(double4 );        \
  int8  _cl_overloadable NAME(double8 );        \
  int16 _cl_overloadable NAME(double16);
#define _CL_DECLARE_FUNC_K_V(NAME)              \
  int    _cl_overloadable NAME(float   );       \
  int2   _cl_overloadable NAME(float2  );       \
  int3   _cl_overloadable NAME(float3  );       \
  int4   _cl_overloadable NAME(float4  );       \
  int8   _cl_overloadable NAME(float8  );       \
  int16  _cl_overloadable NAME(float16 );       \
  int    _cl_overloadable NAME(double  );       \
  long2  _cl_overloadable NAME(double2 );       \
  long3  _cl_overloadable NAME(double3 );       \
  long4  _cl_overloadable NAME(double4 );       \
  long8  _cl_overloadable NAME(double8 );       \
  long16 _cl_overloadable NAME(double16);
#define _CL_DECLARE_FUNC_S_V(NAME)              \
  float  _cl_overloadable NAME(float   );       \
  float  _cl_overloadable NAME(float2  );       \
  float  _cl_overloadable NAME(float3  );       \
  float  _cl_overloadable NAME(float4  );       \
  float  _cl_overloadable NAME(float8  );       \
  float  _cl_overloadable NAME(float16 );       \
  double _cl_overloadable NAME(double  );       \
  double _cl_overloadable NAME(double2 );       \
  double _cl_overloadable NAME(double3 );       \
  double _cl_overloadable NAME(double4 );       \
  double _cl_overloadable NAME(double8 );       \
  double _cl_overloadable NAME(double16);
#define _CL_DECLARE_FUNC_S_VV(NAME)                     \
  float  _cl_overloadable NAME(float   , float   );     \
  float  _cl_overloadable NAME(float2  , float2  );     \
  float  _cl_overloadable NAME(float3  , float3  );     \
  float  _cl_overloadable NAME(float4  , float4  );     \
  float  _cl_overloadable NAME(float8  , float8  );     \
  float  _cl_overloadable NAME(float16 , float16 );     \
  double _cl_overloadable NAME(double  , double  );     \
  double _cl_overloadable NAME(double2 , double2 );     \
  double _cl_overloadable NAME(double3 , double3 );     \
  double _cl_overloadable NAME(double4 , double4 );     \
  double _cl_overloadable NAME(double8 , double8 );     \
  double _cl_overloadable NAME(double16, double16);

/* Move built-in declarations out of the way. (There should be a
   better way of doing so.) These five functions are built-in math
   functions for all Clang languages; see Clang's "Builtin.def".
   */
#define cos _cl_cos
#define fma _cl_fma
#define pow _cl_pow
#define sin _cl_sin
#define sqrt _cl_sqrt

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
// sincos
_CL_DECLARE_FUNC_V_V(sinh)
_CL_DECLARE_FUNC_V_V(sinpi)
_CL_DECLARE_FUNC_V_V(sqrt)
_CL_DECLARE_FUNC_V_V(tan)
_CL_DECLARE_FUNC_V_V(tanh)
_CL_DECLARE_FUNC_V_V(tanpi)
_CL_DECLARE_FUNC_V_V(tgamma)
_CL_DECLARE_FUNC_V_V(trunc)



/* Integer Constants */

#define CHAR_BIT  8
#define CHAR_MAX  SCHAR_MAX
#define CHAR_MIN  SCHAR_MIN
#define INT_MAX   2147483647
#define INT_MIN   (-2147483647 - 1)
#define LONG_MAX  0x7fffffffffffffffL
#define LONG_MIN  (-0x7fffffffffffffffL - 1)
#define SCHAR_MAX 127
#define SCHAR_MIN (-127 - 1)
#define SHRT_MAX  32767
#define SHRT_MIN  (-32767 - 1)
#define UCHAR_MAX 255
#define USHRT_MAX 65535
#define UINT_MAX  0xffffffff
#define ULONG_MAX 0xffffffffffffffffUL



/* Integer Functions */

#define _CL_DECLARE_FUNC_G_G(NAME)              \
  char     _cl_overloadable NAME(char    );     \
  char2    _cl_overloadable NAME(char2   );     \
  char3    _cl_overloadable NAME(char3   );     \
  char4    _cl_overloadable NAME(char4   );     \
  char8    _cl_overloadable NAME(char8   );     \
  char16   _cl_overloadable NAME(char16  );     \
  short    _cl_overloadable NAME(short   );     \
  short2   _cl_overloadable NAME(short2  );     \
  short3   _cl_overloadable NAME(short3  );     \
  short4   _cl_overloadable NAME(short4  );     \
  short8   _cl_overloadable NAME(short8  );     \
  short16  _cl_overloadable NAME(short16 );     \
  int      _cl_overloadable NAME(int     );     \
  int2     _cl_overloadable NAME(int2    );     \
  int3     _cl_overloadable NAME(int3    );     \
  int4     _cl_overloadable NAME(int4    );     \
  int8     _cl_overloadable NAME(int8    );     \
  int16    _cl_overloadable NAME(int16   );     \
  long     _cl_overloadable NAME(long    );     \
  long2    _cl_overloadable NAME(long2   );     \
  long3    _cl_overloadable NAME(long3   );     \
  long4    _cl_overloadable NAME(long4   );     \
  long8    _cl_overloadable NAME(long8   );     \
  long16   _cl_overloadable NAME(long16  );     \
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
  ulong    _cl_overloadable NAME(ulong   );     \
  ulong2   _cl_overloadable NAME(ulong2  );     \
  ulong3   _cl_overloadable NAME(ulong3  );     \
  ulong4   _cl_overloadable NAME(ulong4  );     \
  ulong8   _cl_overloadable NAME(ulong8  );     \
  ulong16  _cl_overloadable NAME(ulong16 );
#define _CL_DECLARE_FUNC_G_GG(NAME)                     \
  char     _cl_overloadable NAME(char    , char    );   \
  char2    _cl_overloadable NAME(char2   , char2   );   \
  char3    _cl_overloadable NAME(char3   , char3   );   \
  char4    _cl_overloadable NAME(char4   , char4   );   \
  char8    _cl_overloadable NAME(char8   , char8   );   \
  char16   _cl_overloadable NAME(char16  , char16  );   \
  short    _cl_overloadable NAME(short   , short   );   \
  short2   _cl_overloadable NAME(short2  , short2  );   \
  short3   _cl_overloadable NAME(short3  , short3  );   \
  short4   _cl_overloadable NAME(short4  , short4  );   \
  short8   _cl_overloadable NAME(short8  , short8  );   \
  short16  _cl_overloadable NAME(short16 , short16 );   \
  int      _cl_overloadable NAME(int     , int     );   \
  int2     _cl_overloadable NAME(int2    , int2    );   \
  int3     _cl_overloadable NAME(int3    , int3    );   \
  int4     _cl_overloadable NAME(int4    , int4    );   \
  int8     _cl_overloadable NAME(int8    , int8    );   \
  int16    _cl_overloadable NAME(int16   , int16   );   \
  long     _cl_overloadable NAME(long    , long    );   \
  long2    _cl_overloadable NAME(long2   , long2   );   \
  long3    _cl_overloadable NAME(long3   , long3   );   \
  long4    _cl_overloadable NAME(long4   , long4   );   \
  long8    _cl_overloadable NAME(long8   , long8   );   \
  long16   _cl_overloadable NAME(long16  , long16  );   \
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
  ulong    _cl_overloadable NAME(ulong   , ulong   );   \
  ulong2   _cl_overloadable NAME(ulong2  , ulong2  );   \
  ulong3   _cl_overloadable NAME(ulong3  , ulong3  );   \
  ulong4   _cl_overloadable NAME(ulong4  , ulong4  );   \
  ulong8   _cl_overloadable NAME(ulong8  , ulong8  );   \
  ulong16  _cl_overloadable NAME(ulong16 , ulong16 );
#define _CL_DECLARE_FUNC_G_GGG(NAME)                            \
  char     _cl_overloadable NAME(char    , char    , char    ); \
  char2    _cl_overloadable NAME(char2   , char2   , char2   ); \
  char3    _cl_overloadable NAME(char3   , char3   , char3   ); \
  char4    _cl_overloadable NAME(char4   , char4   , char4   ); \
  char8    _cl_overloadable NAME(char8   , char8   , char8   ); \
  char16   _cl_overloadable NAME(char16  , char16  , char16  ); \
  short    _cl_overloadable NAME(short   , short   , short   ); \
  short2   _cl_overloadable NAME(short2  , short2  , short2  ); \
  short3   _cl_overloadable NAME(short3  , short3  , short3  ); \
  short4   _cl_overloadable NAME(short4  , short4  , short4  ); \
  short8   _cl_overloadable NAME(short8  , short8  , short8  ); \
  short16  _cl_overloadable NAME(short16 , short16 , short16 ); \
  int      _cl_overloadable NAME(int     , int     , int     ); \
  int2     _cl_overloadable NAME(int2    , int2    , int2    ); \
  int3     _cl_overloadable NAME(int3    , int3    , int3    ); \
  int4     _cl_overloadable NAME(int4    , int4    , int4    ); \
  int8     _cl_overloadable NAME(int8    , int8    , int8    ); \
  int16    _cl_overloadable NAME(int16   , int16   , int16   ); \
  long     _cl_overloadable NAME(long    , long    , long    ); \
  long2    _cl_overloadable NAME(long2   , long2   , long2   ); \
  long3    _cl_overloadable NAME(long3   , long3   , long3   ); \
  long4    _cl_overloadable NAME(long4   , long4   , long4   ); \
  long8    _cl_overloadable NAME(long8   , long8   , long8   ); \
  long16   _cl_overloadable NAME(long16  , long16  , long16  ); \
  uchar    _cl_overloadable NAME(uchar   , uchar   , uchar   ); \
  uchar2   _cl_overloadable NAME(uchar2  , uchar2  , uchar2  ); \
  uchar3   _cl_overloadable NAME(uchar3  , uchar3  , uchar3  ); \
  uchar4   _cl_overloadable NAME(uchar4  , uchar4  , uchar4  ); \
  uchar8   _cl_overloadable NAME(uchar8  , uchar8  , uchar8  ); \
  uchar16  _cl_overloadable NAME(uchar16 , uchar16 , uchar16 ); \
  ushort   _cl_overloadable NAME(ushort  , ushort  , ushort  ); \
  ushort2  _cl_overloadable NAME(ushort2 , ushort2 , ushort2 ); \
  ushort3  _cl_overloadable NAME(ushort3 , ushort3 , ushort3 ); \
  ushort4  _cl_overloadable NAME(ushort4 , ushort4 , ushort4 ); \
  ushort8  _cl_overloadable NAME(ushort8 , ushort8 , ushort8 ); \
  ushort16 _cl_overloadable NAME(ushort16, ushort16, ushort16); \
  uint     _cl_overloadable NAME(uint    , uint    , uint    ); \
  uint2    _cl_overloadable NAME(uint2   , uint2   , uint2   ); \
  uint3    _cl_overloadable NAME(uint3   , uint3   , uint3   ); \
  uint4    _cl_overloadable NAME(uint4   , uint4   , uint4   ); \
  uint8    _cl_overloadable NAME(uint8   , uint8   , uint8   ); \
  uint16   _cl_overloadable NAME(uint16  , uint16  , uint16  ); \
  ulong    _cl_overloadable NAME(ulong   , ulong   , ulong   ); \
  ulong2   _cl_overloadable NAME(ulong2  , ulong2  , ulong2  ); \
  ulong3   _cl_overloadable NAME(ulong3  , ulong3  , ulong3  ); \
  ulong4   _cl_overloadable NAME(ulong4  , ulong4  , ulong4  ); \
  ulong8   _cl_overloadable NAME(ulong8  , ulong8  , ulong8  ); \
  ulong16  _cl_overloadable NAME(ulong16 , ulong16 , ulong16 );
#define _CL_DECLARE_FUNC_G_GS(NAME)                     \
  char2    _cl_overloadable NAME(char2   , char  );     \
  char3    _cl_overloadable NAME(char3   , char  );     \
  char4    _cl_overloadable NAME(char4   , char  );     \
  char8    _cl_overloadable NAME(char8   , char  );     \
  char16   _cl_overloadable NAME(char16  , char  );     \
  short2   _cl_overloadable NAME(short2  , short );     \
  short3   _cl_overloadable NAME(short3  , short );     \
  short4   _cl_overloadable NAME(short4  , short );     \
  short8   _cl_overloadable NAME(short8  , short );     \
  short16  _cl_overloadable NAME(short16 , short );     \
  int2     _cl_overloadable NAME(int2    , int   );     \
  int3     _cl_overloadable NAME(int3    , int   );     \
  int4     _cl_overloadable NAME(int4    , int   );     \
  int8     _cl_overloadable NAME(int8    , int   );     \
  int16    _cl_overloadable NAME(int16   , int   );     \
  long2    _cl_overloadable NAME(long2   , long  );     \
  long3    _cl_overloadable NAME(long3   , long  );     \
  long4    _cl_overloadable NAME(long4   , long  );     \
  long8    _cl_overloadable NAME(long8   , long  );     \
  long16   _cl_overloadable NAME(long16  , long  );     \
  uchar2   _cl_overloadable NAME(uchar2  , uchar );     \
  uchar3   _cl_overloadable NAME(uchar3  , uchar );     \
  uchar4   _cl_overloadable NAME(uchar4  , uchar );     \
  uchar8   _cl_overloadable NAME(uchar8  , uchar );     \
  uchar16  _cl_overloadable NAME(uchar16 , uchar );     \
  ushort2  _cl_overloadable NAME(ushort2 , ushort);     \
  ushort3  _cl_overloadable NAME(ushort3 , ushort);     \
  ushort4  _cl_overloadable NAME(ushort4 , ushort);     \
  ushort8  _cl_overloadable NAME(ushort8 , ushort);     \
  ushort16 _cl_overloadable NAME(ushort16, ushort);     \
  uint2    _cl_overloadable NAME(uint2   , uint  );     \
  uint3    _cl_overloadable NAME(uint3   , uint  );     \
  uint4    _cl_overloadable NAME(uint4   , uint  );     \
  uint8    _cl_overloadable NAME(uint8   , uint  );     \
  uint16   _cl_overloadable NAME(uint16  , uint  );     \
  ulong2   _cl_overloadable NAME(ulong2  , ulong );     \
  ulong3   _cl_overloadable NAME(ulong3  , ulong );     \
  ulong4   _cl_overloadable NAME(ulong4  , ulong );     \
  ulong8   _cl_overloadable NAME(ulong8  , ulong );     \
  ulong16  _cl_overloadable NAME(ulong16 , ulong );
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
  ulong    _cl_overloadable NAME(long    );     \
  ulong2   _cl_overloadable NAME(long2   );     \
  ulong3   _cl_overloadable NAME(long3   );     \
  ulong4   _cl_overloadable NAME(long4   );     \
  ulong8   _cl_overloadable NAME(long8   );     \
  ulong16  _cl_overloadable NAME(long16  );     \
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
  ulong    _cl_overloadable NAME(ulong   );     \
  ulong2   _cl_overloadable NAME(ulong2  );     \
  ulong3   _cl_overloadable NAME(ulong3  );     \
  ulong4   _cl_overloadable NAME(ulong4  );     \
  ulong8   _cl_overloadable NAME(ulong8  );     \
  ulong16  _cl_overloadable NAME(ulong16 );
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
  ulong    _cl_overloadable NAME(long    , long    );   \
  ulong2   _cl_overloadable NAME(long2   , long2   );   \
  ulong3   _cl_overloadable NAME(long3   , long3   );   \
  ulong4   _cl_overloadable NAME(long4   , long4   );   \
  ulong8   _cl_overloadable NAME(long8   , long8   );   \
  ulong16  _cl_overloadable NAME(long16  , long16  );   \
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
  ulong    _cl_overloadable NAME(ulong   , ulong   );   \
  ulong2   _cl_overloadable NAME(ulong2  , ulong2  );   \
  ulong3   _cl_overloadable NAME(ulong3  , ulong3  );   \
  ulong4   _cl_overloadable NAME(ulong4  , ulong4  );   \
  ulong8   _cl_overloadable NAME(ulong8  , ulong8  );   \
  ulong16  _cl_overloadable NAME(ulong16 , ulong16 );
#define _CL_DECLARE_FUNC_LG_GUG(NAME)                   \
  short    _cl_overloadable NAME(char    , uchar    );  \
  short2   _cl_overloadable NAME(char2   , uchar2   );  \
  short3   _cl_overloadable NAME(char3   , uchar3   );  \
  short4   _cl_overloadable NAME(char4   , uchar4   );  \
  short8   _cl_overloadable NAME(char8   , uchar8   );  \
  short16  _cl_overloadable NAME(char16  , uchar16  );  \
  int      _cl_overloadable NAME(short   , ushort   );  \
  int2     _cl_overloadable NAME(short2  , ushort2  );  \
  int3     _cl_overloadable NAME(short3  , ushort3  );  \
  int4     _cl_overloadable NAME(short4  , ushort4  );  \
  int8     _cl_overloadable NAME(short8  , ushort8  );  \
  int16    _cl_overloadable NAME(short16 , ushort16 );  \
  long     _cl_overloadable NAME(int     , uint     );  \
  long2    _cl_overloadable NAME(int2    , uint2    );  \
  long3    _cl_overloadable NAME(int3    , uint3    );  \
  long4    _cl_overloadable NAME(int4    , uint4    );  \
  long8    _cl_overloadable NAME(int8    , uint8    );  \
  long16   _cl_overloadable NAME(int16   , uint16   );  \
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
  ulong    _cl_overloadable NAME(uint    , uint    );   \
  ulong2   _cl_overloadable NAME(uint2   , uint2   );   \
  ulong3   _cl_overloadable NAME(uint3   , uint3   );   \
  ulong4   _cl_overloadable NAME(uint4   , uint4   );   \
  ulong8   _cl_overloadable NAME(uint8   , uint8   );   \
  ulong16  _cl_overloadable NAME(uint16  , uint16  );
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
  int _cl_overloadable NAME(long   );           \
  int _cl_overloadable NAME(long2  );           \
  int _cl_overloadable NAME(long3  );           \
  int _cl_overloadable NAME(long4  );           \
  int _cl_overloadable NAME(long8  );           \
  int _cl_overloadable NAME(long16 );
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
double4 _cl_overloadable cross(double4, double4);
double3 _cl_overloadable cross(double3, double3);
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
_CL_DECLARE_VLOAD(short , __global)
_CL_DECLARE_VLOAD(int   , __global)
_CL_DECLARE_VLOAD(long  , __global)
_CL_DECLARE_VLOAD(uchar , __global)
_CL_DECLARE_VLOAD(ushort, __global)
_CL_DECLARE_VLOAD(uint  , __global)
_CL_DECLARE_VLOAD(ulong , __global)
_CL_DECLARE_VLOAD(float , __global)
_CL_DECLARE_VLOAD(double, __global)

_CL_DECLARE_VLOAD(char  , __local)
_CL_DECLARE_VLOAD(short , __local)
_CL_DECLARE_VLOAD(int   , __local)
_CL_DECLARE_VLOAD(long  , __local)
_CL_DECLARE_VLOAD(uchar , __local)
_CL_DECLARE_VLOAD(ushort, __local)
_CL_DECLARE_VLOAD(uint  , __local)
_CL_DECLARE_VLOAD(ulong , __local)
_CL_DECLARE_VLOAD(float , __local)
_CL_DECLARE_VLOAD(double, __local)

_CL_DECLARE_VLOAD(char  , __constant)
_CL_DECLARE_VLOAD(short , __constant)
_CL_DECLARE_VLOAD(int   , __constant)
_CL_DECLARE_VLOAD(long  , __constant)
_CL_DECLARE_VLOAD(uchar , __constant)
_CL_DECLARE_VLOAD(ushort, __constant)
_CL_DECLARE_VLOAD(uint  , __constant)
_CL_DECLARE_VLOAD(ulong , __constant)
_CL_DECLARE_VLOAD(float , __constant)
_CL_DECLARE_VLOAD(double, __constant)

/* __private is not supported yet               \
_CL_DECLARE_VLOAD(char  , __private)
_CL_DECLARE_VLOAD(short , __private)
_CL_DECLARE_VLOAD(int   , __private)
_CL_DECLARE_VLOAD(long  , __private)
_CL_DECLARE_VLOAD(uchar , __private)
_CL_DECLARE_VLOAD(ushort, __private)
_CL_DECLARE_VLOAD(uint  , __private)
_CL_DECLARE_VLOAD(ulong , __private)
_CL_DECLARE_VLOAD(float , __private)
_CL_DECLARE_VLOAD(double, __private)
*/

#define _CL_DECLARE_VSTORE(TYPE, MOD)                                   \
  void _cl_overloadable vstore2 (TYPE##2  data, size_t offset, MOD TYPE *p); \
  void _cl_overloadable vstore3 (TYPE##3  data, size_t offset, MOD TYPE *p); \
  void _cl_overloadable vstore4 (TYPE##4  data, size_t offset, MOD TYPE *p); \
  void _cl_overloadable vstore8 (TYPE##8  data, size_t offset, MOD TYPE *p); \
  void _cl_overloadable vstore16(TYPE##16 data, size_t offset, MOD TYPE *p);

_CL_DECLARE_VSTORE(char  , __global)
_CL_DECLARE_VSTORE(short , __global)
_CL_DECLARE_VSTORE(int   , __global)
_CL_DECLARE_VSTORE(long  , __global)
_CL_DECLARE_VSTORE(uchar , __global)
_CL_DECLARE_VSTORE(ushort, __global)
_CL_DECLARE_VSTORE(uint  , __global)
_CL_DECLARE_VSTORE(ulong , __global)
_CL_DECLARE_VSTORE(float , __global)
_CL_DECLARE_VSTORE(double, __global)

_CL_DECLARE_VSTORE(char  , __local)
_CL_DECLARE_VSTORE(short , __local)
_CL_DECLARE_VSTORE(int   , __local)
_CL_DECLARE_VSTORE(long  , __local)
_CL_DECLARE_VSTORE(uchar , __local)
_CL_DECLARE_VSTORE(ushort, __local)
_CL_DECLARE_VSTORE(uint  , __local)
_CL_DECLARE_VSTORE(ulong , __local)
_CL_DECLARE_VSTORE(float , __local)
_CL_DECLARE_VSTORE(double, __local)

/* __private is not supported yet
_CL_DECLARE_VSTORE(char  , __private)
_CL_DECLARE_VSTORE(short , __private)
_CL_DECLARE_VSTORE(int   , __private)
_CL_DECLARE_VSTORE(long  , __private)
_CL_DECLARE_VSTORE(uchar , __private)
_CL_DECLARE_VSTORE(ushort, __private)
_CL_DECLARE_VSTORE(uint  , __private)
_CL_DECLARE_VSTORE(ulong , __private)
_CL_DECLARE_VSTORE(float , __private)
_CL_DECLARE_VSTORE(double, __private)
*/



/* Miscellaneous Vector Functions */

// convert a vector type to a scalar type
_CL_DECLARE_FUNC_I_IG(_cl_scalar)
_CL_DECLARE_FUNC_S_V(_cl_scalar)
#define vec_step(a) (sizeof(a) / sizeof(_cl_scalar(a)))



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
