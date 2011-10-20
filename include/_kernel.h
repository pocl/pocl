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

/* Trigonometric functions */

#define cos cl_cos
float __attribute__ ((overloadable)) cl_cos(float a);
float2 __attribute__ ((overloadable)) cl_cos(float2 a);
float3 __attribute__ ((overloadable)) cl_cos(float3 a);
float4 __attribute__ ((overloadable)) cl_cos(float4 a);
float8 __attribute__ ((overloadable)) cl_cos(float8 a);
float16 __attribute__ ((overloadable)) cl_cos(float16 a);
double __attribute__ ((overloadable)) cl_cos(double a);
double2 __attribute__ ((overloadable)) cl_cos(double2 a);
double3 __attribute__ ((overloadable)) cl_cos(double3 a);
double4 __attribute__ ((overloadable)) cl_cos(double4 a);
double8 __attribute__ ((overloadable)) cl_cos(double8 a);
double16 __attribute__ ((overloadable)) cl_cos(double16 a);

#define dot cl_dot
float __attribute__ ((overloadable)) cl_dot(float a, float b);
float __attribute__ ((overloadable)) cl_dot(float2 a, float2 b);
float __attribute__ ((overloadable)) cl_dot(float3 a, float3 b);
float __attribute__ ((overloadable)) cl_dot(float4 a, float4 b);
float __attribute__ ((overloadable)) cl_dot(float8 a, float8 b);
float __attribute__ ((overloadable)) cl_dot(float16 a, float16 b);
double __attribute__ ((overloadable)) cl_dot(double a, double b);
double __attribute__ ((overloadable)) cl_dot(double2 a, double2 b);
double __attribute__ ((overloadable)) cl_dot(double3 a, double3 b);
double __attribute__ ((overloadable)) cl_dot(double4 a, double4 b);
double __attribute__ ((overloadable)) cl_dot(double8 a, double8 b);
double __attribute__ ((overloadable)) cl_dot(double16 a, double16 b);

#define fabs cl_fabs
float __attribute__ ((overloadable)) cl_fabs(float a);
float2 __attribute__ ((overloadable)) cl_fabs(float2 a);
float3 __attribute__ ((overloadable)) cl_fabs(float3 a);
float4 __attribute__ ((overloadable)) cl_fabs(float4 a);
float8 __attribute__ ((overloadable)) cl_fabs(float8 a);
float16 __attribute__ ((overloadable)) cl_fabs(float16 a);
double __attribute__ ((overloadable)) cl_fabs(double a);
double2 __attribute__ ((overloadable)) cl_fabs(double2 a);
double3 __attribute__ ((overloadable)) cl_fabs(double3 a);
double4 __attribute__ ((overloadable)) cl_fabs(double4 a);
double8 __attribute__ ((overloadable)) cl_fabs(double8 a);
double16 __attribute__ ((overloadable)) cl_fabs(double16 a);

#define sin cl_sin
float __attribute__ ((overloadable)) cl_sin(float a);
float2 __attribute__ ((overloadable)) cl_sin(float2 a);
float3 __attribute__ ((overloadable)) cl_sin(float3 a);
float4 __attribute__ ((overloadable)) cl_sin(float4 a);
float8 __attribute__ ((overloadable)) cl_sin(float8 a);
float16 __attribute__ ((overloadable)) cl_sin(float16 a);
double __attribute__ ((overloadable)) cl_sin(double a);
double2 __attribute__ ((overloadable)) cl_sin(double2 a);
double3 __attribute__ ((overloadable)) cl_sin(double3 a);
double4 __attribute__ ((overloadable)) cl_sin(double4 a);
double8 __attribute__ ((overloadable)) cl_sin(double8 a);
double16 __attribute__ ((overloadable)) cl_sin(double16 a);

#define sqrt cl_sqrt
float __attribute__ ((overloadable)) cl_sqrt(float a);
float2 __attribute__ ((overloadable)) cl_sqrt(float2 a);
float3 __attribute__ ((overloadable)) cl_sqrt(float3 a);
float4 __attribute__ ((overloadable)) cl_sqrt(float4 a);
float8 __attribute__ ((overloadable)) cl_sqrt(float8 a);
float16 __attribute__ ((overloadable)) cl_sqrt(float16 a);
double __attribute__ ((overloadable)) cl_sqrt(double a);
double2 __attribute__ ((overloadable)) cl_sqrt(double2 a);
double3 __attribute__ ((overloadable)) cl_sqrt(double3 a);
double4 __attribute__ ((overloadable)) cl_sqrt(double4 a);
double8 __attribute__ ((overloadable)) cl_sqrt(double8 a);
double16 __attribute__ ((overloadable)) cl_sqrt(double16 a);

#define tan cl_tan
float __attribute__ ((overloadable)) cl_tan(float a);
float2 __attribute__ ((overloadable)) cl_tan(float2 a);
float3 __attribute__ ((overloadable)) cl_tan(float3 a);
float4 __attribute__ ((overloadable)) cl_tan(float4 a);
float8 __attribute__ ((overloadable)) cl_tan(float8 a);
float16 __attribute__ ((overloadable)) cl_tan(float16 a);
double __attribute__ ((overloadable)) cl_tan(double a);
double2 __attribute__ ((overloadable)) cl_tan(double2 a);
double3 __attribute__ ((overloadable)) cl_tan(double3 a);
double4 __attribute__ ((overloadable)) cl_tan(double4 a);
double8 __attribute__ ((overloadable)) cl_tan(double8 a);
double16 __attribute__ ((overloadable)) cl_tan(double16 a);

__attribute__ ((noinline)) void barrier (cl_mem_fence_flags flags);
