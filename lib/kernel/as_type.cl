/* OpenCL built-in library: as_type()

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

/* The as_type functions are implemented using union casts, which OpenCL
 * supports as per 6.2.4.1 in the OpenCL 1.2 specification.
 *
 * These map down to the corresponding SPIR/LLVM IR bitcast instruction.
 */

#if (__clang_major__ < 4)

#define DEFINE_AS_TYPE(SRC, DST)                                        \
  _CL_ALWAYSINLINE _CL_OVERLOADABLE                                     \
  DST as_##DST(SRC a)                                                   \
  {                                                                     \
    union { SRC src; DST dst; } cvt;                                    \
    cvt.src = a;                                                        \
    return cvt.dst;                                                     \
  }

/* 1 byte */

#define DEFINE_AS_TYPE_1(DST)                   \
  DEFINE_AS_TYPE(DST, char)                     \
  DEFINE_AS_TYPE(DST, uchar)

DEFINE_AS_TYPE_1(char)
DEFINE_AS_TYPE_1(uchar)

/* 2 bytes */

#define DEFINE_AS_TYPE_2(DST)                   \
  DEFINE_AS_TYPE(DST, char2)                    \
  DEFINE_AS_TYPE(DST, uchar2)                   \
  DEFINE_AS_TYPE(DST, short)                    \
  DEFINE_AS_TYPE(DST, ushort)                   \
  __IF_FP16(DEFINE_AS_TYPE(DST, half))

DEFINE_AS_TYPE_2(char2)
DEFINE_AS_TYPE_2(uchar2)
DEFINE_AS_TYPE_2(short)
DEFINE_AS_TYPE_2(ushort)
__IF_FP16(DEFINE_AS_TYPE_2(half))

/* 4 bytes */

#define DEFINE_AS_TYPE_4(DST)                   \
  DEFINE_AS_TYPE(DST, char4)                    \
  DEFINE_AS_TYPE(DST, uchar4)                   \
  DEFINE_AS_TYPE(DST, char3)                    \
  DEFINE_AS_TYPE(DST, uchar3)                   \
  DEFINE_AS_TYPE(DST, short2)                   \
  DEFINE_AS_TYPE(DST, ushort2)                  \
  __IF_FP16(                                    \
  DEFINE_AS_TYPE(DST, half2))                   \
  DEFINE_AS_TYPE(DST, int)                      \
  DEFINE_AS_TYPE(DST, uint)                     \
  DEFINE_AS_TYPE(DST, float)

DEFINE_AS_TYPE_4(char4)
DEFINE_AS_TYPE_4(uchar4)
DEFINE_AS_TYPE_4(char3)
DEFINE_AS_TYPE_4(uchar3)
DEFINE_AS_TYPE_4(short2)
DEFINE_AS_TYPE_4(ushort2)
__IF_FP16(
DEFINE_AS_TYPE_4(half2))
DEFINE_AS_TYPE_4(int)
DEFINE_AS_TYPE_4(uint)
DEFINE_AS_TYPE_4(float)

/* 8 bytes */

#define DEFINE_AS_TYPE_8(DST)                   \
  DEFINE_AS_TYPE(DST, char8)                    \
  DEFINE_AS_TYPE(DST, uchar8)                   \
  DEFINE_AS_TYPE(DST, short4)                   \
  DEFINE_AS_TYPE(DST, ushort4)                  \
  __IF_FP16(                                    \
  DEFINE_AS_TYPE(DST, half4))                   \
  DEFINE_AS_TYPE(DST, short3)                   \
  DEFINE_AS_TYPE(DST, ushort3)                  \
  __IF_FP16(                                    \
  DEFINE_AS_TYPE(DST, half3))                   \
  DEFINE_AS_TYPE(DST, int2)                     \
  DEFINE_AS_TYPE(DST, uint2)                    \
  __IF_INT64(                                   \
  DEFINE_AS_TYPE(DST, long)                     \
  DEFINE_AS_TYPE(DST, ulong))                   \
  DEFINE_AS_TYPE(DST, float2)                   \
  __IF_FP64(                                    \
  DEFINE_AS_TYPE(DST, double))

DEFINE_AS_TYPE_8(char8)
DEFINE_AS_TYPE_8(uchar8)
DEFINE_AS_TYPE_8(short4)
DEFINE_AS_TYPE_8(ushort4)
__IF_FP16(
DEFINE_AS_TYPE_8(half4))
DEFINE_AS_TYPE_8(short3)
DEFINE_AS_TYPE_8(ushort3)
__IF_FP16(
DEFINE_AS_TYPE_8(half3))
DEFINE_AS_TYPE_8(int2)
DEFINE_AS_TYPE_8(uint2)
__IF_INT64(
DEFINE_AS_TYPE_8(long)
DEFINE_AS_TYPE_8(ulong))
DEFINE_AS_TYPE_8(float2)
__IF_FP64(
DEFINE_AS_TYPE_8(double))

/* 16 bytes */

#define DEFINE_AS_TYPE_16(DST)                  \
  DEFINE_AS_TYPE(DST, char16)                   \
  DEFINE_AS_TYPE(DST, uchar16)                  \
  DEFINE_AS_TYPE(DST, short8)                   \
  DEFINE_AS_TYPE(DST, ushort8)                  \
  __IF_FP16(                                    \
  DEFINE_AS_TYPE(DST, half8))                   \
  DEFINE_AS_TYPE(DST, int4)                     \
  DEFINE_AS_TYPE(DST, uint4)                    \
  DEFINE_AS_TYPE(DST, int3)                     \
  DEFINE_AS_TYPE(DST, uint3)                    \
  __IF_INT64(                                   \
  DEFINE_AS_TYPE(DST, long2)                    \
  DEFINE_AS_TYPE(DST, ulong2))                  \
  DEFINE_AS_TYPE(DST, float4)                   \
  DEFINE_AS_TYPE(DST, float3)                   \
  __IF_FP64(                                    \
  DEFINE_AS_TYPE(DST, double2))

DEFINE_AS_TYPE_16(char16)
DEFINE_AS_TYPE_16(uchar16)
DEFINE_AS_TYPE_16(short8)
DEFINE_AS_TYPE_16(ushort8)
__IF_FP16(
DEFINE_AS_TYPE_16(half8))
DEFINE_AS_TYPE_16(int4)
DEFINE_AS_TYPE_16(uint4)
DEFINE_AS_TYPE_16(int3)
DEFINE_AS_TYPE_16(uint3)
__IF_INT64(
DEFINE_AS_TYPE_16(long2)
DEFINE_AS_TYPE_16(ulong2))
DEFINE_AS_TYPE_16(float4)
DEFINE_AS_TYPE_16(float3)
__IF_FP64(
DEFINE_AS_TYPE_16(double2))

/* 32 bytes */

#define DEFINE_AS_TYPE_32(DST)                  \
  DEFINE_AS_TYPE(DST, short16)                  \
  DEFINE_AS_TYPE(DST, ushort16)                 \
  __IF_FP16(                                    \
  DEFINE_AS_TYPE(DST, half16))                  \
  DEFINE_AS_TYPE(DST, int8)                     \
  DEFINE_AS_TYPE(DST, uint8)                    \
  __IF_INT64(                                   \
  DEFINE_AS_TYPE(DST, long4)                    \
  DEFINE_AS_TYPE(DST, ulong4)                   \
  DEFINE_AS_TYPE(DST, long3)                    \
  DEFINE_AS_TYPE(DST, ulong3))                  \
  DEFINE_AS_TYPE(DST, float8)                   \
  __IF_FP64(                                    \
  DEFINE_AS_TYPE(DST, double4)                  \
  DEFINE_AS_TYPE(DST, double3))

DEFINE_AS_TYPE_32(short16)
DEFINE_AS_TYPE_32(ushort16)
__IF_FP16(
DEFINE_AS_TYPE_32(half16))
DEFINE_AS_TYPE_32(int8)
DEFINE_AS_TYPE_32(uint8)
__IF_INT64(
DEFINE_AS_TYPE_32(long4)
DEFINE_AS_TYPE_32(ulong4)
DEFINE_AS_TYPE_32(long3)
DEFINE_AS_TYPE_32(ulong3))
DEFINE_AS_TYPE_32(float8)
__IF_FP64(
DEFINE_AS_TYPE_32(double4)
DEFINE_AS_TYPE_32(double3))

/* 64 bytes */

#define DEFINE_AS_TYPE_64(DST)                  \
  DEFINE_AS_TYPE(DST, int16)                    \
  DEFINE_AS_TYPE(DST, uint16)                   \
  __IF_INT64(                                   \
  DEFINE_AS_TYPE(DST, long8)                    \
  DEFINE_AS_TYPE(DST, ulong8))                  \
  DEFINE_AS_TYPE(DST, float16)                  \
  __IF_FP64(                                    \
  DEFINE_AS_TYPE(DST, double8))

DEFINE_AS_TYPE_64(int16)
DEFINE_AS_TYPE_64(uint16)
__IF_INT64(
DEFINE_AS_TYPE_64(long8)
DEFINE_AS_TYPE_64(ulong8))
DEFINE_AS_TYPE_64(float16)
__IF_FP64(
DEFINE_AS_TYPE_64(double8))

/* 128 bytes */

#define DEFINE_AS_TYPE_128(DST)                 \
  __IF_INT64(                                   \
  DEFINE_AS_TYPE(DST, long16)                   \
  DEFINE_AS_TYPE(DST, ulong16))                 \
  __IF_FP64(                                    \
  DEFINE_AS_TYPE(DST, double16))

__IF_INT64(
DEFINE_AS_TYPE_128(long16)
DEFINE_AS_TYPE_128(ulong16))
__IF_FP64(
DEFINE_AS_TYPE_128(double16))

#endif
