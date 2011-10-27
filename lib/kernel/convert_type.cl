/* OpenCL built-in library: convert_type()

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

#include "templates.h"

#define DEFINE_CONVERT_TYPE(SRC, DST)                           \
  DST __attribute__ ((overloadable)) convert_##DST(SRC a)       \
  {                                                             \
    return (DST)a;                                              \
  }

#define DEFINE_CONVERT_TYPE_HALF(SRC, DST, HALFDST)                     \
  DST __attribute__ ((overloadable)) convert_##DST(SRC a)               \
  {                                                                     \
    return (DST)(convert_##HALFDST(a.lo), convert_##HALFDST(a.hi));     \
  }

#define DEFINE_CONVERT_TYPE_012(SRC, DST, DST01, DST2)          \
  DST __attribute__ ((overloadable)) convert_##DST(SRC a)       \
  {                                                             \
    return (DST)(convert_##DST01(a.s01), convert_##DST2(a.s2)); \
  }

/* 1 element */
#define DEFINE_CONVERT_TYPE_1(SRC)              \
  DEFINE_CONVERT_TYPE(SRC, char  )              \
  DEFINE_CONVERT_TYPE(SRC, short )              \
  DEFINE_CONVERT_TYPE(SRC, int   )              \
  DEFINE_CONVERT_TYPE(SRC, long  )              \
  DEFINE_CONVERT_TYPE(SRC, uchar )              \
  DEFINE_CONVERT_TYPE(SRC, ushort)              \
  DEFINE_CONVERT_TYPE(SRC, uint  )              \
  DEFINE_CONVERT_TYPE(SRC, ulong )              \
  DEFINE_CONVERT_TYPE(SRC, float )              \
  DEFINE_CONVERT_TYPE(SRC, double)
DEFINE_CONVERT_TYPE_1(char  )
DEFINE_CONVERT_TYPE_1(short )
DEFINE_CONVERT_TYPE_1(int   )
DEFINE_CONVERT_TYPE_1(long  )
DEFINE_CONVERT_TYPE_1(uchar )
DEFINE_CONVERT_TYPE_1(ushort)
DEFINE_CONVERT_TYPE_1(uint  )
DEFINE_CONVERT_TYPE_1(ulong )
DEFINE_CONVERT_TYPE_1(float )
DEFINE_CONVERT_TYPE_1(double)

/* 2 elements */
#define DEFINE_CONVERT_TYPE_2(SRC)                      \
  DEFINE_CONVERT_TYPE_HALF(SRC, char2  , char  )        \
  DEFINE_CONVERT_TYPE_HALF(SRC, short2 , short )        \
  DEFINE_CONVERT_TYPE_HALF(SRC, int2   , int   )        \
  DEFINE_CONVERT_TYPE_HALF(SRC, long2  , long  )        \
  DEFINE_CONVERT_TYPE_HALF(SRC, uchar2 , uchar )        \
  DEFINE_CONVERT_TYPE_HALF(SRC, ushort2, ushort)        \
  DEFINE_CONVERT_TYPE_HALF(SRC, uint2  , uint  )        \
  DEFINE_CONVERT_TYPE_HALF(SRC, ulong2 , ulong )        \
  DEFINE_CONVERT_TYPE_HALF(SRC, float2 , float )        \
  DEFINE_CONVERT_TYPE_HALF(SRC, double2, double)
DEFINE_CONVERT_TYPE_2(char2  )
DEFINE_CONVERT_TYPE_2(short2 )
DEFINE_CONVERT_TYPE_2(int2   )
DEFINE_CONVERT_TYPE_2(long2  )
DEFINE_CONVERT_TYPE_2(uchar2 )
DEFINE_CONVERT_TYPE_2(ushort2)
DEFINE_CONVERT_TYPE_2(uint2  )
DEFINE_CONVERT_TYPE_2(ulong2 )
DEFINE_CONVERT_TYPE_2(float2 )
DEFINE_CONVERT_TYPE_2(double2)

/* 3 elements */
#define DEFINE_CONVERT_TYPE_3(SRC)                              \
  DEFINE_CONVERT_TYPE_012(SRC, char3  , char2  , char  )        \
  DEFINE_CONVERT_TYPE_012(SRC, short3 , short2 , short )        \
  DEFINE_CONVERT_TYPE_012(SRC, int3   , int2   , int   )        \
  DEFINE_CONVERT_TYPE_012(SRC, long3  , long2  , long  )        \
  DEFINE_CONVERT_TYPE_012(SRC, uchar3 , uchar2 , uchar )        \
  DEFINE_CONVERT_TYPE_012(SRC, ushort3, ushort2, ushort)        \
  DEFINE_CONVERT_TYPE_012(SRC, uint3  , uint2  , uint  )        \
  DEFINE_CONVERT_TYPE_012(SRC, ulong3 , ulong2 , ulong )        \
  DEFINE_CONVERT_TYPE_012(SRC, float3 , float2 , float )        \
  DEFINE_CONVERT_TYPE_012(SRC, double3, double2, double)
DEFINE_CONVERT_TYPE_3(char3  )
DEFINE_CONVERT_TYPE_3(short3 )
DEFINE_CONVERT_TYPE_3(int3   )
DEFINE_CONVERT_TYPE_3(long3  )
DEFINE_CONVERT_TYPE_3(uchar3 )
DEFINE_CONVERT_TYPE_3(ushort3)
DEFINE_CONVERT_TYPE_3(uint3  )
DEFINE_CONVERT_TYPE_3(ulong3 )
DEFINE_CONVERT_TYPE_3(float3 )
DEFINE_CONVERT_TYPE_3(double3)

/* 4 elements */
#define DEFINE_CONVERT_TYPE_4(SRC)                      \
  DEFINE_CONVERT_TYPE_HALF(SRC, char4  , char2  )       \
  DEFINE_CONVERT_TYPE_HALF(SRC, short4 , short2 )       \
  DEFINE_CONVERT_TYPE_HALF(SRC, int4   , int2   )       \
  DEFINE_CONVERT_TYPE_HALF(SRC, long4  , long2  )       \
  DEFINE_CONVERT_TYPE_HALF(SRC, uchar4 , uchar2 )       \
  DEFINE_CONVERT_TYPE_HALF(SRC, ushort4, ushort2)       \
  DEFINE_CONVERT_TYPE_HALF(SRC, uint4  , uint2  )       \
  DEFINE_CONVERT_TYPE_HALF(SRC, ulong4 , ulong2 )       \
  DEFINE_CONVERT_TYPE_HALF(SRC, float4 , float2 )       \
  DEFINE_CONVERT_TYPE_HALF(SRC, double4, double2)
DEFINE_CONVERT_TYPE_4(char4  )
DEFINE_CONVERT_TYPE_4(short4 )
DEFINE_CONVERT_TYPE_4(int4   )
DEFINE_CONVERT_TYPE_4(long4  )
DEFINE_CONVERT_TYPE_4(uchar4 )
DEFINE_CONVERT_TYPE_4(ushort4)
DEFINE_CONVERT_TYPE_4(uint4  )
DEFINE_CONVERT_TYPE_4(ulong4 )
DEFINE_CONVERT_TYPE_4(float4 )
DEFINE_CONVERT_TYPE_4(double4)

/* 8 elements */
#define DEFINE_CONVERT_TYPE_8(SRC)                      \
  DEFINE_CONVERT_TYPE_HALF(SRC, char8  , char4  )       \
  DEFINE_CONVERT_TYPE_HALF(SRC, short8 , short4 )       \
  DEFINE_CONVERT_TYPE_HALF(SRC, int8   , int4   )       \
  DEFINE_CONVERT_TYPE_HALF(SRC, long8  , long4  )       \
  DEFINE_CONVERT_TYPE_HALF(SRC, uchar8 , uchar4 )       \
  DEFINE_CONVERT_TYPE_HALF(SRC, ushort8, ushort4)       \
  DEFINE_CONVERT_TYPE_HALF(SRC, uint8  , uint4  )       \
  DEFINE_CONVERT_TYPE_HALF(SRC, ulong8 , ulong4 )       \
  DEFINE_CONVERT_TYPE_HALF(SRC, float8 , float4 )       \
  DEFINE_CONVERT_TYPE_HALF(SRC, double8, double4)
DEFINE_CONVERT_TYPE_8(char8  )
DEFINE_CONVERT_TYPE_8(short8 )
DEFINE_CONVERT_TYPE_8(int8   )
DEFINE_CONVERT_TYPE_8(long8  )
DEFINE_CONVERT_TYPE_8(uchar8 )
DEFINE_CONVERT_TYPE_8(ushort8)
DEFINE_CONVERT_TYPE_8(uint8  )
DEFINE_CONVERT_TYPE_8(ulong8 )
DEFINE_CONVERT_TYPE_8(float8 )
DEFINE_CONVERT_TYPE_8(double8)

/* 16 elements */
#define DEFINE_CONVERT_TYPE_16(SRC)                     \
  DEFINE_CONVERT_TYPE_HALF(SRC, char16  , char8  )      \
  DEFINE_CONVERT_TYPE_HALF(SRC, short16 , short8 )      \
  DEFINE_CONVERT_TYPE_HALF(SRC, int16   , int8   )      \
  DEFINE_CONVERT_TYPE_HALF(SRC, long16  , long8  )      \
  DEFINE_CONVERT_TYPE_HALF(SRC, uchar16 , uchar8 )      \
  DEFINE_CONVERT_TYPE_HALF(SRC, ushort16, ushort8)      \
  DEFINE_CONVERT_TYPE_HALF(SRC, uint16  , uint8  )      \
  DEFINE_CONVERT_TYPE_HALF(SRC, ulong16 , ulong8 )      \
  DEFINE_CONVERT_TYPE_HALF(SRC, float16 , float8 )      \
  DEFINE_CONVERT_TYPE_HALF(SRC, double16, double8)
DEFINE_CONVERT_TYPE_16(char16  )
DEFINE_CONVERT_TYPE_16(short16 )
DEFINE_CONVERT_TYPE_16(int16   )
DEFINE_CONVERT_TYPE_16(long16  )
DEFINE_CONVERT_TYPE_16(uchar16 )
DEFINE_CONVERT_TYPE_16(ushort16)
DEFINE_CONVERT_TYPE_16(uint16  )
DEFINE_CONVERT_TYPE_16(ulong16 )
DEFINE_CONVERT_TYPE_16(float16 )
DEFINE_CONVERT_TYPE_16(double16)
