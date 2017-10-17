/* OpenCL built-in library: select()

   Copyright (c) 2011 Erik Schnetter <eschnetter@perimeterinstitute.ca>
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
   FITNESS FOR A PARTICULAR PURPOSE AND NONORDEREDRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "templates.h"

/* select needs to be implemented slightly differently between scalar,
   signed vector, and unsigned vector types, due to how the relational
   operators differ between them and signed/unsigned arithmetic.

   TODO: clang appears to be generating rather inefficient code for the
   vector versions, it's extracting each component and performing
   piece-wise comparisons instead of emitting the llvm ir "select"
   instruction. It might be worth investigating fixing this.
*/

/* We implement the scalar and the vector version separately, because
   the scalar version selects on the condition being non-zero, while
   the vector version selects on the condition's MSB (i.e. sign
   bit). */

#define IMPLEMENT_SELECT_SCALAR(GTYPE, UIGTYPE) \
  GTYPE _CL_OVERLOADABLE _CL_READNONE           \
  select(GTYPE a, GTYPE b, UIGTYPE c)           \
  {                                             \
    return c ? b : a;                           \
  }

IMPLEMENT_SELECT_SCALAR(char  , char  )
IMPLEMENT_SELECT_SCALAR(char  , uchar )
IMPLEMENT_SELECT_SCALAR(uchar , char  )
IMPLEMENT_SELECT_SCALAR(uchar , uchar )
IMPLEMENT_SELECT_SCALAR(short , short )
IMPLEMENT_SELECT_SCALAR(short , ushort)
IMPLEMENT_SELECT_SCALAR(ushort, short )
IMPLEMENT_SELECT_SCALAR(ushort, ushort)
IMPLEMENT_SELECT_SCALAR(int   , int   )
IMPLEMENT_SELECT_SCALAR(int   , uint  )
IMPLEMENT_SELECT_SCALAR(uint  , int   )
IMPLEMENT_SELECT_SCALAR(uint  , uint  )
__IF_INT64(
IMPLEMENT_SELECT_SCALAR(long  , long  )
IMPLEMENT_SELECT_SCALAR(long  , ulong )
IMPLEMENT_SELECT_SCALAR(ulong , long  )
IMPLEMENT_SELECT_SCALAR(ulong , ulong ))
IMPLEMENT_SELECT_SCALAR(float , int   )
IMPLEMENT_SELECT_SCALAR(float , uint  )
__IF_FP64(
IMPLEMENT_SELECT_SCALAR(double, long  )
IMPLEMENT_SELECT_SCALAR(double, ulong ))

/* clang's ternary operator on extended vectors
 * behaves suitably for OpenCL on x86-64 */
#if defined(__x86_64__) && defined(__clang__)
#define IMPLEMENT_SELECT_VECTOR(GTYPE, UIGTYPE, IGTYPE) \
  IMPLEMENT_SELECT_SCALAR(GTYPE, UIGTYPE)
#else
#define IMPLEMENT_SELECT_VECTOR(GTYPE, UIGTYPE, IGTYPE) \
  GTYPE _CL_OVERLOADABLE _CL_READNONE                   \
  select(GTYPE a, GTYPE b, UIGTYPE c)                   \
  {                                                     \
    return *(IGTYPE*)&c < (IGTYPE)0 ? b : a;            \
  }
#endif

IMPLEMENT_SELECT_VECTOR(char2  , char2  , char2 )
IMPLEMENT_SELECT_VECTOR(char2  , uchar2 , char2 )
IMPLEMENT_SELECT_VECTOR(uchar2 , char2  , char2 )
IMPLEMENT_SELECT_VECTOR(uchar2 , uchar2 , char2 )
IMPLEMENT_SELECT_VECTOR(short2 , short2 , short2)
IMPLEMENT_SELECT_VECTOR(short2 , ushort2, short2)
IMPLEMENT_SELECT_VECTOR(ushort2, short2 , short2)
IMPLEMENT_SELECT_VECTOR(ushort2, ushort2, short2)
IMPLEMENT_SELECT_VECTOR(int2   , int2   , int2  )
IMPLEMENT_SELECT_VECTOR(int2   , uint2  , int2  )
IMPLEMENT_SELECT_VECTOR(uint2  , int2   , int2  )
IMPLEMENT_SELECT_VECTOR(uint2  , uint2  , int2  )
__IF_INT64(
IMPLEMENT_SELECT_VECTOR(long2  , long2  , long2 )
IMPLEMENT_SELECT_VECTOR(long2  , ulong2 , long2 )
IMPLEMENT_SELECT_VECTOR(ulong2 , long2  , long2 )
IMPLEMENT_SELECT_VECTOR(ulong2 , ulong2 , long2 ))
IMPLEMENT_SELECT_VECTOR(float2 , int2   , int2  )
IMPLEMENT_SELECT_VECTOR(float2 , uint2  , int2  )
__IF_FP64(
IMPLEMENT_SELECT_VECTOR(double2, long2  , long2 )
IMPLEMENT_SELECT_VECTOR(double2, ulong2 , long2 ))

IMPLEMENT_SELECT_VECTOR(char3  , char3  , char3 )
IMPLEMENT_SELECT_VECTOR(char3  , uchar3 , char3 )
IMPLEMENT_SELECT_VECTOR(uchar3 , char3  , char3 )
IMPLEMENT_SELECT_VECTOR(uchar3 , uchar3 , char3 )
IMPLEMENT_SELECT_VECTOR(short3 , short3 , short3)
IMPLEMENT_SELECT_VECTOR(short3 , ushort3, short3)
IMPLEMENT_SELECT_VECTOR(ushort3, short3 , short3)
IMPLEMENT_SELECT_VECTOR(ushort3, ushort3, short3)
IMPLEMENT_SELECT_VECTOR(int3   , int3   , int3  )
IMPLEMENT_SELECT_VECTOR(int3   , uint3  , int3  )
IMPLEMENT_SELECT_VECTOR(uint3  , int3   , int3  )
IMPLEMENT_SELECT_VECTOR(uint3  , uint3  , int3  )
__IF_INT64(
IMPLEMENT_SELECT_VECTOR(long3  , long3  , long3 )
IMPLEMENT_SELECT_VECTOR(long3  , ulong3 , long3 )
IMPLEMENT_SELECT_VECTOR(ulong3 , long3  , long3 )
IMPLEMENT_SELECT_VECTOR(ulong3 , ulong3 , long3 ))
IMPLEMENT_SELECT_VECTOR(float3 , int3   , int3  )
IMPLEMENT_SELECT_VECTOR(float3 , uint3  , int3  )
__IF_FP64(
IMPLEMENT_SELECT_VECTOR(double3, long3  , long3 )
IMPLEMENT_SELECT_VECTOR(double3, ulong3 , long3 ))

IMPLEMENT_SELECT_VECTOR(char4  , char4  , char4 )
IMPLEMENT_SELECT_VECTOR(char4  , uchar4 , char4 )
IMPLEMENT_SELECT_VECTOR(uchar4 , char4  , char4 )
IMPLEMENT_SELECT_VECTOR(uchar4 , uchar4 , char4 )
IMPLEMENT_SELECT_VECTOR(short4 , short4 , short4)
IMPLEMENT_SELECT_VECTOR(short4 , ushort4, short4)
IMPLEMENT_SELECT_VECTOR(ushort4, short4 , short4)
IMPLEMENT_SELECT_VECTOR(ushort4, ushort4, short4)
IMPLEMENT_SELECT_VECTOR(int4   , int4   , int4  )
IMPLEMENT_SELECT_VECTOR(int4   , uint4  , int4  )
IMPLEMENT_SELECT_VECTOR(uint4  , int4   , int4  )
IMPLEMENT_SELECT_VECTOR(uint4  , uint4  , int4  )
__IF_INT64(
IMPLEMENT_SELECT_VECTOR(long4  , long4  , long4 )
IMPLEMENT_SELECT_VECTOR(long4  , ulong4 , long4 )
IMPLEMENT_SELECT_VECTOR(ulong4 , long4  , long4 )
IMPLEMENT_SELECT_VECTOR(ulong4 , ulong4 , long4 ))
IMPLEMENT_SELECT_VECTOR(float4 , int4   , int4  )
IMPLEMENT_SELECT_VECTOR(float4 , uint4  , int4  )
__IF_FP64(
IMPLEMENT_SELECT_VECTOR(double4, long4  , long4 )
IMPLEMENT_SELECT_VECTOR(double4, ulong4 , long4 ))

IMPLEMENT_SELECT_VECTOR(char8  , char8  , char8 )
IMPLEMENT_SELECT_VECTOR(char8  , uchar8 , char8 )
IMPLEMENT_SELECT_VECTOR(uchar8 , char8  , char8 )
IMPLEMENT_SELECT_VECTOR(uchar8 , uchar8 , char8 )
IMPLEMENT_SELECT_VECTOR(short8 , short8 , short8)
IMPLEMENT_SELECT_VECTOR(short8 , ushort8, short8)
IMPLEMENT_SELECT_VECTOR(ushort8, short8 , short8)
IMPLEMENT_SELECT_VECTOR(ushort8, ushort8, short8)
IMPLEMENT_SELECT_VECTOR(int8   , int8   , int8  )
IMPLEMENT_SELECT_VECTOR(int8   , uint8  , int8  )
IMPLEMENT_SELECT_VECTOR(uint8  , int8   , int8  )
IMPLEMENT_SELECT_VECTOR(uint8  , uint8  , int8  )
__IF_INT64(
IMPLEMENT_SELECT_VECTOR(long8  , long8  , long8 )
IMPLEMENT_SELECT_VECTOR(long8  , ulong8 , long8 )
IMPLEMENT_SELECT_VECTOR(ulong8 , long8  , long8 )
IMPLEMENT_SELECT_VECTOR(ulong8 , ulong8 , long8 ))
IMPLEMENT_SELECT_VECTOR(float8 , int8   , int8  )
IMPLEMENT_SELECT_VECTOR(float8 , uint8  , int8  )
__IF_FP64(
IMPLEMENT_SELECT_VECTOR(double8, long8  , long8 )
IMPLEMENT_SELECT_VECTOR(double8, ulong8 , long8 ))

IMPLEMENT_SELECT_VECTOR(char16  , char16  , char16 )
IMPLEMENT_SELECT_VECTOR(char16  , uchar16 , char16 )
IMPLEMENT_SELECT_VECTOR(uchar16 , char16  , char16 )
IMPLEMENT_SELECT_VECTOR(uchar16 , uchar16 , char16 )
IMPLEMENT_SELECT_VECTOR(short16 , short16 , short16)
IMPLEMENT_SELECT_VECTOR(short16 , ushort16, short16)
IMPLEMENT_SELECT_VECTOR(ushort16, short16 , short16)
IMPLEMENT_SELECT_VECTOR(ushort16, ushort16, short16)
IMPLEMENT_SELECT_VECTOR(int16   , int16   , int16  )
IMPLEMENT_SELECT_VECTOR(int16   , uint16  , int16  )
IMPLEMENT_SELECT_VECTOR(uint16  , int16   , int16  )
IMPLEMENT_SELECT_VECTOR(uint16  , uint16  , int16  )
__IF_INT64(
IMPLEMENT_SELECT_VECTOR(long16  , long16  , long16 )
IMPLEMENT_SELECT_VECTOR(long16  , ulong16 , long16 )
IMPLEMENT_SELECT_VECTOR(ulong16 , long16  , long16 )
IMPLEMENT_SELECT_VECTOR(ulong16 , ulong16 , long16 ))
IMPLEMENT_SELECT_VECTOR(float16 , int16   , int16  )
IMPLEMENT_SELECT_VECTOR(float16 , uint16  , int16  )
__IF_FP64(
IMPLEMENT_SELECT_VECTOR(double16, long16  , long16 )
IMPLEMENT_SELECT_VECTOR(double16, ulong16 , long16 ))
