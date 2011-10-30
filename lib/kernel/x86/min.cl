/* OpenCL built-in library: min()

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

#include "../templates.h"

#define IMPLEMENT_DIRECT(NAME, TYPE, EXPR)      \
  TYPE _cl_overloadable NAME(TYPE a, TYPE b)    \
  {                                             \
    return EXPR;                                \
  }

#define IMPLEMENT_UPCAST(NAME, TYPE, UPTYPE, LO)        \
  TYPE _cl_overloadable NAME(TYPE a, TYPE b)            \
  {                                                     \
    return NAME(*(UPTYPE*)&a, *(UPTYPE*)&b).LO;         \
  }

#define IMPLEMENT_SPLIT(NAME, TYPE, LO, HI)             \
  TYPE _cl_overloadable NAME(TYPE a, TYPE b)            \
  {                                                     \
    return (TYPE)(NAME(a.LO, b.LO), NAME(a.HI, b.HI));  \
  }



IMPLEMENT_DIRECT(min, char  , a<=b ? a : b)
#ifdef __SSE4_1__
IMPLEMENT_UPCAST(min, char2 , char4 , lo  )
IMPLEMENT_UPCAST(min, char3 , char4 , s012)
IMPLEMENT_UPCAST(min, char4 , char8 , lo  )
IMPLEMENT_UPCAST(min, char8 , char16, lo  )
IMPLEMENT_DIRECT(min, char16, __builtin_ia32_pminsb128(a, b))
#else
IMPLEMENT_DIRECT(min, char2 , (char2 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, char3 , (char3 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, char4 , (char4 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, char8 , (char8 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, char16, (char16)(a<=b) ? a : b)
#endif

IMPLEMENT_DIRECT(min, uchar  , a<=b ? a : b)
#ifdef __SSE__
uchar16 _cl_min_convert_uchar16(char16 a)
{
  return as_uchar16(a);
}
IMPLEMENT_UPCAST(min, uchar2 , uchar4 , lo  )
IMPLEMENT_UPCAST(min, uchar3 , uchar4 , s012)
IMPLEMENT_UPCAST(min, uchar4 , uchar8 , lo  )
IMPLEMENT_UPCAST(min, uchar8 , uchar16, lo  )
IMPLEMENT_DIRECT(min, uchar16, _cl_min_convert_uchar16(__builtin_ia32_pminub128(as_char16(a), as_char16(b))))
#else
IMPLEMENT_DIRECT(min, uchar2 , (char2 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, uchar3 , (char3 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, uchar4 , (char4 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, uchar8 , (char8 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, uchar16, (char16)(a<=b) ? a : b)
#endif

IMPLEMENT_DIRECT(min, short  , a<=b ? a : b)
#ifdef __SSE__
IMPLEMENT_UPCAST(min, short2 , short4, lo  )
IMPLEMENT_UPCAST(min, short3 , short4, s012)
IMPLEMENT_UPCAST(min, short4 , short8, lo  )
IMPLEMENT_DIRECT(min, short8 , __builtin_ia32_pminsw128(a, b))
IMPLEMENT_SPLIT (min, short16, lo, hi)
#else
IMPLEMENT_DIRECT(min, short2 , (short2 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, short3 , (short3 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, short4 , (short4 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, short8 , (short8 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, short16, (short16)(a<=b) ? a : b)
#endif

IMPLEMENT_DIRECT(min, ushort  , a<=b ? a : b)
#ifdef __SSE4_1__
ushort8 _cl_min_convert_ushort8(short8 a)
{
  return as_ushort8(a);
}
IMPLEMENT_UPCAST(min, ushort2 , ushort4, lo  )
IMPLEMENT_UPCAST(min, ushort3 , ushort4, s012)
IMPLEMENT_UPCAST(min, ushort4 , ushort8, lo  )
IMPLEMENT_DIRECT(min, ushort8 , _cl_min_convert_ushort8(__builtin_ia32_pminuw128(as_short8(a), as_short8(b))))
IMPLEMENT_SPLIT (min, ushort16, lo, hi)
#else
IMPLEMENT_DIRECT(min, ushort2 , (short2 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, ushort3 , (short3 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, ushort4 , (short4 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, ushort8 , (short8 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, ushort16, (short16)(a<=b) ? a : b)
#endif

IMPLEMENT_DIRECT(min, int  , a<=b ? a : b)
#ifdef __SSE4_1__
IMPLEMENT_UPCAST(min, int2 , int4, lo  )
IMPLEMENT_UPCAST(min, int3 , int4, s012)
IMPLEMENT_DIRECT(min, int4 , __builtin_ia32_pminsd128(a, b))
IMPLEMENT_SPLIT (min, int8 , lo, hi)
IMPLEMENT_SPLIT (min, int16, lo, hi)
#else
IMPLEMENT_DIRECT(min, int2 , (int2 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, int3 , (int3 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, int4 , (int4 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, int8 , (int8 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, int16, (int16)(a<=b) ? a : b)
#endif

IMPLEMENT_DIRECT(min, uint    , a<=b ? a : b)
#ifdef __SSE4_1__
uint4 _cl_min_convert_uint4(int4 a)
{
  return as_uint4(a);
}
IMPLEMENT_UPCAST(min, uint2 , uint4, lo  )
IMPLEMENT_UPCAST(min, uint3 , uint4, s012)
IMPLEMENT_DIRECT(min, uint4 , _cl_min_convert_uint4(__builtin_ia32_pminud128(as_int4(a), as_int4(b))))
IMPLEMENT_SPLIT (min, uint8 , lo, hi)
IMPLEMENT_SPLIT (min, uint16, lo, hi)
#else
IMPLEMENT_DIRECT(min, uint2 , (int2 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, uint3 , (int3 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, uint4 , (int4 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, uint8 , (int8 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, uint16, (int16)(a<=b) ? a : b)
#endif

IMPLEMENT_DIRECT(min, long  , a<=b ? a : b)
IMPLEMENT_DIRECT(min, long2 , (long2 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, long3 , (long3 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, long4 , (long4 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, long8 , (long8 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, long16, (long16)(a<=b) ? a : b)

IMPLEMENT_DIRECT(min, ulong  , a<=b ? a : b)
IMPLEMENT_DIRECT(min, ulong2 , (long2 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, ulong3 , (long3 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, ulong4 , (long4 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, ulong8 , (long8 )(a<=b) ? a : b)
IMPLEMENT_DIRECT(min, ulong16, (long16)(a<=b) ? a : b)

DEFINE_EXPR_G_GS(min, min(a, (gtype)b))



// Note: min() has no special semantics for inf/nan, even if fmin does

#ifdef __SSE__
float4 _cl_min_ensure_float4(float4 a)
{
  return a;
}
IMPLEMENT_DIRECT(min, float  , _cl_min_ensure_float4(__builtin_ia32_minss(*(float4*)&a, *(float4*)&b)).s0)
IMPLEMENT_UPCAST(min, float2 , float4, lo  )
IMPLEMENT_UPCAST(min, float3 , float4, s012)
IMPLEMENT_DIRECT(min, float4 , __builtin_ia32_minps(a, b))
IMPLEMENT_SPLIT (min, float8 , lo, hi)
IMPLEMENT_SPLIT (min, float16, lo, hi)
#else
IMPLEMENT_DIRECT(min, float  , a<=b ? a : b)
IMPLEMENT_DIRECT(min, float2 , a<=b ? a : b)
IMPLEMENT_DIRECT(min, float3 , a<=b ? a : b)
IMPLEMENT_DIRECT(min, float4 , a<=b ? a : b)
IMPLEMENT_DIRECT(min, float8 , a<=b ? a : b)
IMPLEMENT_DIRECT(min, float16, a<=b ? a : b)
#endif

#ifdef __SSE2__
double2 _cl_min_ensure_double2(double2 a)
{
  return a;
}
IMPLEMENT_DIRECT(min, double  , _cl_min_ensure_double2(__builtin_ia32_minsd(*(double2*)&a, *(double2*)&b)).s0)
IMPLEMENT_DIRECT(min, double2 , __builtin_ia32_minpd(a, b))
IMPLEMENT_SPLIT (min, double3 , lo, s2)
IMPLEMENT_SPLIT (min, double4 , lo, hi)
IMPLEMENT_SPLIT (min, double8 , lo, hi)
IMPLEMENT_SPLIT (min, double16, lo, hi)
#else
IMPLEMENT_DIRECT(min, double  , a<=b ? a : b)
IMPLEMENT_DIRECT(min, double2 , a<=b ? a : b)
IMPLEMENT_DIRECT(min, double3 , a<=b ? a : b)
IMPLEMENT_DIRECT(min, double4 , a<=b ? a : b)
IMPLEMENT_DIRECT(min, double8 , a<=b ? a : b)
IMPLEMENT_DIRECT(min, double16, a<=b ? a : b)
#endif

DEFINE_EXPR_V_VS(min, min(a, (vtype)b))
