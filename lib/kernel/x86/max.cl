/* OpenCL built-in library: max()

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



IMPLEMENT_DIRECT(max, char  , a>=b ? a : b)
#ifdef __SSE4_1__
IMPLEMENT_UPCAST(max, char2 , char4 , lo  )
IMPLEMENT_UPCAST(max, char3 , char4 , s012)
IMPLEMENT_UPCAST(max, char4 , char8 , lo  )
IMPLEMENT_UPCAST(max, char8 , char16, lo  )
IMPLEMENT_DIRECT(max, char16, __builtin_ia32_pmaxsb128(a, b))
#else
IMPLEMENT_DIRECT(max, char2 , (char2 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, char3 , (char3 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, char4 , (char4 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, char8 , (char8 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, char16, (char16)(a>=b) ? a : b)
#endif

IMPLEMENT_DIRECT(max, uchar  , a>=b ? a : b)
#ifdef __SSE__
IMPLEMENT_UPCAST(max, uchar2 , uchar4 , lo  )
IMPLEMENT_UPCAST(max, uchar3 , uchar4 , s012)
IMPLEMENT_UPCAST(max, uchar4 , uchar8 , lo  )
IMPLEMENT_UPCAST(max, uchar8 , uchar16, lo  )
IMPLEMENT_DIRECT(max, uchar16, as_uchar16(__builtin_ia32_pmaxub128(as_char16(a), as_char16(b))))
#else
IMPLEMENT_DIRECT(max, uchar2 , (char2 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, uchar3 , (char3 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, uchar4 , (char4 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, uchar8 , (char8 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, uchar16, (char16)(a>=b) ? a : b)
#endif

IMPLEMENT_DIRECT(max, short  , a>=b ? a : b)
#ifdef __SSE__
IMPLEMENT_UPCAST(max, short2 , short4, lo  )
IMPLEMENT_UPCAST(max, short3 , short4, s012)
IMPLEMENT_UPCAST(max, short4 , short8, lo  )
IMPLEMENT_DIRECT(max, short8 , __builtin_ia32_pmaxsw128(a, b))
IMPLEMENT_SPLIT (max, short16, lo, hi)
#else
IMPLEMENT_DIRECT(max, short2 , (short2 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, short3 , (short3 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, short4 , (short4 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, short8 , (short8 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, short16, (short16)(a>=b) ? a : b)
#endif

IMPLEMENT_DIRECT(max, ushort  , a>=b ? a : b)
#ifdef __SSE4_1__
IMPLEMENT_UPCAST(max, ushort2 , ushort4, lo  )
IMPLEMENT_UPCAST(max, ushort3 , ushort4, s012)
IMPLEMENT_UPCAST(max, ushort4 , ushort8, lo  )
IMPLEMENT_DIRECT(max, ushort8 , as_ushort8(__builtin_ia32_pmaxuw128(as_short8(a), as_short8(b))))
IMPLEMENT_SPLIT (max, ushort16, lo, hi)
#else
IMPLEMENT_DIRECT(max, ushort2 , (short2 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, ushort3 , (short3 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, ushort4 , (short4 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, ushort8 , (short8 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, ushort16, (short16)(a>=b) ? a : b)
#endif

IMPLEMENT_DIRECT(max, int  , a>=b ? a : b)
#ifdef __SSE4_1__
IMPLEMENT_UPCAST(max, int2 , int4, lo  )
IMPLEMENT_UPCAST(max, int3 , int4, s012)
IMPLEMENT_DIRECT(max, int4 , __builtin_ia32_pmaxsd128(a, b))
IMPLEMENT_SPLIT (max, int8 , lo, hi)
IMPLEMENT_SPLIT (max, int16, lo, hi)
#else
IMPLEMENT_DIRECT(max, int2 , (int2 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, int3 , (int3 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, int4 , (int4 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, int8 , (int8 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, int16, (int16)(a>=b) ? a : b)
#endif

IMPLEMENT_DIRECT(max, uint    , a>=b ? a : b)
#ifdef __SSE4_1__
IMPLEMENT_UPCAST(max, uint2 , uint4, lo  )
IMPLEMENT_UPCAST(max, uint3 , uint4, s012)
IMPLEMENT_DIRECT(max, uint4 , as_uint4(__builtin_ia32_pmaxud128(as_int4(a), as_int4(b))))
IMPLEMENT_SPLIT (max, uint8 , lo, hi)
IMPLEMENT_SPLIT (max, uint16, lo, hi)
#else
IMPLEMENT_DIRECT(max, uint2 , (int2 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, uint3 , (int3 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, uint4 , (int4 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, uint8 , (int8 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, uint16, (int16)(a>=b) ? a : b)
#endif

IMPLEMENT_DIRECT(max, long  , a>=b ? a : b)
IMPLEMENT_DIRECT(max, long2 , (long2 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, long3 , (long3 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, long4 , (long4 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, long8 , (long8 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, long16, (long16)(a>=b) ? a : b)

IMPLEMENT_DIRECT(max, ulong  , a>=b ? a : b)
IMPLEMENT_DIRECT(max, ulong2 , (long2 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, ulong3 , (long3 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, ulong4 , (long4 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, ulong8 , (long8 )(a>=b) ? a : b)
IMPLEMENT_DIRECT(max, ulong16, (long16)(a>=b) ? a : b)

DEFINE_EXPR_G_GS(max, max(a, (gtype)b))



// Note: max() has no special semantics for inf/nan, even if fmax does

#ifdef __SSE__
float4 _cl_max_ensure_float4(float4 a)
{
  return a;
}
IMPLEMENT_DIRECT(max, float  , _cl_max_ensure_float4(__builtin_ia32_maxss(*(float4*)&a, *(float4*)&b)).s0)
IMPLEMENT_UPCAST(max, float2 , float4, lo  )
IMPLEMENT_UPCAST(max, float3 , float4, s012)
IMPLEMENT_DIRECT(max, float4 , __builtin_ia32_maxps(a, b))
IMPLEMENT_SPLIT (max, float8 , lo, hi)
IMPLEMENT_SPLIT (max, float16, lo, hi)
#else
IMPLEMENT_DIRECT(max, float  , a>=b ? a : b)
IMPLEMENT_DIRECT(max, float2 , a>=b ? a : b)
IMPLEMENT_DIRECT(max, float3 , a>=b ? a : b)
IMPLEMENT_DIRECT(max, float4 , a>=b ? a : b)
IMPLEMENT_DIRECT(max, float8 , a>=b ? a : b)
IMPLEMENT_DIRECT(max, float16, a>=b ? a : b)
#endif

#ifdef __SSE2__
double2 _cl_max_ensure_double2(double2 a)
{
  return a;
}
IMPLEMENT_DIRECT(max, double  , _cl_max_ensure_double2(__builtin_ia32_maxsd(*(double2*)&a, *(double2*)&b)).s0)
IMPLEMENT_DIRECT(max, double2 , __builtin_ia32_maxpd(a, b))
IMPLEMENT_SPLIT (max, double3 , lo, s2)
IMPLEMENT_SPLIT (max, double4 , lo, hi)
IMPLEMENT_SPLIT (max, double8 , lo, hi)
IMPLEMENT_SPLIT (max, double16, lo, hi)
#else
IMPLEMENT_DIRECT(max, double  , a>=b ? a : b)
IMPLEMENT_DIRECT(max, double2 , a>=b ? a : b)
IMPLEMENT_DIRECT(max, double3 , a>=b ? a : b)
IMPLEMENT_DIRECT(max, double4 , a>=b ? a : b)
IMPLEMENT_DIRECT(max, double8 , a>=b ? a : b)
IMPLEMENT_DIRECT(max, double16, a>=b ? a : b)
#endif

DEFINE_EXPR_V_VS(max, max(a, (vtype)b))
