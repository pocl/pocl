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
    UPTYPE a1, b1;                                      \
    a1.LO = a;                                          \
    b1.LO = b;                                          \
    return NAME(a1, b1).LO;                             \
  }

#define IMPLEMENT_SPLIT(NAME, TYPE, LO, HI)             \
  TYPE _cl_overloadable NAME(TYPE a, TYPE b)            \
  {                                                     \
    return (TYPE)(NAME(a.LO, b.LO), NAME(a.HI, b.HI));  \
  }



#define IMPLEMENT_MIN_DIRECT (a<=b ? a : b)
#define IMPLEMENT_MIN_SSE41_CHAR16              \
  ({                                            \
    __asm__ ("pminsb %[src], %[dst]" :          \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })
#define IMPLEMENT_MIN_SSE_UCHAR16               \
  ({                                            \
    __asm__ ("pminub %[src], %[dst]" :          \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })
#define IMPLEMENT_MIN_SSE_SHORT8                \
  ({                                            \
    __asm__ ("pminsw %[src], %[dst]" :          \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })
#define IMPLEMENT_MIN_SSE41_USHORT8             \
  ({                                            \
    __asm__ ("pminuw %[src], %[dst]" :          \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })
#define IMPLEMENT_MIN_SSE41_INT4                \
  ({                                            \
    __asm__ ("pminsd %[src], %[dst]" :          \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })
#define IMPLEMENT_MIN_SSE41_UINT4               \
  ({                                            \
    __asm__ ("pminud %[src], %[dst]" :          \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })

#define IMPLEMENT_MIN_DIRECT_CAST                               \
  ({                                                            \
    jtype result = (jtype)(a<=b) ? *(jtype*)&a : *(jtype*)&b;   \
    &(type*)&result;                                            \
  })
#define IMPLEMENT_MIN_SSE_FLOAT                 \
  ({                                            \
    __asm__ ("minss %[src], %[dst]" :           \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })
#define IMPLEMENT_MIN_SSE_FLOAT4                \
  ({                                            \
    __asm__ ("minps %[src], %[dst]" :           \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })
#define IMPLEMENT_MIN_AVX_FLOAT8                \
  ({                                            \
    __asm__ ("minps256 %[src], %[dst]" :        \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })
#define IMPLEMENT_MIN_SSE2_DOUBLE               \
  ({                                            \
    __asm__ ("minsd %[src], %[dst]" :           \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })
#define IMPLEMENT_MIN_SSE2_DOUBLE2              \
  ({                                            \
    __asm__ ("minpd %[src], %[dst]" :           \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })
#define IMPLEMENT_MIN_AVX_DOUBLE4               \
  ({                                            \
    __asm__ ("minpd256 %[src], %[dst]" :        \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })



IMPLEMENT_DIRECT(min, char  , IMPLEMENT_MIN_DIRECT)
#ifdef __SSE4_1__
IMPLEMENT_UPCAST(min, char2 , char4 , lo)
IMPLEMENT_UPCAST(min, char3 , char4 , s012)
IMPLEMENT_UPCAST(min, char4 , char8 , lo)
IMPLEMENT_UPCAST(min, char8 , char16, lo)
IMPLEMENT_DIRECT(min, char16, IMPLEMENT_MIN_SSE41_CHAR16)
#else
IMPLEMENT_DIRECT(min, char2 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, char3 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, char4 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, char8 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, char16, IMPLEMENT_MIN_DIRECT)
#endif

IMPLEMENT_DIRECT(min, uchar  , IMPLEMENT_MIN_DIRECT)
#ifdef __SSE__
IMPLEMENT_UPCAST(min, uchar2 , uchar4 ,lo)
IMPLEMENT_UPCAST(min, uchar3 , uchar4 , s012)
IMPLEMENT_UPCAST(min, uchar4 , uchar8 , lo)
IMPLEMENT_UPCAST(min, uchar8 , uchar16, lo)
IMPLEMENT_DIRECT(min, uchar16, IMPLEMENT_MIN_SSE_UCHAR16)
#else
IMPLEMENT_DIRECT(min, uchar2 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, uchar3 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, uchar4 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, uchar8 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, uchar16, IMPLEMENT_MIN_DIRECT)
#endif

IMPLEMENT_DIRECT(min, short  , IMPLEMENT_MIN_DIRECT)
#ifdef __SSE__
IMPLEMENT_UPCAST(min, short2 , short4, lo)
IMPLEMENT_UPCAST(min, short3 , short4, s012)
IMPLEMENT_UPCAST(min, short4 , short8, lo)
IMPLEMENT_DIRECT(min, short8 , IMPLEMENT_MIN_SSE_SHORT8)
IMPLEMENT_SPLIT (min, short16, lo, hi)
#else
IMPLEMENT_DIRECT(min, short2 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, short3 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, short4 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, short8 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, short16, IMPLEMENT_MIN_DIRECT)
#endif

IMPLEMENT_DIRECT(min, ushort  , IMPLEMENT_MIN_DIRECT)
#ifdef __SSE4_1__
IMPLEMENT_UPCAST(min, ushort2 , ushort4, lo)
IMPLEMENT_UPCAST(min, ushort3 , ushort4, s012)
IMPLEMENT_UPCAST(min, ushort4 , ushort8, lo)
IMPLEMENT_DIRECT(min, ushort8 , IMPLEMENT_MIN_SSE41_USHORT8)
IMPLEMENT_SPLIT (min, ushort16, lo, hi)
#else
IMPLEMENT_DIRECT(min, ushort2 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, ushort3 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, ushort4 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, ushort8 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, ushort16, IMPLEMENT_MIN_DIRECT)
#endif

IMPLEMENT_DIRECT(min, int  , IMPLEMENT_MIN_DIRECT)
#ifdef __SSE4_1__
IMPLEMENT_UPCAST(min, int2 , int4, lo)
IMPLEMENT_UPCAST(min, int3 , int4, s012)
IMPLEMENT_DIRECT(min, int4 , IMPLEMENT_MIN_SSE41_INT4)
IMPLEMENT_SPLIT (min, int8 , lo, hi)
IMPLEMENT_SPLIT (min, int16, lo, hi)
#else
IMPLEMENT_DIRECT(min, int2 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, int3 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, int4 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, int8 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, int16, IMPLEMENT_MIN_DIRECT)
#endif

IMPLEMENT_DIRECT(min, uint  , IMPLEMENT_MIN_DIRECT)
#ifdef __SSE4_1__
IMPLEMENT_UPCAST(min, uint2 , uint4, lo)
IMPLEMENT_UPCAST(min, uint3 , uint4, s012)
IMPLEMENT_DIRECT(min, uint4 , IMPLEMENT_MIN_SSE41_UINT4)
IMPLEMENT_SPLIT (min, uint8 , lo, hi)
IMPLEMENT_SPLIT (min, uint16, lo, hi)
#else
IMPLEMENT_DIRECT(min, uint2 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, uint3 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, uint4 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, uint8 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, uint16, IMPLEMENT_MIN_DIRECT)
#endif

#ifdef cles_khr_int64
IMPLEMENT_DIRECT(min, long  , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, long2 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, long3 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, long4 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, long8 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, long16, IMPLEMENT_MIN_DIRECT)

IMPLEMENT_DIRECT(min, ulong  , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, ulong2 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, ulong3 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, ulong4 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, ulong8 , IMPLEMENT_MIN_DIRECT)
IMPLEMENT_DIRECT(min, ulong16, IMPLEMENT_MIN_DIRECT)
#endif

DEFINE_EXPR_G_GS(min, min(a, (gtype)b))



// Note: min() has no special semantics for inf/nan, even if fmin does

#ifdef __SSE__
IMPLEMENT_DIRECT(min, float  , IMPLEMENT_MIN_SSE_FLOAT)
IMPLEMENT_UPCAST(min, float2 , float4, lo)
IMPLEMENT_UPCAST(min, float3 , float4, s012)
IMPLEMENT_DIRECT(min, float4 , IMPLEMENT_MIN_SSE_FLOAT4)
#  ifdef __AVX__
IMPLEMENT_DIRECT(min, float8 , IMPLEMENT_MIN_AVX_FLOAT8)
#  else
IMPLEMENT_SPLIT (min, float8 , lo, hi)
#  endif
IMPLEMENT_SPLIT (min, float16, lo, hi)
#else
IMPLEMENT_DIRECT(min, float  , IMPLEMENT_MIN_DIRECT_CAST)
IMPLEMENT_DIRECT(min, float2 , IMPLEMENT_MIN_DIRECT_CAST)
IMPLEMENT_DIRECT(min, float3 , IMPLEMENT_MIN_DIRECT_CAST)
IMPLEMENT_DIRECT(min, float4 , IMPLEMENT_MIN_DIRECT_CAST)
IMPLEMENT_DIRECT(min, float8 , IMPLEMENT_MIN_DIRECT_CAST)
IMPLEMENT_DIRECT(min, float16, IMPLEMENT_MIN_DIRECT_CAST)
#endif

#ifdef cl_khr_fp64
#ifdef __SSE2__
IMPLEMENT_DIRECT(min, double  , IMPLEMENT_MIN_SSE2_DOUBLE)
IMPLEMENT_DIRECT(min, double2 , IMPLEMENT_MIN_SSE2_DOUBLE2)
#  ifdef __AVX__
IMPLEMENT_UPCAST(min, double3 , double4, s012)
IMPLEMENT_DIRECT(min, double4 , IMPLEMENT_MIN_AVX_DOUBLE4)
#  else
IMPLEMENT_SPLIT (min, double3 , lo, s2)
IMPLEMENT_SPLIT (min, double4 , lo, hi)
#  endif
IMPLEMENT_SPLIT (min, double8 , lo, hi)
IMPLEMENT_SPLIT (min, double16, lo, hi)
#else
IMPLEMENT_DIRECT(min, double  , IMPLEMENT_MIN_DIRECT_CAST)
IMPLEMENT_DIRECT(min, double2 , IMPLEMENT_MIN_DIRECT_CAST)
IMPLEMENT_DIRECT(min, double3 , IMPLEMENT_MIN_DIRECT_CAST)
IMPLEMENT_DIRECT(min, double4 , IMPLEMENT_MIN_DIRECT_CAST)
IMPLEMENT_DIRECT(min, double8 , IMPLEMENT_MIN_DIRECT_CAST)
IMPLEMENT_DIRECT(min, double16, IMPLEMENT_MIN_DIRECT_CAST)
#endif
#endif

DEFINE_EXPR_V_VS(min, min(a, (vtype)b))
