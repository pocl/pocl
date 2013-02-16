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
  TYPE _CL_OVERLOADABLE NAME(TYPE a, TYPE b)    \
  {                                             \
    return EXPR;                                \
  }

#define IMPLEMENT_UPCAST(NAME, TYPE, UPTYPE, LO)        \
  TYPE _CL_OVERLOADABLE NAME(TYPE a, TYPE b)            \
  {                                                     \
    UPTYPE a1, b1;                                      \
    a1.LO = a;                                          \
    b1.LO = b;                                          \
    return NAME(a1, b1).LO;                             \
  }

#define IMPLEMENT_SPLIT(NAME, TYPE, LO, HI)             \
  TYPE _CL_OVERLOADABLE NAME(TYPE a, TYPE b)            \
  {                                                     \
    return (TYPE)(NAME(a.LO, b.LO), NAME(a.HI, b.HI));  \
  }



#define IMPLEMENT_MAX_DIRECT (a>=b ? a : b)
#define IMPLEMENT_MAX_SSE41_CHAR16              \
  ({                                            \
    __asm__ ("pmaxsb %[src], %[dst]" :          \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })
#define IMPLEMENT_MAX_SSE_UCHAR16               \
  ({                                            \
    __asm__ ("pmaxub %[src], %[dst]" :          \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })
#define IMPLEMENT_MAX_SSE_SHORT8                \
  ({                                            \
    __asm__ ("pmaxsw %[src], %[dst]" :          \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })
#define IMPLEMENT_MAX_SSE41_USHORT8             \
  ({                                            \
    __asm__ ("pmaxuw %[src], %[dst]" :          \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })
#define IMPLEMENT_MAX_SSE41_INT4                \
  ({                                            \
    __asm__ ("pmaxsd %[src], %[dst]" :          \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })
#define IMPLEMENT_MAX_SSE41_UINT4               \
  ({                                            \
    __asm__ ("pmaxud %[src], %[dst]" :          \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })

#define IMPLEMENT_MAX_DIRECT_CAST                               \
  ({                                                            \
    jtype result = (jtype)(a>=b) ? *(jtype*)&a : *(jtype*)&b;   \
    &(type*)&result;                                            \
  })
#define IMPLEMENT_MAX_SSE_FLOAT                 \
  ({                                            \
    __asm__ ("maxss %[src], %[dst]" :           \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })
#define IMPLEMENT_MAX_SSE_FLOAT4                \
  ({                                            \
    __asm__ ("maxps %[src], %[dst]" :           \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })
#define IMPLEMENT_MAX_AVX_FLOAT8                \
  ({                                            \
    __asm__ ("maxps256 %[src], %[dst]" :        \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })
#define IMPLEMENT_MAX_SSE2_DOUBLE               \
  ({                                            \
    __asm__ ("maxsd %[src], %[dst]" :           \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })
#define IMPLEMENT_MAX_SSE2_DOUBLE2              \
  ({                                            \
    __asm__ ("maxpd %[src], %[dst]" :           \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })
#define IMPLEMENT_MAX_AVX_DOUBLE4               \
  ({                                            \
    __asm__ ("maxpd256 %[src], %[dst]" :        \
             [dst] "+x" (a) :                   \
             [src] "x" (b));                    \
    a;                                          \
  })



IMPLEMENT_DIRECT(max, char  , IMPLEMENT_MAX_DIRECT)
#ifdef __SSE4_1__
IMPLEMENT_UPCAST(max, char2 , char4 , lo)
IMPLEMENT_UPCAST(max, char3 , char4 , s012)
IMPLEMENT_UPCAST(max, char4 , char8 , lo)
IMPLEMENT_UPCAST(max, char8 , char16, lo)
IMPLEMENT_DIRECT(max, char16, IMPLEMENT_MAX_SSE41_CHAR16)
#else
IMPLEMENT_DIRECT(max, char2 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, char3 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, char4 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, char8 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, char16, IMPLEMENT_MAX_DIRECT)
#endif

IMPLEMENT_DIRECT(max, uchar  , IMPLEMENT_MAX_DIRECT)
#ifdef __SSE__
IMPLEMENT_UPCAST(max, uchar2 , uchar4 ,lo)
IMPLEMENT_UPCAST(max, uchar3 , uchar4 , s012)
IMPLEMENT_UPCAST(max, uchar4 , uchar8 , lo)
IMPLEMENT_UPCAST(max, uchar8 , uchar16, lo)
IMPLEMENT_DIRECT(max, uchar16, IMPLEMENT_MAX_SSE_UCHAR16)
#else
IMPLEMENT_DIRECT(max, uchar2 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, uchar3 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, uchar4 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, uchar8 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, uchar16, IMPLEMENT_MAX_DIRECT)
#endif

IMPLEMENT_DIRECT(max, short  , IMPLEMENT_MAX_DIRECT)
#ifdef __SSE__
IMPLEMENT_UPCAST(max, short2 , short4, lo)
IMPLEMENT_UPCAST(max, short3 , short4, s012)
IMPLEMENT_UPCAST(max, short4 , short8, lo)
IMPLEMENT_DIRECT(max, short8 , IMPLEMENT_MAX_SSE_SHORT8)
IMPLEMENT_SPLIT (max, short16, lo, hi)
#else
IMPLEMENT_DIRECT(max, short2 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, short3 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, short4 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, short8 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, short16, IMPLEMENT_MAX_DIRECT)
#endif

IMPLEMENT_DIRECT(max, ushort  , IMPLEMENT_MAX_DIRECT)
#ifdef __SSE4_1__
IMPLEMENT_UPCAST(max, ushort2 , ushort4, lo)
IMPLEMENT_UPCAST(max, ushort3 , ushort4, s012)
IMPLEMENT_UPCAST(max, ushort4 , ushort8, lo)
IMPLEMENT_DIRECT(max, ushort8 , IMPLEMENT_MAX_SSE41_USHORT8)
IMPLEMENT_SPLIT (max, ushort16, lo, hi)
#else
IMPLEMENT_DIRECT(max, ushort2 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, ushort3 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, ushort4 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, ushort8 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, ushort16, IMPLEMENT_MAX_DIRECT)
#endif

IMPLEMENT_DIRECT(max, int  , IMPLEMENT_MAX_DIRECT)
#ifdef __SSE4_1__
IMPLEMENT_UPCAST(max, int2 , int4, lo)
IMPLEMENT_UPCAST(max, int3 , int4, s012)
IMPLEMENT_DIRECT(max, int4 , IMPLEMENT_MAX_SSE41_INT4)
IMPLEMENT_SPLIT (max, int8 , lo, hi)
IMPLEMENT_SPLIT (max, int16, lo, hi)
#else
IMPLEMENT_DIRECT(max, int2 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, int3 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, int4 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, int8 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, int16, IMPLEMENT_MAX_DIRECT)
#endif

IMPLEMENT_DIRECT(max, uint  , IMPLEMENT_MAX_DIRECT)
#ifdef __SSE4_1__
IMPLEMENT_UPCAST(max, uint2 , uint4, lo)
IMPLEMENT_UPCAST(max, uint3 , uint4, s012)
IMPLEMENT_DIRECT(max, uint4 , IMPLEMENT_MAX_SSE41_UINT4)
IMPLEMENT_SPLIT (max, uint8 , lo, hi)
IMPLEMENT_SPLIT (max, uint16, lo, hi)
#else
IMPLEMENT_DIRECT(max, uint2 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, uint3 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, uint4 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, uint8 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, uint16, IMPLEMENT_MAX_DIRECT)
#endif

#ifdef cles_khr_int64
IMPLEMENT_DIRECT(max, long  , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, long2 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, long3 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, long4 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, long8 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, long16, IMPLEMENT_MAX_DIRECT)

IMPLEMENT_DIRECT(max, ulong  , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, ulong2 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, ulong3 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, ulong4 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, ulong8 , IMPLEMENT_MAX_DIRECT)
IMPLEMENT_DIRECT(max, ulong16, IMPLEMENT_MAX_DIRECT)
#endif

DEFINE_EXPR_G_GS(max, max(a, (gtype)b))



// Note: max() has no special semantics for inf/nan, even if fmax does

#ifdef __SSE__
IMPLEMENT_DIRECT(max, float  , IMPLEMENT_MAX_SSE_FLOAT)
IMPLEMENT_UPCAST(max, float2 , float4, lo)
IMPLEMENT_UPCAST(max, float3 , float4, s012)
IMPLEMENT_DIRECT(max, float4 , IMPLEMENT_MAX_SSE_FLOAT4)
#  ifdef __AVX__
IMPLEMENT_DIRECT(max, float8 , IMPLEMENT_MAX_AVX_FLOAT8)
#  else
IMPLEMENT_SPLIT (max, float8 , lo, hi)
#  endif
IMPLEMENT_SPLIT (max, float16, lo, hi)
#else
IMPLEMENT_DIRECT(max, float  , IMPLEMENT_MAX_DIRECT_CAST)
IMPLEMENT_DIRECT(max, float2 , IMPLEMENT_MAX_DIRECT_CAST)
IMPLEMENT_DIRECT(max, float3 , IMPLEMENT_MAX_DIRECT_CAST)
IMPLEMENT_DIRECT(max, float4 , IMPLEMENT_MAX_DIRECT_CAST)
IMPLEMENT_DIRECT(max, float8 , IMPLEMENT_MAX_DIRECT_CAST)
IMPLEMENT_DIRECT(max, float16, IMPLEMENT_MAX_DIRECT_CAST)
#endif

#ifdef cl_khr_fp64
#ifdef __SSE2__
IMPLEMENT_DIRECT(max, double  , IMPLEMENT_MAX_SSE2_DOUBLE)
IMPLEMENT_DIRECT(max, double2 , IMPLEMENT_MAX_SSE2_DOUBLE2)
#  ifdef __AVX__
IMPLEMENT_UPCAST(max, double3 , double4, s012)
IMPLEMENT_DIRECT(max, double4 , IMPLEMENT_MAX_AVX_DOUBLE4)
#  else
IMPLEMENT_SPLIT (max, double3 , lo, s2)
IMPLEMENT_SPLIT (max, double4 , lo, hi)
#  endif
IMPLEMENT_SPLIT (max, double8 , lo, hi)
IMPLEMENT_SPLIT (max, double16, lo, hi)
#else
IMPLEMENT_DIRECT(max, double  , IMPLEMENT_MAX_DIRECT_CAST)
IMPLEMENT_DIRECT(max, double2 , IMPLEMENT_MAX_DIRECT_CAST)
IMPLEMENT_DIRECT(max, double3 , IMPLEMENT_MAX_DIRECT_CAST)
IMPLEMENT_DIRECT(max, double4 , IMPLEMENT_MAX_DIRECT_CAST)
IMPLEMENT_DIRECT(max, double8 , IMPLEMENT_MAX_DIRECT_CAST)
IMPLEMENT_DIRECT(max, double16, IMPLEMENT_MAX_DIRECT_CAST)
#endif
#endif

DEFINE_EXPR_V_VS(max, max(a, (vtype)b))
