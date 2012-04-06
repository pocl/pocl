/* OpenCL built-in library: copysign()

   Copyright (c) 2012 Universidad Rey Juan Carlos
   
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

#if 0

#include "../templates.h"

// LLVM generates non-optimal code for this implementation
DEFINE_EXPR_V_VV(copysign,
                 ({
                   int bits = CHAR_BIT * sizeof(stype);
                   sjtype sign_mask = (sjtype)1 << (sjtype)(bits - 1);
                   sjtype result =
                     (~sign_mask & *(jtype*)&a) | (sign_mask & *(jtype*)&b);
                   *(vtype*)&result;
                 }))

#endif



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



#define IMPLEMENT_COPYSIGN_DIRECT                                       \
  ({                                                                    \
    int bits = CHAR_BIT * sizeof(stype);                                \
    jtype sign_mask = (jtype)1 << (jtype)(bits - 1);                    \
    jtype result = (~sign_mask & *(jtype*)&a) | (sign_mask & *(jtype*)&b); \
    *(vtype*)&result;                                                   \
  })
#define IMPLEMENT_COPYSIGN_SSE_FLOAT4                                   \
  ({                                                                    \
    uint4 sign_mask = {0x80000000U, 0x80000000U, 0x80000000U, 0x80000000U}; \
    __asm__ ("andps %[src], %[dst]" :                                   \
             [dst] "+x" (a) :                                           \
             [src] "xm" (~sign_mask));                                  \
    __asm__ ("andps %[src], %[dst]" :                                   \
             [dst] "+x" (b) :                                           \
             [src] "xm" (sign_mask));                                   \
    __asm__ ("orps %[src], %[dst]" :                                    \
             [dst] "+x" (a) :                                           \
             [src] "xm" (b));                                           \
    a;                                                                  \
  })
#define IMPLEMENT_COPYSIGN_AVX_FLOAT8                                   \
  ({                                                                    \
    uint8 sign_mask = {0x80000000U, 0x80000000U, 0x80000000U, 0x80000000U, \
                       0x80000000U, 0x80000000U, 0x80000000U, 0x80000000U}; \
    __asm__ ("andps256 %[src], %[dst]" :                                \
             [dst] "+x" (a) :                                           \
             [src] "xm" (~sign_mask));                                  \
    __asm__ ("andps256 %[src], %[dst]" :                                \
             [dst] "+x" (b) :                                           \
             [src] "xm" (sign_mask));                                   \
    __asm__ ("orps256 %[src], %[dst]" :                                 \
             [dst] "+x" (a) :                                           \
             [src] "xm" (b));                                           \
    a;                                                                  \
  })
#define IMPLEMENT_COPYSIGN_SSE2_DOUBLE2                                 \
  ({                                                                    \
    ulong2 sign_mask = {0x8000000000000000UL, 0x8000000000000000UL};    \
    __asm__ ("andpd %[src], %[dst]" :                                   \
             [dst] "+x" (a) :                                           \
             [src] "xm" (~sign_mask));                                  \
    __asm__ ("andpd %[src], %[dst]" :                                   \
             [dst] "+x" (b) :                                           \
             [src] "xm" (sign_mask));                                   \
    __asm__ ("orpd %[src], %[dst]" :                                    \
             [dst] "+x" (a) :                                           \
             [src] "xm" (b));                                           \
    a;                                                                  \
  })
#define IMPLEMENT_COPYSIGN_AVX_DOUBLE4                                  \
  ({                                                                    \
    ulong4 sign_mask = {0x8000000000000000UL, 0x8000000000000000UL,     \
                        0x8000000000000000UL, 0x8000000000000000UL};    \
    __asm__ ("andpd256 %[src], %[dst]" :                                \
             [dst] "+x" (a) :                                           \
             [src] "xm" (~sign_mask));                                  \
    __asm__ ("andpd256 %[src], %[dst]" :                                \
             [dst] "+x" (b) :                                           \
             [src] "xm" (sign_mask));                                   \
    __asm__ ("orpd256 %[src], %[dst]" :                                 \
             [dst] "+x" (a) :                                           \
             [src] "xm" (b));                                           \
    a;                                                                  \
  })



#ifdef __SSE__
IMPLEMENT_UPCAST(copysign, float  , float2, lo)
IMPLEMENT_UPCAST(copysign, float2 , float4, lo)
IMPLEMENT_UPCAST(copysign, float3 , float4, s012)
IMPLEMENT_DIRECT(copysign, float4 , IMPLEMENT_COPYSIGN_SSE_FLOAT4)
#  ifdef __AVX__
IMPLEMENT_DIRECT(copysign, float8 , IMPLEMENT_COPYSIGN_AVX_FLOAT8)
#  else
IMPLEMENT_SPLIT (copysign, float8 , lo, hi)
#  endif
IMPLEMENT_SPLIT (copysign, float16, lo, hi)
#else
IMPLEMENT_DIRECT(copysign, float  , IMPLEMENT_COPYSIGN_DIRECT)
IMPLEMENT_DIRECT(copysign, float2 , IMPLEMENT_COPYSIGN_DIRECT)
IMPLEMENT_DIRECT(copysign, float3 , IMPLEMENT_COPYSIGN_DIRECT)
IMPLEMENT_DIRECT(copysign, float4 , IMPLEMENT_COPYSIGN_DIRECT)
IMPLEMENT_DIRECT(copysign, float8 , IMPLEMENT_COPYSIGN_DIRECT)
IMPLEMENT_DIRECT(copysign, float16, IMPLEMENT_COPYSIGN_DIRECT)
#endif

#ifdef __SSE2__
IMPLEMENT_UPCAST(copysign, double  , double2, lo)
IMPLEMENT_DIRECT(copysign, double2 , IMPLEMENT_COPYSIGN_SSE2_DOUBLE2)
#  ifdef __AVX__
IMPLEMENT_UPCAST(copysign, double3 , double4, s012)
IMPLEMENT_DIRECT(copysign, double4 , IMPLEMENT_COPYSIGN_AVX_DOUBLE4)
#  else
IMPLEMENT_SPLIT (copysign, double3 , lo, s2)
IMPLEMENT_SPLIT (copysign, double4 , lo, hi)
#  endif
IMPLEMENT_SPLIT (copysign, double8 , lo, hi)
IMPLEMENT_SPLIT (copysign, double16, lo, hi)
#else
IMPLEMENT_DIRECT(copysign, double  , IMPLEMENT_COPYSIGN_DIRECT)
IMPLEMENT_DIRECT(copysign, double2 , IMPLEMENT_COPYSIGN_DIRECT)
IMPLEMENT_DIRECT(copysign, double3 , IMPLEMENT_COPYSIGN_DIRECT)
IMPLEMENT_DIRECT(copysign, double4 , IMPLEMENT_COPYSIGN_DIRECT)
IMPLEMENT_DIRECT(copysign, double8 , IMPLEMENT_COPYSIGN_DIRECT)
IMPLEMENT_DIRECT(copysign, double16, IMPLEMENT_COPYSIGN_DIRECT)
#endif
