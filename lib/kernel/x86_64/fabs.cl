/* OpenCL built-in library: fabs()

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
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#if 0

#include "../templates.h"

// LLVM generates non-optimal code for this implementation
DEFINE_EXPR_V_V(fabs,
                ({
                  int bits = CHAR_BIT * sizeof(stype);
                  sjtype sign_mask = (sjtype)1 << (sjtype)(bits - 1);
                  sjtype result = ~sign_mask & *(jtype*)&a;
                  *(vtype*)&result;
                }))

#endif



#define IMPLEMENT_DIRECT(NAME, TYPE, EXPR)      \
  TYPE _CL_OVERLOADABLE NAME(TYPE a)            \
  {                                             \
    return EXPR;                                \
  }

#define IMPLEMENT_UPCAST(NAME, TYPE, UPTYPE, LO)        \
  TYPE _CL_OVERLOADABLE NAME(TYPE a)                    \
  {                                                     \
    UPTYPE a1;                                          \
    a1.LO = a;                                          \
    return NAME(a1).LO;                                 \
  }

#define IMPLEMENT_SPLIT(NAME, TYPE, LO, HI)     \
  TYPE _CL_OVERLOADABLE NAME(TYPE a)            \
  {                                             \
    return (TYPE)(NAME(a.LO), NAME(a.HI));      \
  }



#define IMPLEMENT_FABS_DIRECT                           \
  ({                                                    \
    int bits = CHAR_BIT * sizeof(stype);                \
    jtype sign_mask = (jtype)1 << (jtype)(bits - 1);    \
    jtype result = ~sign_mask & *(jtype*)&a;            \
    *(vtype*)&result;                                   \
  })
#define IMPLEMENT_FABS_SSE_FLOAT4                       \
  ({                                                    \
    uint4 sign_mask = {0x80000000U, 0x80000000U,        \
                       0x80000000U, 0x80000000U};       \
    __asm__ ("andps %[src], %[dst]" :                   \
             [dst] "+x" (a) :                           \
             [src] "xm" (~sign_mask));                  \
    a;                                                  \
  })
#define IMPLEMENT_FABS_AVX_FLOAT8                       \
  ({                                                    \
    uint8 sign_mask = {0x80000000U, 0x80000000U,        \
                       0x80000000U, 0x80000000U,        \
                       0x80000000U, 0x80000000U,        \
                       0x80000000U, 0x80000000U};       \
    __asm__ ("vandps %[src], %[dst], %[dst]" :          \
             [dst] "+x" (a) :                           \
             [src] "xm" (~sign_mask));                  \
    a;                                                  \
  })
#define IMPLEMENT_FABS_SSE2_DOUBLE2                                     \
  ({                                                                    \
    ulong2 sign_mask = {0x8000000000000000UL, 0x8000000000000000UL};    \
    __asm__ ("andpd %[src], %[dst]" :                                   \
             [dst] "+x" (a) :                                           \
             [src] "xm" (~sign_mask));                                  \
    a;                                                                  \
  })
#define IMPLEMENT_FABS_AVX_DOUBLE4                                      \
  ({                                                                    \
    ulong4 sign_mask = {0x8000000000000000UL, 0x8000000000000000UL,     \
                        0x8000000000000000UL, 0x8000000000000000UL};    \
    __asm__ ("vandpd %[src], %[dst], %[dst]" :                          \
             [dst] "+x" (a) :                                           \
             [src] "xm" (~sign_mask));                                  \
    a;                                                                  \
  })



#ifdef __SSE__
IMPLEMENT_UPCAST(fabs, float  , float2, lo)
IMPLEMENT_UPCAST(fabs, float2 , float4, lo)
IMPLEMENT_UPCAST(fabs, float3 , float4, s012)
IMPLEMENT_DIRECT(fabs, float4 , IMPLEMENT_FABS_SSE_FLOAT4)
#  ifdef __AVX__
IMPLEMENT_DIRECT(fabs, float8 , IMPLEMENT_FABS_AVX_FLOAT8)
#  else
IMPLEMENT_SPLIT (fabs, float8 , lo, hi)
#  endif
IMPLEMENT_SPLIT (fabs, float16, lo, hi)
#else
IMPLEMENT_DIRECT(fabs, float  , IMPLEMENT_FABS_DIRECT)
IMPLEMENT_DIRECT(fabs, float2 , IMPLEMENT_FABS_DIRECT)
IMPLEMENT_DIRECT(fabs, float3 , IMPLEMENT_FABS_DIRECT)
IMPLEMENT_DIRECT(fabs, float4 , IMPLEMENT_FABS_DIRECT)
IMPLEMENT_DIRECT(fabs, float8 , IMPLEMENT_FABS_DIRECT)
IMPLEMENT_DIRECT(fabs, float16, IMPLEMENT_FABS_DIRECT)
#endif

#ifdef cl_khr_fp64
#ifdef __SSE2__
IMPLEMENT_UPCAST(fabs, double  , double2, lo)
IMPLEMENT_DIRECT(fabs, double2 , IMPLEMENT_FABS_SSE2_DOUBLE2)
#  ifdef __AVX__
IMPLEMENT_UPCAST(fabs, double3 , double4, s012)
IMPLEMENT_DIRECT(fabs, double4 , IMPLEMENT_FABS_AVX_DOUBLE4)
#  else
IMPLEMENT_SPLIT (fabs, double3 , lo, s2)
IMPLEMENT_SPLIT (fabs, double4 , lo, hi)
#  endif
IMPLEMENT_SPLIT (fabs, double8 , lo, hi)
IMPLEMENT_SPLIT (fabs, double16, lo, hi)
#else
IMPLEMENT_DIRECT(fabs, double  , IMPLEMENT_FABS_DIRECT)
IMPLEMENT_DIRECT(fabs, double2 , IMPLEMENT_FABS_DIRECT)
IMPLEMENT_DIRECT(fabs, double3 , IMPLEMENT_FABS_DIRECT)
IMPLEMENT_DIRECT(fabs, double4 , IMPLEMENT_FABS_DIRECT)
IMPLEMENT_DIRECT(fabs, double8 , IMPLEMENT_FABS_DIRECT)
IMPLEMENT_DIRECT(fabs, double16, IMPLEMENT_FABS_DIRECT)
#endif
#endif
