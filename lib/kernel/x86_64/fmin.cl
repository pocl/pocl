/* OpenCL built-in library: fmin()

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



#define IMPLEMENT_FMIN_SSE_FLOAT1               \
  ({                                            \
    __asm__ ("minss %[src], %[dst]" :           \
             [dst] "+x" (a) :                   \
             [src] "xm" (b));                   \
    a;                                          \
  })
#define IMPLEMENT_FMIN_SSE_FLOAT4               \
  ({                                            \
    __asm__ ("minps %[src], %[dst]" :           \
             [dst] "+x" (a) :                   \
             [src] "xm" (b));                   \
    a;                                          \
  })
#define IMPLEMENT_FMIN_AVX_FLOAT8               \
  ({                                            \
    __asm__ ("vminps %[src], %[dst], %[dst]" :  \
             [dst] "+x" (a) :                   \
             [src] "xm" (b));                   \
    a;                                          \
  })
#define IMPLEMENT_FMIN_SSE2_DOUBLE1             \
  ({                                            \
    __asm__ ("minsd %[src], %[dst]" :           \
             [dst] "+x" (a) :                   \
             [src] "xm" (b));                   \
    a;                                          \
  })
#define IMPLEMENT_FMIN_SSE2_DOUBLE2             \
  ({                                            \
    __asm__ ("minpd %[src], %[dst]" :           \
             [dst] "+x" (a) :                   \
             [src] "xm" (b));                   \
    a;                                          \
  })
#define IMPLEMENT_FMIN_AVX_DOUBLE4              \
  ({                                            \
    __asm__ ("vminpd %[src], %[dst], %[dst]" :  \
             [dst] "+x" (a) :                   \
             [src] "xm" (b));                   \
    a;                                          \
  })



#ifdef __SSE__
IMPLEMENT_DIRECT(fmin, float  , IMPLEMENT_FMIN_SSE_FLOAT1)
IMPLEMENT_UPCAST(fmin, float2 , float4, lo)
IMPLEMENT_UPCAST(fmin, float3 , float4, s012)
IMPLEMENT_DIRECT(fmin, float4 , IMPLEMENT_FMIN_SSE_FLOAT4)
#  ifdef __AVX__
IMPLEMENT_DIRECT(fmin, float8 , IMPLEMENT_FMIN_AVX_FLOAT8)
#  else
IMPLEMENT_SPLIT (fmin, float8 , lo, hi)
#  endif
IMPLEMENT_SPLIT (fmin, float16, lo, hi)
#else
#  error "SSE not supported"
#endif

#ifdef cl_khr_fp64
#ifdef __SSE2__
IMPLEMENT_DIRECT(fmin, double  , IMPLEMENT_FMIN_SSE2_DOUBLE1)
IMPLEMENT_DIRECT(fmin, double2 , IMPLEMENT_FMIN_SSE2_DOUBLE2)
#  ifdef __AVX__
IMPLEMENT_UPCAST(fmin, double3 , double4, s012)
IMPLEMENT_DIRECT(fmin, double4 , IMPLEMENT_FMIN_AVX_DOUBLE4)
#  else
IMPLEMENT_SPLIT (fmin, double3 , lo, s2)
IMPLEMENT_SPLIT (fmin, double4 , lo, hi)
#  endif
IMPLEMENT_SPLIT (fmin, double8 , lo, hi)
IMPLEMENT_SPLIT (fmin, double16, lo, hi)
#else
#  error "SSE2 not supported"
#endif
#endif
