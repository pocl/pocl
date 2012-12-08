/* OpenCL built-in library: fmax()

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



#define IMPLEMENT_FMAX_SSE_FLOAT1               \
  ({                                            \
    __asm__ ("maxss %[src], %[dst]" :           \
             [dst] "+x" (a) :                   \
             [src] "xm" (b));                   \
    a;                                          \
  })
#define IMPLEMENT_FMAX_SSE_FLOAT4               \
  ({                                            \
    __asm__ ("maxps %[src], %[dst]" :           \
             [dst] "+x" (a) :                   \
             [src] "xm" (b));                   \
    a;                                          \
  })
#define IMPLEMENT_FMAX_AVX_FLOAT8               \
  ({                                            \
    __asm__ ("maxps256 %[src], %[dst]" :        \
             [dst] "+x" (a) :                   \
             [src] "xm" (b));                   \
    a;                                          \
  })
#define IMPLEMENT_FMAX_SSE2_DOUBLE1             \
  ({                                            \
    __asm__ ("maxsd %[src], %[dst]" :           \
             [dst] "+x" (a) :                   \
             [src] "xm" (b));                   \
    a;                                          \
  })
#define IMPLEMENT_FMAX_SSE2_DOUBLE2             \
  ({                                            \
    __asm__ ("maxpd %[src], %[dst]" :           \
             [dst] "+x" (a) :                   \
             [src] "xm" (b));                   \
    a;                                          \
  })
#define IMPLEMENT_FMAX_AVX_DOUBLE4              \
  ({                                            \
    __asm__ ("maxpd256 %[src], %[dst]" :        \
             [dst] "+x" (a) :                   \
             [src] "xm" (b));                   \
    a;                                          \
  })



#ifdef __SSE__
IMPLEMENT_DIRECT(fmax, float  , IMPLEMENT_FMAX_SSE_FLOAT1)
IMPLEMENT_UPCAST(fmax, float2 , float4, lo)
IMPLEMENT_UPCAST(fmax, float3 , float4, s012)
IMPLEMENT_DIRECT(fmax, float4 , IMPLEMENT_FMAX_SSE_FLOAT4)
#  ifdef __AVX__
IMPLEMENT_DIRECT(fmax, float8 , IMPLEMENT_FMAX_AVX_FLOAT8)
#  else
IMPLEMENT_SPLIT (fmax, float8 , lo, hi)
#  endif
IMPLEMENT_SPLIT (fmax, float16, lo, hi)
#else
#  error "SSE not supported"
#endif

#ifdef __SSE2__
IMPLEMENT_DIRECT(fmax, double  , IMPLEMENT_FMAX_SSE2_DOUBLE1)
IMPLEMENT_DIRECT(fmax, double2 , IMPLEMENT_FMAX_SSE2_DOUBLE2)
#  ifdef __AVX__
IMPLEMENT_UPCAST(fmax, double3 , double4, s012)
IMPLEMENT_DIRECT(fmax, double4 , IMPLEMENT_FMAX_AVX_DOUBLE4)
#  else
IMPLEMENT_SPLIT (fmax, double3 , lo, s2)
IMPLEMENT_SPLIT (fmax, double4 , lo, hi)
#  endif
IMPLEMENT_SPLIT (fmax, double8 , lo, hi)
IMPLEMENT_SPLIT (fmax, double16, lo, hi)
#else
#  error "SSE2 not supported"
#endif
