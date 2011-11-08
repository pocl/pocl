/* OpenCL built-in library: sqrt()

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

#define IMPLEMENT_DIRECT(NAME, TYPE, EXPR)      \
  TYPE _cl_overloadable NAME(TYPE a)            \
  {                                             \
    typedef TYPE type;                          \
    return EXPR;                                \
  }

#define IMPLEMENT_UPCAST(NAME, TYPE, UPTYPE, LO)        \
  TYPE _cl_overloadable NAME(TYPE a)                    \
  {                                                     \
    return NAME(*(UPTYPE*)&a).LO;                       \
  }

#define IMPLEMENT_SPLIT(NAME, TYPE, LO, HI)     \
  TYPE _cl_overloadable NAME(TYPE a)            \
  {                                             \
    return (TYPE)(NAME(a.LO), NAME(a.HI));      \
  }



#define IMPLEMENT_SQRT_DIRECT_FLOAT  __builtin_sqrtf(a)
#define IMPLEMENT_SQRT_DIRECT_DOUBLE __builtin_sqrt(a)
#define IMPLEMENT_SQRT_SSE_FLOAT                \
  ({                                            \
    __asm__ ("sqrtss %[dst], %[dst]" :          \
             [dst] "+x" (a));                   \
    a;                                          \
  })
#define IMPLEMENT_SQRT_SSE_FLOAT4               \
  ({                                            \
    __asm__ ("sqrtps %[dst], %[dst]" :          \
             [dst] "+x" (a));                   \
    a;                                          \
  })
#define IMPLEMENT_SQRT_AVX_FLOAT8               \
  ({                                            \
    __asm__ ("sqrtps256 %[dst], %[dst]" :       \
             [dst] "+x" (a));                   \
    a;                                          \
  })
#define IMPLEMENT_SQRT_SSE2_DOUBLE              \
  ({                                            \
    __asm__ ("sqrtsd %[dst], %[dst]" :          \
             [dst] "+x" (a));                   \
    a;                                          \
  })
#define IMPLEMENT_SQRT_SSE2_DOUBLE2             \
  ({                                            \
    __asm__ ("sqrtpd %[dst], %[dst]" :          \
             [dst] "+x" (a));                   \
    a;                                          \
  })
#define IMPLEMENT_SQRT_AVX_DOUBLE4              \
  ({                                            \
    __asm__ ("sqrtpd256 %[dst], %[dst]" :       \
             [dst] "+x" (a));                   \
    a;                                          \
  })



#ifdef __SSE__
IMPLEMENT_DIRECT(sqrt, float  , IMPLEMENT_SQRT_SSE_FLOAT)
IMPLEMENT_UPCAST(sqrt, float2 , float4, lo)
IMPLEMENT_UPCAST(sqrt, float3 , float4, s012)
IMPLEMENT_DIRECT(sqrt, float4 , IMPLEMENT_SQRT_SSE_FLOAT4)
#  ifdef __AVX__
IMPLEMENT_DIRECT(sqrt, float8 , IMPLEMENT_SQRT_AVX_FLOAT8)
#  else
IMPLEMENT_SPLIT (sqrt, float8 , lo, hi)
#  endif
#else
IMPLEMENT_DIRECT(sqrt, float  , IMPLEMENT_SQRT_DIRECT_FLOAT)
IMPLEMENT_SPLIT (sqrt, float2 , lo, hi)
IMPLEMENT_SPLIT (sqrt, float3 , lo, s2)
IMPLEMENT_SPLIT (sqrt, float4 , lo, hi)
IMPLEMENT_SPLIT (sqrt, float8 , lo, hi)
#endif
IMPLEMENT_SPLIT (sqrt, float16, lo, hi)

#ifdef __SSE2__
IMPLEMENT_DIRECT(sqrt, double  , IMPLEMENT_SQRT_SSE2_DOUBLE)
IMPLEMENT_DIRECT(sqrt, double2 , IMPLEMENT_SQRT_SSE2_DOUBLE2)
#  ifdef __AVX__
IMPLEMENT_UPCAST(sqrt, double3 , double4, s012)
IMPLEMENT_DIRECT(sqrt, double4 , IMPLEMENT_SQRT_AVX_DOUBLE4)
#  else
IMPLEMENT_SPLIT (sqrt, double3 , lo, s2)
IMPLEMENT_SPLIT (sqrt, double4 , lo, hi)
#  endif
#else
IMPLEMENT_DIRECT(sqrt, double  , IMPLEMENT_SQRT_DIRECT_DOUBLE)
IMPLEMENT_SPLIT (sqrt, double2 , lo, hi)
IMPLEMENT_SPLIT (sqrt, double3 , lo, s2)
IMPLEMENT_SPLIT (sqrt, double4 , lo, hi)
#endif
IMPLEMENT_SPLIT (sqrt, double8 , lo, hi)
IMPLEMENT_SPLIT (sqrt, double16, lo, hi)
