/* OpenCL built-in library: ceil()

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
    UPTYPE a1;                                          \
    a1.LO = a;                                          \
    return NAME(a1).LO;                                 \
  }

#define IMPLEMENT_SPLIT(NAME, TYPE, LO, HI)     \
  TYPE _cl_overloadable NAME(TYPE a)            \
  {                                             \
    return (TYPE)(NAME(a.LO), NAME(a.HI));      \
  }



#define _MM_FROUND_TO_NEAREST_INT 0x00
#define _MM_FROUND_TO_NEG_INF     0x01
#define _MM_FROUND_TO_POS_INF     0x02
#define _MM_FROUND_TO_ZERO        0x03
#define _MM_FROUND_CUR_DIRECTION  0x04

#define _MM_FROUND_RAISE_EXC 0x00
#define _MM_FROUND_NO_EXC    0x08

#define _MM_FROUND_NINT      (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_FLOOR     (_MM_FROUND_TO_NEG_INF     | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_CEIL      (_MM_FROUND_TO_POS_INF     | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_TRUNC     (_MM_FROUND_TO_ZERO        | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_RINT      (_MM_FROUND_CUR_DIRECTION  | _MM_FROUND_RAISE_EXC)
#define _MM_FROUND_NEARBYINT (_MM_FROUND_CUR_DIRECTION  | _MM_FROUND_NO_EXC   )



#define IMPLEMENT_CEIL_DIRECT_FLOAT  __builtin_ceilf(a)
#define IMPLEMENT_CEIL_DIRECT_DOUBLE __builtin_ceil(a)
// Using only a single asm operand leads to better code, since LLVM
// doesn't seem to allocate input and output operands to the same
// register
#define IMPLEMENT_CEIL_SSE41_FLOAT                      \
  ({                                                    \
    __asm__ ("roundss %[dst], %[dst], %[mode]" :        \
             [dst] "+x" (a) :                           \
             [mode] "n" (_MM_FROUND_CEIL));             \
    a;                                                  \
  })
#define IMPLEMENT_CEIL_SSE41_FLOAT4                     \
  ({                                                    \
    __asm__ ("roundps %[dst], %[dst], %[mode]" :        \
             [dst] "+x" (a) :                           \
             [mode] "n" (_MM_FROUND_CEIL));             \
    a;                                                  \
  })
#define IMPLEMENT_CEIL_AVX_FLOAT8                       \
  ({                                                    \
    __asm__ ("roundps256 %[dst], %[dst], %[mode]" :     \
             [dst] "+x" (a) :                           \
             [mode] "n" (_MM_FROUND_CEIL));             \
    a;                                                  \
  })
#define IMPLEMENT_CEIL_SSE41_DOUBLE                     \
  ({                                                    \
    __asm__ ("roundsd %[dst], %[dst], %[mode]" :        \
             [dst] "+x" (a) :                           \
             [mode] "n" (_MM_FROUND_CEIL));             \
    a;                                                  \
  })
#define IMPLEMENT_CEIL_SSE41_DOUBLE2                    \
  ({                                                    \
    __asm__ ("roundpd %[dst], %[dst], %[mode]" :        \
             [dst] "+x" (a) :                           \
             [mode] "n" (_MM_FROUND_CEIL));             \
    a;                                                  \
  })
#define IMPLEMENT_CEIL_AVX_DOUBLE4                      \
  ({                                                    \
    __asm__ ("roundpd256 %[dst], %[dst], %[mode]" :     \
             [dst] "+x" (a) :                           \
             [mode] "n" (_MM_FROUND_CEIL));             \
    a;                                                  \
  })



#ifdef __SSE4_1__
IMPLEMENT_DIRECT(ceil, float  , IMPLEMENT_CEIL_SSE41_FLOAT)
IMPLEMENT_UPCAST(ceil, float2 , float4, lo)
IMPLEMENT_UPCAST(ceil, float3 , float4, s012)
IMPLEMENT_DIRECT(ceil, float4 , IMPLEMENT_CEIL_SSE41_FLOAT4)
#  ifdef __AVX__
IMPLEMENT_DIRECT(ceil, float8 , IMPLEMENT_CEIL_AVX_FLOAT8)
#  else
IMPLEMENT_SPLIT (ceil, float8 , lo, hi)
#  endif
#else
IMPLEMENT_DIRECT(ceil, float  , IMPLEMENT_CEIL_DIRECT_FLOAT)
IMPLEMENT_SPLIT (ceil, float2 , lo, hi)
IMPLEMENT_SPLIT (ceil, float3 , lo, s2)
IMPLEMENT_SPLIT (ceil, float4 , lo, hi)
IMPLEMENT_SPLIT (ceil, float8 , lo, hi)
#endif
IMPLEMENT_SPLIT (ceil, float16, lo, hi)

#ifdef __SSE4_1__
IMPLEMENT_DIRECT(ceil, double  , IMPLEMENT_CEIL_SSE41_DOUBLE)
IMPLEMENT_DIRECT(ceil, double2 , IMPLEMENT_CEIL_SSE41_DOUBLE2)
#  ifdef __AVX__
IMPLEMENT_UPCAST(ceil, double3 , double4, s012)
IMPLEMENT_DIRECT(ceil, double4 , IMPLEMENT_CEIL_AVX_DOUBLE4)
#  else
IMPLEMENT_SPLIT (ceil, double3 , lo, s2)
IMPLEMENT_SPLIT (ceil, double4 , lo, hi)
#  endif
#else
IMPLEMENT_DIRECT(ceil, double  , IMPLEMENT_CEIL_DIRECT_DOUBLE)
IMPLEMENT_SPLIT (ceil, double2 , lo, hi)
IMPLEMENT_SPLIT (ceil, double3 , lo, s2)
IMPLEMENT_SPLIT (ceil, double4 , lo, hi)
#endif
IMPLEMENT_SPLIT (ceil, double8 , lo, hi)
IMPLEMENT_SPLIT (ceil, double16, lo, hi)
