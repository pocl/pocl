/* OpenCL built-in library: abs()

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

#include "../templates.h"

#define IMPLEMENT_DIRECT(NAME, TYPE, UTYPE, EXPR)       \
  UTYPE _CL_OVERLOADABLE NAME(TYPE a)                   \
  {                                                     \
    typedef TYPE gtype;                                 \
    typedef UTYPE ugtype;                               \
    return EXPR;                                        \
  }

#define IMPLEMENT_UPCAST(NAME, TYPE, UTYPE, UPTYPE, LO) \
  UTYPE _CL_OVERLOADABLE NAME(TYPE a)                   \
  {                                                     \
    UPTYPE a1;                                          \
    a1.LO = a;                                          \
    return NAME(a1).LO;                                 \
  }

#define IMPLEMENT_SPLIT(NAME, TYPE, UTYPE, LO, HI)      \
  UTYPE _CL_OVERLOADABLE NAME(TYPE a)                   \
  {                                                     \
    return (UTYPE)(NAME(a.LO), NAME(a.HI));             \
  }



#define IMPLEMENT_ABS_DIRECT_UNSIGNED (a)

#define IMPLEMENT_ABS_BUILTIN_INT  __builtin_abs(a)

#define IMPLEMENT_ABS_DIRECT                    \
  ({                                            \
    a = a<(gtype)0 ? -a : a;                    \
    *(ugtype*)&a;                               \
  })

#define IMPLEMENT_ABS_SSSE3_CHAR16              \
  ({                                            \
    __asm__ ("pabsb %[src], %[dst]" :           \
             [dst] "=x" (a) :                   \
             [src] "x" (a));                    \
    *(ugtype*)&a;                               \
  })
#define IMPLEMENT_ABS_AVX2_CHAR32               \
  ({                                            \
    __asm__ ("pabsb256 %[src], %[dst]" :        \
             [dst] "=x" (a) :                   \
             [src] "x" (a));                    \
    *(ugtype)&a;                                \
  })
#define IMPLEMENT_ABS_SSSE3_SHORT8              \
  ({                                            \
    __asm__ ("pabsw %[src], %[dst]" :           \
             [dst] "=x" (a) :                   \
             [src] "x" (a));                    \
    *(ugtype*)&a;                               \
  })
#define IMPLEMENT_ABS_AVX2_SHORT16              \
  ({                                            \
    __asm__ ("pabsw256 %[src], %[dst]" :        \
             [dst] "=x" (a) :                   \
             [src] "x" (a));                    \
    *(ugtype*)&a;                               \
  })
#define IMPLEMENT_ABS_SSSE3_INT4                \
  ({                                            \
    __asm__ ("pabsd %[src], %[dst]" :           \
             [dst] "=x" (a) :                   \
             [src] "x" (a));                    \
    *(ugtype*)&a;                               \
  })
#define IMPLEMENT_ABS_AVX2_INT8                 \
  ({                                            \
    __asm__ ("pabsd256 %[src], %[dst]" :        \
             [dst] "=x" (a) :                   \
             [src] "x" (a));                    \
    *(ugtype*)&a;                               \
  })



IMPLEMENT_DIRECT(abs, char  , uchar  , IMPLEMENT_ABS_DIRECT)
#ifdef __SSSE3__
IMPLEMENT_UPCAST(abs, char2 , uchar2 , char4 , lo)
IMPLEMENT_UPCAST(abs, char3 , uchar3 , char4 , s012)
IMPLEMENT_UPCAST(abs, char4 , uchar4 , char8 , lo)
IMPLEMENT_UPCAST(abs, char8 , uchar8 , char16, lo)
IMPLEMENT_DIRECT(abs, char16, uchar16, IMPLEMENT_ABS_SSSE3_CHAR16)
#else
IMPLEMENT_DIRECT(abs, char2 , uchar2 , IMPLEMENT_ABS_DIRECT)
IMPLEMENT_DIRECT(abs, char3 , uchar3 , IMPLEMENT_ABS_DIRECT)
IMPLEMENT_DIRECT(abs, char4 , uchar4 , IMPLEMENT_ABS_DIRECT)
IMPLEMENT_DIRECT(abs, char8 , uchar8 , IMPLEMENT_ABS_DIRECT)
IMPLEMENT_DIRECT(abs, char16, uchar16, IMPLEMENT_ABS_DIRECT)
#endif

IMPLEMENT_DIRECT(abs, uchar  , uchar  , IMPLEMENT_ABS_DIRECT_UNSIGNED)
IMPLEMENT_DIRECT(abs, uchar2 , uchar2 , IMPLEMENT_ABS_DIRECT_UNSIGNED)
IMPLEMENT_DIRECT(abs, uchar3 , uchar3 , IMPLEMENT_ABS_DIRECT_UNSIGNED)
IMPLEMENT_DIRECT(abs, uchar4 , uchar4 , IMPLEMENT_ABS_DIRECT_UNSIGNED)
IMPLEMENT_DIRECT(abs, uchar8 , uchar8 , IMPLEMENT_ABS_DIRECT_UNSIGNED)
IMPLEMENT_DIRECT(abs, uchar16, uchar16, IMPLEMENT_ABS_DIRECT_UNSIGNED)

IMPLEMENT_DIRECT(abs, short  , ushort  , IMPLEMENT_ABS_DIRECT)
#ifdef __SSSE3__
IMPLEMENT_UPCAST(abs, short2 , ushort2 , short4, lo)
IMPLEMENT_UPCAST(abs, short3 , ushort3 , short4, s012)
IMPLEMENT_UPCAST(abs, short4 , ushort4 , short8, lo)
IMPLEMENT_DIRECT(abs, short8 , ushort8 , IMPLEMENT_ABS_SSSE3_SHORT8)
#  ifdef __AVX2__
IMPLEMENT_DIRECT(abs, short16, ushort16, IMPLEMENT_ABS_AVX2_SHORT16)
#  else
IMPLEMENT_SPLIT (abs, short16, ushort16, lo, hi)
#  endif
#else
IMPLEMENT_DIRECT(abs, short2 , ushort2 , IMPLEMENT_ABS_DIRECT)
IMPLEMENT_DIRECT(abs, short3 , ushort3 , IMPLEMENT_ABS_DIRECT)
IMPLEMENT_DIRECT(abs, short4 , ushort4 , IMPLEMENT_ABS_DIRECT)
IMPLEMENT_DIRECT(abs, short8 , ushort8 , IMPLEMENT_ABS_DIRECT)
IMPLEMENT_DIRECT(abs, short16, ushort16, IMPLEMENT_ABS_DIRECT)
#endif

IMPLEMENT_DIRECT(abs, ushort  , ushort  , IMPLEMENT_ABS_DIRECT_UNSIGNED)
IMPLEMENT_DIRECT(abs, ushort2 , ushort2 , IMPLEMENT_ABS_DIRECT_UNSIGNED)
IMPLEMENT_DIRECT(abs, ushort3 , ushort3 , IMPLEMENT_ABS_DIRECT_UNSIGNED)
IMPLEMENT_DIRECT(abs, ushort4 , ushort4 , IMPLEMENT_ABS_DIRECT_UNSIGNED)
IMPLEMENT_DIRECT(abs, ushort8 , ushort8 , IMPLEMENT_ABS_DIRECT_UNSIGNED)
IMPLEMENT_DIRECT(abs, ushort16, ushort16, IMPLEMENT_ABS_DIRECT_UNSIGNED)

IMPLEMENT_DIRECT(abs, int  , uint  , IMPLEMENT_ABS_BUILTIN_INT)
#ifdef __SSSE3__
IMPLEMENT_UPCAST(abs, int2 , uint2 , int4, lo)
IMPLEMENT_UPCAST(abs, int3 , uint3 , int4, s012)
IMPLEMENT_DIRECT(abs, int4 , uint4 , IMPLEMENT_ABS_SSSE3_INT4)
#  ifdef __AVX2__
IMPLEMENT_DIRECT(abs, int8 , uint8 , IMPLEMENT_ABS_AVX2_INT8)
#  else
IMPLEMENT_SPLIT (abs, int8 , uint8 , lo, hi)
#endif
IMPLEMENT_SPLIT (abs, int16, uint16, lo, hi)
#else
IMPLEMENT_DIRECT(abs, int2 , uint2 , IMPLEMENT_ABS_DIRECT)
IMPLEMENT_DIRECT(abs, int3 , uint3 , IMPLEMENT_ABS_DIRECT)
IMPLEMENT_DIRECT(abs, int4 , uint4 , IMPLEMENT_ABS_DIRECT)
IMPLEMENT_DIRECT(abs, int8 , uint8 , IMPLEMENT_ABS_DIRECT)
IMPLEMENT_DIRECT(abs, int16, uint16, IMPLEMENT_ABS_DIRECT)
#endif

IMPLEMENT_DIRECT(abs, uint  , uint  , IMPLEMENT_ABS_DIRECT_UNSIGNED)
IMPLEMENT_DIRECT(abs, uint2 , uint2 , IMPLEMENT_ABS_DIRECT_UNSIGNED)
IMPLEMENT_DIRECT(abs, uint3 , uint3 , IMPLEMENT_ABS_DIRECT_UNSIGNED)
IMPLEMENT_DIRECT(abs, uint4 , uint4 , IMPLEMENT_ABS_DIRECT_UNSIGNED)
IMPLEMENT_DIRECT(abs, uint8 , uint8 , IMPLEMENT_ABS_DIRECT_UNSIGNED)
IMPLEMENT_DIRECT(abs, uint16, uint16, IMPLEMENT_ABS_DIRECT_UNSIGNED)

#ifdef cles_khr_int64 
IMPLEMENT_DIRECT(abs, long  , ulong  , IMPLEMENT_ABS_DIRECT)
IMPLEMENT_DIRECT(abs, long2 , ulong2 , IMPLEMENT_ABS_DIRECT)
IMPLEMENT_DIRECT(abs, long3 , ulong3 , IMPLEMENT_ABS_DIRECT)
IMPLEMENT_DIRECT(abs, long4 , ulong4 , IMPLEMENT_ABS_DIRECT)
IMPLEMENT_DIRECT(abs, long8 , ulong8 , IMPLEMENT_ABS_DIRECT)
IMPLEMENT_DIRECT(abs, long16, ulong16, IMPLEMENT_ABS_DIRECT)

IMPLEMENT_DIRECT(abs, ulong  , ulong  , IMPLEMENT_ABS_DIRECT_UNSIGNED)
IMPLEMENT_DIRECT(abs, ulong2 , ulong2 , IMPLEMENT_ABS_DIRECT_UNSIGNED)
IMPLEMENT_DIRECT(abs, ulong3 , ulong3 , IMPLEMENT_ABS_DIRECT_UNSIGNED)
IMPLEMENT_DIRECT(abs, ulong4 , ulong4 , IMPLEMENT_ABS_DIRECT_UNSIGNED)
IMPLEMENT_DIRECT(abs, ulong8 , ulong8 , IMPLEMENT_ABS_DIRECT_UNSIGNED)
IMPLEMENT_DIRECT(abs, ulong16, ulong16, IMPLEMENT_ABS_DIRECT_UNSIGNED)
#endif

