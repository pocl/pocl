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



#ifdef __SSE4_1__
float4 _cl_ceil_ensure_float4(float4 a)
{
  return a;
}
IMPLEMENT_DIRECT(ceil, float  , _cl_ceil_ensure_float4(__builtin_ia32_roundss(*(float4*)&a, *(float4*)&a, _MM_FROUND_CEIL)).s0)
IMPLEMENT_UPCAST(ceil, float2 , float4, lo  )
IMPLEMENT_UPCAST(ceil, float3 , float4, s012)
IMPLEMENT_DIRECT(ceil, float4 , __builtin_ia32_roundps(a, _MM_FROUND_CEIL))
#else
IMPLEMENT_DIRECT(ceil, float  , __builtin_ceilf(a))
IMPLEMENT_SPLIT (ceil, float2 , lo, hi)
IMPLEMENT_SPLIT (ceil, float3 , lo, s2)
IMPLEMENT_SPLIT (ceil, float4 , lo, hi)
#endif
IMPLEMENT_SPLIT (ceil, float8 , lo, hi)
IMPLEMENT_SPLIT (ceil, float16, lo, hi)

#ifdef __SSE4_1__
double2 _cl_ceil_ensure_double2(double2 a)
{
  return a;
}
IMPLEMENT_DIRECT(ceil, double  , _cl_ceil_ensure_double2(__builtin_ia32_roundsd(*(double2*)&a, *(double2*)&a, _MM_FROUND_CEIL)).s0)
IMPLEMENT_DIRECT(ceil, double2 , __builtin_ia32_roundpd(a, _MM_FROUND_CEIL))
#else
IMPLEMENT_DIRECT(ceil, double  , __builtin_ceil(a))
IMPLEMENT_SPLIT (ceil, double2 , lo, hi)
#endif
IMPLEMENT_SPLIT (ceil, double3 , lo, s2)
IMPLEMENT_SPLIT (ceil, double4 , lo, hi)
IMPLEMENT_SPLIT (ceil, double8 , lo, hi)
IMPLEMENT_SPLIT (ceil, double16, lo, hi)
