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



#ifdef __SSE__
float4 _cl_sqrt_ensure_float4(float4 a)
{
  return a;
}
IMPLEMENT_DIRECT(sqrt, float  , _cl_sqrt_ensure_float4(__builtin_ia32_sqrtss(*(float4*)&a)).s0)
IMPLEMENT_UPCAST(sqrt, float2 , float4, lo  )
IMPLEMENT_UPCAST(sqrt, float3 , float4, s012)
IMPLEMENT_DIRECT(sqrt, float4 , __builtin_ia32_sqrtps(a))
#else
IMPLEMENT_DIRECT(sqrt, float  , __builtin_sqrtf(a))
IMPLEMENT_SPLIT (sqrt, float2 , lo, hi)
IMPLEMENT_SPLIT (sqrt, float3 , lo, s2)
IMPLEMENT_SPLIT (sqrt, float4 , lo, hi)
#endif
IMPLEMENT_SPLIT (sqrt, float8 , lo, hi)
IMPLEMENT_SPLIT (sqrt, float16, lo, hi)

#ifdef __SSE2__
double2 _cl_sqrt_ensure_double2(double2 a)
{
  return a;
}
IMPLEMENT_DIRECT(sqrt, double  , _cl_sqrt_ensure_double2(__builtin_ia32_sqrtsd(*(double2*)&a)).s0)
IMPLEMENT_DIRECT(sqrt, double2 , __builtin_ia32_sqrtpd(a))
IMPLEMENT_SPLIT (sqrt, double3 , lo, s2)
IMPLEMENT_SPLIT (sqrt, double4 , lo, hi)
#else
IMPLEMENT_DIRECT(sqrt, double  , __builtin_sqrt(a))
IMPLEMENT_SPLIT (sqrt, double2 , lo, hi)
IMPLEMENT_SPLIT (sqrt, double3 , lo, s2)
IMPLEMENT_SPLIT (sqrt, double4 , lo, hi)
#endif
IMPLEMENT_SPLIT (sqrt, double8 , lo, hi)
IMPLEMENT_SPLIT (sqrt, double16, lo, hi)
