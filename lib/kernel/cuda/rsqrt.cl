/* OpenCL built-in library: rsqrt()

   Copyright (c) 2011 Erik Schnetter <eschnetter@perimeterinstitute.ca>
                      Perimeter Institute for Theoretical Physics
                 2022 Isuru Fernando <isuruf@gmail.com>

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

#undef ENABLE_BUILTIN
float _CL_OVERLOADABLE rsqrt(float a)
{
  return __nvvm_rsqrt_approx_f(a);
}

float2 _CL_OVERLOADABLE rsqrt(float2 a)
{
  return (float2)(rsqrt(a.x), rsqrt(a.y));
}

float3 _CL_OVERLOADABLE rsqrt(float3 a)
{
  return (float3)(rsqrt(a.x), rsqrt(a.y), rsqrt(a.z));
}

float4 _CL_OVERLOADABLE rsqrt(float4 a)
{
  return (float4)(rsqrt(a.x), rsqrt(a.y), rsqrt(a.z), rsqrt(a.w));
}

float8 _CL_OVERLOADABLE rsqrt(float8 a)
{
  return (float8)(rsqrt(a.x), rsqrt(a.y), rsqrt(a.z), rsqrt(a.w),
                   rsqrt(a.s4), rsqrt(a.s5), rsqrt(a.s6), rsqrt(a.s7));
}

float16 _CL_OVERLOADABLE rsqrt(float16 a)
{
  return (float16)(rsqrt(a.x), rsqrt(a.y), rsqrt(a.z), rsqrt(a.w),
                    rsqrt(a.s4), rsqrt(a.s5), rsqrt(a.s6), rsqrt(a.s7),
                    rsqrt(a.s8), rsqrt(a.s9), rsqrt(a.sa), rsqrt(a.sb),
                    rsqrt(a.sc), rsqrt(a.sd), rsqrt(a.se), rsqrt(a.sf));
}



#ifdef cl_khr_fp64
double _CL_OVERLOADABLE rsqrt(double a)
{
  return __nvvm_rsqrt_approx_d(a);
}

double2 _CL_OVERLOADABLE rsqrt(double2 a)
{
  return (double2)(rsqrt(a.x), rsqrt(a.y));
}

double3 _CL_OVERLOADABLE rsqrt(double3 a)
{
  return (double3)(rsqrt(a.x), rsqrt(a.y), rsqrt(a.z));
}

double4 _CL_OVERLOADABLE rsqrt(double4 a)
{
  return (double4)(rsqrt(a.x), rsqrt(a.y), rsqrt(a.z), rsqrt(a.w));
}

double8 _CL_OVERLOADABLE rsqrt(double8 a)
{
  return (double8)(rsqrt(a.x), rsqrt(a.y), rsqrt(a.z), rsqrt(a.w),
                   rsqrt(a.s4), rsqrt(a.s5), rsqrt(a.s6), rsqrt(a.s7));
}

double16 _CL_OVERLOADABLE rsqrt(double16 a)
{
  return (double16)(rsqrt(a.x), rsqrt(a.y), rsqrt(a.z), rsqrt(a.w),
                    rsqrt(a.s4), rsqrt(a.s5), rsqrt(a.s6), rsqrt(a.s7),
                    rsqrt(a.s8), rsqrt(a.s9), rsqrt(a.sa), rsqrt(a.sb),
                    rsqrt(a.sc), rsqrt(a.sd), rsqrt(a.se), rsqrt(a.sf));
}
#endif

DEFINE_EXPR_F_F(half_rsqrt, rsqrt(a))
