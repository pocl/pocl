/* OpenCL built-in library: sqrt()

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

#include "templates.h"

#undef ENABLE_BUILTIN
float sqrtf(float);

float _CL_OVERLOADABLE sqrt(float a)
{
#if defined(ENABLE_BUILTIN) && __has_builtin(__builtin_sqrtf)
  return __builtin_sqrtf(a);
#else
  return sqrtf(a);
#endif
}

float2 _CL_OVERLOADABLE sqrt(float2 a)
{
  return (float2)(sqrt(a.x), sqrt(a.y));
}

float3 _CL_OVERLOADABLE sqrt(float3 a)
{
  return (float3)(sqrt(a.x), sqrt(a.y), sqrt(a.z));
}

float4 _CL_OVERLOADABLE sqrt(float4 a)
{
  return (float4)(sqrt(a.x), sqrt(a.y), sqrt(a.z), sqrt(a.w));
}

float8 _CL_OVERLOADABLE sqrt(float8 a)
{
  return (float8)(sqrt(a.x), sqrt(a.y), sqrt(a.z), sqrt(a.w),
                   sqrt(a.s4), sqrt(a.s5), sqrt(a.s6), sqrt(a.s7));
}

float16 _CL_OVERLOADABLE sqrt(float16 a)
{
  return (float16)(sqrt(a.x), sqrt(a.y), sqrt(a.z), sqrt(a.w),
                    sqrt(a.s4), sqrt(a.s5), sqrt(a.s6), sqrt(a.s7),
                    sqrt(a.s8), sqrt(a.s9), sqrt(a.sa), sqrt(a.sb),
                    sqrt(a.sc), sqrt(a.sd), sqrt(a.se), sqrt(a.sf));
}



#ifdef cl_khr_fp64
double _CL_OVERLOADABLE sqrt(double a)
{
  return __builtin_sqrt(a);
}

double2 _CL_OVERLOADABLE sqrt(double2 a)
{
  return (double2)(sqrt(a.x), sqrt(a.y));
}

double3 _CL_OVERLOADABLE sqrt(double3 a)
{
  return (double3)(sqrt(a.x), sqrt(a.y), sqrt(a.z));
}

double4 _CL_OVERLOADABLE sqrt(double4 a)
{
  return (double4)(sqrt(a.x), sqrt(a.y), sqrt(a.z), sqrt(a.w));
}

double8 _CL_OVERLOADABLE sqrt(double8 a)
{
  return (double8)(sqrt(a.x), sqrt(a.y), sqrt(a.z), sqrt(a.w),
                   sqrt(a.s4), sqrt(a.s5), sqrt(a.s6), sqrt(a.s7));
}

double16 _CL_OVERLOADABLE sqrt(double16 a)
{
  return (double16)(sqrt(a.x), sqrt(a.y), sqrt(a.z), sqrt(a.w),
                    sqrt(a.s4), sqrt(a.s5), sqrt(a.s6), sqrt(a.s7),
                    sqrt(a.s8), sqrt(a.s9), sqrt(a.sa), sqrt(a.sb),
                    sqrt(a.sc), sqrt(a.sd), sqrt(a.se), sqrt(a.sf));
}
#endif

DEFINE_EXPR_F_F(half_sqrt, sqrt(a))
