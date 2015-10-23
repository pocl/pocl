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

/* TODO:

   Clang does not recognize __builtin_sqrt* in CGBuiltin.cpp (around line 1285)
   to be a call to sqrt and cannot selectively convert it to the llvm.sqrt.*
   intrinsics with looser math flags. Therefore we have to call it as a libcall
   so it regonized it. For the double we still call the __builtin_sqrt() to
   disambiguate from the sqrt() to avoid infinite recursion. Probably the
   correct fix is to patch CGBuiltin.cpp to recognize also the call via
   __builtin_sqrt*. */

float sqrtf(float);



#ifdef cl_khr_fp16
half _CL_OVERLOADABLE sqrt(half a)
{
  return sqrtf(a);
}

half2 _CL_OVERLOADABLE sqrt(half2 a)
{
  return (half2)(sqrt(a.x), sqrt(a.y));
}

half3 _CL_OVERLOADABLE sqrt(half3 a)
{
  return (half3)(sqrt(a.x), sqrt(a.y), sqrt(a.z));
}

half4 _CL_OVERLOADABLE sqrt(half4 a)
{
  return (half4)(sqrt(a.x), sqrt(a.y), sqrt(a.z), sqrt(a.w));
}

half8 _CL_OVERLOADABLE sqrt(half8 a)
{
  return (half8)(sqrt(a.x), sqrt(a.y), sqrt(a.z), sqrt(a.w),
                 sqrt(a.s4), sqrt(a.s5), sqrt(a.s6), sqrt(a.s7));
}

half16 _CL_OVERLOADABLE sqrt(half16 a)
{
  return (half16)(sqrt(a.x), sqrt(a.y), sqrt(a.z), sqrt(a.w),
                  sqrt(a.s4), sqrt(a.s5), sqrt(a.s6), sqrt(a.s7),
                  sqrt(a.s8), sqrt(a.s9), sqrt(a.sa), sqrt(a.sb),
                  sqrt(a.sc), sqrt(a.sd), sqrt(a.se), sqrt(a.sf));
}
#endif



float _CL_OVERLOADABLE sqrt(float a)
{
  return sqrtf(a);
}

float2 _CL_OVERLOADABLE sqrt(float2 a)
{
  return (float2)(sqrtf(a.x), sqrtf(a.y));
}

float3 _CL_OVERLOADABLE sqrt(float3 a)
{
  return (float3)(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z));
}

float4 _CL_OVERLOADABLE sqrt(float4 a)
{
  return (float4)(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z), sqrtf(a.w));
}

float8 _CL_OVERLOADABLE sqrt(float8 a)
{
  return (float8)(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z), sqrtf(a.w),
		  sqrtf(a.s4), sqrtf(a.s5), sqrtf(a.s6), sqrtf(a.s7));
}

float16 _CL_OVERLOADABLE sqrt(float16 a)
{
  return (float16)(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z), sqrtf(a.w),
		   sqrtf(a.s4), sqrtf(a.s5), sqrtf(a.s6), sqrtf(a.s7),
		   sqrtf(a.s8), sqrtf(a.s9), sqrtf(a.sa), sqrtf(a.sb),
		   sqrtf(a.sc), sqrtf(a.sd), sqrtf(a.se), sqrtf(a.sf));

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
