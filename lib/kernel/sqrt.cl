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

float _CL_OVERLOADABLE sqrt(float a)
{
  return sqrtf(a);
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
  return (float16)(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z), sqrtf(a.w),
		   sqrtf(a.s4), sqrtf(a.s5), sqrtf(a.s6), sqrtf(a.s7),
		   sqrtf(a.s8), sqrtf(a.s9), sqrtf(a.sa), sqrtf(a.sb),
		   sqrtf(a.sc), sqrtf(a.sd), sqrtf(a.se), sqrtf(a.sf));

}

DEFINE_EXPR_F_F(half_sqrt, sqrt(a))
DEFINE_EXPR_F_F(native_sqrt, sqrt(a))
