/* OpenCL built-in library: fract()

   Copyright (c) 2015 Michal Babej / Tampere University of Technology

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

#include "hsail_templates.h"


float _CL_OVERLOADABLE fract(float a, __global float *b)
{
  *b = floor(a);
  return fmin(a - floor(a), 0x1.fffffep-1f);
}

double _CL_OVERLOADABLE fract(double a, __global double *b)
{
  *b = floor(a);
  return fmin(a - floor(a), 0x1.fffffffffffffp-1);
}


float _CL_OVERLOADABLE fract(float a, __local float *b)
{
  *b = floor(a);
  return fmin(a - floor(a), 0x1.fffffep-1f);
}

double _CL_OVERLOADABLE fract(double a, __local double *b)
{
  *b = floor(a);
  return fmin(a - floor(a), 0x1.fffffffffffffp-1);
}


float _CL_OVERLOADABLE fract(float a, __private float *b)
{
  *b = floor(a);
  return fmin(a - floor(a), 0x1.fffffep-1f);
}

double _CL_OVERLOADABLE fract(double a, __private double *b)
{
  *b = floor(a);
  return fmin(a - floor(a), 0x1.fffffffffffffp-1);
}

IMPLEMENT_EXPR_V_VP_ALL(fract, float, float)

IMPLEMENT_EXPR_V_VP_ALL(fract, double, double)
