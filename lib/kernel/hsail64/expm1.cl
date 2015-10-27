/* OpenCL built-in library: expm1()

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

// public domain code from http://www.johndcook.com/blog/cpp_expm1/

float _cl_builtin_expm1f(float x)
{
  if (fabs(x) < 1e-4f)
    {
      float xx = x*x;
      return fma(0.5f, xx, x);
    }
  else
    return exp(x) - 1.0f;
}

double _cl_builtin_expm1(double x)
{
  if (fabs(x) < 1e-10)  // TODO find the proper value to compare against
    {
      double xx = x*x;
      return fma(0.5, xx, x);
    }
  else
    return exp(x) - 1.0;
}

IMPLEMENT_EXPR_ALL(expm1, V_V, _cl_builtin_expm1f(a), _cl_builtin_expm1(a))
