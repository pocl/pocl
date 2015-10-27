/* OpenCL built-in library: log1p()

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

// DEFINE_EXPR_V_V(log1p, (log((vtype)(1.0) + a)))

// public domain from http://www.johndcook.com/blog/cpp_log_one_plus_x/

float _cl_builtin_log1pf(float x) {
  if (x < -1.0f)
    return NAN;

  if (fabs(x) > 1e-4f)  // TODO find the proper value here
    return log(1.0f + x);
  else
    {
      float xx = x*x;
      return fma(-0.5f, xx, x);
    }
}

double _cl_builtin_log1p(double x) {
  if (x < -1.0)
    return NAN;

  if (fabs(x) > 1e-8) // TODO find the proper value here
    return log(1.0 + x);
  else
    {
      double xx = x*x;
      return fma(-0.5, xx, x);
    }
}


IMPLEMENT_EXPR_ALL(log1p, V_V, _cl_builtin_log1pf(a), _cl_builtin_log1p(a))
