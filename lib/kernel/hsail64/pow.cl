/* OpenCL built-in library: pow()

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

float _CL_OVERLOADABLE pow(float x, float y)
  {
    // Handle zero
    if (x == 0.0f)
      return 0.0f;

    float r = exp(log(fabs(x)) * y);

    // The result is negative if x<0 and if y is integer and odd
    float mod_y = fabs(y) - (2.0f * floor(0.5f * fabs(y)));
    float sign = copysign(mod_y, x) + 0.5f;
    r = copysign(r, sign);

    // Handle zero
    // r = ifthen(is_zero, (float)(0.0), r);

    return r;
  }


double _CL_OVERLOADABLE pow(double x, double y)
  {
    // Handle zero
    if (x == 0.0)
      return 0.0;

    double r = exp(log(fabs(x)) * y);

    // The result is negative if x<0 and if y is integer and odd
    double mod_y = fabs(y) - (2.0 * floor(0.5 * fabs(y)));
    double sign = copysign(mod_y, x) + 0.5;
    r = copysign(r, sign);

    // Handle zero
    // r = ifthen(is_zero, (double)(0.0), r);

    return r;
  }

IMPLEMENT_VECWITHSCALARS(pow, V_VV, float, int)
IMPLEMENT_VECWITHSCALARS(pow, V_VV, double, int)
