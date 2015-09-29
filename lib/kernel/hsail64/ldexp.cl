/* OpenCL built-in library: ldexp()

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

#include "vml_constants.h"

float _CL_OVERLOADABLE ldexp(float x, int n)
  {
    // TODO: Check SLEEF 2.80 algorithm
    float r = as_float(as_int(x) + (n << (unsigned)(PROPS_FLOAT_MANTISSA_BITS)));
    int max_n = PROPS_FLOAT_MAX_EXPONENT - PROPS_FLOAT_MIN_EXPONENT;
    bool underflow = n < (-max_n);
    bool overflow = n > max_n;
    int old_exp = LSR_F(x);
    int new_exp = old_exp + n;
    // TODO: check bit patterns instead
    underflow =
      underflow || new_exp < (int)(PROPS_FLOAT_MIN_EXPONENT + PROPS_FLOAT_EXPONENT_OFFSET);
    overflow =
      overflow || new_exp > (int)(PROPS_FLOAT_MAX_EXPONENT + PROPS_FLOAT_EXPONENT_OFFSET);
    //r = ifthen(underflow, copysign(RV(R(0.0)), x), r);
    if (underflow)
      r = copysign(0.0f, x);
    //r = ifthen(overflow, copysign(RV(FP::infinity()), x), r);
    if (overflow)
      r = copysign(PROPS_FLOAT_INFINITY, x);
    //bool dont_change = (x == RV(R(0.0)) || isinf(x) || isnan(x));
    // r = ifthen(dont_change, x, r);
    if (x == 0.0f || isinf(x) || isnan(x))
      r = x;
    return r;
  }

double _CL_OVERLOADABLE ldexp(double x, int n)
  {
    // TODO: Check SLEEF 2.80 algorithm
    double r = as_double(as_long(x) + (n << (unsigned)(PROPS_DOUBLE_MANTISSA_BITS)));
    int max_n = PROPS_DOUBLE_MAX_EXPONENT - PROPS_DOUBLE_MIN_EXPONENT;
    bool underflow = n < (-max_n);
    bool overflow = n > max_n;
    int old_exp = LSR_D(x);
    int new_exp = old_exp + n;
    // TODO: check bit patterns instead
    underflow =
      underflow || new_exp < (int)(PROPS_DOUBLE_MIN_EXPONENT + PROPS_DOUBLE_EXPONENT_OFFSET);
    overflow =
      overflow || new_exp > (int)(PROPS_DOUBLE_MAX_EXPONENT + PROPS_DOUBLE_EXPONENT_OFFSET);
    //r = ifthen(underflow, copysign(RV(R(0.0)), x), r);
    if (underflow)
      r = copysign(0.0, x);
    //r = ifthen(overflow, copysign(RV(FP::infinity()), x), r);
    if (overflow)
      r = copysign(PROPS_DOUBLE_INFINITY, x);
    //bool dont_change = (x == RV(R(0.0)) || isinf(x) || isnan(x));
    // r = ifthen(dont_change, x, r);
    if (x == 0.0 || isinf(x) || isnan(x))
      r = x;
    return r;
  }

IMPLEMENT_VECWITHSCALARS(ldexp, V_VI, double, int)
IMPLEMENT_VECWITHSCALARS(ldexp, V_VI, float, int)
