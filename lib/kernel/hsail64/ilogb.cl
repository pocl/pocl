/* OpenCL built-in library: ilogb()

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

int _CL_OVERLOADABLE ilogb(float x)
{
  // TODO: Check SLEEF 2.80 algorithm
  //intvec_t e = lsr(as_int(x) & IV(FP::exponent_mask), FP::mantissa_bits);
  int e = LSR_F(x);
  //intvec_t r = e - IV(FP::exponent_offset);
  int r = e - (int)(PROPS_FLOAT_EXPONENT_OFFSET);
  //r = ifthen((e), r, IV(std::numeric_limits<int_t>::min()));
  if (!e)
    r = INT_MIN;
#if defined VML_HAVE_INF
  //r = ifthen(isinf(x), IV(std::numeric_limits<int_t>::max()), r);
  if (isinf(x))
    r = INT_MAX;
#endif
#if defined VML_HAVE_NAN
  // r = ifthen(isnan(x), IV(std::numeric_limits<int_t>::min()), r);
  if (isnan(x))
    r = INT_MIN;
#endif
  return r;
}

int _CL_OVERLOADABLE ilogb(double x)
{
  // TODO: Check SLEEF 2.80 algorithm
  //intvec_t e = lsr(as_int(x) & IV(FP::exponent_mask), FP::mantissa_bits);
  int e = LSR_D(x);
  //intvec_t r = e - IV(FP::exponent_offset);
  int r = e - (int)(PROPS_DOUBLE_EXPONENT_OFFSET);
  //r = ifthen((e), r, IV(std::numeric_limits<int_t>::min()));
  if (!e)
    r = INT_MIN;
#if defined VML_HAVE_INF
  //r = ifthen(isinf(x), IV(std::numeric_limits<int_t>::max()), r);
  if (isinf(x))
    r = INT_MAX;
#endif
#if defined VML_HAVE_NAN
  // r = ifthen(isnan(x), IV(std::numeric_limits<int_t>::min()), r);
  if (isnan(x))
    r = INT_MIN;
#endif
  return r;
}

IMPLEMENT_VECWITHSCALARS(ilogb, I_V, float, int)
IMPLEMENT_VECWITHSCALARS(ilogb, I_V, double, int)
