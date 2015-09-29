/* OpenCL built-in library: log2()

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

float _CL_OVERLOADABLE log2(float x)
{
  // Algorithm inspired by SLEEF 2.80

  // Rescale
  int ilogb_x = ilogb(x * (float)M_SQRT2);
  x = ldexp(x, -ilogb_x);
  //VML_ASSERT(all(x >= (float)(M_SQRT1_2) && x <= (float)(M_SQRT2)));

  float y = (x - 1.0f) / (x + 1.0f);
  float y2 = y*y;

  float r;
  // float, error=7.09807175879142775648452461821e-8
  r = (float)(0.59723611417135718739797302426);
  r = mad(r, y2, (float)(0.961524413175528426101613434));
  r = mad(r, y2, (float)(2.88539097665498228703236701));
  r *= y;

  // Undo rescaling
  r += convert_float(ilogb_x);

  if (x>0)
    return r;
  else
    return NAN;
}

double _CL_OVERLOADABLE log2(double x)
{
  // Algorithm inspired by SLEEF 2.80

  // Rescale
  int ilogb_x = ilogb(x * (double)M_SQRT2);
  x = ldexp(x, -ilogb_x);
  // VML_ASSERT(all(x >= (double)(M_SQRT1_2) && x <= (double)(M_SQRT2)));

  double y = (x - 1.0) / (x + 1.0);
  double y2 = y*y;

  double r;
#ifdef VML_HAVE_FP_CONTRACT
  // double, error=1.48294180185938512675770096324e-16
  r = (double)(0.243683403415639178527756320773);
  r = mad(r, y2, (double)(0.26136626803870009948502658));
  r = mad(r, y2, (double)(0.320619429891299265439389));
  r = mad(r, y2, (double)(0.4121983452028499242926));
  r = mad(r, y2, (double)(0.577078017761894161436));
  r = mad(r, y2, (double)(0.96179669392233355927));
  r = mad(r, y2, (double)(2.8853900817779295236));
#else
  // double, error=2.1410114030383689267772704676e-14
  r = (double)(0.283751646449323373643963474845);
  r = mad(r, y2, (double)(0.31983138095551191299118812));
  r = mad(r, y2, (double)(0.412211603844146279666022));
  r = mad(r, y2, (double)(0.5770779098948940070516));
  r = mad(r, y2, (double)(0.961796694295973716912));
  r = mad(r, y2, (double)(2.885390081777562819196));
#endif
  r *= y;

  // Undo rescaling
  r += convert_double(ilogb_x);

  if (x>0)
    return r;
  else
    return NAN;
}

IMPLEMENT_VECWITHSCALARS(log2, V_V, float, int)

IMPLEMENT_VECWITHSCALARS(log2, V_V, double, int)
