/* OpenCL built-in library: exp2()

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

float _CL_OVERLOADABLE exp2(float x)
{
  // TODO: Check SLEEF 2.80 algorithm
  // (in particular the improved-precision truncation)

  // Rescale
  float x0 = x;

  // Round by adding, then subtracting again a large number
  // Add a large number to move the mantissa bits to the right
  int large = ((int)1 << PROPS_FLOAT_MANTISSA_BITS) + PROPS_FLOAT_EXPONENT_OFFSET;
  float tmp = x + (float)large;
  // tmp.barrier();
  __asm__ ("");

  float round_x = tmp - (float)large;
  x -= round_x;

  //VML_ASSERT(all(x >= (float)(-0.5) && x <= (float)(0.5)));

  // Polynomial expansion
  float r;
#ifdef VML_HAVE_FP_CONTRACT
  // float, error=4.55549108005200277750378992345e-9
  r = (float)(0.000154653240842602623787395880898);
  r = mad(r, x, (float)(0.00133952915439234389712105060319));
  r = mad(r, x, (float)(0.0096180399118156827664944870552));
  r = mad(r, x, (float)(0.055503406540531310853149866446));
  r = mad(r, x, (float)(0.240226511015459465468737123346));
  r = mad(r, x, (float)(0.69314720007380208630542805293));
  r = mad(r, x, (float)(0.99999999997182023878745628977));
#else
  // float, error=1.62772721960621336664735896836e-7
  r = (float)(0.00133952915439234389712105060319);
  r = mad(r, x, (float)(0.009670773148229417605024318985));
  r = mad(r, x, (float)(0.055503406540531310853149866446));
  r = mad(r, x, (float)(0.240222115700585316818177639177));
  r = mad(r, x, (float)(0.69314720007380208630542805293));
  r = mad(r, x, (float)(1.00000005230745711373079206024));
#endif

  // Use direct integer manipulation
  // Extract integer as lowest mantissa bits (highest bits still
  // contain offset, exponent, and sign)
  int itmp = as_int(tmp);
  // Construct scale factor by setting exponent (this shifts out the
  // highest bits)
  float scale = as_float(itmp << PROPS_FLOAT_MANTISSA_BITS);
  r *= scale;

  if (x0 < (float)PROPS_FLOAT_MIN_EXPONENT)
    return 0.0f;
  else
    return r;
}

double _CL_OVERLOADABLE exp2(double x)
{
  // TODO: Check SLEEF 2.80 algorithm
  // (in particular the improved-precision truncation)

  // Rescale
  double x0 = x;

  // Round by adding, then subtracting again a large number
  // Add a large number to move the mantissa bits to the right
  long large = ((long)1 << PROPS_DOUBLE_MANTISSA_BITS) + PROPS_DOUBLE_EXPONENT_OFFSET;
  double tmp = x + (double)large;
  // tmp.barrier();
  __asm__ ("");

  double round_x = tmp - (double)large;
  x -= round_x;

  // Polynomial expansion
  double r;
#ifdef VML_HAVE_FP_CONTRACT
  // double, error=9.32016781355638010975628074746e-18
  r = (double)(4.45623165388261696886670014471e-10);
  r = mad(r, x, (double)(7.0733589360775271430968224806e-9));
  r = mad(r, x, (double)(1.01780540270960163558119510246e-7));
  r = mad(r, x, (double)(1.3215437348041505269462510712e-6));
  r = mad(r, x, (double)(0.000015252733849766201174247690629));
  r = mad(r, x, (double)(0.000154035304541242555115696403795));
  r = mad(r, x, (double)(0.00133335581463968601407096905671));
  r = mad(r, x, (double)(0.0096181291075949686712855561931));
  r = mad(r, x, (double)(0.055504108664821672870565883052));
  r = mad(r, x, (double)(0.240226506959101382690753994082));
  r = mad(r, x, (double)(0.69314718055994530864272481773));
  r = mad(r, x, (double)(0.9999999999999999978508676375));
#else
  // double, error=3.74939899823302048807873981077e-14
  r = (double)(1.02072375599725694063203809188e-7);
  r = mad(r, x, (double)(1.32573274434801314145133004073e-6));
  r = mad(r, x, (double)(0.0000152526647170731944840736190013));
  r = mad(r, x, (double)(0.000154034441925859828261898614555));
  r = mad(r, x, (double)(0.00133335582175770747495287552557));
  r = mad(r, x, (double)(0.0096181291794939392517233403183));
  r = mad(r, x, (double)(0.055504108664525029438908798685));
  r = mad(r, x, (double)(0.240226506957026959772247598695));
  r = mad(r, x, (double)(0.6931471805599487321347668143));
  r = mad(r, x, (double)(1.00000000000000942892870993489));
#endif

  // Use direct integer manipulation
  // Extract integer as lowest mantissa bits (highest bits still
  // contain offset, exponent, and sign)
  long itmp = as_long(tmp);
  // Construct scale factor by setting exponent (this shifts out the
  // highest bits)
  double scale = as_double(itmp << PROPS_DOUBLE_MANTISSA_BITS);
  r *= scale;

  if (x0 < (float)PROPS_DOUBLE_MIN_EXPONENT)
    return 0.0;
  else
    return r;
}

IMPLEMENT_VECWITHSCALARS(exp2, V_V, float, int)

IMPLEMENT_VECWITHSCALARS(exp2, V_V, double, int)
