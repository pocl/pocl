/* OpenCL built-in library: tgamma()

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

#define M_SQRT_PI 1.7724538509055159
#define M_SQRT_PI_F 1.7724538509055159f

double _cl_builtin_tgamma(double g)
{
  double x = g;
  if (g < 0.5)
    x = (1.0 - g);

  double a = 0.99999999999980993;
  a += 676.5203681218851 / x;
  x += 1.0;
  a += -1259.1392167224028 / x;
  x += 1.0;
  a += 771.32342877765313 / x;
  x += 1.0;
  a += -176.61502916214059 / x;
  x += 1.0;
  a += 12.507343278686905 / x;
  x += 1.0;
  a +=  -0.13857109526572012/ x;
  x += 1.0;
  a += 9.9843695780195716e-6 / x;
  x += 1.0;
  a += 1.5056327351493116e-7 / x;

  double t = x - 0.5;
  double res = M_SQRT_PI * M_SQRT2 * pow(t, (x - 7.5)) * exp(-t) * a;

  if (g < 0.5)
    return (M_PI / (sin(M_PI*x) * res));
  else
    return ((fabs(g) > 40.0f) ? INFINITY : res);  // TODO proper range
}

float _cl_builtin_tgammaf(float g)
{
  float x = g;
  if (g < 0.5f)
    x = (1.0f - g);

  float a = 0.99999999999980993f;
  a += 676.5203681218851f / x;
  x += 1.0f;
  a += -1259.1392167224028f / x;
  x += 1.0f;
  a += 771.32342877765313f / x;
  x += 1.0f;
  a += -176.61502916214059f / x;
  x += 1.0f;
  a += 12.507343278686905f / x;
  x += 1.0f;
  a +=  -0.13857109526572012/ x;
  x += 1.0f;
  a += 9.9843695780195716e-6f / x;
  x += 1.0f;
  a += 1.5056327351493116e-7f / x;

  float t = x - 0.5f;
  float  res = M_SQRT_PI_F * M_SQRT2_F * pow(t, (x - 7.5f)) * exp(-t) * a;

  if (g < 0.5f)
    return (M_PI_F / (sin(M_PI_F*x) * res));
  else
    return ((fabs(g) > 25.0f) ? INFINITY : res);

}

IMPLEMENT_EXPR_ALL(tgamma, V_V, _cl_builtin_tgammaf(a), _cl_builtin_tgamma(a))
