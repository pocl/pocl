/* OpenCL built-in library: atan2()

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

MULSIGN(float, uint, PROPS_FLOAT_SIGNBIT_MASK)

MULSIGN(double, ulong, PROPS_DOUBLE_SIGNBIT_MASK)

// double
double _CL_OVERLOADABLE _cl_atan2k(double y, double x)
{
  // Algorithm taken from SLEEF 2.80

  double q = 0.0;

  if (signbit(x))
    q = -2.0;
  x = fabs(x);

  bool q0 = (y > x);
  double x0 = x;
  double y0 = y;
  x = q0 ? y0 : x0;
  y = q0 ? -x0 : y0;
  if (q0)
    q += 1.0;

  double s = y / x;
  double t = s * s;

  double u;
    u = (-1.88796008463073496563746e-05);
    u = mad(u, t, (0.000209850076645816976906797));
    u = mad(u, t, (-0.00110611831486672482563471));
    u = mad(u, t, (0.00370026744188713119232403));
    u = mad(u, t, (-0.00889896195887655491740809));
    u = mad(u, t, (0.016599329773529201970117));
    u = mad(u, t, (-0.0254517624932312641616861));
    u = mad(u, t, (0.0337852580001353069993897));
    u = mad(u, t, (-0.0407629191276836500001934));
    u = mad(u, t, (0.0466667150077840625632675));
    u = mad(u, t, (-0.0523674852303482457616113));
    u = mad(u, t, (0.0587666392926673580854313));
    u = mad(u, t, (-0.0666573579361080525984562));
    u = mad(u, t, (0.0769219538311769618355029));
    u = mad(u, t, (-0.090908995008245008229153));
    u = mad(u, t, (0.111111105648261418443745));
    u = mad(u, t, (-0.14285714266771329383765));
    u = mad(u, t, (0.199999999996591265594148));
    u = mad(u, t, (-0.333333333333311110369124));

  t = mad(u, t * s, s);
  t = mad(q, M_PI_2, t);

  return t;
}

// Note: the order of arguments is y, x, as is convention for atan2 float
float _CL_OVERLOADABLE _cl_atan2k(float y, float x)
{
  // Algorithm taken from SLEEF 2.80

  float q = 0.0f;

  if (signbit(x))
    q = -2.0;
  x = fabs(x);

  bool q0 = (y > x);
  float x0 = x;
  float y0 = y;
  x = q0 ? y0 : x0;
  y = q0 ? -x0 : y0;
  if (q0)
    q += 1.0;

  float s = y / x;
  float t = s * s;

  float u;
  u = (0.00282363896258175373077393f);
  u = mad(u, t, (-0.0159569028764963150024414f));
  u = mad(u, t, (0.0425049886107444763183594f));
  u = mad(u, t, (-0.0748900920152664184570312f));
  u = mad(u, t, (0.106347933411598205566406f));
  u = mad(u, t, (-0.142027363181114196777344f));
  u = mad(u, t, (0.199926957488059997558594f));
  u = mad(u, t, (-0.333331018686294555664062f));

  t = mad(u, t * s, s);
  t = mad(q, (float)M_PI_2, t);

  return t;
}

// Note: the order of arguments is y, x, as is convention for atan2
float _CL_OVERLOADABLE atan2(float y, float x)
{
  // Algorithm taken from SLEEF 2.80
  float r = _cl_atan2k(fabs(y), x);
  r = mulsign(r, x);
  if(isinf(x) || x == 0.0f)
    {
      if (isinf(x))
        r = (float)M_PI_2 - copysign((float)M_PI_2, x);
      else
        r = (float)M_PI_2;
    }
  if (isinf(y))
    {
      if (isinf(x))
        r = (float)M_PI_2 - copysign((float)M_PI_4, x);
      else
        r = (float)M_PI_2;
    }
  if (y == 0.0)
    {
      if (signbit(x))
        r = (float)M_PI;
      else
        r = 0.0;
    }
  if (isnan(x) || isnan(y))
    return NAN;
  else
    return mulsign(r, y);
}

  // Note: the order of arguments is y, x, as is convention for atan2
double _CL_OVERLOADABLE atan2(double y, double x)
{
  // Algorithm taken from SLEEF 2.80
  double r = _cl_atan2k(fabs(y), x);
  r = mulsign(r, x);
  if(isinf(x) || x == 0.0f)
    {
      if (isinf(x))
        r = (double)M_PI_2 - copysign((double)M_PI_2, x);
      else
        r = (double)M_PI_2;
    }
  if (isinf(y))
    {
      if (isinf(x))
        r = (double)M_PI_2 - copysign((double)M_PI_4, x);
      else
        r = (double)M_PI_2;
    }
  if (y == 0.0)
    {
      if (signbit(x))
        r = (double)M_PI;
      else
        r = 0.0;
    }
  if (isnan(x) || isnan(y))
    return NAN;
  else
    return mulsign(r, y);
}

IMPLEMENT_VECWITHSCALARS(atan2, V_VV, float, int)

IMPLEMENT_VECWITHSCALARS(atan2, V_VV, double, long)
