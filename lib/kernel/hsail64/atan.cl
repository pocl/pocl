/* OpenCL built-in library: atan()

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

float _CL_OVERLOADABLE atan(float s)
{
  // Algorithm taken from SLEEF 2.80

  float q1 = s;
  s = fabs(s);

  bool q0 = (s > (1.0f));
  if (q0)
    s = 1.0f / s;

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

  t = s + s * (t * u);

  if (q0)
    t = (float)M_PI_2 - t;

  return copysign(t, q1);
}

double _CL_OVERLOADABLE atan(double s)
{
  // Algorithm taken from SLEEF 2.80

  double q1 = s;
  s = fabs(s);

  bool q0 = (s > 1.0f);
  if (q0)
    s = 1.0 / s;

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

  t = s + s * (t * u);

  if (q0)
    t = M_PI_2 - t;

  return copysign(t, q1);
}

IMPLEMENT_VECWITHSCALARS(atan, V_V, float, int)

IMPLEMENT_VECWITHSCALARS(atan, V_V, double, long)
