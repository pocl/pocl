/* OpenCL built-in library: cos()

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

float _CL_OVERLOADABLE cos(float d)
{
  // Algorithm taken from SLEEF 2.80

  float PI4_A, PI4_B, PI4_C, PI4_D;
  PI4_A = 0.78515625f;
  PI4_B = 0.00024187564849853515625f;
  PI4_C = 3.7747668102383613586e-08f;
  PI4_D = 1.2816720341285448015e-12f;

  float q = mad(2.0f, rint(mad(d, (float)M_1_PI, -0.5f)), 1.0f);
  int iq = convert_int(q);

#ifdef VML_HAVE_FP_CONTRACT
  d = mad(q, (float)(-PI4_A*2), d);
  d = mad(q, (float)(-PI4_B*2), d);
  d = mad(q, (float)(-PI4_C*2), d);
  d = mad(q, (float)(-PI4_D*2), d);
#else
  d = mad(q, (float)(-M_PI_2), d);
#endif

  float s = d * d;

  //d = ifthen(convert_bool(iq & IV(I(2))), d, -d);
  if (!(iq & 1))
    d = -d;

  float u;
  u = (float)(2.6083159809786593541503e-06f);
  u = mad(u, s, (float)(-0.0001981069071916863322258f));
  u = mad(u, s, (float)(0.00833307858556509017944336f));
  u = mad(u, s, (float)(-0.166666597127914428710938f));

  u = mad(s, u * d, d);

  if (isinf(d))
    return NAN;
  else
    return u;
}

double _CL_OVERLOADABLE cos(double d)
{
  // Algorithm taken from SLEEF 2.80

  double PI4_A, PI4_B, PI4_C, PI4_D;
  PI4_A = 0.78539816290140151978;
  PI4_B = 4.9604678871439933374e-10;
  PI4_C = 1.1258708853173288931e-18;
  PI4_D = 1.7607799325916000908e-27;

  double q = mad((double)(2.0), rint(mad(d, (double)(M_1_PI), (double)(-0.5))), (double)(1.0));
  int iq = convert_int(q);

#ifdef VML_HAVE_FP_CONTRACT
  d = mad(q, (double)(-PI4_A*2), d);
  d = mad(q, (double)(-PI4_B*2), d);
  d = mad(q, (double)(-PI4_C*2), d);
  d = mad(q, (double)(-PI4_D*2), d);
#else
  d = mad(q, (double)(-M_PI_2), d);
#endif

  double s = d * d;

  //d = ifthen(convert_bool(iq & IV(I(2))), d, -d);
  if (!(iq & 2))
    d = -d;

  double u;
  u = (double)(-7.97255955009037868891952e-18);
  u = mad(u, s, (double)(2.81009972710863200091251e-15));
  u = mad(u, s, (double)(-7.64712219118158833288484e-13));
  u = mad(u, s, (double)(1.60590430605664501629054e-10));
  u = mad(u, s, (double)(-2.50521083763502045810755e-08));
  u = mad(u, s, (double)(2.75573192239198747630416e-06));
  u = mad(u, s, (double)(-0.000198412698412696162806809));
  u = mad(u, s, (double)(0.00833333333333332974823815));
  u = mad(u, s, (double)(-0.166666666666666657414808));

  u = mad(s, u * d, d);

  //const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
  //u = ifthen(isinf(d), (double)(nan), u);

  if (isinf(d))
    return NAN;
  else
    return u;
}

IMPLEMENT_VECWITHSCALARS(cos, V_V, float, int)
IMPLEMENT_VECWITHSCALARS(cos, V_V, double, int)
