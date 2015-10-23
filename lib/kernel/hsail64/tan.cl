/* OpenCL built-in library: tan()

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

  float _CL_OVERLOADABLE tan(float d)
  {
    // Algorithm taken from SLEEF 2.80

    float PI4_A, PI4_B, PI4_C, PI4_D;
      PI4_A = 0.78515625f;
      PI4_B = 0.00024187564849853515625f;
      PI4_C = 3.7747668102383613586e-08f;
      PI4_D = 1.2816720341285448015e-12f;

    float q = rint(d * (float)(2 * M_1_PI));
    int ib = (convert_int(q) & 1);

    float x = d;

#ifdef VML_HAVE_FP_CONTRACT
    x = mad(q, (float)(-PI4_A*2), x);
    x = mad(q, (float)(-PI4_B*2), x);
    x = mad(q, (float)(-PI4_C*2), x);
    x = mad(q, (float)(-PI4_D*2), x);
#else
    x = mad(q, (float)(-M_PI_2), x);
#endif

    float s = x * x;

    //x = ifthen(convert_bool(iq & IV(I(1))), -x, x);
    if (ib)
      x = -x;

    float u;
      u = (float)(0.00927245803177356719970703f);
      u = mad(u, s, (float)(0.00331984995864331722259521f));
      u = mad(u, s, (float)(0.0242998078465461730957031f));
      u = mad(u, s, (float)(0.0534495301544666290283203f));
      u = mad(u, s, (float)(0.133383005857467651367188f));
      u = mad(u, s, (float)(0.333331853151321411132812f));

    u = mad(s, u * x, x);

    //u = ifthen(convert_bool(iq & IV(I(1))), rcp(u), u);
    if (ib)
      u = 1.0f / u;

    //const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
    //u = ifthen(isinf(d), (float)(nan), u);
    if (isinf(d))
      return NAN;
    else
      return u;
  }

  double _CL_OVERLOADABLE tan(double d)
  {
    // Algorithm taken from SLEEF 2.80

    double PI4_A, PI4_B, PI4_C, PI4_D;
      PI4_A = 0.78539816290140151978;
      PI4_B = 4.9604678871439933374e-10;
      PI4_C = 1.1258708853173288931e-18;
      PI4_D = 1.7607799325916000908e-27;

    double q = rint(d * (double)(2 * M_1_PI));
    //int iq = convert_int(q);
    int ib = (convert_int(q) & 1);

    double x = d;

#ifdef VML_HAVE_FP_CONTRACT
    x = mad(q, (double)(-PI4_A*2), x);
    x = mad(q, (double)(-PI4_B*2), x);
    x = mad(q, (double)(-PI4_C*2), x);
    x = mad(q, (double)(-PI4_D*2), x);
#else
    x = mad(q, (double)(-M_PI_2), x);
#endif

    double s = x * x;

    //x = ifthen(convert_bool(iq & IV(I(1))), -x, x);
    if (ib)
      x = -x;

    double u;
      u = (double)(1.01419718511083373224408e-05);
      u = mad(u, s, (double)(-2.59519791585924697698614e-05));
      u = mad(u, s, (double)(5.23388081915899855325186e-05));
      u = mad(u, s, (double)(-3.05033014433946488225616e-05));
      u = mad(u, s, (double)(7.14707504084242744267497e-05));
      u = mad(u, s, (double)(8.09674518280159187045078e-05));
      u = mad(u, s, (double)(0.000244884931879331847054404));
      u = mad(u, s, (double)(0.000588505168743587154904506));
      u = mad(u, s, (double)(0.00145612788922812427978848));
      u = mad(u, s, (double)(0.00359208743836906619142924));
      u = mad(u, s, (double)(0.00886323944362401618113356));
      u = mad(u, s, (double)(0.0218694882853846389592078));
      u = mad(u, s, (double)(0.0539682539781298417636002));
      u = mad(u, s, (double)(0.133333333333125941821962));
      u = mad(u, s, (double)(0.333333333333334980164153));

    u = mad(s, u * x, x);

    //u = ifthen(convert_bool(iq & IV(I(1))), rcp(u), u);
    if (ib)
      u = 1.0 / u;

    //const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
    //u = ifthen(isinf(d), (double)(nan), u);
    if (isinf(d))
      return NAN;
    else
      return u;
  }

IMPLEMENT_VECWITHSCALARS(tan, V_V, float, int)
IMPLEMENT_VECWITHSCALARS(tan, V_V, double, int)
