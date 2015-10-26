// -*-C++-*-

#ifndef MATHFUNCS_SIN_H
#define MATHFUNCS_SIN_H

#include "mathfuncs_base.h"

#include <cmath>

namespace vecmathlib {

template <typename realvec_t>
realvec_t mathfuncs<realvec_t>::vml_sin(realvec_t d) {
  // Algorithm taken from SLEEF 2.80

  real_t PI4_A, PI4_B, PI4_C, PI4_D;
  switch (sizeof(real_t)) {
  default:
    __builtin_unreachable();
  case sizeof(float):
    PI4_A = 0.78515625f;
    PI4_B = 0.00024187564849853515625f;
    PI4_C = 3.7747668102383613586e-08f;
    PI4_D = 1.2816720341285448015e-12f;
    break;
  case sizeof(double):
    PI4_A = 0.78539816290140151978;
    PI4_B = 4.9604678871439933374e-10;
    PI4_C = 1.1258708853173288931e-18;
    PI4_D = 1.7607799325916000908e-27;
    break;
  }

  realvec_t q = rint(d * RV(M_1_PI));
  intvec_t iq = convert_int(q);

#ifdef VML_HAVE_FP_CONTRACT
  d = mad(q, RV(-PI4_A * 4), d);
  d = mad(q, RV(-PI4_B * 4), d);
  d = mad(q, RV(-PI4_C * 4), d);
  d = mad(q, RV(-PI4_D * 4), d);
#else
  d = mad(q, RV(-M_PI), d);
#endif

  realvec_t s = d * d;

  d = ifthen(convert_bool(iq & IV(I(1))), -d, d);

  realvec_t u;
  switch (sizeof(real_t)) {
  default:
    __builtin_unreachable();
  case sizeof(float):
    u = RV(2.6083159809786593541503e-06f);
    u = mad(u, s, RV(-0.0001981069071916863322258f));
    u = mad(u, s, RV(0.00833307858556509017944336f));
    u = mad(u, s, RV(-0.166666597127914428710938f));
    break;
  case sizeof(double):
    u = RV(-7.97255955009037868891952e-18);
    u = mad(u, s, RV(2.81009972710863200091251e-15));
    u = mad(u, s, RV(-7.64712219118158833288484e-13));
    u = mad(u, s, RV(1.60590430605664501629054e-10));
    u = mad(u, s, RV(-2.50521083763502045810755e-08));
    u = mad(u, s, RV(2.75573192239198747630416e-06));
    u = mad(u, s, RV(-0.000198412698412696162806809));
    u = mad(u, s, RV(0.00833333333333332974823815));
    u = mad(u, s, RV(-0.166666666666666657414808));
    break;
  }

  u = mad(s, u * d, d);

  const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
  u = ifthen(isinf(d), RV(nan), u);

  return u;
}

template <typename realvec_t>
realvec_t mathfuncs<realvec_t>::vml_cos(realvec_t d) {
  // Algorithm taken from SLEEF 2.80

  real_t PI4_A, PI4_B, PI4_C, PI4_D;
  switch (sizeof(real_t)) {
  default:
    __builtin_unreachable();
  case sizeof(float):
    PI4_A = 0.78515625f;
    PI4_B = 0.00024187564849853515625f;
    PI4_C = 3.7747668102383613586e-08f;
    PI4_D = 1.2816720341285448015e-12f;
    break;
  case sizeof(double):
    PI4_A = 0.78539816290140151978;
    PI4_B = 4.9604678871439933374e-10;
    PI4_C = 1.1258708853173288931e-18;
    PI4_D = 1.7607799325916000908e-27;
    break;
  }

  realvec_t q = mad(RV(2.0), rint(mad(d, RV(M_1_PI), RV(-0.5))), RV(1.0));
  intvec_t iq = convert_int(q);

#ifdef VML_HAVE_FP_CONTRACT
  d = mad(q, RV(-PI4_A * 2), d);
  d = mad(q, RV(-PI4_B * 2), d);
  d = mad(q, RV(-PI4_C * 2), d);
  d = mad(q, RV(-PI4_D * 2), d);
#else
  d = mad(q, RV(-M_PI_2), d);
#endif

  realvec_t s = d * d;

  d = ifthen(convert_bool(iq & IV(I(2))), d, -d);

  realvec_t u;
  switch (sizeof(real_t)) {
  default:
    __builtin_unreachable();
  case sizeof(float):
    u = RV(2.6083159809786593541503e-06f);
    u = mad(u, s, RV(-0.0001981069071916863322258f));
    u = mad(u, s, RV(0.00833307858556509017944336f));
    u = mad(u, s, RV(-0.166666597127914428710938f));
    break;
  case sizeof(double):
    u = RV(-7.97255955009037868891952e-18);
    u = mad(u, s, RV(2.81009972710863200091251e-15));
    u = mad(u, s, RV(-7.64712219118158833288484e-13));
    u = mad(u, s, RV(1.60590430605664501629054e-10));
    u = mad(u, s, RV(-2.50521083763502045810755e-08));
    u = mad(u, s, RV(2.75573192239198747630416e-06));
    u = mad(u, s, RV(-0.000198412698412696162806809));
    u = mad(u, s, RV(0.00833333333333332974823815));
    u = mad(u, s, RV(-0.166666666666666657414808));
    break;
  }

  u = mad(s, u * d, d);

  const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
  u = ifthen(isinf(d), RV(nan), u);

  return u;
}

template <typename realvec_t>
realvec_t mathfuncs<realvec_t>::vml_tan(realvec_t d) {
  // Algorithm taken from SLEEF 2.80

  real_t PI4_A, PI4_B, PI4_C, PI4_D;
  switch (sizeof(real_t)) {
  default:
    __builtin_unreachable();
  case sizeof(float):
    PI4_A = 0.78515625f;
    PI4_B = 0.00024187564849853515625f;
    PI4_C = 3.7747668102383613586e-08f;
    PI4_D = 1.2816720341285448015e-12f;
    break;
  case sizeof(double):
    PI4_A = 0.78539816290140151978;
    PI4_B = 4.9604678871439933374e-10;
    PI4_C = 1.1258708853173288931e-18;
    PI4_D = 1.7607799325916000908e-27;
    break;
  }

  realvec_t q = rint(d * RV(2 * M_1_PI));
  intvec_t iq = convert_int(q);

  realvec_t x = d;

#ifdef VML_HAVE_FP_CONTRACT
  x = mad(q, RV(-PI4_A * 2), x);
  x = mad(q, RV(-PI4_B * 2), x);
  x = mad(q, RV(-PI4_C * 2), x);
  x = mad(q, RV(-PI4_D * 2), x);
#else
  x = mad(q, RV(-M_PI_2), x);
#endif

  realvec_t s = x * x;

  x = ifthen(convert_bool(iq & IV(I(1))), -x, x);

  realvec_t u;
  switch (sizeof(real_t)) {
  default:
    __builtin_unreachable();
  case sizeof(float):
    u = RV(0.00927245803177356719970703f);
    u = mad(u, s, RV(0.00331984995864331722259521f));
    u = mad(u, s, RV(0.0242998078465461730957031f));
    u = mad(u, s, RV(0.0534495301544666290283203f));
    u = mad(u, s, RV(0.133383005857467651367188f));
    u = mad(u, s, RV(0.333331853151321411132812f));
    break;
  case sizeof(double):
    u = RV(1.01419718511083373224408e-05);
    u = mad(u, s, RV(-2.59519791585924697698614e-05));
    u = mad(u, s, RV(5.23388081915899855325186e-05));
    u = mad(u, s, RV(-3.05033014433946488225616e-05));
    u = mad(u, s, RV(7.14707504084242744267497e-05));
    u = mad(u, s, RV(8.09674518280159187045078e-05));
    u = mad(u, s, RV(0.000244884931879331847054404));
    u = mad(u, s, RV(0.000588505168743587154904506));
    u = mad(u, s, RV(0.00145612788922812427978848));
    u = mad(u, s, RV(0.00359208743836906619142924));
    u = mad(u, s, RV(0.00886323944362401618113356));
    u = mad(u, s, RV(0.0218694882853846389592078));
    u = mad(u, s, RV(0.0539682539781298417636002));
    u = mad(u, s, RV(0.133333333333125941821962));
    u = mad(u, s, RV(0.333333333333334980164153));
    break;
  }

  u = mad(s, u * x, x);

  u = ifthen(convert_bool(iq & IV(I(1))), rcp(u), u);

  const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
  u = ifthen(isinf(d), RV(nan), u);

  return u;
}

}; // namespace vecmathlib

#endif // #ifndef MATHFUNCS_SIN_H
