// -*-C++-*-

#ifndef MATHFUNCS_ASIN_H
#define MATHFUNCS_ASIN_H

#include "mathfuncs_base.h"

#include <cmath>

namespace vecmathlib {

namespace {

template <typename realvec_t> realvec_t mulsign(realvec_t x, realvec_t y) {
  typedef typename realvec_t::real_t real_t;
  typedef typename realvec_t::intvec_t intvec_t;
  typedef intvec_t IV;
  typedef floatprops<real_t> FP;

  intvec_t value = as_int(x);
  intvec_t sign = as_int(y) & IV(FP::signbit_mask);
  return as_float(value ^ sign);
}

// Note: the order of arguments is y, x, as is convention for atan2
template <typename realvec_t> realvec_t atan2k(realvec_t y, realvec_t x) {
  // Algorithm taken from SLEEF 2.80

  typedef typename realvec_t::real_t real_t;
  typedef typename realvec_t::boolvec_t boolvec_t;
  typedef realvec_t RV;

  realvec_t q = RV(0.0);

  q = ifthen(signbit(x), RV(-2.0), q);
  x = fabs(x);

  boolvec_t cond = y > x;
  realvec_t x0 = x;
  realvec_t y0 = y;
  x = ifthen(cond, y0, x0);
  y = ifthen(cond, -x0, y0);
  q += ifthen(cond, RV(1.0), RV(0.0));

  realvec_t s = y / x;
  realvec_t t = s * s;

  realvec_t u;
  switch (sizeof(real_t)) {
  default:
    __builtin_unreachable();
  case sizeof(float):
    u = RV(0.00282363896258175373077393f);
    u = mad(u, t, RV(-0.0159569028764963150024414f));
    u = mad(u, t, RV(0.0425049886107444763183594f));
    u = mad(u, t, RV(-0.0748900920152664184570312f));
    u = mad(u, t, RV(0.106347933411598205566406f));
    u = mad(u, t, RV(-0.142027363181114196777344f));
    u = mad(u, t, RV(0.199926957488059997558594f));
    u = mad(u, t, RV(-0.333331018686294555664062f));
    break;
  case sizeof(double):
    u = RV(-1.88796008463073496563746e-05);
    u = mad(u, t, RV(0.000209850076645816976906797));
    u = mad(u, t, RV(-0.00110611831486672482563471));
    u = mad(u, t, RV(0.00370026744188713119232403));
    u = mad(u, t, RV(-0.00889896195887655491740809));
    u = mad(u, t, RV(0.016599329773529201970117));
    u = mad(u, t, RV(-0.0254517624932312641616861));
    u = mad(u, t, RV(0.0337852580001353069993897));
    u = mad(u, t, RV(-0.0407629191276836500001934));
    u = mad(u, t, RV(0.0466667150077840625632675));
    u = mad(u, t, RV(-0.0523674852303482457616113));
    u = mad(u, t, RV(0.0587666392926673580854313));
    u = mad(u, t, RV(-0.0666573579361080525984562));
    u = mad(u, t, RV(0.0769219538311769618355029));
    u = mad(u, t, RV(-0.090908995008245008229153));
    u = mad(u, t, RV(0.111111105648261418443745));
    u = mad(u, t, RV(-0.14285714266771329383765));
    u = mad(u, t, RV(0.199999999996591265594148));
    u = mad(u, t, RV(-0.333333333333311110369124));
    break;
  }

  t = mad(u, t * s, s);
  t = mad(q, RV(M_PI_2), t);

  return t;
}
}

template <typename realvec_t>
realvec_t mathfuncs<realvec_t>::vml_asin(realvec_t d) {
  // Algorithm taken from SLEEF 2.80
  return mulsign(atan2k(fabs(d), sqrt((RV(1.0) + d) * (RV(1.0) - d))), d);
}

template <typename realvec_t>
realvec_t mathfuncs<realvec_t>::vml_acos(realvec_t d) {
  // negative zero has the same (positive) result as positive zero
  d = ifthen(d == RV(-0.0), RV(0.0), d);
  // Algorithm taken from SLEEF 2.80
  return (mulsign(atan2k(sqrt((RV(1.0) + d) * (RV(1.0) - d)), fabs(d)), d) +
          ifthen(d < RV(0.0), RV(M_PI), RV(0.0)));
}

template <typename realvec_t>
realvec_t mathfuncs<realvec_t>::vml_atan(realvec_t s) {
  // Algorithm taken from SLEEF 2.80

  realvec_t q1 = s;
  s = fabs(s);

  boolvec_t q0 = s > RV(1.0);
  s = ifthen(q0, rcp(s), s);

  realvec_t t = s * s;

  realvec_t u;
  switch (sizeof(real_t)) {
  default:
    __builtin_unreachable();
  case sizeof(float):
    u = RV(0.00282363896258175373077393f);
    u = mad(u, t, RV(-0.0159569028764963150024414f));
    u = mad(u, t, RV(0.0425049886107444763183594f));
    u = mad(u, t, RV(-0.0748900920152664184570312f));
    u = mad(u, t, RV(0.106347933411598205566406f));
    u = mad(u, t, RV(-0.142027363181114196777344f));
    u = mad(u, t, RV(0.199926957488059997558594f));
    u = mad(u, t, RV(-0.333331018686294555664062f));
    break;
  case sizeof(double):
    u = RV(-1.88796008463073496563746e-05);
    u = mad(u, t, RV(0.000209850076645816976906797));
    u = mad(u, t, RV(-0.00110611831486672482563471));
    u = mad(u, t, RV(0.00370026744188713119232403));
    u = mad(u, t, RV(-0.00889896195887655491740809));
    u = mad(u, t, RV(0.016599329773529201970117));
    u = mad(u, t, RV(-0.0254517624932312641616861));
    u = mad(u, t, RV(0.0337852580001353069993897));
    u = mad(u, t, RV(-0.0407629191276836500001934));
    u = mad(u, t, RV(0.0466667150077840625632675));
    u = mad(u, t, RV(-0.0523674852303482457616113));
    u = mad(u, t, RV(0.0587666392926673580854313));
    u = mad(u, t, RV(-0.0666573579361080525984562));
    u = mad(u, t, RV(0.0769219538311769618355029));
    u = mad(u, t, RV(-0.090908995008245008229153));
    u = mad(u, t, RV(0.111111105648261418443745));
    u = mad(u, t, RV(-0.14285714266771329383765));
    u = mad(u, t, RV(0.199999999996591265594148));
    u = mad(u, t, RV(-0.333333333333311110369124));
    break;
  }

  t = s + s * (t * u);

  t = ifthen(q0, RV(M_PI_2) - t, t);
  t = copysign(t, q1);

  return t;
}

// Note: the order of arguments is y, x, as is convention for atan2
template <typename realvec_t>
realvec_t mathfuncs<realvec_t>::vml_atan2(realvec_t y, realvec_t x) {
  // Algorithm taken from SLEEF 2.80

  realvec_t r = atan2k(fabs(y), x);

  r = mulsign(r, x);

  r = ifthen(isinf(x) || x == RV(0.0),
             ifthen(isinf(x), RV(M_PI_2) - copysign(RV(M_PI_2), x), RV(M_PI_2)),
             r);

  r = ifthen(isinf(y),
             ifthen(isinf(x), RV(M_PI_2) - copysign(RV(M_PI_4), x), RV(M_PI_2)),
             r);

  r = ifthen(y == RV(0.0), ifthen(signbit(x), RV(M_PI), RV(0.0)), r);

  const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
  return ifthen(isnan(x) || isnan(y), RV(nan), mulsign(r, y));
}

}; // namespace vecmathlib

#endif // #ifndef MATHFUNCS_ASIN_H
