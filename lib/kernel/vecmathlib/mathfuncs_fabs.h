// -*-C++-*-

#ifndef MATHFUNCS_FABS_H
#define MATHFUNCS_FABS_H

#include "mathfuncs_base.h"

#include <cmath>

namespace vecmathlib {

template <typename realvec_t>
realvec_t mathfuncs<realvec_t>::vml_copysign(realvec_t x, realvec_t y) {
  intvec_t value = as_int(x) & IV(U(~FP::signbit_mask));
  intvec_t sign = as_int(y) & IV(FP::signbit_mask);
  return as_float(sign | value);
}

template <typename realvec_t>
realvec_t mathfuncs<realvec_t>::vml_fabs(realvec_t x) {
  return as_float(as_int(x) & IV(U(~FP::signbit_mask)));
}

template <typename realvec_t>
realvec_t mathfuncs<realvec_t>::vml_fdim(realvec_t x, realvec_t y) {
  // return ifthen(x > y, x - y, RV(0.0));
  realvec_t res = fmax(x - y, RV(0.0));
#if defined VML_HAVE_NAN
  res = ifthen(isnan(x), RV(NAN), res);
  res = ifthen(isnan(y), RV(NAN), res);
#endif
  return res;
}

template <typename realvec_t>
realvec_t mathfuncs<realvec_t>::vml_fma(realvec_t x, realvec_t y, realvec_t z) {
  return x * y + z;
}

template <typename realvec_t>
realvec_t mathfuncs<realvec_t>::vml_fmax(realvec_t x, realvec_t y) {
  return ifthen(x < y, y, x);
}

template <typename realvec_t>
realvec_t mathfuncs<realvec_t>::vml_fmin(realvec_t x, realvec_t y) {
  return ifthen(y < x, y, x);
}

template <typename realvec_t>
realvec_t mathfuncs<realvec_t>::vml_frexp(realvec_t x,
                                          typename realvec_t::intvec_t *irp) {
  intvec_t e = lsr(as_int(x) & IV(FP::exponent_mask), FP::mantissa_bits);
  intvec_t ir = e - IV(FP::exponent_offset - 1);
  ir = ifthen(convert_bool(e), ir, IV(std::numeric_limits<int_t>::min()));
#if defined VML_HAVE_INF
  ir = ifthen(isinf(x), IV(std::numeric_limits<int_t>::max()), ir);
#endif
#if defined VML_HAVE_NAN
  ir = ifthen(isnan(x), IV(std::numeric_limits<int_t>::min()), ir);
#endif
  realvec_t r =
      as_float((as_int(x) & IV(FP::signbit_mask | FP::mantissa_mask)) |
               IV(FP::as_int(R(0.5)) & FP::exponent_mask));
  boolvec_t iszero = x == RV(0.0);
  ir = ifthen(iszero, IV(I(0)), ir);
  r = ifthen(iszero, copysign(RV(R(0.0)), r), r);
  *irp = ir;
  return r;
}

template <typename realvec_t>
typename realvec_t::intvec_t mathfuncs<realvec_t>::vml_ilogb(realvec_t x) {
  // TODO: Check SLEEF 2.80 algorithm
  intvec_t e = lsr(as_int(x) & IV(FP::exponent_mask), FP::mantissa_bits);
  intvec_t r = e - IV(FP::exponent_offset);
  r = ifthen(convert_bool(e), r, IV(std::numeric_limits<int_t>::min()));
#if defined VML_HAVE_INF
  r = ifthen(isinf(x), IV(std::numeric_limits<int_t>::max()), r);
#endif
#if defined VML_HAVE_NAN
  r = ifthen(isnan(x), IV(std::numeric_limits<int_t>::min()), r);
#endif
  return r;
}

template <typename realvec_t>
typename realvec_t::boolvec_t
mathfuncs<realvec_t>::vml_ieee_isfinite(realvec_t x) {
  return (as_int(x) & IV(FP::exponent_mask)) != IV(FP::exponent_mask);
}

template <typename realvec_t>
typename realvec_t::boolvec_t
mathfuncs<realvec_t>::vml_ieee_isinf(realvec_t x) {
  return (as_int(x) & IV(I(~FP::signbit_mask))) == IV(FP::exponent_mask);
}

template <typename realvec_t>
typename realvec_t::boolvec_t
mathfuncs<realvec_t>::vml_ieee_isnan(realvec_t x) {
  return (as_int(x) & IV(FP::exponent_mask)) == IV(FP::exponent_mask) &&
         (as_int(x) & IV(FP::mantissa_mask)) != IV(I(0));
}

template <typename realvec_t>
typename realvec_t::boolvec_t
mathfuncs<realvec_t>::vml_ieee_isnormal(realvec_t x) {
  return (as_int(x) & IV(FP::exponent_mask)) != IV(FP::exponent_mask) &&
         (as_int(x) & IV(FP::exponent_mask)) != IV(I(0));
}

template <typename realvec_t>
typename realvec_t::boolvec_t mathfuncs<realvec_t>::vml_isfinite(realvec_t x) {
#if defined VML_HAVE_INF || defined VML_HAVE_NAN
  return vml_ieee_isfinite(x);
#else
  return BV(true);
#endif
}

template <typename realvec_t>
typename realvec_t::boolvec_t mathfuncs<realvec_t>::vml_isinf(realvec_t x) {
#if defined VML_HAVE_INF
  return vml_ieee_isinf(x);
#else
  return BV(false);
#endif
}

template <typename realvec_t>
typename realvec_t::boolvec_t mathfuncs<realvec_t>::vml_isnan(realvec_t x) {
#if defined VML_HAVE_NAN
  return vml_ieee_isnan(x);
#else
  return BV(false);
#endif
}

template <typename realvec_t>
typename realvec_t::boolvec_t mathfuncs<realvec_t>::vml_isnormal(realvec_t x) {
#if defined VML_HAVE_DENORMALS || defined VML_HAVE_INF || defined VML_HAVE_NAN
  return vml_ieee_isnormal(x);
#else
  return BV(true);
#endif
}

template <typename realvec_t>
realvec_t mathfuncs<realvec_t>::vml_ldexp(realvec_t x, intvec_t n) {
// TODO: Check SLEEF 2.80 algorithm
#if 0
    realvec_t r = as_float(as_int(x) + (n << I(FP::mantissa_bits)));
    r = ifthen((as_int(x) & IV(FP::exponent_mask)) == IV(I(0)), x, r);
    return r;
#endif
  realvec_t r = as_float(as_int(x) + (n << U(FP::mantissa_bits)));
  int max_n = FP::max_exponent - FP::min_exponent;
  boolvec_t underflow = n < IV(I(-max_n));
  boolvec_t overflow = n > IV(I(max_n));
  intvec_t old_exp = lsr(as_int(x) & IV(FP::exponent_mask), FP::mantissa_bits);
  intvec_t new_exp = old_exp + n;
  // TODO: check bit patterns instead
  underflow =
      underflow || new_exp < IV(I(FP::min_exponent + FP::exponent_offset));
  overflow =
      overflow || new_exp > IV(I(FP::max_exponent + FP::exponent_offset));
  r = ifthen(underflow, copysign(RV(R(0.0)), x), r);
  r = ifthen(overflow, copysign(RV(FP::infinity()), x), r);
  boolvec_t dont_change = x == RV(R(0.0)) || isinf(x) || isnan(x);
  r = ifthen(dont_change, x, r);
  return r;
}

template <typename realvec_t>
realvec_t mathfuncs<realvec_t>::vml_mad(realvec_t x, realvec_t y, realvec_t z) {
  return x * y + z;
}

template <typename realvec_t>
typename realvec_t::boolvec_t mathfuncs<realvec_t>::vml_signbit(realvec_t x) {
  return convert_bool(as_int(x) & IV(FP::signbit_mask));
}

}; // namespace vecmathlib

#endif // #ifndef MATHFUNCS_FABS_H
