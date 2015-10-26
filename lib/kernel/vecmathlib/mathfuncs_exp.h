// -*-C++-*-

#ifndef MATHFUNCS_EXP_H
#define MATHFUNCS_EXP_H

#include "mathfuncs_base.h"

#include <cmath>

namespace vecmathlib {

template <typename realvec_t>
realvec_t mathfuncs<realvec_t>::vml_exp2(realvec_t x) {
  // TODO: Check SLEEF 2.80 algorithm
  // (in particular the improved-precision truncation)

  // Rescale
  realvec_t x0 = x;

// realvec_t round_x = rint(x);
// intvec_t iround_x = convert_int(round_x);
// r = ldexp(r, iround_x);

#if 0
    // Straightforward implementation
    realvec_t round_x = rint(x);
    x -= round_x;
#elif 1
  // Round by adding, then subtracting again a large number
  // Add a large number to move the mantissa bits to the right
  int_t large = (U(1) << FP::mantissa_bits) + FP::exponent_offset;
  realvec_t tmp = x + RV(R(large));
  tmp.barrier();

  realvec_t round_x = tmp - RV(R(large));
  x -= round_x;
#else
  // Straightforward implementation, using round instead of rint,
  // since round is faster for QPX
  realvec_t round_x = round(x);
  x -= round_x;
#endif
  VML_ASSERT(all(x >= RV(-0.5) && x <= RV(0.5)));

  // Polynomial expansion
  realvec_t r;
  switch (sizeof(real_t)) {
  case 4:
#ifdef VML_HAVE_FP_CONTRACT
    // float, error=4.55549108005200277750378992345e-9
    r = RV(0.000154653240842602623787395880898);
    r = mad(r, x, RV(0.00133952915439234389712105060319));
    r = mad(r, x, RV(0.0096180399118156827664944870552));
    r = mad(r, x, RV(0.055503406540531310853149866446));
    r = mad(r, x, RV(0.240226511015459465468737123346));
    r = mad(r, x, RV(0.69314720007380208630542805293));
    r = mad(r, x, RV(0.99999999997182023878745628977));
#else
    // float, error=1.62772721960621336664735896836e-7
    r = RV(0.00133952915439234389712105060319);
    r = mad(r, x, RV(0.009670773148229417605024318985));
    r = mad(r, x, RV(0.055503406540531310853149866446));
    r = mad(r, x, RV(0.240222115700585316818177639177));
    r = mad(r, x, RV(0.69314720007380208630542805293));
    r = mad(r, x, RV(1.00000005230745711373079206024));
#endif
    break;
  case 8:
#ifdef VML_HAVE_FP_CONTRACT
    // double, error=9.32016781355638010975628074746e-18
    r = RV(4.45623165388261696886670014471e-10);
    r = mad(r, x, RV(7.0733589360775271430968224806e-9));
    r = mad(r, x, RV(1.01780540270960163558119510246e-7));
    r = mad(r, x, RV(1.3215437348041505269462510712e-6));
    r = mad(r, x, RV(0.000015252733849766201174247690629));
    r = mad(r, x, RV(0.000154035304541242555115696403795));
    r = mad(r, x, RV(0.00133335581463968601407096905671));
    r = mad(r, x, RV(0.0096181291075949686712855561931));
    r = mad(r, x, RV(0.055504108664821672870565883052));
    r = mad(r, x, RV(0.240226506959101382690753994082));
    r = mad(r, x, RV(0.69314718055994530864272481773));
    r = mad(r, x, RV(0.9999999999999999978508676375));
#else
    // double, error=3.74939899823302048807873981077e-14
    r = RV(1.02072375599725694063203809188e-7);
    r = mad(r, x, RV(1.32573274434801314145133004073e-6));
    r = mad(r, x, RV(0.0000152526647170731944840736190013));
    r = mad(r, x, RV(0.000154034441925859828261898614555));
    r = mad(r, x, RV(0.00133335582175770747495287552557));
    r = mad(r, x, RV(0.0096181291794939392517233403183));
    r = mad(r, x, RV(0.055504108664525029438908798685));
    r = mad(r, x, RV(0.240226506957026959772247598695));
    r = mad(r, x, RV(0.6931471805599487321347668143));
    r = mad(r, x, RV(1.00000000000000942892870993489));
#endif
    break;
  default:
    __builtin_unreachable();
  }

// Undo rescaling
#if 0
    // Straightforward implementation
    r = ldexp(r, convert_int(round_x));
#elif 1
  // Use direct integer manipulation
  // Extract integer as lowest mantissa bits (highest bits still
  // contain offset, exponent, and sign)
  intvec_t itmp = as_int(tmp);
  // Construct scale factor by setting exponent (this shifts out the
  // highest bits)
  realvec_t scale = as_float(itmp << I(FP::mantissa_bits));
  r *= scale;
#else
  // Use floating point operations instead of integer operations,
  // since these are faster for QPX
  real_t exponent_factor = R(I(1) << I(FP::mantissa_bits));
  real_t exponent_offset = R(I(FP::exponent_offset) << I(FP::mantissa_bits));
  realvec_t exponent = mad(round_x, RV(exponent_factor), RV(exponent_offset));
  realvec_t scale = as_float(convert_int(exponent));
  r *= scale;
#endif

  r = ifthen(x0 < RV(R(FP::min_exponent)), RV(0.0), r);

  return r;
}

template <typename realvec_t>
inline realvec_t mathfuncs<realvec_t>::vml_exp(realvec_t x) {
  return exp2(RV(M_LOG2E) * x);
}

template <typename realvec_t>
inline realvec_t mathfuncs<realvec_t>::vml_exp10(realvec_t x) {
  return exp2(RV(M_LOG2E * M_LN10) * x);
}

template <typename realvec_t>
inline realvec_t mathfuncs<realvec_t>::vml_expm1(realvec_t x) {
  // TODO: improve this
  return exp(x) - RV(1.0);
#if 0
    r = exp(x) - RV(1.0);
    return ifthen(r == RV(0.0), x, r);
#endif
}

}; // namespace vecmathlib

#endif // #ifndef MATHFUNCS_EXP_H
