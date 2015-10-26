// -*-C++-*-

#ifndef MATHFUNCS_LOG_H
#define MATHFUNCS_LOG_H

#include "mathfuncs_base.h"

#include <cmath>

namespace vecmathlib {

template <typename realvec_t>
realvec_t mathfuncs<realvec_t>::vml_log2(realvec_t x) {
  // Algorithm inspired by SLEEF 2.80

  // Rescale
  intvec_t ilogb_x = ilogb(x * RV(M_SQRT2));
  x = ldexp(x, -ilogb_x);
  VML_ASSERT(all(x >= RV(M_SQRT1_2) && x <= RV(M_SQRT2)));

  realvec_t y = (x - RV(1.0)) / (x + RV(1.0));
  realvec_t y2 = y * y;

  realvec_t r;
  switch (sizeof(real_t)) {
  case 4:
    // float, error=7.09807175879142775648452461821e-8
    r = RV(0.59723611417135718739797302426);
    r = mad(r, y2, RV(0.961524413175528426101613434));
    r = mad(r, y2, RV(2.88539097665498228703236701));
    break;
  case 8:
#ifdef VML_HAVE_FP_CONTRACT
    // double, error=1.48294180185938512675770096324e-16
    r = RV(0.243683403415639178527756320773);
    r = mad(r, y2, RV(0.26136626803870009948502658));
    r = mad(r, y2, RV(0.320619429891299265439389));
    r = mad(r, y2, RV(0.4121983452028499242926));
    r = mad(r, y2, RV(0.577078017761894161436));
    r = mad(r, y2, RV(0.96179669392233355927));
    r = mad(r, y2, RV(2.8853900817779295236));
#else
    // double, error=2.1410114030383689267772704676e-14
    r = RV(0.283751646449323373643963474845);
    r = mad(r, y2, RV(0.31983138095551191299118812));
    r = mad(r, y2, RV(0.412211603844146279666022));
    r = mad(r, y2, RV(0.5770779098948940070516));
    r = mad(r, y2, RV(0.961796694295973716912));
    r = mad(r, y2, RV(2.885390081777562819196));
#endif
    break;
  default:
    __builtin_unreachable();
  }
  r *= y;

  // Undo rescaling
  r += convert_float(ilogb_x);

  return r;
}

template <typename realvec_t>
inline realvec_t mathfuncs<realvec_t>::vml_log(realvec_t x) {
  return log2(x) * RV(M_LN2);
}

template <typename realvec_t>
inline realvec_t mathfuncs<realvec_t>::vml_log10(realvec_t x) {
  return log(x) * RV(M_LOG10E);
}

template <typename realvec_t>
inline realvec_t mathfuncs<realvec_t>::vml_log1p(realvec_t x) {
  // TODO: Check SLEEF 2.80 algorithm

  return log(RV(1.0) + x);
#if 0
    // Goldberg, theorem 4
    realvec_t x1 = RV(1.0) + x;
    x1.barrier();
    return ifthen(x1 == x, x, x * log(x1) / (x1 - RV(1.0)));
#endif
}

}; // namespace vecmathlib

#endif // #ifndef MATHFUNCS_LOG_H
