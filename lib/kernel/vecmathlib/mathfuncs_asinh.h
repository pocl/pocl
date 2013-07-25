// -*-C++-*-

#ifndef MATHFUNCS_ASINH_H
#define MATHFUNCS_ASINH_H

#include "mathfuncs_base.h"

#include <cmath>



namespace vecmathlib {
  
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_acosh(realvec_t x)
  {
    return log(x + sqrt(x*x - RV(1.0)));
  }
  
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_asinh(realvec_t x)
  {
    // Reduce range
    realvec_t r = fabs(x);
    r = log(r + sqrt(r*r + RV(1.0)));
    r = copysign(r, x);
    return r;
  }
  
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_atanh(realvec_t x)
  {
    // Reduce range
    realvec_t r = fabs(x);
    r = RV(0.5) * log((RV(1.0) + r) / (RV(1.0) - r));
    r = copysign(r, x);
    return r;
  }
  
}; // namespace vecmathlib

#endif  // #ifndef MATHFUNCS_ASINH_H
