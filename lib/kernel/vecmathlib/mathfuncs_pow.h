// -*-C++-*-

#ifndef MATHFUNCS_POW_H
#define MATHFUNCS_POW_H

#include "mathfuncs_base.h"

#include <cmath>



namespace vecmathlib {
  
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_pow(realvec_t x, realvec_t y)
  {
    // Handle zero
    boolvec_t is_zero = x == RV(0.0);
    x = ifthen(is_zero, RV(1.0), x);
    
    realvec_t r = exp(log(fabs(x)) * y);
    
    // The result is negative if x<0 and if y is integer and odd
    realvec_t mod_y = fabs(y) - RV(2.0) * floor(RV(0.5) * fabs(y));
    realvec_t sign = copysign(mod_y, x) + RV(0.5);
    r = copysign(r, sign);
    
    // Handle zero
    r = ifthen(is_zero, RV(0.0), r);
    
    return r;
  }
  
}; // namespace vecmathlib

#endif  // #ifndef MATHFUNCS_POW_H
