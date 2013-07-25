// -*-C++-*-

#ifndef MATHFUNCS_SINH_H
#define MATHFUNCS_SINH_H

#include "mathfuncs_base.h"

#include <cmath>



namespace vecmathlib {
  
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_cosh(realvec_t x)
  {
    return RV(0.5) * (exp(x) + exp(-x));
  }
  
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_sinh(realvec_t x)
  {
    return RV(0.5) * (exp(x) - exp(-x));
  }
  
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_tanh(realvec_t x)
  {
    return sinh(x) / cosh(x);
  }
  
}; // namespace vecmathlib

#endif  // #ifndef MATHFUNCS_SINH_H
