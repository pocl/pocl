// -*-C++-*-

#ifndef MATHFUNCS_SQRT_H
#define MATHFUNCS_SQRT_H

#include "mathfuncs_base.h"

#include <cmath>



namespace vecmathlib {
  
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_sqrt(realvec_t x)
  {
#if 0
    // Handle special case: zero
    boolvec_t is_zero = x <= RV(0.0);
    x = ifthen(is_zero, RV(1.0), x);
    
    // Initial guess
    VML_ASSERT(all(x > RV(0.0)));
#if 0
    intvec_t ilogb_x = ilogb(x);
    realvec_t r = ldexp(RV(M_SQRT2), ilogb_x >> 1);
    // TODO: divide by M_SQRT2 if ilogb_x % 2 == 1 ?
#else
    real_t correction =
      std::ldexp(R(FP::exponent_offset & 1 ? M_SQRT2 : 1.0),
                 FP::exponent_offset >> 1);
    realvec_t r = lsr(x.as_int(), 1).as_float() * RV(correction);
#endif
    
    // Iterate
    // nmax iterations give an accuracy of 2^nmax binary digits. 4
    // iterations suffice for double precision with its 53 digits.
    int const nmax = sizeof(real_t)==4 ? 2 : 4;
    for (int n=0; n<nmax; ++n) {
      // Step
      VML_ASSERT(all(r > RV(0.0)));
      // Newton method:
      // Solve   f(r) = 0   for   f(r) = x - r^2
      //    r <- r - f(r) / f'(r)
      //    r <- (r + x / r) / 2
      r = RV(0.5) * (r + x / r);
    }
    
    // Handle special case: zero
    r = ifthen(is_zero, RV(0.0), r);
#endif
    
    realvec_t r = x * rsqrt(x);
    // Handle special case: zero
    r = ifthen(x == RV(0.0), RV(0.0), r);
    
    return r;
  }
  
  
  
  // TODO: Use "Halley's method with cubic convergence":
  // <http://press.mcs.anl.gov/gswjanuary12/files/2012/01/Optimizing-Single-Node-Performance-on-BlueGene.pdf>
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_cbrt(realvec_t x)
  {
    return pow(x, RV(1.0/3.0));
  }
  
  
  
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_rsqrt(realvec_t x)
  {
    // Initial guess
    VML_ASSERT(all(x > RV(0.0)));
    intvec_t ilogb_x = ilogb(x);
    realvec_t s =
      ifthen(convert_bool(ilogb_x & IV(I(1))), RV(R(0.583)), RV(R(0.824)));
    realvec_t r = ldexp(s, -(ilogb_x >> I(1)));
    
    realvec_t x_2 = RV(0.5) * x;
    
    // Iterate
    // nmax iterations give an accuracy of 2^nmax binary digits. 5
    // iterations suffice for double precision with its 53 digits.
    int const nmax = sizeof(real_t)==4 ? 4 : 5;
    for (int n=0; n<nmax; ++n) {
      // Step
      VML_ASSERT(all(r > RV(0.0)));
      // Newton method:
      // Solve   f(r) = 0   for   f(r) = x - 1/r^2
      //    r <- r - f(r) / f'(r)
      //    r <- (3 r - r^3 x) / 2
      //    r <- r (3/2 - r^2 x/2)
      
      // Note: don't rewrite this expression, this may introduce
      // cancellation errors (says who?)
      // r *= RV(1.5) - x_2 * r*r;
      r += r * (RV(0.5) - x_2 * r*r);
    }
    
    return r;
  }
  
  
  
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_hypot(realvec_t x, realvec_t y)
  {
    return sqrt(x*x + y*y);
  }
  
}; // namespace vecmathlib

#endif  // #ifndef MATHFUNCS_SQRT_H
