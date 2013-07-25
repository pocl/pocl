// -*-C++-*-

#ifndef MATHFUNCS_ASIN_H
#define MATHFUNCS_ASIN_H

#include "mathfuncs_base.h"

#include <cmath>



namespace vecmathlib {
  
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_atan(realvec_t x)
  {
    // Handle negative values
    realvec_t x0 = x;
    x = fabs(x);
    
    // Reduce range using 1/x identity
    VML_ASSERT(all(x >= RV(0.0)));
    boolvec_t gt_one = x > RV(1.0);
    x = ifthen(gt_one, rcp(x), x);
    
    // Reduce range again using half-angle formula; see
    // <https://en.wikipedia.org/wiki/Inverse_trigonometric_functions>.
    // This is necessary for good convergence below.
    x = x / (RV(1.0) + sqrt(RV(1.0) + x*x));
    
#if 0
    // Taylor expansion; see
    // <https://en.wikipedia.org/wiki/Inverse_trigonometric_functions>.
    VML_ASSERT(all(x >= RV(0.0) && x <= RV(0.5)));
    int const nmax = 30;        // ???
    realvec_t y = x / (RV(1.0) + x*x);
    realvec_t x2 = x * y;
    realvec_t r = y;
    for (int n=3; n<nmax; n+=2) {
      y *= RV(R(n-1) / R(n)) * x2;
      r += y;
    }
#endif
    
    // Polynomial expansion
    realvec_t x2 = x*x;
    realvec_t r;
    switch (sizeof(real_t)) {
    case 4:
#ifdef VML_HAVE_FP_CONTRACT
      // float, error=6.66422646286979497624922530951e-8
      r = RV(+0.067880041389077294203913658751);
      r = fma(r, x2, RV(-0.133733063898947623461317627449));
      r = fma(r, x2, RV(+0.19911334683762018553047223812));
      r = fma(r, x2, RV(-0.333297629421461914979541552214));
      r = fma(r, x2, RV(0.99999959717590068040974191128));
#else
      // float, error=1.32698047768409645072438892571e-6
      r = RV(-0.097792979486911722224672843246);
      r = fma(r, x2, RV(+0.192823203439066255014185816489));
      r = fma(r, x2, RV(-0.332894374801791377010848071499));
      r = fma(r, x2, RV(+0.99999272283064606320969693166));
#endif
      break;
    case 8:
#ifdef VML_HAVE_FP_CONTRACT
      // double, error=7.23827307971781482562906889298e-17
      r = RV(-0.0119268581772947474118625342812);
      r = fma(r, x2, RV(+0.0314136659341247717203573238714));
      r = fma(r, x2, RV(-0.0471488852137420698546535537847));
      r = fma(r, x2, RV(+0.057569335614537634720962389105));
      r = fma(r, x2, RV(-0.066469674694325315726074277265));
      r = fma(r, x2, RV(+0.076901807118525604168645555045));
      r = fma(r, x2, RV(-0.090907534052423738503603248332));
      r = fma(r, x2, RV(+0.111111036293577807846564325197));
      r = fma(r, x2, RV(-0.142857140628324608456323319753));
      r = fma(r, x2, RV(+0.199999999962771461194546102341));
      r = fma(r, x2, RV(-0.333333333333044876533973016233));
      r = fma(r, x2, RV(+0.9999999999999993386168118967));
#else
      // double, error=2.55716065822130171460427941931e-14
      r = RV(-0.0181544668501977245345114956236);
      r = fma(r, x2, RV(+0.0437292245148812058302877081114));
      r = fma(r, x2, RV(-0.062544339168176953821092935889));
      r = fma(r, x2, RV(+0.076201106915857900897507167054));
      r = fma(r, x2, RV(-0.090827503061359848291718661563));
      r = fma(r, x2, RV(+0.11110526662726539375601075137));
      r = fma(r, x2, RV(-0.142856889699649116820344831251));
      r = fma(r, x2, RV(+0.199999993955745494607716573019));
      r = fma(r, x2, RV(-0.333333333267116176139371545145));
      r = fma(r, x2, RV(+0.99999999999978652742783656623));
#endif
      break;
    default:
      __builtin_unreachable();
    }
    r *= x;
    
    // Undo second range reduction
    // TODO: put this into the coefficients
    r = RV(2.0) * r;
    
    // Undo range reduction
    r = ifthen(gt_one, RV(M_PI_2) - r, r);
    
    // Handle negative values
    r = copysign(r, x0);
    
    return r;
  }
  
  
  
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_acos(realvec_t x)
  {
    return RV(M_PI_2) - asin(x);
  }
  
  
  
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_asin(realvec_t x)
  {
    return RV(2.0) * atan(x / (RV(1.0) + sqrt(RV(1.0) - x*x)));
  }
  
  
  
  // Note: the order of arguments is y, x, as is convention for atan2
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_atan2(realvec_t y, realvec_t x)
  {
    realvec_t r = atan(y/x);
    realvec_t offset = copysign(ifthen(signbit(x), RV(M_PI), RV(0.0)), y);
    r = r + offset;
    // Note: the case x=y=0 is implemented via the second if
    // condition; thus, the order of the two if conditions cannot be
    // exchanged
    r = ifthen(x==RV(0.0), copysign(RV(M_PI_2), y), r);
    r = ifthen(y==RV(0.0), offset, r);
    return r;
  }
  
}; // namespace vecmathlib

#endif  // #ifndef MATHFUNCS_ASIN_H
