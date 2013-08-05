// -*-C++-*-

#ifndef MATHFUNCS_SIN_H
#define MATHFUNCS_SIN_H

#include "mathfuncs_base.h"

#include <cmath>



namespace vecmathlib {
  
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_sin(realvec_t x)
  {
    // Rescale input
    x *= RV(1.0/(2.0*M_PI));
    
    // Reduce range: sin(x) = sin(x + 2pi)
    x -= rint(x);
    VML_ASSERT(all(x >= RV(-0.5) && x <= RV(+0.5)));
    
    // Reduce range: sin(x) = -sin(-x)
    realvec_t sign = x;
    x = fabs(x);
    VML_ASSERT(all(x >= RV(0.0) && x <= RV(0.5)));
    
    // Reduce range: sin(x) = sin(pi - x)
    x = fmin(x, RV(0.5)-x);
    VML_ASSERT(all(x >= RV(0.0) && x <= RV(0.25)));
    
    // Polynomial expansion
    realvec_t x2 = x*x;
    realvec_t r;
    switch (sizeof(real_t)) {
    case 4:
#ifdef VML_HAVE_FP_CONTRACT
      // float, error=9.87410297844129295403342670375e-9
      r = RV(+39.6528001753697560443995227219);
      r = fma(r, x2, RV(-76.56466773344087533597673461));
      r = fma(r, x2, RV(+81.601631640211881069182778798));
      r = fma(r, x2, RV(-41.3416646547590947908779056914));
      r = fma(r, x2, RV(+6.2831851989440238363581945604));
#else
      // float, error=1.58016215200483927797284866108e-6, 
      r = RV(-71.315978441592194407247888986);
      r = fma(r, x2, RV(+81.371978532544677342208302432));
      r = fma(r, x2, RV(-41.3379839486638336886792146778));
      r = fma(r, x2, RV(+6.2831695125493457846550448487));
#endif
      break;
    case 8:
#ifdef VML_HAVE_FP_CONTRACT
      // double, error=6.23674794993677351325343779135e-19
      r = RV(+0.100807907479216992437490038615);
      r = fma(r, x2, RV(-0.71770901464828231655437134119));
      r = fma(r, x2, RV(+3.81992525864144427125360186953));
      r = fma(r, x2, RV(-15.0946415041050550042911122547));
      r = fma(r, x2, RV(+42.0586939194912275451623859862));
      r = fma(r, x2, RV(-76.705859752708750470270935774));
      r = fma(r, x2, RV(+81.605249276072410674251626791));
      r = fma(r, x2, RV(-41.3417022403997512576544306479));
      r = fma(r, x2, RV(+6.2831853071795864680224671155));
#else
      // double, error=1.35052419895760612440936163371e-13
      r = RV(+3.66062453812577851620993390211);
      r = fma(r, x2, RV(-15.0803528951943294831983114707));
      r = fma(r, x2, RV(+42.0580411171712349980374012262));
      r = fma(r, x2, RV(-76.705843817504942181869384713));
      r = fma(r, x2, RV(+81.605249077126372940457689074));
      r = fma(r, x2, RV(-41.3417022393099629778945111838));
      r = fma(r, x2, RV(+6.2831853071778704147024863797));
#endif
      break;
    default:
      __builtin_unreachable();
    }
    r *= x;
    
    // Undo range reduction
    r = copysign(r, sign);
    
    return r;
  }

  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_cos(realvec_t x)
  {
    return vml_sin(x + RV(M_PI_2));
  }
  
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_tan(realvec_t x)
  {
    return sin(x) / cos(x);
  }
  
}; // namespace vecmathlib

#endif  // #ifndef MATHFUNCS_SIN_H
