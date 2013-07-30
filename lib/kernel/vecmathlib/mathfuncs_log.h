// -*-C++-*-

#ifndef MATHFUNCS_LOG_H
#define MATHFUNCS_LOG_H

#include "mathfuncs_base.h"

#include <cmath>



namespace vecmathlib {
  
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_log2(realvec_t x)
  {
    // Rescale
    VML_ASSERT(all(x > RV(0.0)));
    // intvec_t ilogb_x = ilogb(x);
    // x = ldexp(x, -ilogb_x);
    // sign bit is known to be zero
    intvec_t ilogb_x = (lsr(as_int(x), I(FP::mantissa_bits)) -
                        IV(FP::exponent_offset));
    x = as_float((as_int(x) & IV(FP::mantissa_mask)) |
                 IV(I(FP::exponent_offset) << I(FP::mantissa_bits)));
    VML_ASSERT(all(x >= RV(1.0) && x < RV(2.0)));
    
    realvec_t y = (x - RV(1.0)) / (x + RV(1.0));
    realvec_t y2 = y*y;
    
    realvec_t r;
    switch (sizeof(real_t)) {
    case 4:
#ifdef VML_HAVE_FP_CONTRACT
      // float, error=5.98355642684398209498469870525e-9
      r = RV(0.410981538282433293325329456838);
      r = fma(r, y2, RV(0.402155483172044562892705980539));
      r = fma(r, y2, RV(0.57755014627178237959721643293));
      r = fma(r, y2, RV(0.96178780600659929206930296869));
      r = fma(r, y2, RV(2.88539012786343587248965772685));
#else
      //flaot, error=2.25468184051947656525068987795e-7
      r = RV(0.498866687070343238590910977481);
      r = fma(r, y2, RV(0.57002741193682764193895550312));
      r = fma(r, y2, RV(0.96200215034262628756932169194));
      r = fma(r, y2, RV(2.88538850388042106595516956395));
#endif
      break;
    case 8:
#ifdef VML_HAVE_FP_CONTRACT
      // double, error=9.45037202901655672811489051683e-17
      r = RV(0.259935726478127940817401224248);
      r = fma(r, y2, RV(0.140676370079882918464564658472));
      r = fma(r, y2, RV(0.196513478841924000569879320851));
      r = fma(r, y2, RV(0.221596471338300882039273355617));
      r = fma(r, y2, RV(0.262327298560598641020007602127));
      r = fma(r, y2, RV(0.320598261015170101859472461613));
      r = fma(r, y2, RV(0.412198595799726905825871956187));
      r = fma(r, y2, RV(0.57707801621733949207376840932));
      r = fma(r, y2, RV(0.96179669392666302667713134701));
      r = fma(r, y2, RV(2.88539008177792581277410991327));
#else
      // double, error=1.21820548287702216975532695788e-13
      r = RV(0.293251364683280430617251942017);
      r = fma(r, y2, RV(0.201364223624519571276587631354));
      r = fma(r, y2, RV(0.264443947645547871780098560836));
      r = fma(r, y2, RV(0.320475051320227723946459855458));
      r = fma(r, y2, RV(0.412202612052105347480086431555));
      r = fma(r, y2, RV(0.57707794741938820005328259256));
      r = fma(r, y2, RV(0.96179669445173881282808321929));
      r = fma(r, y2, RV(2.88539008177676567117601117274));
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
  
  
  
  template<typename realvec_t>
  inline
  realvec_t mathfuncs<realvec_t>::vml_log(realvec_t x)
  {
    return log2(x) * RV(M_LN2);
  }

  template<typename realvec_t>
  inline
  realvec_t mathfuncs<realvec_t>::vml_log10(realvec_t x)
  {
    return log(x) * RV(M_LOG10E);
  }

  template<typename realvec_t>
  inline
  realvec_t mathfuncs<realvec_t>::vml_log1p(realvec_t x)
  {
    return log(RV(1.0) + x);
#if 0
    // Goldberg, theorem 4
    realvec_t x1 = RV(1.0) + x;
    x1.barrier();
    return ifthen(x1 == x, x, x * log(x1) / (x1 - RV(1.0)));
#endif
  }
  
}; // namespace vecmathlib

#endif  // #ifndef MATHFUNCS_LOG_H
