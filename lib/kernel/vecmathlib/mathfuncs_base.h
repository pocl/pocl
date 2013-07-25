// -*-C++-*-

#ifndef MATHFUNCS_BASE_H
#define MATHFUNCS_BASE_H

#include "floatprops.h"
#include "vec_base.h"



namespace vecmathlib {
  
  template<typename realvec_t>
  struct mathfuncs {
    typedef floatprops<typename realvec_t::real_t> FP;
    
    typedef typename FP::real_t real_t;
    typedef typename FP::int_t int_t;
    typedef typename FP::uint_t uint_t;
    
    static int const size = realvec_t::size;
    
    // typedef realvec<real_t, size> realvec_t;
    typedef typename realvec_t::intvec_t intvec_t;
    typedef typename realvec_t::boolvec_t boolvec_t;
    
    // Short names for type casts
    typedef real_t R;
    typedef int_t I;
    typedef uint_t U;
    typedef realvec_t RV;
    typedef intvec_t IV;
    typedef boolvec_t BV;
    // static real_t R(double a) { return real_t(a); }
    // static int_t I(int a) { return int_t(a); }
    // static uint_t U(int a) { return uint_t(a); }
    // static realvec_t RV(real_t a) { return realvec_t(a); }
    // static intvec_t IV(int_t a) { return intvec_t(a); }
    // static boolvec_t BV(bool a) { return boolvec_t(a); }
    
    // asin
    static realvec_t vml_acos(realvec_t x);
    static realvec_t vml_asin(realvec_t x);
    static realvec_t vml_atan(realvec_t x);
    static realvec_t vml_atan2(realvec_t y, realvec_t x);
    
    // asinh
    static realvec_t vml_acosh(realvec_t x);
    static realvec_t vml_asinh(realvec_t x);
    static realvec_t vml_atanh(realvec_t x);
    
    // convert
    static realvec_t vml_antitrunc(realvec_t x);
    static realvec_t vml_ceil(realvec_t x);
    static realvec_t vml_convert_float(intvec_t x);
    static intvec_t vml_convert_int(realvec_t x);
    static realvec_t vml_floor(realvec_t x);
    static realvec_t vml_rint(realvec_t x);
    static realvec_t vml_round(realvec_t x);
    static realvec_t vml_nextafter(realvec_t x, realvec_t y);
    static realvec_t vml_trunc(realvec_t x);
    
    // fabs
    static realvec_t vml_copysign(realvec_t x, realvec_t y);
    static realvec_t vml_fabs(realvec_t x);
    static realvec_t vml_fdim(realvec_t x, realvec_t y);
    static realvec_t vml_fma(realvec_t x, realvec_t y, realvec_t z);
    static realvec_t vml_fmax(realvec_t x, realvec_t y);
    static realvec_t vml_fmin(realvec_t x, realvec_t y);
    static realvec_t vml_frexp(realvec_t x, intvec_t& r);
    static intvec_t vml_ilogb(realvec_t x);
    static boolvec_t vml_ieee_isfinite(realvec_t x);
    static boolvec_t vml_ieee_isinf(realvec_t x);
    static boolvec_t vml_ieee_isnan(realvec_t x);
    static boolvec_t vml_ieee_isnormal(realvec_t x);
    static boolvec_t vml_isfinite(realvec_t x);
    static boolvec_t vml_isinf(realvec_t x);
    static boolvec_t vml_isnan(realvec_t x);
    static boolvec_t vml_isnormal(realvec_t x);
    static realvec_t vml_ldexp(realvec_t x, intvec_t n);
    static boolvec_t vml_signbit(realvec_t x);
    
    // exp
    static realvec_t vml_exp(realvec_t x);
    static realvec_t vml_exp10(realvec_t x);
    static realvec_t vml_exp2(realvec_t x);
    static realvec_t vml_expm1(realvec_t x);
    
    // log
    static realvec_t vml_log(realvec_t x);
    static realvec_t vml_log10(realvec_t x);
    static realvec_t vml_log1p(realvec_t x);
    static realvec_t vml_log2(realvec_t x);
    
    // pow
    static realvec_t vml_pow(realvec_t x, realvec_t y);
    
    // rcp
    static realvec_t vml_fmod(realvec_t x, realvec_t y);
    static realvec_t vml_rcp(realvec_t x);
    static realvec_t vml_remainder(realvec_t x, realvec_t y);
    
    // sin
    static realvec_t vml_cos(realvec_t x);
    static realvec_t vml_sin(realvec_t x);
    static realvec_t vml_tan(realvec_t x);
    
    // sinh
    static realvec_t vml_cosh(realvec_t x);
    static realvec_t vml_sinh(realvec_t x);
    static realvec_t vml_tanh(realvec_t x);
    
    // sqrt
    static realvec_t vml_cbrt(realvec_t x);
    static realvec_t vml_hypot(realvec_t x, realvec_t y);
    static realvec_t vml_rsqrt(realvec_t x);
    static realvec_t vml_sqrt(realvec_t x);
  };
  
} // namespace vecmathlib

#endif  // #ifndef MATHFUNCS_BASE_H
