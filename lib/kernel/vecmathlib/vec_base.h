// -*-C++-*-

#ifndef VEC_BASE_H
#define VEC_BASE_H

#include <iostream>

#include "vec_mask.h"



namespace vecmathlib {
  
  template<typename real_t, int size>
  struct boolvec {
  };
  
  template<typename real_t, int size>
  struct intvec {
  };
  
  template<typename real_t, int size>
  struct realvec {
  };
  

  
  // boolvec wrappers
  
  template<typename real_t, int size>
  inline intvec<real_t, size> as_int(boolvec<real_t, size> x)
  {
    return x.as_int();
  }
  
  template<typename real_t, int size>
  inline intvec<real_t, size> convert_int(boolvec<real_t, size> x)
  {
    return x.convert_int();
  }
  
  template<typename real_t, int size>
  inline bool all(boolvec<real_t, size> x) { return x.all(); }
  
  template<typename real_t, int size>
  inline bool any(boolvec<real_t, size> x) { return x.any(); }
  
  template<typename real_t, int size>
  inline
  boolvec<real_t, size> ifthen(boolvec<real_t, size> c,
                               boolvec<real_t, size> x,
                               boolvec<real_t, size> y)
  {
    return c.ifthen(x, y);
  }
  
  template<typename real_t, int size>
  inline
  intvec<real_t, size> ifthen(boolvec<real_t, size> c,
                              intvec<real_t, size> x,
                              intvec<real_t, size> y)
  {
    return c.ifthen(x, y);
  }
  
  template<typename real_t, int size>
  inline
  realvec<real_t, size> ifthen(boolvec<real_t, size> c,
                               realvec<real_t, size> x,
                               realvec<real_t, size> y)
  {
    return c.ifthen(x, y);
  }
  
  
  
  // intvec wrappers
  
  template<typename real_t, int size>
  inline boolvec<real_t, size> as_bool(intvec<real_t, size> x)
  {
    return x.as_bool();
  }
  
  template<typename real_t, int size>
  inline boolvec<real_t, size> convert_bool(intvec<real_t, size> x)
  {
    return x.convert_bool();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> as_float(intvec<real_t, size> x)
  {
    return x.as_float();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> convert_float(intvec<real_t, size> x)
  {
    return x.convert_float();
  }
  
  template<typename real_t, int size>
  inline intvec<real_t, size> lsr(intvec<real_t, size> x,
                                  typename intvec<real_t, size>::int_t n)
  {
    return x.lsr(n);
  }
  
  template<typename real_t, int size>
  inline intvec<real_t, size> lsr(intvec<real_t, size> x,
                                  intvec<real_t, size> n)
  {
    return x.lsr(n);
  }
  
  template<typename real_t, int size>
  inline boolvec<real_t, size> signbit(intvec<real_t, size> x)
  {
    return x.signbit();
  }
  
  
  
  // realvec wrappers
  
  template<typename real_t, int size>
  inline realvec<real_t, size>
  loada(real_t const* p,
        realvec<real_t, size> x,
        typename realvec<real_t, size>::mask_t const& m)
  {
    return x.loada(p, m);
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size>
  loadu(real_t const* p,
        realvec<real_t, size> x,
        typename realvec<real_t, size>::mask_t const& m)
  {
    return x.loadu(p, m);
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size>
  loadu(real_t const* p, size_t ioff,
        realvec<real_t, size> x,
        typename realvec<real_t, size>::mask_t const& m)
  {
    return x.loadu(p, ioff, m);
  }
  
  template<typename real_t, int size>
  inline void storea(realvec<real_t, size> x, real_t* p)
  {
    x.storea(p);
  }
  
  template<typename real_t, int size>
  inline void storeu(realvec<real_t, size> x, real_t* p)
  {
    x.storeu(p);
  }
  
  template<typename real_t, int size>
  inline void storeu(realvec<real_t, size> x, real_t* p, size_t ioff)
  {
    x.storeu(p, ioff);
  }
  
  template<typename real_t, int size>
  inline void storea(realvec<real_t, size> x, real_t* p,
                     typename realvec<real_t, size>::mask_t const& m)
  {
    x.storea(p, m);
  }
  
  template<typename real_t, int size>
  inline void storeu(realvec<real_t, size> x, real_t* p,
                     typename realvec<real_t, size>::mask_t const& m)
  {
    x.storeu(p, m);
  }
  
  template<typename real_t, int size>
  inline void storeu(realvec<real_t, size> x, real_t* p, size_t ioff,
                     typename realvec<real_t, size>::mask_t const &m)
  {
    x.storeu(p, ioff, m);
  }
  
  
  
  template<typename real_t, int size>
  inline intvec<real_t, size> as_int(realvec<real_t, size> x)
  {
    return x.as_int();
  }
  
  template<typename real_t, int size>
  inline intvec<real_t, size> convert_int(realvec<real_t, size> x)
  {
    return x.convert_int();
  }
  
  template<typename real_t, int size>
  inline
  typename realvec<real_t, size>::real_t maxval(realvec<real_t, size> x)
  {
    return x.maxval();
  }
  
  template<typename real_t, int size>
  inline
  typename realvec<real_t, size>::real_t minval(realvec<real_t, size> x)
  {
    return x.minval();
  }
  
  template<typename real_t, int size>
  inline
  typename realvec<real_t, size>::real_t prod(realvec<real_t, size> x)
  {
    return x.prod();
  }
  
  template<typename real_t, int size>
  inline
  typename realvec<real_t, size>::real_t sum(realvec<real_t, size> x)
  {
    return x.sum();
  }
  
  
  
  template<typename real_t, int size>
  inline realvec<real_t, size> acos(realvec<real_t, size> x)
  {
    return x.acos();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> acosh(realvec<real_t, size> x)
  {
    return x.acosh();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> asin(realvec<real_t, size> x)
  {
    return x.asin();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> asinh(realvec<real_t, size> x)
  {
    return x.asinh();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> atan(realvec<real_t, size> x)
  {
    return x.atan();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> atan2(realvec<real_t, size> x,
                                     realvec<real_t, size> y)
  {
    return x.atan2(y);
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> atanh(realvec<real_t, size> x)
  {
    return x.atanh();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> cbrt(realvec<real_t, size> x)
  {
    return x.cbrt();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> ceil(realvec<real_t, size> x)
  {
    return x.ceil();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> copysign(realvec<real_t, size> x,
                                        realvec<real_t, size> y)
  {
    return x.copysign(y);
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> cos(realvec<real_t, size> x)
  {
    return x.cos();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> cosh(realvec<real_t, size> x)
  {
    return x.cosh();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> exp(realvec<real_t, size> x)
  {
    return x.exp();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> exp10(realvec<real_t, size> x)
  {
    return x.exp10();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> exp2(realvec<real_t, size> x)
  {
    return x.exp2();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> expm1(realvec<real_t, size> x)
  {
    return x.expm1();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> fabs(realvec<real_t, size> x)
  {
    return x.fabs();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> floor(realvec<real_t, size> x)
  {
    return x.floor();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> fdim(realvec<real_t, size> x,
                                    realvec<real_t, size> y)
  {
    return x.fdim(y);
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> fma(realvec<real_t, size> x,
                                   realvec<real_t, size> y,
                                   realvec<real_t, size> z)
  {
    return x.fma(y, z);
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> fmax(realvec<real_t, size> x,
                                    realvec<real_t, size> y)
  {
    return x.fmax(y);
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> fmin(realvec<real_t, size> x,
                                    realvec<real_t, size> y)
  {
    return x.fmin(y);
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> fmod(realvec<real_t, size> x,
                                    realvec<real_t, size> y)
  {
    return x.fmod(y);
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> frexp(realvec<real_t, size> x,
                                     intvec<real_t, size>& r)
  {
    return x.frexp(r);
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> hypot(realvec<real_t, size> x,
                                     realvec<real_t, size> y)
  {
    return x.hypot(y);
  }
  
  template<typename real_t, int size>
  inline intvec<real_t, size> ilogb(realvec<real_t, size> x)
  {
    return x.ilogb();
  }
  
  template<typename real_t, int size>
  inline boolvec<real_t, size> isfinite(realvec<real_t, size> x)
  {
    return x.isfinite();
  }
  
  template<typename real_t, int size>
  inline boolvec<real_t, size> isinf(realvec<real_t, size> x)
  {
    return x.isinf();
  }
  
  template<typename real_t, int size>
  inline boolvec<real_t, size> isnan(realvec<real_t, size> x)
  {
    return x.isnan();
  }
  
  template<typename real_t, int size>
  inline boolvec<real_t, size> isnormal(realvec<real_t, size> x)
  {
    return x.isnormal();
  }
  
  template<typename real_t, int size>
  inline
  realvec<real_t, size> ldexp(realvec<real_t, size> x,
                              typename intvec<real_t, size>::int_t n)
  {
    return x.ldexp(n);
  }
  
  template<typename real_t, int size>
  inline
  realvec<real_t, size> ldexp(realvec<real_t, size> x,
                               intvec<real_t, size> n)
  {
    return x.ldexp(n);
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> log(realvec<real_t, size> x)
  {
    return x.log();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> log10(realvec<real_t, size> x)
  {
    return x.log10();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> log1p(realvec<real_t, size> x)
  {
    return x.log1p();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> log2(realvec<real_t, size> x)
  {
    return x.log2();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> nextafter(realvec<real_t, size> x,
                                         realvec<real_t, size> y)
  {
    return x.nextafter(y);
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> pow(realvec<real_t, size> x,
                                   realvec<real_t, size> y)
  {
    return x.pow(y);
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> rcp(realvec<real_t, size> x)
  {
    return x.rcp();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> remainder(realvec<real_t, size> x,
                                         realvec<real_t, size> y)
  {
    return x.remainder(y);
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> rint(realvec<real_t, size> x)
  {
    return x.rint();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> round(realvec<real_t, size> x)
  {
    return x.round();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> rsqrt(realvec<real_t, size> x)
  {
    return x.rsqrt();
  }
  
  template<typename real_t, int size>
  inline boolvec<real_t, size> signbit(realvec<real_t, size> x)
  {
    return x.signbit();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> sin(realvec<real_t, size> x)
  {
    return x.sin();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> sinh(realvec<real_t, size> x)
  {
    return x.sinh();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> sqrt(realvec<real_t, size> x)
  {
    return x.sqrt();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> tan(realvec<real_t, size> x)
  {
    return x.tan();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> tanh(realvec<real_t, size> x)
  {
    return x.tanh();
  }
  
  template<typename real_t, int size>
  inline realvec<real_t, size> trunc(realvec<real_t, size> x)
  {
    return x.trunc();
  }
  
  
  
  template<typename real_t, int size>
  std::ostream& operator<<(std::ostream& os, boolvec<real_t, size> const& x)
  {
    os << "[";
    for (int i=0; i<size; ++i) {
      if (i!=0) os << ",";
      os << x[i];
    }
    os << "]";
    return os;
  }
  
  template<typename real_t, int size>
  std::ostream& operator<<(std::ostream& os, intvec<real_t, size> const& x)
  {
    os << "[";
    for (int i=0; i<size; ++i) {
      if (i!=0) os << ",";
      os << x[i];
    }
    os << "]";
    return os;
  }
  
  template<typename real_t, int size>
  std::ostream& operator<<(std::ostream& os, realvec<real_t, size> const& x)
  {
    os << "[";
    for (int i=0; i<size; ++i) {
      if (i!=0) os << ",";
      os << x[i];
    }
    os << "]";
    return os;
  }
  
} // namespace vecmathlib

#endif  // #ifndef VEC_BASE_H
