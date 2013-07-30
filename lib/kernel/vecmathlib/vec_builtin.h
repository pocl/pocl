// -*-C++-*-

#ifndef VEC_BUILTIN_H
#define VEC_BUILTIN_H

#include "floatprops.h"
#include "mathfuncs.h"
#include "vec_base.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <string>



namespace vecmathlib {
  
  template<typename T, int N> struct boolbuiltinvec;
  template<typename T, int N> struct intbuiltinvec;
  template<typename T, int N> struct realbuiltinvec;
  
  
  
  template<typename T, int N>
  struct boolbuiltinvec: floatprops<T>
  {
    typedef typename floatprops<T>::int_t int_t;
    typedef typename floatprops<T>::uint_t uint_t;
    typedef typename floatprops<T>::real_t real_t;
    
    static int const size = N;
    typedef bool scalar_t;
    // true values are -1, false values are 0
#ifdef __clang__
    typedef int_t bvector_t __attribute__((__ext_vector_type__(N)));
#else
    typedef int_t bvector_t __attribute__((__vector_size__(N*sizeof(int_t))));
#endif
    static int const alignment = sizeof(bvector_t);
    
    static_assert(size * sizeof(real_t) == sizeof(bvector_t),
                  "vector size is wrong");
    
    typedef boolbuiltinvec boolvec_t;
    typedef intbuiltinvec<real_t, size> intvec_t;
    typedef realbuiltinvec<real_t, size> realvec_t;
    
    // Short names for type casts
    typedef real_t R;
    typedef int_t I;
    typedef uint_t U;
    typedef realvec_t RV;
    typedef intvec_t IV;
    typedef boolvec_t BV;
    typedef floatprops<real_t> FP;
    typedef mathfuncs<realvec_t> MF;
    
    static boolvec_t wrap(bvector_t x)
    {
      boolvec_t res;
      res.v = x;
      return res;
    }
    
    
    
    bvector_t v;
    
    boolbuiltinvec() {}
    // Can't have a non-trivial copy constructor; if so, objects won't
    // be passed in registers
    // boolbuiltinvec(boolbuiltinvec const& x): v(x.v) {}
    // boolbuiltinvec& operator=(boolbuiltinvec const& x) { return v=x.v, *this; }
    boolbuiltinvec(bool a): v(-(int_t)a) {}
    boolbuiltinvec(bool const* as)
    {
      for (int d=0; d<size; ++d) set_elt(d, as[d]);
    }
    
#ifdef __clang__
    bool operator[](int n) const { return v[n]; }
    boolvec_t& set_elt(int n, bool a) { return v[n]=-(int_t)a, *this; }
#else
    bool operator[](int n) const { return ((int_t const*)&v)[n]; }
    boolvec_t& set_elt(int n, bool a) { return ((int_t*)&v)[n]=a, *this; }
#endif
    
    
    
    intvec_t as_int() const;      // defined after intbuiltinvec
    intvec_t convert_int() const; // defined after intbuiltinvec
    
    
    
    boolvec_t operator!() const { return wrap(!v); }
    
    boolvec_t operator&&(boolvec_t x) const
    {
      return wrap((typename intvec_t::ivector_t)(v && x.v));
    }
    boolvec_t operator||(boolvec_t x) const
    {
      return wrap((typename intvec_t::ivector_t)(v || x.v));
    }
    boolvec_t operator==(boolvec_t x) const
    {
      return wrap((typename intvec_t::ivector_t)(v == x.v));
    }
    boolvec_t operator!=(boolvec_t x) const
    {
      return wrap((typename intvec_t::ivector_t)(v != x.v));
    }
    
    bool all() const
    {
      bool res = true;
      for (int d=0; d<size; ++d) res = res && (*this)[d];
      return res;
    }
    bool any() const
    {
      bool res = false;
      for (int d=0; d<size; ++d) res = res || (*this)[d];
      return res;
    }
    
    
    
    // ifthen(condition, then-value, else-value)
    intvec_t ifthen(intvec_t x, intvec_t y) const; // defined after intbuiltinvec
    realvec_t ifthen(realvec_t x, realvec_t y) const; // defined after realbuiltinvec
  };
  
  
  
  template<typename T, int N>
  struct intbuiltinvec: floatprops<T>
  {
    typedef typename floatprops<T>::int_t int_t;
    typedef typename floatprops<T>::uint_t uint_t;
    typedef typename floatprops<T>::real_t real_t;
    
    static int const size = N;
    typedef int_t scalar_t;
#ifdef __clang__
    typedef int_t ivector_t __attribute__((__ext_vector_type__(N)));
    typedef uint_t uvector_t __attribute__((__ext_vector_type__(N)));
#else
    typedef int_t ivector_t __attribute__((__vector_size__(N*sizeof(int_t))));
    typedef uint_t uvector_t __attribute__((__vector_size__(N*sizeof(uint_t))));
#endif
    static int const alignment = sizeof(ivector_t);
    
    static_assert(size * sizeof(real_t) == sizeof(ivector_t),
                  "vector size is wrong");
    static_assert(size * sizeof(real_t) == sizeof(uvector_t),
                  "vector size is wrong");
    
    typedef boolbuiltinvec<real_t, size> boolvec_t;
    typedef intbuiltinvec intvec_t;
    typedef realbuiltinvec<real_t, size> realvec_t;
    
    // Short names for type casts
    typedef real_t R;
    typedef int_t I;
    typedef uint_t U;
    typedef realvec_t RV;
    typedef intvec_t IV;
    typedef boolvec_t BV;
    typedef floatprops<real_t> FP;
    typedef mathfuncs<realvec_t> MF;
    
    static intvec_t wrap(ivector_t x)
    {
      intvec_t res;
      res.v = x;
      return res;
    }
    
    
    
    ivector_t v;
    
    intbuiltinvec() {}
    // Can't have a non-trivial copy constructor; if so, objects won't
    // be passed in registers
    // intbuiltinvec(intbuiltinvec const& x): v(x.v) {}
    // intbuiltinvec& operator=(intbuiltinvec const& x) { return v=x.v, *this; }
    intbuiltinvec(int_t a): v(ivector_t {a}) {}
    intbuiltinvec(int_t const* as) { std::memcpy(&v, as, sizeof v); }
    static intvec_t iota()
    {
      intvec_t res;
      for (int d=0; d<size; ++d) res.set_elt(d, d);
      return res;
    }
    
    // int_t operator[](int n) const { return ((int_t const*)&v)[n]; }
    // intvec_t& set_elt(int n, int_t a) { return ((int_t*)&v)[n]=a, *this; }
    int_t operator[](int n) const { return v[n]; }
    intvec_t& set_elt(int n, int_t a) { return v[n]=a, *this; }
    
    
    
    boolvec_t as_bool() const
    {
      boolvec_t res;
      std::memcpy(&res.v, &v, sizeof v);
      return res;
    }
    boolvec_t convert_bool() const { return boolvec_t::wrap(ivector_t(!!v)); }
    realvec_t as_float() const;      // defined after realbuiltinvec
    realvec_t convert_float() const; // defined after realbuiltinvec
    
    
    
    intvec_t operator+() const { return wrap(+v); }
    intvec_t operator-() const { return wrap(-v); }
    
    intvec_t operator+(intvec_t x) const { return wrap(v+x.v); }
    intvec_t operator-(intvec_t x) const { return wrap(v-x.v); }
    intvec_t operator*(intvec_t x) const { return wrap(v*x.v); }
    intvec_t operator/(intvec_t x) const { return wrap(v/x.v); }
    intvec_t operator%(intvec_t x) const { return wrap(v%x.v); }
    
    intvec_t& operator+=(intvec_t const& x) { return *this=*this+x; }
    intvec_t& operator-=(intvec_t const& x) { return *this=*this-x; }
    intvec_t& operator*=(intvec_t const& x) { return *this=*this*x; }
    intvec_t& operator/=(intvec_t const& x) { return *this=*this/x; }
    intvec_t& operator%=(intvec_t const& x) { return *this=*this%x; }
    
    
    
    intvec_t operator~() const { return wrap(~v); }
    
    intvec_t operator&(intvec_t x) const { return wrap(v&x.v); }
    intvec_t operator|(intvec_t x) const { return wrap(v|x.v); }
    intvec_t operator^(intvec_t x) const { return wrap(v^x.v); }
    
    intvec_t& operator&=(intvec_t const& x) { return *this=*this&x; }
    intvec_t& operator|=(intvec_t const& x) { return *this=*this|x; }
    intvec_t& operator^=(intvec_t const& x) { return *this=*this^x; }
    
    
    
    intvec_t lsr(int_t n) const { return wrap(ivector_t(uvector_t(v)>>U(n))); }
    intvec_t operator>>(int_t n) const { return wrap(v>>n); }
    intvec_t operator<<(int_t n) const { return wrap(v<<n); }
    
    intvec_t& operator>>=(int_t n) { return *this=*this>>n; }
    intvec_t& operator<<=(int_t n) { return *this=*this<<n; }
    
    intvec_t lsr(intvec_t n) const
    {
      return wrap(ivector_t(uvector_t(v)>>uvector_t(n.v)));
    }
    intvec_t operator>>(intvec_t n) const { return wrap(v>>n.v); }
    intvec_t operator<<(intvec_t n) const { return wrap(v<<n.v); }
    
    intvec_t& operator>>=(intvec_t n) { return *this=*this>>n; }
    intvec_t& operator<<=(intvec_t n) { return *this=*this<<n; }
    
    
    
    boolvec_t operator==(intvec_t const& x) const
    {
      return boolvec_t::wrap((ivector_t)(v==x.v));
    }
    boolvec_t operator!=(intvec_t const& x) const
    {
      return boolvec_t::wrap((ivector_t)(v!=x.v));
    }
    boolvec_t operator<(intvec_t const& x) const
    {
      return boolvec_t::wrap((ivector_t)(v<x.v));
    }
    boolvec_t operator<=(intvec_t const& x) const
    {
      return boolvec_t::wrap((ivector_t)(v<=x.v));
    }
    boolvec_t operator>(intvec_t const& x) const
    {
      return boolvec_t::wrap((ivector_t)(v>x.v));
    }
    boolvec_t operator>=(intvec_t const& x) const
    {
      return boolvec_t::wrap((ivector_t)(v>=x.v));
    }
  };
  
  
  
  template<typename T, int N>
  struct realbuiltinvec: floatprops<T>
  {
    typedef typename floatprops<T>::int_t int_t;
    typedef typename floatprops<T>::uint_t uint_t;
    typedef typename floatprops<T>::real_t real_t;
    
    static int const size = N;
    typedef real_t scalar_t;
#ifdef __clang__
    typedef real_t vector_t __attribute__((__ext_vector_type__(N)));
#else
    typedef real_t vector_t __attribute__((__vector_size__(N*sizeof(real_t))));
#endif
    static int const alignment = sizeof(vector_t);
    
    static_assert(size * sizeof(real_t) == sizeof(vector_t),
                  "vector size is wrong");
    
    static char const* name()
    {
      static std::string name_;
      if (name_.empty()) {
        std::stringstream buf;
        buf << "<builtin:" << N << "*" << FP::name() << ">";
        name_ = buf.str();
      }
      return name_.c_str();
    }
    void barrier() { volatile vector_t x __attribute__((__unused__)) = v; }
    
    typedef boolbuiltinvec<real_t, size> boolvec_t;
    typedef intbuiltinvec<real_t, size> intvec_t;
    typedef realbuiltinvec realvec_t;
    
    // Short names for type casts
    typedef real_t R;
    typedef int_t I;
    typedef uint_t U;
    typedef realvec_t RV;
    typedef intvec_t IV;
    typedef boolvec_t BV;
    typedef floatprops<real_t> FP;
    typedef mathfuncs<realvec_t> MF;
    
    static realvec_t wrap(vector_t x)
    {
      realvec_t res;
      res.v = x;
      return res;
    }
    
    
    
    vector_t v;
    
    realbuiltinvec() {}
    // Can't have a non-trivial copy constructor; if so, objects won't
    // be passed in registers
    realbuiltinvec(realbuiltinvec const& x): v(x.v) {}
    realbuiltinvec& operator=(realbuiltinvec const& x) { return v=x.v, *this; }
    realbuiltinvec(real_t a): v(vector_t {a}) {}
    realbuiltinvec(real_t const* as) { std::memcpy(&v, as, sizeof v); }
    
#ifdef __clang__
    real_t operator[](int n) const { return v[n]; }
    realvec_t& set_elt(int n, real_t a) { return v[n]=a, *this; }
#else
    real_t operator[](int n) const { return ((real_t const*)&v)[n]; }
    realvec_t& set_elt(int n, real_t a) { return ((real_t*)&v)[n]=a, *this; }
#endif
    
    
    
    typedef vecmathlib::mask_t<realvec_t> mask_t;
    
    static realvec_t loada(real_t const* p)
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
#ifdef __clang__
#else
      p = (real_t const*)__builtin_assume_aligned(p, sizeof(realvec_t));
#endif
      return wrap(*(vector_t const*)p);
    }
    static realvec_t loadu(real_t const* p)
    {
      // realvec_t res;
      // for (int d=0; d<size; ++d) res.set_elt(d, p[d]);
      // return res;
      return wrap(*(vector_t const*)p);
    }
    static realvec_t loadu(real_t const* p, size_t ioff)
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      return loadu(p+ioff);
    }
    realvec_t loada(real_t const* p, mask_t const& m) const
    {
      return m.m.ifthen(loada(p), *this);
    }
    realvec_t loadu(real_t const* p, mask_t const& m) const
    {
      return m.m.ifthen(loadu(p), *this);
    }
    realvec_t loadu(real_t const* p, size_t ioff, mask_t const& m) const
    {
      return m.m.ifthen(loadu(p, ioff), *this);
    }
    
    void storea(real_t* p) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
#ifdef __clang__
#else
      p = __builtin_assume_aligned(p, sizeof(realvec_t));
#endif
      *(vector_t*)p = v;
    }
    void storeu(real_t* p) const
    {
      // for (int d=0; d<size; ++d) p[d] = v[d];
      *(vector_t*)p = v;
    }
    void storeu(real_t* p, size_t ioff) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      storeu(p+ioff);
    }
    void storea(real_t* p, mask_t const& m) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      storeu(p, m);
    }
    void storeu(real_t* p, mask_t const& m) const
    {
      for (int d=0; d<size; ++d) if (m.m[d]) p[d] = v[d];
    }
    void storeu(real_t* p, size_t ioff, mask_t const& m) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      storeu(p+ioff, m);
    }
    
    
    
    intvec_t as_int() const
    {
      intvec_t res;
      std::memcpy(&res.v, &v, sizeof v);
      return res;
    }
    intvec_t convert_int() const
    {
      return intvec_t::wrap((typename intvec_t::ivector_t)v);
    }
    
    
    
    realvec_t operator+() const { return wrap(+v); }
    realvec_t operator-() const { return wrap(-v); }
    
    realvec_t operator+(realvec_t x) const { return wrap(v+x.v); }
    realvec_t operator-(realvec_t x) const { return wrap(v-x.v); }
    realvec_t operator*(realvec_t x) const { return wrap(v*x.v); }
    realvec_t operator/(realvec_t x) const { return wrap(v/x.v); }
    
    realvec_t& operator+=(realvec_t const& x) { return *this=*this+x; }
    realvec_t& operator-=(realvec_t const& x) { return *this=*this-x; }
    realvec_t& operator*=(realvec_t const& x) { return *this=*this*x; }
    realvec_t& operator/=(realvec_t const& x) { return *this=*this/x; }
    
    real_t prod() const
    {
      real_t res = R(1.0);
      for (int d=0; d<size; ++d) res *= (*this)[d];
      return res;
    }
    real_t sum() const
    {
      real_t res = R(0.0);
      for (int d=0; d<size; ++d) res += (*this)[d];
      return res;
    }
    
    
    
    boolvec_t operator==(realvec_t const& x) const
    {
      return boolvec_t::wrap((typename intvec_t::ivector_t)(v==x.v));
    }
    boolvec_t operator!=(realvec_t const& x) const
    {
      return boolvec_t::wrap((typename intvec_t::ivector_t)(v!=x.v));
    }
    boolvec_t operator<(realvec_t const& x) const
    {
      return boolvec_t::wrap((typename intvec_t::ivector_t)(v<x.v));
    }
    boolvec_t operator<=(realvec_t const& x) const
    {
      return boolvec_t::wrap((typename intvec_t::ivector_t)(v<=x.v));
    }
    boolvec_t operator>(realvec_t const& x) const
    {
      return boolvec_t::wrap((typename intvec_t::ivector_t)(v>x.v));
    }
    boolvec_t operator>=(realvec_t const& x) const
    {
      return boolvec_t::wrap((typename intvec_t::ivector_t)(v>=x.v));
    }
    
    
    
    realvec_t acos() const { return MF::vml_acos(*this); }
    realvec_t acosh() const { return MF::vml_acosh(*this); }
    realvec_t asin() const { return MF::vml_asin(*this); }
    realvec_t asinh() const { return MF::vml_asinh(*this); }
    realvec_t atan() const { return MF::vml_atan(*this); }
    realvec_t atan2(realvec_t y) const { return MF::vml_atan(*this, y); }
    realvec_t atanh() const { return MF::vml_atanh(*this); }
    realvec_t cbrt() const { return MF::vml_cbrt(*this); }
    realvec_t ceil() const { return MF::vml_ceil(*this); }
    realvec_t copysign(realvec_t y) const { return MF::vml_copysign(*this, y); }
    realvec_t cos() const { return MF::vml_cos(*this); }
    realvec_t cosh() const { return MF::vml_cosh(*this); }
    realvec_t exp() const { return MF::vml_exp(*this); }
    realvec_t exp10() const { return MF::vml_exp10(*this); }
    realvec_t exp2() const { return MF::vml_exp2(*this); }
    realvec_t expm1() const { return MF::vml_expm1(*this); }
    realvec_t fabs() const { return MF::vml_fabs(*this); }
    realvec_t fdim(realvec_t y) const { return MF::vml_fdim(*this, y); }
    realvec_t floor() const { return MF::vml_floor(*this); }
    realvec_t fma(realvec_t y, realvec_t z) const
    {
      return MF::vml_fma(*this, y, z);
    }
    realvec_t fmax(realvec_t y) const { return MF::vml_fmax(*this, y); }
    realvec_t fmin(realvec_t y) const { return MF::vml_fmin(*this, y); }
    realvec_t fmod(realvec_t y) const { return MF::vml_fmod(*this, y); }
    realvec frexp(intvec_t& r) const { return MF::vml_frexp(*this, r); }
    realvec_t hypot(realvec_t y) const { return MF::vml_hypot(*this, y); }
    intvec_t ilogb() const { return MF::vml_ilogb(*this); }
    boolvec_t isfinite() const { return MF::vml_isfinite(*this); }
    boolvec_t isinf() const { return MF::vml_isinf(*this); }
    boolvec_t isnan() const { return MF::vml_isnan(*this); }
    boolvec_t isnormal() const { return MF::vml_isnormal(*this); }
    realvec_t ldexp(int_t n) const { return MF::vml_ldexp(*this, n); }
    realvec_t ldexp(intvec_t n) const { return MF::vml_ldexp(*this, n); }
    realvec_t log() const { return MF::vml_log(*this); }
    realvec_t log10() const { return MF::vml_log10(*this); }
    realvec_t log1p() const { return MF::vml_log1p(*this); }
    realvec_t log2() const { return MF::vml_log2(*this); }
    realvec_t pow(realvec_t y) const { return MF::vml_pow(*this, y); }
    realvec_t rcp() const { return MF::vml_rcp(*this); }
    realvec_t remainder(realvec_t y) const
    {
      return MF::vml_remainder(*this, y);
    }
    realvec_t rint() const { return MF::vml_rint(*this); }
    realvec_t round() const { return MF::vml_round(*this); }
    realvec_t rsqrt() const { return MF::vml_rsqrt(*this); }
    boolvec_t signbit() const { return MF::vml_signbit(*this); }
    realvec_t sin() const { return MF::vml_sin(*this); }
    realvec_t sinh() const { return MF::vml_sinh(*this); }
    realvec_t sqrt() const { return MF::vml_sqrt(*this); }
    realvec_t tan() const { return MF::vml_tan(*this); }
    realvec_t tanh() const { return MF::vml_tanh(*this); }
    realvec_t trunc() const { return MF::vml_trunc(*this); }
  };
  
  
  
  // boolbuiltinvec definitions
  
  template<typename T, int N>
  inline
  auto boolbuiltinvec<T,N>::as_int() const -> intvec_t
  {
    intvec_t res;
    std::memcpy(&res.v, &v, sizeof v);
    return res;
  }
  
  template<typename T, int N>
  inline
  auto boolbuiltinvec<T,N>::convert_int() const -> intvec_t
  {
    return intvec_t::wrap(-v);
  }
  
  template<typename T, int N>
  inline
  auto boolbuiltinvec<T,N>::ifthen(intvec_t x, intvec_t y) const -> intvec_t
  {
#ifdef __clang__
    intvec_t mask = as_int();
    return (mask & x) | (~mask & y);
 #else
    return intvec_t::wrap(v ? x.v : y.v);
#endif
  }
  
  template<typename T, int N>
  inline
  auto boolbuiltinvec<T,N>::ifthen(realvec_t x, realvec_t y) const -> realvec_t
  {
#ifdef __clang__
    intvec_t mask = as_int();
    return as_float((mask & x.as_int()) | (~mask & y.as_int()));
 #else
    return realvec_t::wrap(v ? x.v : y.v);
#endif
  }
  
  
  
  // intbuiltinvec definitions
  
  template<typename T, int N>
  inline auto intbuiltinvec<T,N>::as_float() const -> realvec_t
  {
    realvec_t res;
    std::memcpy(&res.v, &v, sizeof v);
    return res;
  }
  
  template<typename T, int N>
  inline auto intbuiltinvec<T,N>::convert_float() const -> realvec_t
  {
    return realvec_t::wrap((typename realvec_t::vector_t)v);
  }
  
  
  
  // Wrappers
  
  // boolbuiltinvec wrappers
  
  template<typename real_t, int size>
  inline intbuiltinvec<real_t, size> as_int(boolbuiltinvec<real_t, size> x)
  {
    return x.as_int();
  }
  
  template<typename real_t, int size>
  inline intbuiltinvec<real_t, size> convert_int(boolbuiltinvec<real_t, size> x)
  {
    return x.convert_int();
  }
  
  template<typename real_t, int size>
  inline bool all(boolbuiltinvec<real_t, size> x) { return x.all(); }
  
  template<typename real_t, int size>
  inline bool any(boolbuiltinvec<real_t, size> x) { return x.any(); }
  
  template<typename real_t, int size>
  inline
  intbuiltinvec<real_t, size> ifthen(boolbuiltinvec<real_t, size> c,
                                     intbuiltinvec<real_t, size> x,
                                     intbuiltinvec<real_t, size> y)
  {
    return c.ifthen(x, y);
  }
  
  template<typename real_t, int size>
  inline
  realbuiltinvec<real_t, size> ifthen(boolbuiltinvec<real_t, size> c,
                                      realbuiltinvec<real_t, size> x,
                                      realbuiltinvec<real_t, size> y)
  {
    return c.ifthen(x, y);
  }
  
  
  
  // intbuiltinvec wrappers
  
  template<typename real_t, int size>
  inline boolbuiltinvec<real_t, size> as_bool(intbuiltinvec<real_t, size> x)
  {
    return x.as_bool();
  }
  
  template<typename real_t, int size>
  inline boolbuiltinvec<real_t, size> convert_bool(intbuiltinvec<real_t, size> x)
  {
    return x.convert_bool();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> as_float(intbuiltinvec<real_t, size> x)
  {
    return x.as_float();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> convert_float(intbuiltinvec<real_t, size> x)
  {
    return x.convert_float();
  }
  
  template<typename real_t, int size>
  inline
  intbuiltinvec<real_t, size> lsr(intbuiltinvec<real_t, size> x,
                                 typename intbuiltinvec<real_t, size>::int_t n)
  {
    return x.lsr(n);
  }
  
  template<typename real_t, int size>
  inline intbuiltinvec<real_t, size> lsr(intbuiltinvec<real_t, size> x,
                                        intbuiltinvec<real_t, size> n)
  {
    return x.lsr(n);
  }
  
  
  
  // realbuiltinvec wrappers
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size>
  loada(real_t const* p,
        realbuiltinvec<real_t, size> x,
        typename realbuiltinvec<real_t, size>::mask_t const& m)
  {
    return x.loada(p, m);
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size>
  loadu(real_t const* p,
        realbuiltinvec<real_t, size> x,
        typename realbuiltinvec<real_t, size>::mask_t const& m)
  {
    return x.loadu(p, m);
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size>
  loadu(real_t const* p, size_t ioff,
        realbuiltinvec<real_t, size> x,
        typename realbuiltinvec<real_t, size>::mask_t const& m)
  {
    return x.loadu(p, ioff, m);
  }
  
  template<typename real_t, int size>
  inline void storea(realbuiltinvec<real_t, size> x, real_t* p)
  {
    x.storea(p);
  }
  
  template<typename real_t, int size>
  inline void storeu(realbuiltinvec<real_t, size> x, real_t* p)
  {
    x.storeu(p);
  }
  
  template<typename real_t, int size>
  inline void storeu(realbuiltinvec<real_t, size> x, real_t* p, size_t ioff)
  {
    x.storeu(p, ioff);
  }
  
  template<typename real_t, int size>
  inline void storea(realbuiltinvec<real_t, size> x, real_t* p,
                     typename realbuiltinvec<real_t, size>::mask_t const& m)
  {
    x.storea(p, m);
  }
  
  template<typename real_t, int size>
  inline void storeu(realbuiltinvec<real_t, size> x, real_t* p,
                     typename realbuiltinvec<real_t, size>::mask_t const& m)
  {
    x.storeu(p, m);
  }
  
  template<typename real_t, int size>
  inline void storeu(realbuiltinvec<real_t, size> x, real_t* p, size_t ioff,
                     typename realbuiltinvec<real_t, size>::mask_t const& m)
  {
    x.storeu(p, ioff, m);
  }
  
  
  
  template<typename real_t, int size>
  inline intbuiltinvec<real_t, size> as_int(realbuiltinvec<real_t, size> x)
  {
    return x.as_int();
  }
  
  template<typename real_t, int size>
  inline intbuiltinvec<real_t, size> convert_int(realbuiltinvec<real_t, size> x)
  {
    return x.convert_int();
  }
  
  template<typename real_t, int size>
  inline auto prod(realbuiltinvec<real_t, size> x) -> real_t
  {
    return x.prod();
  }
  
  template<typename real_t, int size>
  inline auto sum(realbuiltinvec<real_t, size> x) -> real_t
  {
    return x.sum();
  }
  
  
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> acos(realbuiltinvec<real_t, size> x)
  {
    return x.acos();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> acosh(realbuiltinvec<real_t, size> x)
  {
    return x.acosh();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> asin(realbuiltinvec<real_t, size> x)
  {
    return x.asin();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> asinh(realbuiltinvec<real_t, size> x)
  {
    return x.asinh();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> atan(realbuiltinvec<real_t, size> x)
  {
    return x.atan();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> atan2(realbuiltinvec<real_t, size> x,
                                            realbuiltinvec<real_t, size> y)
  {
    return x.atan2(y);
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> atanh(realbuiltinvec<real_t, size> x)
  {
    return x.atanh();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> cbrt(realbuiltinvec<real_t, size> x)
  {
    return x.cbrt();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> ceil(realbuiltinvec<real_t, size> x)
  {
    return x.ceil();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> copysign(realbuiltinvec<real_t, size> x,
                                               realbuiltinvec<real_t, size> y)
  {
    return x.copysign(y);
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> cos(realbuiltinvec<real_t, size> x)
  {
    return x.cos();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> cosh(realbuiltinvec<real_t, size> x)
  {
    return x.cosh();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> exp(realbuiltinvec<real_t, size> x)
  {
    return x.exp();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> exp10(realbuiltinvec<real_t, size> x)
  {
    return x.exp10();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> exp2(realbuiltinvec<real_t, size> x)
  {
    return x.exp2();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> expm1(realbuiltinvec<real_t, size> x)
  {
    return x.expm1();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> fabs(realbuiltinvec<real_t, size> x)
  {
    return x.fabs();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> floor(realbuiltinvec<real_t, size> x)
  {
    return x.floor();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> fdim(realbuiltinvec<real_t, size> x,
                                           realbuiltinvec<real_t, size> y)
  {
    return x.fdim(y);
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> fma(realbuiltinvec<real_t, size> x,
                                          realbuiltinvec<real_t, size> y,
                                          realbuiltinvec<real_t, size> z)
  {
    return x.fma(y, z);
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> fmax(realbuiltinvec<real_t, size> x,
                                           realbuiltinvec<real_t, size> y)
  {
    return x.fmax(y);
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> fmin(realbuiltinvec<real_t, size> x,
                                           realbuiltinvec<real_t, size> y)
  {
    return x.fmin(y);
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> fmod(realbuiltinvec<real_t, size> x,
                                           realbuiltinvec<real_t, size> y)
  {
    return x.fmod(y);
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> hypot(realbuiltinvec<real_t, size> x,
                                            realbuiltinvec<real_t, size> y)
  {
    return x.hypot(y);
  }
  
  template<typename real_t, int size>
  inline intbuiltinvec<real_t, size> ilogb(realbuiltinvec<real_t, size> x)
  {
    return x.ilogb();
  }
  
  template<typename real_t, int size>
  inline boolbuiltinvec<real_t, size> isfinite(realbuiltinvec<real_t, size> x)
  {
    return x.isfinite();
  }
  
  template<typename real_t, int size>
  inline boolbuiltinvec<real_t, size> isinf(realbuiltinvec<real_t, size> x)
  {
    return x.isinf();
  }
  
  template<typename real_t, int size>
  inline boolbuiltinvec<real_t, size> isnan(realbuiltinvec<real_t, size> x)
  {
    return x.isnan();
  }
  
  template<typename real_t, int size>
  inline boolbuiltinvec<real_t, size> isnormal(realbuiltinvec<real_t, size> x)
  {
    return x.isnormal();
  }
  
  template<typename real_t, int size>
  inline
  realbuiltinvec<real_t, size> ldexp(realbuiltinvec<real_t, size> x,
                                     typename intbuiltinvec<real_t, size>::int_t n)
  {
    return x.ldexp(n);
  }
  
  template<typename real_t, int size>
  inline
  realbuiltinvec<real_t, size> ldexp(realbuiltinvec<real_t, size> x,
                                     intbuiltinvec<real_t, size> n)
  {
    return x.ldexp(n);
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> log(realbuiltinvec<real_t, size> x)
  {
    return x.log();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> log10(realbuiltinvec<real_t, size> x)
  {
    return x.log10();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> log1p(realbuiltinvec<real_t, size> x)
  {
    return x.log1p();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> log2(realbuiltinvec<real_t, size> x)
  {
    return x.log2();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> pow(realbuiltinvec<real_t, size> x,
                                          realbuiltinvec<real_t, size> y)
  {
    return x.pow(y);
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> rcp(realbuiltinvec<real_t, size> x)
  {
    return x.rcp();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> remainder(realbuiltinvec<real_t, size> x,
                                                realbuiltinvec<real_t, size> y)
  {
    return x.remainder(y);
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> rint(realbuiltinvec<real_t, size> x)
  {
    return x.rint();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> round(realbuiltinvec<real_t, size> x)
  {
    return x.round();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> rsqrt(realbuiltinvec<real_t, size> x)
  {
    return x.rsqrt();
  }
  
  template<typename real_t, int size>
  inline boolbuiltinvec<real_t, size> signbit(realbuiltinvec<real_t, size> x)
  {
    return x.signbit();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> sin(realbuiltinvec<real_t, size> x)
  {
    return x.sin();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> sinh(realbuiltinvec<real_t, size> x)
  {
    return x.sinh();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> sqrt(realbuiltinvec<real_t, size> x)
  {
    return x.sqrt();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> tan(realbuiltinvec<real_t, size> x)
  {
    return x.tan();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> tanh(realbuiltinvec<real_t, size> x)
  {
    return x.tanh();
  }
  
  template<typename real_t, int size>
  inline realbuiltinvec<real_t, size> trunc(realbuiltinvec<real_t, size> x)
  {
    return x.trunc();
  }
  
  
  
  template<typename real_t, int size>
  std::ostream& operator<<(std::ostream& os,
                           boolbuiltinvec<real_t, size> const& x)
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
  std::ostream& operator<<(std::ostream& os,
                           intbuiltinvec<real_t, size> const& x)
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
  std::ostream& operator<<(std::ostream& os,
                           realbuiltinvec<real_t, size> const& x)
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

#endif  // #ifndef VEC_BUILTIN_H
