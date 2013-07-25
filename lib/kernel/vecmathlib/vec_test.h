// -*-C++-*-

#ifndef VEC_TEST_H
#define VEC_TEST_H

#include "floatprops.h"
#include "mathfuncs.h"
#include "vec_base.h"

#include <cmath>



namespace vecmathlib {
  
  template<typename T, int N> struct booltestvec;
  template<typename T, int N> struct inttestvec;
  template<typename T, int N> struct realtestvec;
  
  
  
  template<typename T, int N>
  struct booltestvec: floatprops<T>
  {
    typedef typename floatprops<T>::int_t int_t;
    typedef typename floatprops<T>::uint_t uint_t;
    typedef typename floatprops<T>::real_t real_t;
    
    static int const size = N;
    typedef bool scalar_t;
    typedef bool bvector_t[size];
    static int const alignment = sizeof(bool);
    
    typedef booltestvec boolvec_t;
    typedef inttestvec<real_t, size> intvec_t;
    typedef realtestvec<real_t, size> realvec_t;
    
    // short names for type casts
    typedef real_t R;
    typedef int_t I;
    typedef uint_t U;
    typedef realvec_t RV;
    typedef intvec_t IV;
    typedef boolvec_t BV;
    typedef floatprops<real_t> FP;
    typedef mathfuncs<realvec_t> MF;
    
    
    
    bvector_t v;
    
    booltestvec() {}
    // can't have a non-trivial copy constructor; if so, objects won't
    // be passed in registers
    // booltestvec(booltestvec const& x): v(x.v) {}
    // booltestvec& operator=(booltestvec const& x) { return v=x.v, *this; }
    //booltestvec(vector_t x): v(x) {}
    booltestvec(bool a) { for (int d=0; d<size; ++d) v[d]=a; }
    booltestvec(bool const* as) { for (int d=0; d<size; ++d) v[d]=as[d]; }
    
    bool operator[](int n) const { return v[n]; }
    booltestvec& set_elt(int n, bool a) { return v[n]=a, *this; }
    
    
    
    intvec_t as_int() const;      // defined after inttestvec
    intvec_t convert_int() const; // defined after inttestvec
    
    
    
    booltestvec operator!() const
    {
      booltestvec res;
      for (int d=0; d<size; ++d) res.v[d] = !v[d];
      return res;
    }
    
    booltestvec operator&&(booltestvec x) const
    {
      booltestvec res;
      for (int d=0; d<size; ++d) res.v[d] = v[d] && x.v[d];
      return res;
    }
    booltestvec operator||(booltestvec x) const
    {
      booltestvec res;
      for (int d=0; d<size; ++d) res.v[d] = v[d] || x.v[d];
      return res;
    }
    booltestvec operator==(booltestvec x) const
    {
      booltestvec res;
      for (int d=0; d<size; ++d) res.v[d] = v[d] == x.v[d];
      return res;
    }
    booltestvec operator!=(booltestvec x) const
    {
      booltestvec res;
      for (int d=0; d<size; ++d) res.v[d] = v[d] != x.v[d];
      return res;
    }
    
    bool all() const
    {
      bool res = v[0];
      for (int d=1; d<size; ++d) res = res && v[d];
      return res;
    }
    bool any() const
    {
      bool res = v[0];
      for (int d=1; d<size; ++d) res = res || v[d];
      return res;
    }
    
    
    
    // ifthen(condition, then-value, else-value)
    boolvec_t ifthen(boolvec_t x, boolvec_t y) const;
    intvec_t ifthen(intvec_t x, intvec_t y) const; // defined after inttestvec
    realvec_t ifthen(realvec_t x, realvec_t y) const; // defined after realtestvec
  };
  
  
  
  template<typename T, int N>
  struct inttestvec: floatprops<T>
  {
    typedef typename floatprops<T>::int_t int_t;
    typedef typename floatprops<T>::uint_t uint_t;
    typedef typename floatprops<T>::real_t real_t;
    
    static int const size = N;
    typedef int_t scalar_t;
    typedef int_t ivector_t[size];
    static int const alignment = sizeof(int_t);
    
    typedef booltestvec<real_t, size> boolvec_t;
    typedef inttestvec intvec_t;
    typedef realtestvec<real_t, size> realvec_t;
    
    // short names for type casts
    typedef real_t R;
    typedef int_t I;
    typedef uint_t U;
    typedef realvec_t RV;
    typedef intvec_t IV;
    typedef boolvec_t BV;
    typedef floatprops<real_t> FP;
    typedef mathfuncs<realvec_t> MF;
    
    
    
    ivector_t v;
    
    inttestvec() {}
    // can't have a non-trivial copy constructor; if so, objects won't
    // be passed in registers
    // inttestvec(inttestvec const& x): v(x.v) {}
    // inttestvec& operator=(inttestvec const& x) { return v=x.v, *this; }
    //inttestvec(vector_t x): v(x) {}
    inttestvec(int_t a) { for (int d=0; d<size; ++d) v[d]=a; }
    inttestvec(int_t const* as) { for (int d=0; d<size; ++d) v[d]=as[d]; }
    static inttestvec iota()
    {
      inttestvec res;
      for (int d=0; d<size; ++d) res.v[d]=d;
      return res;
    }
    
    int_t operator[](int n) const { return v[n]; }
    inttestvec& set_elt(int n, int_t a) { return v[n]=a, *this; }
    
    
    
    boolvec_t as_bool() const { return convert_bool(); }
    boolvec_t convert_bool() const
    {
      // result: convert_bool(0)=false, convert_bool(else)=true
      boolvec_t res;
      for (int d=0; d<size; ++d) res.v[d]=v[d];
      return res;
    }
    realvec_t as_float() const;      // defined after realtestvec
    realvec_t convert_float() const; // defined after realtestvec
    
    
    
    inttestvec operator+() const
    {
      inttestvec res;
      for (int d=0; d<size; ++d) res.v[d] = + v[d];
      return res;
    }
    inttestvec operator-() const
    {
      inttestvec res;
      for (int d=0; d<size; ++d) res.v[d] = - v[d];
      return res;
    }
    
    inttestvec& operator+=(inttestvec const& x)
    {
      for (int d=0; d<size; ++d) v[d] += x.v[d];
      return *this;
    }
    inttestvec& operator-=(inttestvec const& x)
    {
      for (int d=0; d<size; ++d) v[d] -= x.v[d];
      return *this;
    }
    inttestvec& operator*=(inttestvec const& x)
    {
      for (int d=0; d<size; ++d) v[d] *= x.v[d];
      return *this;
    }
    inttestvec& operator/=(inttestvec const& x)
    {
      for (int d=0; d<size; ++d) v[d] /= x.v[d];
      return *this;
    }
    inttestvec& operator%=(inttestvec const& x)
    {
      for (int d=0; d<size; ++d) v[d] %= x.v[d];
      return *this;
    }
    
    inttestvec operator+(inttestvec x) const
    {
      inttestvec res = *this;
      return res += x;
    }
    inttestvec operator-(inttestvec x) const
    {
      inttestvec res = *this;
      return res -= x;
    }
    inttestvec operator*(inttestvec x) const
    {
      inttestvec res = *this;
      return res *= x;
    }
    inttestvec operator/(inttestvec x) const
    {
      inttestvec res = *this;
      return res /= x;
    }
    inttestvec operator%(inttestvec x) const
    {
      inttestvec res = *this;
      return res %= x;
    }
    
    
    
    inttestvec operator~() const
    {
      inttestvec res;
      for (int d=0; d<size; ++d) res.v[d] = ~ v[d];
      return res;
    }
    
    inttestvec& operator&=(inttestvec const& x)
    {
      for (int d=0; d<size; ++d) v[d] &= x.v[d];
      return *this;
    }
    inttestvec& operator|=(inttestvec const& x)
    {
      for (int d=0; d<size; ++d) v[d] |= x.v[d];
      return *this;
    }
    inttestvec& operator^=(inttestvec const& x)
    {
      for (int d=0; d<size; ++d) v[d] ^= x.v[d];
      return *this;
    }
    
    inttestvec operator&(inttestvec x) const
    {
      inttestvec res = *this;
      return res &= x;
    }
    inttestvec operator|(inttestvec x) const
    {
      inttestvec res = *this;
      return res |= x;
    }
    inttestvec operator^(inttestvec x) const
    {
      inttestvec res = *this;
      return res ^= x;
    }
    
    
    
    inttestvec lsr(int_t n) const
    {
      inttestvec res;
      for (int d=0; d<size; ++d) res.v[d] = I(U(v[d]) >> U(n));
      return res;
    }
    inttestvec& operator>>=(int_t n)
    {
      for (int d=0; d<size; ++d) v[d] >>= n;
      return *this;
    }
    inttestvec& operator<<=(int_t n)
    {
      for (int d=0; d<size; ++d) v[d] <<= n;
      return *this;
    }
    inttestvec operator>>(int_t n) const
    {
      inttestvec res = *this;
      return res >>= n;
    }
    inttestvec operator<<(int_t n) const
    {
      inttestvec res = *this;
      return res <<= n;
    }
    
    inttestvec lsr(inttestvec n) const
    {
      inttestvec res;
      for (int d=0; d<size; ++d) res.v[d] = I(U(v[d]) >> U(n.v[d]));
      return res;
    }
    inttestvec& operator>>=(inttestvec n)
    {
      for (int d=0; d<size; ++d) v[d] >>= n.v[d];
      return *this;
    }
    inttestvec& operator<<=(inttestvec n)
    {
      for (int d=0; d<size; ++d) v[d] <<= n.v[d];
      return *this;
    }
    inttestvec operator>>(inttestvec n) const
    {
      inttestvec res = *this;
      return res >>= n;
    }
    inttestvec operator<<(inttestvec n) const
    {
      inttestvec res = *this;
      return res <<= n;
    }
    
    
    
    boolvec_t signbit() const
    {
      boolvec_t res;
      for (int d=0; d<size; ++d) res.set_elt(d, v[d] < 0);
      return res;
    }
    
    boolvec_t operator==(inttestvec const& x) const
    {
      boolvec_t res;
      for (int d=0; d<size; ++d) res.v[d] = v[d] == x.v[d];
      return res;
    }
    boolvec_t operator!=(inttestvec const& x) const
    {
      boolvec_t res;
      for (int d=0; d<size; ++d) res.v[d] = v[d] != x.v[d];
      return res;
    }
    boolvec_t operator<(inttestvec const& x) const
    {
      boolvec_t res;
      for (int d=0; d<size; ++d) res.v[d] = v[d] < x.v[d];
      return res;
    }
    boolvec_t operator<=(inttestvec const& x) const
    {
      boolvec_t res;
      for (int d=0; d<size; ++d) res.v[d] = v[d] <= x.v[d];
      return res;
    }
    boolvec_t operator>(inttestvec const& x) const
    {
      boolvec_t res;
      for (int d=0; d<size; ++d) res.v[d] = v[d] > x.v[d];
      return res;
    }
    boolvec_t operator>=(inttestvec const& x) const
    {
      boolvec_t res;
      for (int d=0; d<size; ++d) res.v[d] = v[d] >= x.v[d];
      return res;
    }
  };
  
  
  
  template<typename T, int N>
  struct realtestvec: floatprops<T>
  {
    typedef typename floatprops<T>::int_t int_t;
    typedef typename floatprops<T>::uint_t uint_t;
    typedef typename floatprops<T>::real_t real_t;
    
    static int const size = N;
    typedef real_t scalar_t;
    typedef real_t vector_t[size];
    static int const alignment = sizeof(real_t);
    
    static char const* name()
    {
      static std::string name_;
      if (name_.empty()) {
        std::stringstream buf;
        buf << "<VML:" << N << "*" << FP::name() << ">";
        name_ = buf.str();
      }
      return name_.c_str();
    }
    void barrier()
    {
#if defined __GNUC__ && !defined __clang__ && !defined __ICC
      // GCC crashes when +X is used as constraint
#  if defined __SSE2__
      for (int d=0; d<size; ++d) __asm__("": "+x"(v[d]));
#  elif defined __PPC64__       // maybe also __PPC__
      for (int d=0; d<size; ++d) __asm__("": "+f"(v[d]));
#  elif defined __arm__
      for (int d=0; d<size; ++d) __asm__("": "+w"(v[d]));
#  else
#    error "Floating point barrier undefined on this architecture"
#  endif
#elif defined __clang__
      for (int d=0; d<size; ++d) __asm__("": "+X"(v[d]));
#elif defined __ICC
      for (int d=0; d<size; ++d) {
        real_t tmp = v[d];
        __asm__("": "+X"(tmp));
        v[d] = tmp;
      }
#elif defined __IBMCPP__
      for (int d=0; d<size; ++d) __asm__("": "+f"(v[d]));
#else
#  error "Floating point barrier undefined on this architecture"
#endif
    }
    
    typedef booltestvec<real_t, size> boolvec_t;
    typedef inttestvec<real_t, size> intvec_t;
    typedef realtestvec realvec_t;
    
    // short names for type casts
    typedef real_t R;
    typedef int_t I;
    typedef uint_t U;
    typedef realvec_t RV;
    typedef intvec_t IV;
    typedef boolvec_t BV;
    typedef floatprops<real_t> FP;
    typedef mathfuncs<realvec_t> MF;
    
    
    
    vector_t v;
    
    realtestvec() {}
    // can't have a non-trivial copy constructor; if so, objects won't
    // be passed in registers
    // realtestvec(realtestvec const& x): v(x.v) {}
    // realtestvec& operator=(realtestvec const& x) { return v=x.v, *this; }
    //realtestvec(vector_t x): v(x) {}
    realtestvec(real_t a) { for (int d=0; d<size; ++d) v[d]=a; }
    realtestvec(real_t const* as) { for (int d=0; d<size; ++d) v[d]=as[d]; }
    
    real_t operator[](int n) const { return v[n]; }
    realtestvec& set_elt(int n, real_t a) { return v[n]=a, *this; }
    
    
    
    typedef vecmathlib::mask_t<realvec_t> mask_t;
    
    static realvec_t loada(real_t const* p)
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      return loadu(p);
    }
    static realvec_t loadu(real_t const* p)
    {
      realvec_t res;
      for (int d=0; d<size; ++d) res.v[d] = p[d];
      return res;
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
      storeu(p);
    }
    void storeu(real_t* p) const
    {
      for (int d=0; d<size; ++d) p[d] = v[d];
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
      for (int d=0; d<size; ++d) res.v[d] = FP::as_int(v[d]);
      return res;
    }
    intvec_t convert_int() const { return MF::vml_convert_int(*this); }
    
    
    
    realtestvec operator+() const
    {
      realtestvec res;
      for (int d=0; d<size; ++d) res.v[d] = + v[d];
      return res;
    }
    realtestvec operator-() const
    {
      realtestvec res;
      for (int d=0; d<size; ++d) res.v[d] = - v[d];
      return res;
    }
    
    realtestvec& operator+=(realtestvec const& x)
    {
      for (int d=0; d<size; ++d) v[d] += x.v[d];
      return *this;
    }
    realtestvec& operator-=(realtestvec const& x)
    {
      for (int d=0; d<size; ++d) v[d] -= x.v[d];
      return *this;
    }
    realtestvec& operator*=(realtestvec const& x)
    {
      for (int d=0; d<size; ++d) v[d] *= x.v[d];
      return *this;
    }
    realtestvec& operator/=(realtestvec const& x)
    {
      for (int d=0; d<size; ++d) v[d] /= x.v[d];
      return *this;
    }
    
    realtestvec operator+(realtestvec x) const
    {
      realtestvec res = *this;
      return res += x;
    }
    realtestvec operator-(realtestvec x) const
    {
      realtestvec res = *this;
      return res -= x;
    }
    realtestvec operator*(realtestvec x) const
    {
      realtestvec res = *this;
      return res *= x;
    }
    realtestvec operator/(realtestvec x) const
    {
      realtestvec res = *this;
      return res /= x;
    }
    
    real_t maxval() const
    {
      real_t res = v[0];
      for (int d=1; d<size; ++d) res = std::fmax(res, v[d]);
      return res;
    }
    real_t minval() const
    {
      real_t res = v[0];
      for (int d=1; d<size; ++d) res = std::fmin(res, v[d]);
      return res;
    }
    real_t prod() const
    {
      real_t res = v[0];
      for (int d=1; d<size; ++d) res *= v[d];
      return res;
    }
    real_t sum() const
    {
      real_t res = v[0];
      for (int d=1; d<size; ++d) res += v[d];
      return res;
    }
    
    
    
    boolvec_t operator==(realtestvec const& x) const
    {
      boolvec_t res;
      for (int d=0; d<size; ++d) res.v[d] = v[d] == x.v[d];
      return res;
    }
    boolvec_t operator!=(realtestvec const& x) const
    {
      boolvec_t res;
      for (int d=0; d<size; ++d) res.v[d] = v[d] != x.v[d];
      return res;
    }
    boolvec_t operator<(realtestvec const& x) const
    {
      boolvec_t res;
      for (int d=0; d<size; ++d) res.v[d] = v[d] < x.v[d];
      return res;
    }
    boolvec_t operator<=(realtestvec const& x) const
    {
      boolvec_t res;
      for (int d=0; d<size; ++d) res.v[d] = v[d] <= x.v[d];
      return res;
    }
    boolvec_t operator>(realtestvec const& x) const
    {
      boolvec_t res;
      for (int d=0; d<size; ++d) res.v[d] = v[d] > x.v[d];
      return res;
    }
    boolvec_t operator>=(realtestvec const& x) const
    {
      boolvec_t res;
      for (int d=0; d<size; ++d) res.v[d] = v[d] >= x.v[d];
      return res;
    }
    
    
    
    realtestvec acos() const { return MF::vml_acos(*this); }
    realtestvec acosh() const { return MF::vml_acosh(*this); }
    realtestvec asin() const { return MF::vml_asin(*this); }
    realtestvec asinh() const { return MF::vml_asinh(*this); }
    realtestvec atan() const { return MF::vml_atan(*this); }
    realtestvec atan2(realtestvec y) const { return MF::vml_atan2(*this, y); }
    realtestvec atanh() const { return MF::vml_atanh(*this); }
    realtestvec cbrt() const { return MF::vml_cbrt(*this); }
    realtestvec ceil() const { return MF::vml_ceil(*this); }
    realtestvec copysign(realtestvec y) const
    {
      return MF::vml_copysign(*this, y);
    }
    realtestvec cos() const { return MF::vml_cos(*this); }
    realtestvec cosh() const { return MF::vml_cosh(*this); }
    realtestvec exp() const { return MF::vml_exp(*this); }
    realtestvec exp10() const { return MF::vml_exp10(*this); }
    realtestvec exp2() const { return MF::vml_exp2(*this); }
    realtestvec expm1() const { return MF::vml_expm1(*this); }
    realtestvec fabs() const { return MF::vml_fabs(*this); }
    realtestvec fdim(realtestvec y) const { return MF::vml_fdim(*this, y); }
    realtestvec floor() const { return MF::vml_floor(*this); }
    realtestvec fma(realtestvec y, realtestvec z) const 
    {
      return MF::vml_fma(*this, y, z);
    }
    realtestvec fmax(realtestvec y) const { return MF::vml_fmax(*this, y); }
    realtestvec fmin(realtestvec y) const { return MF::vml_fmin(*this, y); }
    realtestvec fmod(realtestvec y) const { return MF::vml_fmod(*this, y); }
    realtestvec frexp(intvec_t& r) const { return MF::vml_frexp(*this, r); }
    realtestvec hypot(realtestvec y) const { return MF::vml_hypot(*this, y); }
    intvec_t ilogb() const { return MF::vml_ilogb(*this); }
    boolvec_t isfinite() const { return MF::vml_isfinite(*this); }
    boolvec_t isinf() const { return MF::vml_isinf(*this); }
    boolvec_t isnan() const { return MF::vml_isnan(*this); }
    boolvec_t isnormal() const { return MF::vml_isnormal(*this); }
    realtestvec ldexp(int_t n) const { return MF::vml_ldexp(*this, n); }
    realtestvec ldexp(intvec_t n) const { return MF::vml_ldexp(*this, n); }
    realtestvec log() const { return MF::vml_log(*this); }
    realtestvec log10() const { return MF::vml_log10(*this); }
    realtestvec log1p() const { return MF::vml_log1p(*this); }
    realtestvec log2() const { return MF::vml_log2(*this); }
    realtestvec nextafter(realtestvec y) const
    {
      return MF::vml_nextafter(*this, y);
    }
    realtestvec pow(realtestvec y) const { return MF::vml_pow(*this, y); }
    realtestvec rcp() const { return MF::vml_rcp(*this); }
    realtestvec remainder(realtestvec y) const
    {
      return MF::vml_remainder(*this, y);
    }
    realtestvec rint() const { return MF::vml_rint(*this); }
    realtestvec round() const { return MF::vml_round(*this); }
    realtestvec rsqrt() const { return MF::vml_rsqrt(*this); }
    boolvec_t signbit() const { return MF::vml_signbit(*this); }
    realtestvec sin() const { return MF::vml_sin(*this); }
    realtestvec sinh() const { return MF::vml_sinh(*this); }
    realtestvec sqrt() const { return MF::vml_sqrt(*this); }
    realtestvec tan() const { return MF::vml_tan(*this); }
    realtestvec tanh() const { return MF::vml_tanh(*this); }
    realtestvec trunc() const { return MF::vml_trunc(*this); }
  };
  
  
  
  // booltestvec definitions
  
  template<typename T, int N>
  inline
  typename booltestvec<T,N>::intvec_t booltestvec<T,N>::as_int() const
  {
    return convert_int();
  }
  
  template<typename T, int N>
  inline
  typename booltestvec<T,N>::intvec_t booltestvec<T,N>::convert_int() const
  {
    intvec_t res;
    for (int d=0; d<size; ++d) res.v[d] = v[d];
    return res;
  }
  
  template<typename T, int N>
  inline
  typename booltestvec<T,N>::boolvec_t
  booltestvec<T,N>::ifthen(boolvec_t x, boolvec_t y) const
  {
    boolvec_t res;
    for (int d=0; d<size; ++d) res.v[d] = v[d] ? x.v[d] : y.v[d];
    return res;
  }
  
  template<typename T, int N>
  inline
  typename booltestvec<T,N>::intvec_t
  booltestvec<T,N>::ifthen(intvec_t x, intvec_t y) const
  {
    intvec_t res;
    for (int d=0; d<size; ++d) res.v[d] = v[d] ? x.v[d] : y.v[d];
    return res;
  }
  
  template<typename T, int N>
  inline
  typename booltestvec<T,N>::realvec_t
  booltestvec<T,N>::ifthen(realvec_t x, realvec_t y) const
  {
    realvec_t res;
    for (int d=0; d<size; ++d) res.v[d] = v[d] ? x.v[d] : y.v[d];
    return res;
  }

  
  
  // inttestvec definitions
  
  template<typename T, int N>
  inline
  typename inttestvec<T,N>::realvec_t inttestvec<T,N>::as_float() const
  {
    realvec_t res;
    for (int d=0; d<size; ++d) res.v[d] = FP::as_float(v[d]);
    return res;
  }
  
  template<typename T, int N>
  inline
  typename inttestvec<T,N>::realvec_t inttestvec<T,N>::convert_float() const
  {
    return MF::vml_convert_float(*this);
  }
  


  // Wrappers
  
  // booltestvec wrappers
  
  template<typename real_t, int size>
  inline inttestvec<real_t, size> as_int(booltestvec<real_t, size> x)
  {
    return x.as_int();
  }
  
  template<typename real_t, int size>
  inline inttestvec<real_t, size> convert_int(booltestvec<real_t, size> x)
  {
    return x.convert_int();
  }
  
  template<typename real_t, int size>
  inline bool all(booltestvec<real_t, size> x) { return x.all(); }
  
  template<typename real_t, int size>
  inline bool any(booltestvec<real_t, size> x) { return x.any(); }
  
  template<typename real_t, int size>
  inline
  booltestvec<real_t, size> ifthen(booltestvec<real_t, size> c,
                                   booltestvec<real_t, size> x,
                                   booltestvec<real_t, size> y)
  {
    return c.ifthen(x, y);
  }
  
  template<typename real_t, int size>
  inline
  inttestvec<real_t, size> ifthen(booltestvec<real_t, size> c,
                                  inttestvec<real_t, size> x,
                                  inttestvec<real_t, size> y)
  {
    return c.ifthen(x, y);
  }
  
  template<typename real_t, int size>
  inline
  realtestvec<real_t, size> ifthen(booltestvec<real_t, size> c,
                                   realtestvec<real_t, size> x,
                                   realtestvec<real_t, size> y)
  {
    return c.ifthen(x, y);
  }
  
  
  
  // inttestvec wrappers
  
  template<typename real_t, int size>
  inline booltestvec<real_t, size> as_bool(inttestvec<real_t, size> x)
  {
    return x.as_bool();
  }
  
  template<typename real_t, int size>
  inline booltestvec<real_t, size> convert_bool(inttestvec<real_t, size> x)
  {
    return x.convert_bool();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> as_float(inttestvec<real_t, size> x)
  {
    return x.as_float();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> convert_float(inttestvec<real_t, size> x)
  {
    return x.convert_float();
  }
  
  template<typename real_t, int size>
  inline
  inttestvec<real_t, size> lsr(inttestvec<real_t, size> x,
                               typename inttestvec<real_t, size>::int_t n)
  {
    return x.lsr(n);
  }
  
  template<typename real_t, int size>
  inline inttestvec<real_t, size> lsr(inttestvec<real_t, size> x,
                                      inttestvec<real_t, size> n)
  {
    return x.lsr(n);
  }
  
  template<typename real_t, int size>
  inline booltestvec<real_t, size> signbit(inttestvec<real_t, size> x)
  {
    return x.signbit();
  }
  
  
  
  // realtestvec wrappers
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size>
  loada(real_t const* p,
        realtestvec<real_t, size> x,
        typename realtestvec<real_t, size>::mask_t const& m)
  {
    return x.loada(p, m);
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size>
  loadu(real_t const* p,
        realtestvec<real_t, size> x,
        typename realtestvec<real_t, size>::mask_t const& m)
  {
    return x.loadu(p, m);
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size>
  loadu(real_t const* p, size_t ioff,
        realtestvec<real_t, size> x,
        typename realtestvec<real_t, size>::mask_t const& m)
  {
    return x.loadu(p, ioff, m);
  }
  
  template<typename real_t, int size>
  inline void storea(realtestvec<real_t, size> x, real_t* p)
  {
    return x.storea(p);
  }
  
  template<typename real_t, int size>
  inline void storeu(realtestvec<real_t, size> x, real_t* p)
  {
    return x.storeu(p);
  }
  
  template<typename real_t, int size>
  inline void storeu(realtestvec<real_t, size> x, real_t* p, size_t ioff)
  {
    return x.storeu(p, ioff);
  }
  
  template<typename real_t, int size>
  inline void storea(realtestvec<real_t, size> x, real_t* p,
                     typename realtestvec<real_t, size>::mask_t const& m)
  {
    return x.storea(p, m);
  }
  
  template<typename real_t, int size>
  inline void storeu(realtestvec<real_t, size> x, real_t* p,
                     typename realtestvec<real_t, size>::mask_t const& m)
  {
    return x.storeu(p, m);
  }
  
  template<typename real_t, int size>
  inline void storeu(realtestvec<real_t, size> x, real_t* p, size_t ioff,
                     typename realtestvec<real_t, size>::mask_t const& m)
  {
    return x.storeu(p, ioff, m);
  }
  
  
  
  template<typename real_t, int size>
  inline inttestvec<real_t, size> as_int(realtestvec<real_t, size> x)
  {
    return x.as_int();
  }
  
  template<typename real_t, int size>
  inline inttestvec<real_t, size> convert_int(realtestvec<real_t, size> x)
  {
    return x.convert_int();
  }
  
  template<typename real_t, int size>
  inline real_t maxval(realtestvec<real_t, size> x)
  {
    return x.maxval();
  }
  
  template<typename real_t, int size>
  inline real_t minval(realtestvec<real_t, size> x)
  {
    return x.minval();
  }
  
  template<typename real_t, int size>
  inline real_t prod(realtestvec<real_t, size> x)
  {
    return x.prod();
  }
  
  template<typename real_t, int size>
  inline real_t sum(realtestvec<real_t, size> x)
  {
    return x.sum();
  }
  
  
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> acos(realtestvec<real_t, size> x)
  {
    return x.acos();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> acosh(realtestvec<real_t, size> x)
  {
    return x.acosh();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> asin(realtestvec<real_t, size> x)
  {
    return x.asin();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> asinh(realtestvec<real_t, size> x)
  {
    return x.asinh();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> atan(realtestvec<real_t, size> x)
  {
    return x.atan();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> atan2(realtestvec<real_t, size> x,
                                         realtestvec<real_t, size> y)
  {
    return x.atan2(y);
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> atanh(realtestvec<real_t, size> x)
  {
    return x.atanh();
  }
    
  template<typename real_t, int size>
  inline realtestvec<real_t, size> cbrt(realtestvec<real_t, size> x)
  {
    return x.cbrt();
  }
    
  template<typename real_t, int size>
  inline realtestvec<real_t, size> ceil(realtestvec<real_t, size> x)
  {
    return x.ceil();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> copysign(realtestvec<real_t, size> x,
                                            realtestvec<real_t, size> y)
  {
    return x.copysign(y);
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> cos(realtestvec<real_t, size> x)
  {
    return x.cos();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> cosh(realtestvec<real_t, size> x)
  {
    return x.cosh();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> exp(realtestvec<real_t, size> x)
  {
    return x.exp();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> exp10(realtestvec<real_t, size> x)
  {
    return x.exp10();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> exp2(realtestvec<real_t, size> x)
  {
    return x.exp2();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> expm1(realtestvec<real_t, size> x)
  {
    return x.expm1();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> fabs(realtestvec<real_t, size> x)
  {
    return x.fabs();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> floor(realtestvec<real_t, size> x)
  {
    return x.floor();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> fdim(realtestvec<real_t, size> x,
                                        realtestvec<real_t, size> y)
  {
    return x.fdim(y);
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> fma(realtestvec<real_t, size> x,
                                       realtestvec<real_t, size> y,
                                       realtestvec<real_t, size> z)
  {
    return x.fma(y, z);
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> fmax(realtestvec<real_t, size> x,
                                        realtestvec<real_t, size> y)
  {
    return x.fmax(y);
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> fmin(realtestvec<real_t, size> x,
                                        realtestvec<real_t, size> y)
  {
    return x.fmin(y);
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> fmod(realtestvec<real_t, size> x,
                                        realtestvec<real_t, size> y)
  {
    return x.fmod(y);
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> frexp(realtestvec<real_t, size> x,
                                         inttestvec<real_t, size>& r)
  {
    return x.frexp(r);
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> hypot(realtestvec<real_t, size> x,
                                         realtestvec<real_t, size> y)
  {
    return x.hypot(y);
  }
  
  template<typename real_t, int size>
  inline inttestvec<real_t, size> ilogb(realtestvec<real_t, size> x)
  {
    return x.ilogb();
  }
  
  template<typename real_t, int size>
  inline booltestvec<real_t, size> isfinite(realtestvec<real_t, size> x)
  {
    return x.isfinite();
  }
  
  template<typename real_t, int size>
  inline booltestvec<real_t, size> isinf(realtestvec<real_t, size> x)
  {
    return x.isinf();
  }
  
  template<typename real_t, int size>
  inline booltestvec<real_t, size> isnan(realtestvec<real_t, size> x)
  {
    return x.isnan();
  }
  
  template<typename real_t, int size>
  inline booltestvec<real_t, size> isnormal(realtestvec<real_t, size> x)
  {
    return x.isnormal();
  }
  
  template<typename real_t, int size>
  inline
  realtestvec<real_t, size> ldexp(realtestvec<real_t, size> x,
                                  typename inttestvec<real_t, size>::int_t n)
  {
    return x.ldexp(n);
  }
  
  template<typename real_t, int size>
  inline
  realtestvec<real_t, size> ldexp(realtestvec<real_t, size> x,
                                  inttestvec<real_t, size> n)
  {
    return x.ldexp(n);
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> log(realtestvec<real_t, size> x)
  {
    return x.log();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> log10(realtestvec<real_t, size> x)
  {
    return x.log10();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> log1p(realtestvec<real_t, size> x)
  {
    return x.log1p();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> log2(realtestvec<real_t, size> x)
  {
    return x.log2();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> nextafter(realtestvec<real_t, size> x,
                                             realtestvec<real_t, size> y)
  {
    return x.nextafter(y);
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> pow(realtestvec<real_t, size> x,
                                       realtestvec<real_t, size> y)
  {
    return x.pow(y);
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> rcp(realtestvec<real_t, size> x)
  {
    return x.rcp();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> remainder(realtestvec<real_t, size> x,
                                             realtestvec<real_t, size> y)
  {
    return x.remainder(y);
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> rint(realtestvec<real_t, size> x)
  {
    return x.rint();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> round(realtestvec<real_t, size> x)
  {
    return x.round();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> rsqrt(realtestvec<real_t, size> x)
  {
    return x.rsqrt();
  }
  
  template<typename real_t, int size>
  inline booltestvec<real_t, size> signbit(realtestvec<real_t, size> x)
  {
    return x.signbit();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> sin(realtestvec<real_t, size> x)
  {
    return x.sin();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> sinh(realtestvec<real_t, size> x)
  {
    return x.sinh();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> sqrt(realtestvec<real_t, size> x)
  {
    return x.sqrt();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> tan(realtestvec<real_t, size> x)
  {
    return x.tan();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> tanh(realtestvec<real_t, size> x)
  {
    return x.tanh();
  }
  
  template<typename real_t, int size>
  inline realtestvec<real_t, size> trunc(realtestvec<real_t, size> x)
  {
    return x.trunc();
  }
  
  
  
  template<typename real_t, int size>
  std::ostream& operator<<(std::ostream& os,
                           booltestvec<real_t, size> const& x)
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
                           inttestvec<real_t, size> const& x)
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
                           realtestvec<real_t, size> const& x)
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

#endif  // #ifndef VEC_TEST_H
