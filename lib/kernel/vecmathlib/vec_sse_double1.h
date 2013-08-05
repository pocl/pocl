// -*-C++-*-

#ifndef VEC_SSE_DOUBLE1_H
#define VEC_SSE_DOUBLE1_H

#include "floatprops.h"
#include "mathfuncs.h"
#include "vec_base.h"

#include <cmath>
#include <climits>

// SSE2 intrinsics
#include <emmintrin.h>
#ifdef __SSE3__                 // Intel's SSE 3
#  include <pmmintrin.h>
#endif
#ifdef __SSE4_1__               // Intel's SSE 4.1
#  include <smmintrin.h>
#endif
#ifdef __SSE4A__                // AMD's SSE 4a
#  include <ammintrin.h>
#endif
#if defined __AVX__             // Intel's AVX
#  include <immintrin.h>
#endif



namespace vecmathlib {
  
#define VECMATHLIB_HAVE_VEC_DOUBLE_1
  template<> struct boolvec<double,1>;
  template<> struct intvec<double,1>;
  template<> struct realvec<double,1>;
  
  
  
  template<>
  struct boolvec<double,1>: floatprops<double>
  {
    static int const size = 1;
    typedef bool scalar_t;
    typedef uint_t bvector_t;
    static int const alignment = sizeof(bvector_t);
    
    static_assert(size * sizeof(real_t) == sizeof(bvector_t),
                  "vector size is wrong");
    
    // true values are non-zero, false values are zero
    
    typedef boolvec boolvec_t;
    typedef intvec<real_t, size> intvec_t;
    typedef realvec<real_t, size> realvec_t;
    
    // Short names for type casts
    typedef real_t R;
    typedef int_t I;
    typedef uint_t U;
    typedef realvec_t RV;
    typedef intvec_t IV;
    typedef boolvec_t BV;
    typedef floatprops<real_t> FP;
    typedef mathfuncs<realvec_t> MF;
    
    
    
    bvector_t v;
    
    boolvec() {}
    // Can't have a non-trivial copy constructor; if so, objects won't
    // be passed in registers
    // boolvec(boolvec const& x): v(x.v) {}
    // boolvec& operator=(boolvec const& x) { return v=x.v, *this; }
    boolvec(bvector_t x): v(x) {}
    boolvec(bool a): v(a) {}
    boolvec(bool const* as): v(as[0]) {}
    
    operator bvector_t() const { return v; }
    bool operator[](int n) const { return v; }
    boolvec& set_elt(int n, bool a) { return v=a, *this; }
    
    
    
    intvec_t as_int() const;      // defined after intvec
    intvec_t convert_int() const; // defined after intvec
    
    
    
    boolvec operator!() const { return !v; }
    
    boolvec operator&&(boolvec x) const { return v && x.v; }
    boolvec operator||(boolvec x) const { return v || x.v; }
    boolvec operator==(boolvec x) const { return bool(v) == bool(x.v); }
    boolvec operator!=(boolvec x) const { return bool(v) != bool(x.v); }
    
    bool all() const { return *this; }
    bool any() const { return *this; }
    
    
    
    // ifthen(condition, then-value, else-value)
    boolvec_t ifthen(boolvec_t x, boolvec_t y) const;
    intvec_t ifthen(intvec_t x, intvec_t y) const; // defined after intvec
    realvec_t ifthen(realvec_t x, realvec_t y) const; // defined after realvec
  };
  
  
  
  template<>
  struct intvec<double,1>: floatprops<double>
  {
    static int const size = 1;
    typedef int_t scalar_t;
    typedef int_t ivector_t;
    static int const alignment = sizeof(ivector_t);
    
    static_assert(size * sizeof(real_t) == sizeof(ivector_t),
                  "vector size is wrong");
    
    typedef boolvec<real_t, size> boolvec_t;
    typedef intvec intvec_t;
    typedef realvec<real_t, size> realvec_t;
    
    // Short names for type casts
    typedef real_t R;
    typedef int_t I;
    typedef uint_t U;
    typedef realvec_t RV;
    typedef intvec_t IV;
    typedef boolvec_t BV;
    typedef floatprops<real_t> FP;
    typedef mathfuncs<realvec_t> MF;
    
    
    
    ivector_t v;
    
    intvec() {}
    // Can't have a non-trivial copy constructor; if so, objects won't
    // be passed in registers
    // intvec(intvec const& x): v(x.v) {}
    // intvec& operator=(intvec const& x) { return v=x.v, *this; }
    intvec(int_t a): v(a) {}
    intvec(int_t const* as): v(as[0]) {}
    static intvec iota() { return intvec(I(0)); }
    
    operator ivector_t() const { return v; }
    int_t operator[](int n) const { return v; }
    intvec& set_elt(int n, int_t a) { return v=a, *this; }
    
    
    
    boolvec_t as_bool() const { return U(v); }
    boolvec_t convert_bool() const { return bool(v); }
    realvec_t as_float() const;      // defined after realvec
    realvec_t convert_float() const; // defined after realvec
    
    
    
    intvec operator+() const { return +v; }
    intvec operator-() const { return -v; }
    
    intvec operator+(intvec x) const { return v+x.v; }
    intvec operator-(intvec x) const { return v-x.v; }
    intvec operator*(intvec x) const { return v*x.v; }
    intvec operator/(intvec x) const { return v/x.v; }
    intvec operator%(intvec x) const { return v%x.v; }
    
    intvec& operator+=(intvec const& x) { return *this=*this+x; }
    intvec& operator-=(intvec const& x) { return *this=*this-x; }
    intvec& operator*=(intvec const& x) { return *this=*this*x; }
    intvec& operator/=(intvec const& x) { return *this=*this/x; }
    intvec& operator%=(intvec const& x) { return *this=*this%x; }
    
    
    
    intvec operator~() const { return ~v; }
    
    intvec operator&(intvec x) const { return v&x.v; }
    intvec operator|(intvec x) const { return v|x.v; }
    intvec operator^(intvec x) const { return v^x.v; }
    
    intvec& operator&=(intvec const& x) { return *this=*this&x; }
    intvec& operator|=(intvec const& x) { return *this=*this|x; }
    intvec& operator^=(intvec const& x) { return *this=*this^x; }
    
    
    
    intvec lsr(int_t n) const { return U(v) >> U(n); }
    intvec operator>>(int_t n) const { return v>>n; }
    intvec operator<<(int_t n) const { return v<<n; }
    
    intvec& operator>>=(int_t n) { return *this=*this>>n; }
    intvec& operator<<=(int_t n) { return *this=*this<<n; }
    
    intvec lsr(intvec n) const { return U(v) >> U(n); }
    intvec operator>>(intvec n) const { return v>>n; }
    intvec operator<<(intvec n) const { return v<<n; }
    
    intvec& operator>>=(intvec n) { return *this=*this>>n; }
    intvec& operator<<=(intvec n) { return *this=*this<<n; }
    
    
    
    boolvec_t signbit() const { return *this < IV(I(0)); }
    
    boolvec_t operator==(intvec const& x) const { return v==x.v; }
    boolvec_t operator!=(intvec const& x) const { return v!=x.v; }
    boolvec_t operator<(intvec const& x) const { return v<x.v; }
    boolvec_t operator<=(intvec const& x) const { return v<=x.v; }
    boolvec_t operator>(intvec const& x) const { return v>x.v; }
    boolvec_t operator>=(intvec const& x) const { return v>=x.v; }
  };
  
  
  
  template<>
  struct realvec<double,1>: floatprops<double>
  {
    static int const size = 1;
    typedef real_t scalar_t;
    typedef double vector_t;
    static int const alignment = sizeof(vector_t);
    
    static char const* name() { return "<SSE2:1*double>"; }
    void barrier() { __asm__("": "+x"(v)); }
    
    static_assert(size * sizeof(real_t) == sizeof(vector_t),
                  "vector size is wrong");
    
  private:
    static __m128d from_double(double a) { return _mm_set_sd(a); }
    static double to_double(__m128d a) { return _mm_cvtsd_f64(a); }
  public:
    
    typedef boolvec<real_t, size> boolvec_t;
    typedef intvec<real_t, size> intvec_t;
    typedef realvec realvec_t;
    
    // Short names for type casts
    typedef real_t R;
    typedef int_t I;
    typedef uint_t U;
    typedef realvec_t RV;
    typedef intvec_t IV;
    typedef boolvec_t BV;
    typedef floatprops<real_t> FP;
    typedef mathfuncs<realvec_t> MF;
    
    
    
    vector_t v;
    
    realvec() {}
    // Can't have a non-trivial copy constructor; if so, objects won't
    // be passed in registers
    // realvec(realvec const& x): v(x.v) {}
    // realvec& operator=(realvec const& x) { return v=x.v, *this; }
    realvec(real_t a): v(a) {}
    realvec(real_t const* as): v(as[0]) {}
    
    operator vector_t() const { return v; }
    real_t operator[](int n) const { return v; }
    realvec& set_elt(int n, real_t a) { return v=a, *this; }
    
    
    
    typedef vecmathlib::mask_t<realvec_t> mask_t;
    
    static realvec_t loada(real_t const* p)
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      return *p;
    }
    static realvec_t loadu(real_t const* p)
    {
      return *p;
    }
    static realvec_t loadu(real_t const* p, std::ptrdiff_t ioff)
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      return loada(p+ioff);
    }
    realvec_t loada(real_t const* p, mask_t const& m) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      if (__builtin_expect(all(m.m), true)) {
        return loada(p);
      } else {
        return *this;
      }
    }
    realvec_t loadu(real_t const* p, mask_t const& m) const
    {
      if (__builtin_expect(m.all_m, true)) {
        return loadu(p);
      } else {
        return *this;
      }
    }
    realvec_t loadu(real_t const* p, std::ptrdiff_t ioff, mask_t const& m) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      return loada(p+ioff, m);
    }
    
    void storea(real_t* p) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      *p = v;
    }
    void storeu(real_t* p) const
    {
      *p = v;
    }
    void storeu(real_t* p, std::ptrdiff_t ioff) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      storea(p+ioff);
    }
    void storea(real_t* p, mask_t const& m) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      if (__builtin_expect(m.all_m, true)) {
        storea(p);
      }
    }
    void storeu(real_t* p, mask_t const& m) const
    {
      if (__builtin_expect(m.all_m, true)) {
        storeu(p);
      }
    }
    void storeu(real_t* p, std::ptrdiff_t ioff, mask_t const& m) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      storea(p+ioff, m);
    }
    
    
    
    intvec_t as_int() const { return floatprops::as_int(v); }
    intvec_t convert_int() const {
#ifdef __x86_64__
      return _mm_cvttsd_si64(_mm_set_sd(v));
#else
      return floatprops::convert_int(v);
#endif
    }
    
    
    
    realvec operator+() const { return +v; }
    realvec operator-() const { return -v; }
    
    realvec operator+(realvec x) const { return v+x.v; }
    realvec operator-(realvec x) const { return v-x.v; }
    realvec operator*(realvec x) const { return v*x.v; }
    realvec operator/(realvec x) const { return v/x.v; }
    
    realvec& operator+=(realvec const& x) { return *this=*this+x; }
    realvec& operator-=(realvec const& x) { return *this=*this-x; }
    realvec& operator*=(realvec const& x) { return *this=*this*x; }
    realvec& operator/=(realvec const& x) { return *this=*this/x; }
    
    real_t maxval() const { return *this; }
    real_t minval() const { return *this; }
    real_t prod() const { return *this; }
    real_t sum() const { return *this; }
    
    
    
    boolvec_t operator==(realvec const& x) const { return v==x.v; }
    boolvec_t operator!=(realvec const& x) const { return v!=x.v; }
    boolvec_t operator<(realvec const& x) const { return v<x.v; }
    boolvec_t operator<=(realvec const& x) const { return v<=x.v; }
    boolvec_t operator>(realvec const& x) const { return v>x.v; }
    boolvec_t operator>=(realvec const& x) const { return v>=x.v; }
    
    
    
    realvec acos() const { return MF::vml_acos(*this); }
    realvec acosh() const { return MF::vml_acosh(*this); }
    realvec asin() const { return MF::vml_asin(*this); }
    realvec asinh() const { return MF::vml_asinh(*this); }
    realvec atan() const { return MF::vml_atan(*this); }
    realvec atan2(realvec y) const { return MF::vml_atan2(*this, y); }
    realvec atanh() const { return MF::vml_atanh(*this); }
    realvec cbrt() const { return MF::vml_cbrt(*this); }
    realvec ceil() const
    {
#ifdef __SSE4_1__
      return to_double(_mm_ceil_sd(from_double(v), from_double(v)));
#else
      return std::ceil(v);
#endif
    }
    realvec copysign(realvec y) const { return std::copysign(v, y.v); }
    realvec cos() const { return MF::vml_cos(*this); }
    realvec cosh() const { return MF::vml_cosh(*this); }
    realvec exp() const { return MF::vml_exp(*this); }
    realvec exp10() const { return MF::vml_exp10(*this); }
    realvec exp2() const { return MF::vml_exp2(*this); }
    realvec expm1() const { return MF::vml_expm1(*this); }
    realvec fabs() const { return std::fabs(v); }
    realvec fdim(realvec y) const { return MF::vml_fdim(*this, y); }
    realvec floor() const
    {
#ifdef __SSE4_1__
      return to_double(_mm_floor_sd(from_double(v), from_double(v)));
#else
      return std::floor(v);
#endif
    }
    realvec fma(realvec y, realvec z) const { return MF::vml_fma(*this, y, z); }
    realvec fmax(realvec y) const
    {
      return to_double(_mm_max_sd(from_double(v), from_double(y.v)));
    }
    realvec fmin(realvec y) const
    {
      return to_double(_mm_min_sd(from_double(v), from_double(y.v)));
    }
    realvec fmod(realvec y) const { return std::fmod(v, y.v); }
    realvec frexp(intvec_t& ir) const
    {
      int iri;
      realvec r = std::frexp(v, &iri);
      ir.v = iri;
      if (isinf()) ir.v = std::numeric_limits<int_t>::max();
      if (isnan()) ir.v = std::numeric_limits<int_t>::min();
      return r;
    }
    realvec hypot(realvec y) const { return MF::vml_hypot(*this, y); }
    intvec_t ilogb() const
    {
      int_t r = std::ilogb(v);
      typedef std::numeric_limits<int_t> NL;
      if (FP_ILOGB0 != NL::min() and v == R(0.0)) {
        r = NL::min();
#if defined VML_HAVE_INF
      } else if (INT_MAX != NL::max() and std::isinf(v)) {
        r = NL::max();
#endif
#if defined VML_HAVE_NAN
      } else if (FP_ILOGBNAN != NL::min() and std::isnan(v)) {
        r = NL::min();
#endif
      }
      return r;
    }
    boolvec_t isfinite() const { return std::isfinite(v); }
    boolvec_t isinf() const { return std::isinf(v); }
    boolvec_t isnan() const
    {
      // This is wrong:
      // return _mm_ucomineq_sd(from_double(v), from_double(v));
      // This works:
      // char r;
      // __asm__("ucomisd %[v],%[v]; setp %[r]": [r]"=q"(r): [v]"x"(v));
      // return boolvec_t::scalar_t(r);
      // This works as well:
      return std::isnan(v);
    }
    boolvec_t isnormal() const { return std::isnormal(v); }
    realvec ldexp(int_t n) const { return std::ldexp(v, n); }
    realvec ldexp(intvec_t n) const { return std::ldexp(v, n); }
    realvec log() const { return MF::vml_log(*this); }
    realvec log10() const { return MF::vml_log10(*this); }
    realvec log1p() const { return MF::vml_log1p(*this); }
    realvec log2() const { return MF::vml_log2(*this); }
    realvec nextafter(realvec y) const { return MF::vml_nextafter(*this, y); }
    realvec pow(realvec y) const { return MF::vml_pow(*this, y); }
    realvec rcp() const { return R(1.0)/v; }
    realvec remainder(realvec y) const { return std::remainder(v, y.v); }
    realvec rint() const
    {
#ifdef __SSE4_1__
      return to_double(_mm_round_sd(from_double(v), from_double(v),
                                    _MM_FROUND_TO_NEAREST_INT));
#else
      return MF::vml_rint(*this);
#endif
    }
    realvec round() const { return MF::vml_round(*this); }
    realvec rsqrt() const { return MF::vml_rsqrt(*this); }
    boolvec_t signbit() const { return std::signbit(v); }
    realvec sin() const { return MF::vml_sin(*this); }
    realvec sinh() const { return MF::vml_sinh(*this); }
    realvec sqrt() const
    {
      return to_double(_mm_sqrt_sd(from_double(v), from_double(v)));
    }
    realvec tan() const { return MF::vml_tan(*this); }
    realvec tanh() const { return MF::vml_tanh(*this); }
    realvec trunc() const
    {
#ifdef __SSE4_1__
      return to_double(_mm_round_sd(from_double(v), from_double(v),
                                    _MM_FROUND_TO_ZERO));
#else
      return MF::vml_trunc(*this);
#endif
    }
  };
  
  
  
  // boolvec definitions
  
  inline intvec<double,1> boolvec<double,1>::as_int() const
  {
    return I(v);
  }
  
  inline intvec<double,1> boolvec<double,1>::convert_int() const
  {
    return v;
  }
  
  inline
  boolvec<double,1> boolvec<double,1>::ifthen(boolvec_t x, boolvec_t y) const
  {
    return v ? x : y;
  }
  
  inline
  intvec<double,1> boolvec<double,1>::ifthen(intvec_t x, intvec_t y) const
  {
    return v ? x : y;
  }
  
  inline
  realvec<double,1> boolvec<double,1>::ifthen(realvec_t x, realvec_t y) const
  {
    return v ? x : y;
  }
  
  
  
  // intvec definitions
  
  inline realvec<double,1> intvec<double,1>::as_float() const
  {
    return FP::as_float(v);
  }
  
  inline realvec<double,1> intvec<double,1>::convert_float() const
  {
#ifdef __x86_64__
    return _mm_cvtsd_f64(_mm_cvtsi64_sd(_mm_setzero_pd(), v));
#else
    return FP::convert_float(v);
#endif
  }
  
} // namespace vecmathlib

#endif  // #ifndef VEC_SSE_DOUBLE1_H
