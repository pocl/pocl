// -*-C++-*-

#ifndef VEC_ALTIVEC_FLOAT4_H
#define VEC_ALTIVEC_FLOAT4_H

#include "floatprops.h"
#include "mathfuncs.h"
#include "vec_base.h"

#include <cmath>

// Altivec intrinsics
#include <altivec.h>
#ifdef __clang__
#  define __vector vector
#  define __pixel pixel
#  define __bool bool
#else
#  undef vector
#  undef pixel
#  undef bool
#endif



namespace vecmathlib {
  
#define VECMATHLIB_HAVE_VEC_FLOAT_4
  template<> struct boolvec<float,4>;
  template<> struct intvec<float,4>;
  template<> struct realvec<float,4>;
  
  
  
  template<>
  struct boolvec<float,4>: floatprops<float>
  {
    static int const size = 4;
    typedef bool scalar_t;
    typedef __vector __bool int bvector_t;
    static int const alignment = sizeof(bvector_t);
    
    static_assert(size * sizeof(real_t) == sizeof(bvector_t),
                  "vector size is wrong");
    
  private:
    // true values are -1, false values are 0
    static uint_t from_bool(bool a) { return -int_t(a); }
    static bool to_bool(uint_t a) { return a; }
  public:
    
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
    boolvec(bool a): v(vec_splats(from_bool(a))) {}
    boolvec(bool const* as)
    {
      for (int d=0; d<size; ++d) set_elt(d, as[d]);
    }
    
    operator bvector_t() const { return v; }
    bool operator[](int n) const
    {
      return to_bool(vecmathlib::get_elt<BV,bvector_t,uint_t>(v, n));
    }
    boolvec& set_elt(int n, bool a)
    {
      return
        vecmathlib::set_elt<BV,bvector_t,uint_t>(v, n, from_bool(a)), *this;
    }
    
    
    
    intvec_t as_int() const;      // defined after intvec
    intvec_t convert_int() const; // defined after intvec
    
    
    
    boolvec operator!() const { return vec_nor(v, v); }
    
    boolvec operator&&(boolvec x) const { return vec_and(v, x.v); }
    boolvec operator||(boolvec x) const { return vec_or(v, x.v); }
    // boolvec operator==(boolvec x) const { return !(*this!=x); }
    boolvec operator==(boolvec x) const; // defined after intvec
    boolvec operator!=(boolvec x) const { return vec_xor(v, x.v); }
    
    bool all() const { return vec_all_ne(v, BV(false).v); }
    bool any() const { return vec_any_ne(v, BV(false).v); }
    
    
    
    // ifthen(condition, then-value, else-value)
    boolvec_t ifthen(boolvec_t x, boolvec_t y) const;
    intvec_t ifthen(intvec_t x, intvec_t y) const; // defined after intvec
    realvec_t ifthen(realvec_t x, realvec_t y) const; // defined after realvec
  };
  
  
  
  template<>
  struct intvec<float,4>: floatprops<float>
  {
    static int const size = 4;
    typedef int_t scalar_t;
    typedef __vector int ivector_t;
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
    intvec(ivector_t x): v(x) {}
    intvec(int_t a): v(vec_splats(a)) {}
    intvec(int_t const* as)
    {
      for (int d=0; d<size; ++d) set_elt(d, as[d]);
    }
    static intvec iota() { return (__vector int){0, 1, 2, 3}; }
    
    operator ivector_t() const { return v; }
    int_t operator[](int n) const
    {
      return vecmathlib::get_elt<IV,ivector_t,int_t>(v, n);
    }
    intvec_t& set_elt(int n, int_t a)
    {
      return vecmathlib::set_elt<IV,ivector_t,int_t>(v, n, a), *this;
    }
    
    
    
    // Vector casts do not change the bit battern
    boolvec_t as_bool() const { return (__vector __bool int)v; }
    boolvec_t convert_bool() const { return *this != IV(0); }
    realvec_t as_float() const;      // defined after realvec
    realvec_t convert_float() const; // defined after realvec
    
    
    
    intvec operator+() const { return *this; }
    intvec operator-() const { return IV(0) - *this; }
    
    intvec operator+(intvec x) const { return vec_add(v, x.v); }
    intvec operator-(intvec x) const { return vec_sub(v, x.v); }
    
    intvec& operator+=(intvec const& x) { return *this=*this+x; }
    intvec& operator-=(intvec const& x) { return *this=*this-x; }
    
    
    
    intvec operator~() const { return vec_nor(v, v); }
    
    intvec operator&(intvec x) const { return vec_and(v, x.v); }
    intvec operator|(intvec x) const { return vec_or(v, x.v); }
    intvec operator^(intvec x) const { return vec_xor(v, x.v); }
    
    intvec& operator&=(intvec const& x) { return *this=*this&x; }
    intvec& operator|=(intvec const& x) { return *this=*this|x; }
    intvec& operator^=(intvec const& x) { return *this=*this^x; }
    
    
    
    intvec lsr(int_t n) const { return lsr(IV(n)); }
    intvec operator>>(int_t n) const { return *this >> IV(n); }
    intvec operator<<(int_t n) const { return *this << IV(n); }
    intvec& operator>>=(int_t n) { return *this=*this>>n; }
    intvec& operator<<=(int_t n) { return *this=*this<<n; }
    
    intvec lsr(intvec n) const
    {
      return vec_sr(v, (__vector unsigned int)n.v);
    }
    intvec operator>>(intvec n) const
    {
      return vec_sra(v, (__vector unsigned int)n.v);
    }
    intvec operator<<(intvec n) const
    {
      return vec_sl(v, (__vector unsigned int)n.v);
    }
    intvec& operator>>=(intvec n) { return *this=*this>>n; }
    intvec& operator<<=(intvec n) { return *this=*this<<n; }
    
    
    
    boolvec_t signbit() const
    {
      return *this < IV(I(0));
    }
    
    boolvec_t operator==(intvec const& x) const { return vec_cmpeq(v, x.v); }
    boolvec_t operator!=(intvec const& x) const { return !(*this == x); }
    boolvec_t operator<(intvec const& x) const { return vec_cmplt(v, x.v); }
    boolvec_t operator<=(intvec const& x) const { return !(*this > x); }
    boolvec_t operator>(intvec const& x) const { return vec_cmpgt(v, x.v); }
    boolvec_t operator>=(intvec const& x) const { return !(*this < x); }
  };
  
  
  
  template<>
  struct realvec<float,4>: floatprops<float>
  {
    static int const size = 4;
    typedef real_t scalar_t;
    typedef __vector float vector_t;
    static int const alignment = sizeof(vector_t);
    
    static char const* name() { return "<Altivec:4*float>"; }
    void barrier() { __asm__("": "+v"(v)); }
    
    static_assert(size * sizeof(real_t) == sizeof(vector_t),
                  "vector size is wrong");
    
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
    realvec(vector_t x): v(x) {}
    realvec(real_t a): v(vec_splats(a)) {}
    realvec(real_t const* as)
    {
      for (int d=0; d<size; ++d) set_elt(d, as[d]);
    }
    
    operator vector_t() const { return v; }
    real_t operator[](int n) const
    {
      return vecmathlib::get_elt<RV,vector_t,real_t>(v, n);
    }
    realvec_t& set_elt(int n, real_t a)
    {
      return vecmathlib::set_elt<RV,vector_t,real_t>(v, n, a), *this;
    }
    
    
    
    typedef vecmathlib::mask_t<realvec_t> mask_t;
    
    static realvec_t loada(real_t const* p)
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      return vec_ld(0, p);
    }
    static realvec_t loadu(real_t const* p)
    {
      realvec_t v0 = vec_ld(0, p);
      realvec_t v1 = vec_ld(15, p);
      return vec_perm(v0.v, v1.v, vec_lvsl(0, p));
    }
    static realvec_t loadu(real_t const* p, std::ptrdiff_t ioff)
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      if (ioff % realvec::size == 0) return loada(p+ioff);
      return loadu(p+ioff);
    }
    realvec_t loada(real_t const* p, mask_t const& m) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      if (__builtin_expect(all(m.m), true)) {
        return loada(p);
      } else {
        return m.m.ifthen(loada(p), *this);
      }
    }
    realvec_t loadu(real_t const* p, mask_t const& m) const
    {
      if (__builtin_expect(m.all_m, true)) {
        return loadu(p);
      } else {
        return m.m.ifthen(loadu(p), *this);
      }
    }
    realvec_t loadu(real_t const* p, std::ptrdiff_t ioff, mask_t const& m) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      if (ioff % realvec::size == 0) return loada(p+ioff, m);
      return loadu(p+ioff, m);
    }
    
    void storea(real_t* p) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      vec_st(v, 0, p);
    }
    void storeu(real_t* p) const
    {
      // Vector stores would require vector loads, which would need to
      // be atomic
      // TODO: see <https://developer.apple.com/hardwaredrivers/ve/alignment.html> for good ideas
      p[0] = (*this)[0];
      p[1] = (*this)[1];
      p[2] = (*this)[2];
      p[3] = (*this)[3];
    }
    void storeu(real_t* p, std::ptrdiff_t ioff) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      if (ioff % realvec::size == 0) return storea(p+ioff);
      storeu(p+ioff);
    }
    void storea(real_t* p, mask_t const& m) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      if (__builtin_expect(m.all_m, true)) {
        storea(p);
      } else {
	// Use vec_ste?
        if (m.m[0]) p[0] = (*this)[0];
        if (m.m[1]) p[1] = (*this)[1];
        if (m.m[2]) p[2] = (*this)[2];
        if (m.m[3]) p[3] = (*this)[3];
      }
    }
    void storeu(real_t* p, mask_t const& m) const
    {
      if (__builtin_expect(m.all_m, true)) {
        storeu(p);
      } else {
	// Use vec_ste?
        if (m.m[0]) p[0] = (*this)[0];
        if (m.m[1]) p[1] = (*this)[1];
        if (m.m[2]) p[2] = (*this)[2];
        if (m.m[3]) p[3] = (*this)[3];
      }
    }
    void storeu(real_t* p, std::ptrdiff_t ioff, mask_t const& m) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      if (ioff % realvec::size == 0) return storea(p+ioff, m);
      storeu(p+ioff, m);
    }
    
    
    
    intvec_t as_int() const { return (__vector int) v; }
    intvec_t convert_int() const { return vec_cts(v, 0); }
    
    
    
    realvec operator+() const { return *this; }
    realvec operator-() const { return RV(0.0) - *this; }
    
    realvec operator+(realvec x) const { return vec_add(v, x.v); }
    realvec operator-(realvec x) const { return vec_sub(v, x.v); }
    realvec operator*(realvec x) const {
#if defined __VSX__
      return vec_mul(v, x.v);
#else
      return vec_madd(v, x.v, RV(0.0).v);
#endif
    }
    realvec operator/(realvec x) const {
#if defined __VSX__
      return vec_div(v, x.v);
#else
      return *this * x.rcp();
#endif
    }
    
    realvec& operator+=(realvec const& x) { return *this=*this+x; }
    realvec& operator-=(realvec const& x) { return *this=*this-x; }
    realvec& operator*=(realvec const& x) { return *this=*this*x; }
    realvec& operator/=(realvec const& x) { return *this=*this/x; }
    
    real_t maxval() const
    {
      return std::fmax(std::fmax((*this)[0], (*this)[1]),
                       std::fmax((*this)[2], (*this)[3]));
    }
    real_t minval() const
    {
      return std::fmin(std::fmin((*this)[0], (*this)[1]),
                       std::fmin((*this)[2], (*this)[3]));
    }
    real_t prod() const
    {
      return (*this)[0] * (*this)[1] * (*this)[2] * (*this)[3];
    }
    real_t sum() const
    {
      return (*this)[0] + (*this)[1] + (*this)[2] + (*this)[3];
    }
    
    
    
    boolvec_t operator==(realvec const& x) const { return vec_cmpeq(v, x.v); }
    boolvec_t operator!=(realvec const& x) const { return ! (*this == x); }
    boolvec_t operator<(realvec const& x) const { return vec_cmplt(v, x.v); }
    boolvec_t operator<=(realvec const& x) const { return vec_cmple(v, x.v); }
    boolvec_t operator>(realvec const& x) const { return vec_cmpgt(v, x.v); }
    boolvec_t operator>=(realvec const& x) const { return vec_cmpge(v, x.v); }
    
    
    
    realvec acos() const { return MF::vml_acos(*this); }
    realvec acosh() const { return MF::vml_acosh(*this); }
    realvec asin() const { return MF::vml_asin(*this); }
    realvec asinh() const { return MF::vml_asinh(*this); }
    realvec atan() const { return MF::vml_atan(*this); }
    realvec atan2(realvec y) const { return MF::vml_atan2(*this, y); }
    realvec atanh() const { return MF::vml_atanh(*this); }
    realvec cbrt() const { return MF::vml_cbrt(*this); }
    realvec ceil() const { return vec_ceil(v); }
    realvec copysign(realvec y) const { return MF::vml_copysign(*this, y); }
    realvec cos() const { return MF::vml_cos(*this); }
    realvec cosh() const { return MF::vml_cosh(*this); }
    realvec exp() const { return MF::vml_exp(*this); }
    realvec exp10() const { return MF::vml_exp10(*this); }
    realvec exp2() const { return MF::vml_exp2(*this); }
    realvec expm1() const { return MF::vml_expm1(*this); }
    realvec fabs() const { return vec_abs(v); }
    realvec fdim(realvec y) const { return MF::vml_fdim(*this, y); }
    realvec floor() const { return vec_floor(v); }
    realvec fma(realvec y, realvec z) const { return vec_madd(v, y.v, z.v); }
    realvec fmax(realvec y) const { return vec_max(v, y.v); }
    realvec fmin(realvec y) const { return vec_min(v, y.v); }
    realvec fmod(realvec y) const { return MF::vml_fmod(*this, y); }
    realvec frexp(intvec_t& r) const { return MF::vml_frexp(*this, r); }
    realvec hypot(realvec y) const { return MF::vml_hypot(*this, y); }
    intvec_t ilogb() const { return MF::vml_ilogb(*this); }
    boolvec_t isfinite() const { return MF::vml_isfinite(*this); }
    boolvec_t isinf() const { return MF::vml_isinf(*this); }
    boolvec_t isnan() const { return MF::vml_isnan(*this); }
    boolvec_t isnormal() const { return MF::vml_isnormal(*this); }
    realvec ldexp(int_t n) const { return MF::vml_ldexp(*this, n); }
    realvec ldexp(intvec_t n) const { return MF::vml_ldexp(*this, n); }
    realvec log() const { return MF::vml_log(*this); }
    realvec log10() const { return MF::vml_log10(*this); }
    realvec log1p() const { return MF::vml_log1p(*this); }
    realvec log2() const { return MF::vml_log2(*this); }
    realvec nextafter(realvec y) const { return MF::vml_nextafter(*this, y); }
    realvec pow(realvec y) const { return MF::vml_pow(*this, y); }
    realvec rcp() const
    {
      realvec x = *this;
      realvec r = vec_re(v);    // this is only an approximation
      // TODO: use fma
      // Note: don't rewrite this expression, this may introduce
      // cancellation errors
      r += r * (RV(1.0) - x*r); // one Newton iteration (see vml_rcp)
      return r;
    }
    realvec remainder(realvec y) const { return MF::vml_remainder(*this, y); }
    realvec rint() const { return vec_round(v); }
    realvec round() const { return MF::vml_round(*this); }
    realvec rsqrt() const
    {
#if defined __VSX__
      return vec_rsqrt(v);
#else
      realvec x = *this;
      realvec r = vec_rsqrte(x.v); // this is only an approximation
      // TODO: use fma
      // one Newton iteration (see vml_rsqrt)
      r += RV(0.5)*r * (RV(1.0) - x * r*r);
      return r;
#endif
    }
    boolvec_t signbit() const { return MF::vml_signbit(*this); }
    realvec sin() const { return MF::vml_sin(*this); }
    realvec sinh() const { return MF::vml_sinh(*this); }
    realvec sqrt() const {
#if defined __VSX__
      return vec_sqrt(v);
#else
      return *this * rsqrt();
#endif
    }
    realvec tan() const { return MF::vml_tan(*this); }
    realvec tanh() const { return MF::vml_tanh(*this); }
    realvec trunc() const { return vec_trunc(v); }
  };
  
  
  
  // boolvec definitions
  
  inline intvec<float,4> boolvec<float,4>::as_int() const
  {
    return (__vector int) v;
  }
  
  inline intvec<float,4> boolvec<float,4>::convert_int() const
  {
    return -(__vector int)v;
  }
  
  inline boolvec<float,4> boolvec<float,4>::operator==(boolvec_t x) const
  {
    return as_int() == x.as_int();
  }
  
  inline
  boolvec<float,4> boolvec<float,4>::ifthen(boolvec_t x, boolvec_t y) const
  {
    return vec_sel(y.v, x.v, v);
  }
  
  inline
  intvec<float,4> boolvec<float,4>::ifthen(intvec_t x, intvec_t y) const
  {
    return vec_sel(y.v, x.v, v);
  }
  
  inline
  realvec<float,4> boolvec<float,4>::ifthen(realvec_t x, realvec_t y) const
  {
    return vec_sel(y.v, x.v, v);
  }
  
  
  
  // intvec definitions
  
  inline realvec<float,4> intvec<float,4>::as_float() const
  {
    return (__vector float)v;
  }
  
  inline realvec<float,4> intvec<float,4>::convert_float() const
  {
    return vec_ctf(v, 0);
  }
  
} // namespace vecmathlib

#endif  // #ifndef VEC_ALTIVEC_FLOAT4_H
