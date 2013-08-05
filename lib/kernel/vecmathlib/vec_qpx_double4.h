// -*-C++-*-

#ifndef VEC_QPX_DOUBLE4_H
#define VEC_QPX_DOUBLE4_H

#include "floatprops.h"
#include "mathfuncs.h"
#include "vec_base.h"

#include <cmath>
#warning "TODO"
#include <iostream>

// QPX intrinsics
#ifdef __clang__
#  include <qpxintrin.h>
#else
#  include <builtins.h>
#endif
#include <mass_simd.h>



namespace vecmathlib {
  
#define VECMATHLIB_HAVE_VEC_DOUBLE_4
  template<> struct boolvec<double,4>;
  template<> struct intvec<double,4>;
  template<> struct realvec<double,4>;
  
  
  
  template<>
  struct boolvec<double,4>: floatprops<double>
  {
    static int const size = 4;
    typedef bool scalar_t;
    typedef vector4double bvector_t;
    static int const alignment = sizeof(bvector_t);
    
    static_assert(size * sizeof(real_t) == sizeof(bvector_t),
                  "vector size is wrong");
    
  private:
    // canonical true is +1.0, canonical false is -1.0
    // >=0 is true, -0 is true, nan is false
    static real_t from_bool(bool a) { return a ? +1.0 : -1.0; }
    static bool to_bool(real_t a) { return a>=0.0; }
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
      return to_bool(v[n]);
    }
    boolvec& set_elt(int n, bool a)
    {
      return v[n]=from_bool(a), *this;
    }
    
    
    
    intvec_t as_int() const;      // defined after intvec
    intvec_t convert_int() const; // defined after intvec
    
    
    
    boolvec operator!() const { return vec_not(v); }
    
    boolvec operator&&(boolvec x) const { return vec_and(v, x.v); }
    boolvec operator||(boolvec x) const { return vec_or(v, x.v); }
    boolvec operator==(boolvec x) const { return vec_logical(v, x.v, 0x9); }
    boolvec operator!=(boolvec x) const { return vec_xor(v, x.v); }
    
    bool all() const
    {
      return (*this)[0] && (*this)[1] && (*this)[2] && (*this)[3];
    }
    bool any() const
    {
      return (*this)[0] || (*this)[1] || (*this)[2] || (*this)[3];
    }
    
    
    
    // ifthen(condition, then-value, else-value)
    boolvec_t ifthen(boolvec_t x, boolvec_t y) const;
    intvec_t ifthen(intvec_t x, intvec_t y) const; // defined after intvec
    realvec_t ifthen(realvec_t x, realvec_t y) const; // defined after realvec
  };
  
  
  
  template<>
  struct intvec<double,4>: floatprops<double>
  {
    static int const size = 4;
    typedef int_t scalar_t;
    typedef vector4double ivector_t;
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
    intvec(int_t a): v(vec_splats(FP::as_float(a))) {}
    intvec(int_t const* as)
    {
      for (int d=0; d<size; ++d) set_elt(d, as[d]);
    }
    static intvec iota()
    {
      const int_t iota_[] = {0, 1, 2, 3};
      return intvec(iota_);
    }
    
    operator ivector_t() const { return v; }
    int_t operator[](int n) const
    {
      return FP::as_int(v[n]);
    }
    intvec& set_elt(int n, int_t a)
    {
      return v[n]=FP::as_float(a), *this;
    }
    
    
    
    // Vector casts do not change the bit battern
    boolvec_t as_bool() const { return v; }
    boolvec_t convert_bool() const { return *this != IV(I(0)); }
    realvec_t as_float() const;      // defined after realvec
    realvec_t convert_float() const; // defined after realvec
    
    
    
    intvec operator+() const { return *this; }
    intvec operator-() const
    {
      intvec r;
      for (int d=0; d<size; ++d) r.set_elt(d, -(*this)[d]);
      return r;
    }
    
    intvec operator+(intvec x) const
    {
      intvec r;
      for (int d=0; d<size; ++d) r.set_elt(d, (*this)[d] + x[d]);
      return r;
    }
    intvec operator-(intvec x) const
    {
      intvec r;
      for (int d=0; d<size; ++d) r.set_elt(d, (*this)[d] - x[d]);
      return r;
    }
    
    intvec& operator+=(intvec const& x) { return *this=*this+x; }
    intvec& operator-=(intvec const& x) { return *this=*this-x; }
    
    
    
    intvec operator~() const
    {
      intvec r;
      for (int d=0; d<size; ++d) r.set_elt(d, ~(*this)[d]);
      return r;
    }
    
    intvec operator&(intvec x) const
    {
      intvec r;
      for (int d=0; d<size; ++d) r.set_elt(d, (*this)[d] & x[d]);
      return r;
    }
    intvec operator|(intvec x) const
    {
      intvec r;
      for (int d=0; d<size; ++d) r.set_elt(d, (*this)[d] | x[d]);
      return r;
    }
    intvec operator^(intvec x) const
    {
      intvec r;
      for (int d=0; d<size; ++d) r.set_elt(d, (*this)[d] ^ x[d]);
      return r;
    }
    
    intvec& operator&=(intvec const& x) { return *this=*this&x; }
    intvec& operator|=(intvec const& x) { return *this=*this|x; }
    intvec& operator^=(intvec const& x) { return *this=*this^x; }
    
    
    
    intvec lsr(int_t n) const
    {
      intvec r;
      for (int d=0; d<size; ++d) r.set_elt(d, U((*this)[d]) >> U(n));
      return r;
    }
    intvec operator>>(int_t n) const
    {
      intvec r;
      for (int d=0; d<size; ++d) r.set_elt(d, (*this)[d] >> n);
      return r;
    }
    intvec operator<<(int_t n) const
    {
      intvec r;
      for (int d=0; d<size; ++d) r.set_elt(d, (*this)[d] << n);
      return r;
    }
    intvec& operator>>=(int_t n) { return *this=*this>>n; }
    intvec& operator<<=(int_t n) { return *this=*this<<n; }
    
    intvec lsr(intvec n) const
    {
      intvec r;
      for (int d=0; d<size; ++d) r.set_elt(d, U((*this)[d]) >> U(n[d]));
      return r;
    }
    intvec operator>>(intvec n) const
    {
      intvec r;
      for (int d=0; d<size; ++d) r.set_elt(d, (*this)[d] >> n[d]);
      return r;
    }
    intvec operator<<(intvec n) const
    {
      intvec r;
      for (int d=0; d<size; ++d) r.set_elt(d, (*this)[d] << n[d]);
      return r;
    }
    intvec& operator>>=(intvec n) { return *this=*this>>n; }
    intvec& operator<<=(intvec n) { return *this=*this<<n; }
    
    
    
    boolvec_t signbit() const
    {
      return *this < IV(I(0));
    }
    
    boolvec_t operator==(intvec const& x) const
    {
      boolvec_t r;
      for (int d=0; d<size; ++d) r.set_elt(d, (*this)[d] == x[d]);
      return r;
    }
    boolvec_t operator!=(intvec const& x) const
    {
      boolvec_t r;
      for (int d=0; d<size; ++d) r.set_elt(d, (*this)[d] != x[d]);
      return r;
    }
    boolvec_t operator<(intvec const& x) const
    {
      boolvec_t r;
      for (int d=0; d<size; ++d) r.set_elt(d, (*this)[d] < x[d]);
      return r;
    }
    boolvec_t operator<=(intvec const& x) const
    {
      boolvec_t r;
      for (int d=0; d<size; ++d) r.set_elt(d, (*this)[d] <= x[d]);
      return r;
    }
    boolvec_t operator>(intvec const& x) const
    {
      boolvec_t r;
      for (int d=0; d<size; ++d) r.set_elt(d, (*this)[d] > x[d]);
      return r;
    }
    boolvec_t operator>=(intvec const& x) const
    {
      boolvec_t r;
      for (int d=0; d<size; ++d) r.set_elt(d, (*this)[d] >= x[d]);
      return r;
    }
  };
  
  
  
  template<>
  struct realvec<double,4>: floatprops<double>
  {
    static int const size = 4;
    typedef real_t scalar_t;
    typedef vector4double vector_t;
    static int const alignment = sizeof(vector_t);
    
    static char const* name() { return "<QPX:4*double>"; }
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
      return v[n];
    }
    realvec& set_elt(int n, real_t a)
    {
      return v[n]=a, *this;
    }
    
    
    
    typedef vecmathlib::mask_t<realvec_t> mask_t;
    
    static realvec_t loada(real_t const* p)
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      return vec_lda(0, (real_t*)p);
    }
    static realvec_t loadu(real_t const* p)
    {
      realvec_t v0 = vec_ld(0, (real_t*)p);
      realvec_t v1 = vec_ld(31, (real_t*)p);
      return vec_perm(v0.v, v1.v, vec_lvsl(0, (real_t*)p));
    }
    static realvec_t loadu(real_t const* p, std::ptrdiff_t ioff)
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      if (ioff % realvec::size == 0) return loada(p+ioff);
      // TODO: use load instruction with fixed offset
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
      // TODO: use load instruction with fixed offset
      return loadu(p+ioff, m);
    }
    
    void storea(real_t* p) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
#warning "TODO"
      std::cout << "yes this is storea\n";
      vec_sta(v, 0, p);
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
    
    
    
    intvec_t as_int() const { return v; }
    intvec_t convert_int() const { return vec_ctid(v); }
    
    
    
    realvec operator+() const { return *this; }
    realvec operator-() const { return vec_neg(v); }
    
    realvec operator+(realvec x) const { return vec_add(v, x.v); }
    realvec operator-(realvec x) const { return vec_sub(v, x.v); }
    realvec operator*(realvec x) const { return vec_mul(v, x.v); }
    realvec operator/(realvec x) const
    {
      // return vec_swdiv_nochk(v, x.v);
      return div_fastd4(v, x.v);
    }
    
    realvec& operator+=(realvec const& x) { return *this=*this+x; }
    realvec& operator-=(realvec const& x) { return *this=*this-x; }
    realvec& operator*=(realvec const& x) { return *this=*this*x; }
    realvec& operator/=(realvec const& x) { return *this=*this/x; }
    
    real_t maxval() const
    {
      // return std::fmax(std::fmax((*this)[0], (*this)[1]),
      //                  std::fmax((*this)[2], (*this)[3]));
      realvec_t x0123 = *this;
      realvec_t x1032 = vec_perm(x0123, x0123, vec_gpci(01032));
      realvec_t y0022 = x0123.fmax(x1032);
      return std::fmax(y0022[0], y0022[2]);
    }
    real_t minval() const
    {
      // return std::fmin(std::fmin((*this)[0], (*this)[1]),
      //                  std::fmin((*this)[2], (*this)[3]));
      realvec_t x0123 = *this;
      realvec_t x1032 = vec_perm(x0123, x0123, vec_gpci(01032));
      realvec_t y0022 = x0123.fmin(x1032);
      return std::fmin(y0022[0], y0022[2]);
    }
    real_t prod() const
    {
      // return (*this)[0] * (*this)[1] * (*this)[2] * (*this)[3];
      realvec_t x = vec_xmul(v, v);
      return x[1] * x[3];
    }
    real_t sum() const
    {
      // return (*this)[0] + (*this)[1] + (*this)[2] + (*this)[3];
      realvec_t c1 = vec_logical(v, v, 0xf);
      realvec_t x = vec_xxmadd(v, c1, v);
      return x[0] + x[2];
    }
    
    
    
    boolvec_t operator==(realvec const& x) const { return vec_cmpeq(v, x.v); }
    boolvec_t operator!=(realvec const& x) const { return ! (*this == x); }
    boolvec_t operator<(realvec const& x) const { return vec_cmplt(v, x.v); }
    boolvec_t operator<=(realvec const& x) const { return ! (*this > x); }
    boolvec_t operator>(realvec const& x) const { return vec_cmpgt(v, x.v); }
    boolvec_t operator>=(realvec const& x) const { return ! (*this < x); }
    
    
    
    realvec acos() const { return acosd4(v); }
    realvec acosh() const { return acoshd4(v); }
    realvec asin() const { return asind4(v); }
    realvec asinh() const { return asinhd4(v); }
    realvec atan() const { return atand4(v); }
    realvec atan2(realvec y) const { return atan2d4(v, y.v); }
    realvec atanh() const { return atanhd4(v); }
    realvec cbrt() const { return cbrtd4(v); }
    realvec ceil() const { return vec_ceil(v); }
    realvec copysign(realvec y) const { return vec_cpsgn(v, y.v); }
    realvec cos() const { return cosd4(v); }
    realvec cosh() const { return coshd4(v); }
    realvec exp() const { return expd4(v); }
    realvec exp10() const { return exp10d4(v); }
    realvec exp2() const { return exp2d4(v); }
    realvec expm1() const { return expm1d4(v); }
    realvec fabs() const { return vec_abs(v); }
    realvec fdim(realvec y) const { return MF::vml_fdim(*this, y); }
    realvec floor() const { return vec_floor(v); }
    realvec fma(realvec y, realvec z) const { return vec_madd(v, y.v, z.v); }
    realvec fmax(realvec y) const { return MF::vml_fmax(v, y.v); }
    realvec fmin(realvec y) const { return MF::vml_fmin(v, y.v); }
    realvec fmod(realvec y) const { return MF::vml_fmod(*this, y); }
    realvec frexp(intvec_t& r) const { return MF::vml_frexp(*this, r); }
    realvec hypot(realvec y) const { return hypotd4(v, y.v); }
    intvec_t ilogb() const
    {
      int_t ilogb_[] = {
	::ilogb((*this)[0]),
	::ilogb((*this)[1]),
	::ilogb((*this)[2]),
	::ilogb((*this)[3])
      };
      return intvec_t(ilogb_);
    }
    boolvec_t isfinite() const { return MF::vml_isfinite(*this); }
    boolvec_t isinf() const { return MF::vml_isinf(*this); }
    boolvec_t isnan() const
    {
#ifdef VML_HAVE_NAN
      return vec_tstnan(v, v);
#else
      return BV(false);
#endif
    }
    boolvec_t isnormal() const { return MF::vml_isnormal(*this); }
    realvec ldexp(int_t n) const { return ldexp(intvec_t(n)); }
    realvec ldexp(intvec_t n) const
    {
      real_t ldexp_[] = {
	std::ldexp((*this)[0], n[0]),
        std::ldexp((*this)[1], n[1]),
        std::ldexp((*this)[2], n[2]),
        std::ldexp((*this)[3], n[3])
      };
      return realvec_t(ldexp_);
    }
    realvec log() const { return logd4(v); }
    realvec log10() const { return log10d4(v); }
    realvec log1p() const { return log1pd4(v); }
    realvec log2() const { return log2d4(v); }
    realvec nextafter(realvec y) const { return MF::vml_nextafter(*this, y); }
    realvec pow(realvec y) const { return powd4(v, y.v); }
    realvec rcp() const { return recip_fastd4(v); }
    realvec remainder(realvec y) const { return MF::vml_remainder(*this, y); }
    realvec rint() const
    {
      return MF::vml_rint(*this);
      // This is tempting, but seems too invasive
      // #ifdef VML_HAVE_FP_CONTRACT
      //       return MF::vml_rint(*this);
      // #else
      //       return vec_round(v);      // use round instead of rint
      // #endif
    }
    realvec round() const { return vec_round(v); }
    realvec rsqrt() const
    {
      realvec x = *this;
      realvec r = vec_rsqrte(x.v); // this is only an approximation
      // TODO: use fma
      // one Newton iteration (see vml_rsqrt)
      r += RV(0.5)*r * (RV(1.0) - x * r*r);
      return r;
    }
    boolvec_t signbit() const { return !copysign(RV(1.0)).as_int().as_bool(); }
    realvec sin() const { return sind4(v); }
    realvec sinh() const { return sinhd4(v); }
    realvec sqrt() const
    {
      // return vec_sqrtsw_nochk(v);
      return *this * rsqrt();
    }
    realvec tan() const { return tand4(v); }
    realvec tanh() const { return tanhd4(v); }
    realvec trunc() const { return vec_trunc(v); }
  };
  
  
  
  // boolvec definitions
  
  inline intvec<double,4> boolvec<double,4>::as_int() const
  {
    return v;
  }
  
  inline intvec<double,4> boolvec<double,4>::convert_int() const
  {
    return ifthen(IV(I(1)), IV(I(0)));
  }
  
  inline
  boolvec<double,4> boolvec<double,4>::ifthen(boolvec_t x, boolvec_t y) const
  {
    return ifthen(x.as_int(), y.as_int()).as_bool();
  }
  
  inline
  intvec<double,4> boolvec<double,4>::ifthen(intvec_t x, intvec_t y) const
  {
    return ifthen(x.as_float(), y.as_float()).as_int();
  }
  
  inline
  realvec<double,4> boolvec<double,4>::ifthen(realvec_t x, realvec_t y) const
  {
    return vec_sel(y.v, x.v, v);
  }
  
  
  
  // intvec definitions
  
  inline realvec<double,4> intvec<double,4>::as_float() const
  {
    return v;
  }
  
  inline realvec<double,4> intvec<double,4>::convert_float() const
  {
    return vec_cfid(v);
  }
  
} // namespace vecmathlib

#endif  // #ifndef VEC_QPX_DOUBLE4_H
