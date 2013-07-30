// -*-C++-*-

#ifndef VEC_SSE_DOUBLE2_H
#define VEC_SSE_DOUBLE2_H

#include "floatprops.h"
#include "mathfuncs.h"
#include "vec_base.h"

#include <cmath>

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
  
#define VECMATHLIB_HAVE_VEC_DOUBLE_2
  template<> struct boolvec<double,2>;
  template<> struct intvec<double,2>;
  template<> struct realvec<double,2>;
  
  
  
  template<>
  struct boolvec<double,2>: floatprops<double>
  {
    static int const size = 2;
    typedef bool scalar_t;
    typedef __m128d bvector_t;
    static int const alignment = sizeof(bvector_t);
    
    static_assert(size * sizeof(real_t) == sizeof(bvector_t),
                  "vector size is wrong");
    
  private:
    // true values have the sign bit set, false values have it unset
    static uint_t from_bool(bool a) { return - uint_t(a); }
    static bool to_bool(uint_t a) { return int_t(a) < int_t(0); }
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
    boolvec(bool a):
    v(_mm_castsi128_pd(_mm_set1_epi64x(from_bool(a)))) {}
    boolvec(bool const* as):
    v(_mm_castsi128_pd(_mm_set_epi64x(from_bool(as[1]), from_bool(as[0])))) {}
    
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
    
    
    
    boolvec operator!() const { return _mm_xor_pd(boolvec(true), v); }
    
    boolvec operator&&(boolvec x) const { return _mm_and_pd(v, x.v); }
    boolvec operator||(boolvec x) const { return _mm_or_pd(v, x.v); }
    boolvec operator==(boolvec x) const { return !(*this!=x); }
    boolvec operator!=(boolvec x) const { return _mm_xor_pd(v, x.v); }
    
    bool all() const
    {
#if defined __AVX__
      return ! (! *this).any();
#else
      return (*this)[0] && (*this)[1];
#endif
    }
    bool any() const
    {
#if defined __AVX__
      return ! _mm_testz_pd(v, v);
#else
      return (*this)[0] || (*this)[1];
#endif
    }
    
    
    
    // ifthen(condition, then-value, else-value)
    boolvec_t ifthen(boolvec_t x, boolvec_t y) const;
    intvec_t ifthen(intvec_t x, intvec_t y) const; // defined after intvec
    realvec_t ifthen(realvec_t x, realvec_t y) const; // defined after realvec
  };
  
  
  
  template<>
  struct intvec<double,2>: floatprops<double>
  {
    static int const size = 2;
    typedef int_t scalar_t;
    typedef __m128i ivector_t;
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
    intvec(int_t a): v(_mm_set1_epi64x(a)) {}
    intvec(int_t const* as): v(_mm_set_epi64x(as[1], as[0])) {}
    static intvec iota() { return _mm_set_epi64x(1, 0); }
    
    operator ivector_t() const { return v; }
    int_t operator[](int n) const
    {
      return vecmathlib::get_elt<IV,ivector_t,int_t>(v, n);
    }
    intvec_t& set_elt(int n, int_t a)
    {
      return vecmathlib::set_elt<IV,ivector_t,int_t>(v, n, a), *this;
    }
    
    
    
    boolvec_t as_bool() const { return _mm_castsi128_pd(v); }
    boolvec_t convert_bool() const
    {
      // Result: convert_bool(0)=false, convert_bool(else)=true
      // There is no intrinsic to compare with zero. Instead, we check
      // whether x is positive and x-1 is negative.
      intvec x = *this;
      // We know that boolvec values depend only on the sign bit
      // return (~(x-1) | x).as_bool();
      // return x.as_bool() || !(x-1).as_bool();
      return x.as_bool() || (x + (FP::signbit_mask - 1)).as_bool();
    }
    realvec_t as_float() const;      // defined after realvec
    realvec_t convert_float() const; // defined after realvec
    
    
    
    // Note: not all arithmetic operations are supported!
    
    intvec operator+() const { return *this; }
    intvec operator-() const { return IV(I(0)) - *this; }
    
    intvec operator+(intvec x) const { return _mm_add_epi64(v, x.v); }
    intvec operator-(intvec x) const { return _mm_sub_epi64(v, x.v); }
    
    intvec& operator+=(intvec const& x) { return *this=*this+x; }
    intvec& operator-=(intvec const& x) { return *this=*this-x; }
    
    
    
    intvec operator~() const { return IV(~U(0)) ^ *this; }
    
    intvec operator&(intvec x) const
    {
      return _mm_castpd_si128(_mm_and_pd(_mm_castsi128_pd(v),
                                         _mm_castsi128_pd(x.v)));
    }
    intvec operator|(intvec x) const
    {
      return _mm_castpd_si128(_mm_or_pd(_mm_castsi128_pd(v),
                                        _mm_castsi128_pd(x.v)));
    }
    intvec operator^(intvec x) const
    {
      return _mm_castpd_si128(_mm_xor_pd(_mm_castsi128_pd(v),
                                         _mm_castsi128_pd(x.v)));
    }
    
    intvec& operator&=(intvec const& x) { return *this=*this&x; }
    intvec& operator|=(intvec const& x) { return *this=*this|x; }
    intvec& operator^=(intvec const& x) { return *this=*this^x; }
    
    
    
    intvec lsr(int_t n) const { return _mm_srli_epi64(v, n); }
    intvec operator>>(int_t n) const
    {
      // There is no _mm_srai_epi64. To emulate it, add 0x80000000
      // before shifting, and subtract the shifted 0x80000000 after
      // shifting
      intvec x = *this;
      // Convert signed to unsiged
      x += U(1) << (bits-1);
      // Shift
      x = x.lsr(n);
      // Undo conversion
      x -= U(1) << (bits-1-n);
      return x;
    }
    intvec operator<<(int_t n) const { return _mm_slli_epi64(v, n); }
    intvec& operator>>=(int_t n) { return *this=*this>>n; }
    intvec& operator<<=(int_t n) { return *this=*this<<n; }
    
    intvec lsr(intvec n) const
    {
      intvec r;
      for (int i=0; i<size; ++i) {
        r.set_elt(i, U((*this)[i]) >> U(n[i]));
      }
      return r;
    }
    intvec operator>>(intvec n) const
    {
      intvec r;
      for (int i=0; i<size; ++i) {
        r.set_elt(i, (*this)[i] >> n[i]);
      }
      return r;
    }
    intvec operator<<(intvec n) const
    {
      intvec r;
      for (int i=0; i<size; ++i) {
        r.set_elt(i, (*this)[i] << n[i]);
      }
      return r;
    }
    intvec& operator>>=(intvec n) { return *this=*this>>n; }
    intvec& operator<<=(intvec n) { return *this=*this<<n; }
    
    
    
    boolvec_t signbit() const
    {
      return as_bool();
    }
    
    boolvec_t operator==(intvec const& x) const
    {
      return ! (*this != x);
    }
    boolvec_t operator!=(intvec const& x) const
    {
      return (*this ^ x).convert_bool();
    }
    boolvec_t operator<(intvec const& x) const
    {
      // return (*this - x).as_bool();
      boolvec_t r;
      for (int i=0; i<size; ++i) {
        r.set_elt(i, (*this)[i] < x[i]);
      }
      return r;
    }
    boolvec_t operator<=(intvec const& x) const
    {
      return ! (*this > x);
    }
    boolvec_t operator>(intvec const& x) const
    {
      return x < *this;
    }
    boolvec_t operator>=(intvec const& x) const
    {
      return ! (*this < x);
    }
  };
  
  
  
  template<>
  struct realvec<double,2>: floatprops<double>
  {
    static int const size = 2;
    typedef real_t scalar_t;
    typedef __m128d vector_t;
    static int const alignment = sizeof(vector_t);
    
    static char const* name() { return "<SSE2:2*double>"; }
    void barrier() { __asm__("": "+x"(v)); }
    
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
    realvec(real_t a): v(_mm_set1_pd(a)) {}
    realvec(real_t const* as): v(_mm_set_pd(as[1], as[0])) {}
    
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
      return _mm_load_pd(p);
    }
    static realvec_t loadu(real_t const* p)
    {
      return _mm_loadu_pd(p);
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
      _mm_store_pd(p, v);
    }
    void storeu(real_t* p) const
    {
      return _mm_storeu_pd(p, v);
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
#if defined __AVX__
        _mm_maskstore_pd(p, m.m.as_int(), v);
#else
        if      (m.m[0]) _mm_storel_pd(p  , v);
        else if (m.m[1]) _mm_storeh_pd(p+1, v);
#endif
      }
    }
    void storeu(real_t* p, mask_t const& m) const
    {
      if (__builtin_expect(m.all_m, true)) {
        storeu(p);
      } else {
        if      (m.m[0]) _mm_storel_pd(p  , v);
        else if (m.m[1]) _mm_storeh_pd(p+1, v);
      }
    }
    void storeu(real_t* p, std::ptrdiff_t ioff, mask_t const& m) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      if (ioff % realvec::size == 0) return storea(p+ioff, m);
      storeu(p+ioff, m);
    }
    
    
    
    intvec_t as_int() const { return _mm_castpd_si128(v); }
    intvec_t convert_int() const
    {
      intvec_t r;
      r.set_elt(0, floatprops::convert_int((*this)[0]));
      r.set_elt(1, floatprops::convert_int((*this)[1]));
      return r;
    }
    
    
    
    realvec operator+() const { return *this; }
    realvec operator-() const { return RV(0.0) - *this; }
    
    realvec operator+(realvec x) const { return _mm_add_pd(v, x.v); }
    realvec operator-(realvec x) const { return _mm_sub_pd(v, x.v); }
    realvec operator*(realvec x) const { return _mm_mul_pd(v, x.v); }
    realvec operator/(realvec x) const { return _mm_div_pd(v, x.v); }
    
    realvec& operator+=(realvec const& x) { return *this=*this+x; }
    realvec& operator-=(realvec const& x) { return *this=*this-x; }
    realvec& operator*=(realvec const& x) { return *this=*this*x; }
    realvec& operator/=(realvec const& x) { return *this=*this/x; }
    
    real_t maxval() const
    {
      return std::fmax((*this)[0], (*this)[1]);
    }
    real_t minval() const
    {
      return std::fmin((*this)[0], (*this)[1]);
    }
    real_t prod() const
    {
      return (*this)[0] * (*this)[1];
    }
    real_t sum() const
    {
#ifdef __SSE3__
      return _mm_cvtsd_f64(_mm_hadd_pd(v, v));
#else
      return (*this)[0] + (*this)[1];
#endif
    }
    
    
    
    boolvec_t operator==(realvec const& x) const
    {
      return _mm_cmpeq_pd(v, x.v);
    }
    boolvec_t operator!=(realvec const& x) const
    {
      return _mm_cmpneq_pd(v, x.v);
    }
    boolvec_t operator<(realvec const& x) const
    {
      return _mm_cmplt_pd(v, x.v);
    }
    boolvec_t operator<=(realvec const& x) const
    {
      return _mm_cmple_pd(v, x.v);
    }
    boolvec_t operator>(realvec const& x) const
    {
      return _mm_cmpgt_pd(v, x.v);
    }
    boolvec_t operator>=(realvec const& x) const
    {
      return _mm_cmpge_pd(v, x.v);
    }
    
    
    
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
      return _mm_ceil_pd(v);
#else
      return MF::vml_ceil(*this);
#endif
 }
    realvec copysign(realvec y) const { return MF::vml_copysign(*this, y); }
    realvec cos() const { return MF::vml_cos(*this); }
    realvec cosh() const { return MF::vml_cosh(*this); }
    realvec exp() const { return MF::vml_exp(*this); }
    realvec exp10() const { return MF::vml_exp10(*this); }
    realvec exp2() const { return MF::vml_exp2(*this); }
    realvec expm1() const { return MF::vml_expm1(*this); }
    realvec fabs() const { return MF::vml_fabs(*this); }
    realvec fdim(realvec y) const { return MF::vml_fdim(*this, y); }
    realvec floor() const
    {
#ifdef __SSE4_1__
      return _mm_floor_pd(v);
#else
      return MF::vml_floor(*this);
#endif
 }
    realvec fma(realvec y, realvec z) const { return MF::vml_fma(*this, y, z); }
    realvec fmax(realvec y) const { return _mm_max_pd(v, y.v); }
    realvec fmin(realvec y) const { return _mm_min_pd(v, y.v); }
    realvec fmod(realvec y) const { return MF::vml_fmod(*this, y); }
    realvec frexp(intvec_t& r) const { return MF::vml_frexp(*this, r); }
    realvec hypot(realvec y) const { return MF::vml_hypot(*this, y); }
    intvec_t ilogb() const { return MF::vml_ilogb(*this); }
    boolvec_t isfinite() const { return MF::vml_isfinite(*this); }
    boolvec_t isinf() const { return MF::vml_isinf(*this); }
    boolvec_t isnan() const
    {
#ifdef VML_HAVE_NAN
      return _mm_cmpunord_pd(v, v);
#else
      return BV(false);
#endif
    }
    boolvec_t isnormal() const { return MF::vml_isnormal(*this); }
    realvec ldexp(int_t n) const { return MF::vml_ldexp(*this, n); }
    realvec ldexp(intvec_t n) const { return MF::vml_ldexp(*this, n); }
    realvec log() const { return MF::vml_log(*this); }
    realvec log10() const { return MF::vml_log10(*this); }
    realvec log1p() const { return MF::vml_log1p(*this); }
    realvec log2() const { return MF::vml_log2(*this); }
    realvec nextafter(realvec y) const { return MF::vml_nextafter(*this, y); }
    realvec pow(realvec y) const { return MF::vml_pow(*this, y); }
    realvec rcp() const { return _mm_div_pd(_mm_set1_pd(1.0), v); }
    realvec remainder(realvec y) const { return MF::vml_remainder(*this, y); }
    realvec rint() const
    {
#ifdef __SSE4_1__
      return _mm_round_pd(v, _MM_FROUND_TO_NEAREST_INT);
#else
      return MF::vml_rint(*this);
#endif
    }
    realvec round() const { return MF::vml_round(*this); }
    realvec rsqrt() const { return MF::vml_rsqrt(*this); }
    boolvec_t signbit() const { return v; }
    realvec sin() const { return MF::vml_sin(*this); }
    realvec sinh() const { return MF::vml_sinh(*this); }
    realvec sqrt() const { return _mm_sqrt_pd(v); }
    realvec tan() const { return MF::vml_tan(*this); }
    realvec tanh() const { return MF::vml_tanh(*this); }
    realvec trunc() const
    {
#ifdef __SSE4_1__
      return _mm_round_pd(v, _MM_FROUND_TO_ZERO);
#else
      return MF::vml_trunc(*this);
#endif
 }
  };
  
  
  
  // boolvec definitions
  
  inline intvec<double,2> boolvec<double,2>::as_int() const
  {
    return _mm_castpd_si128(v);
  }
  
  inline intvec<double,2> boolvec<double,2>::convert_int() const
  {
    //return ifthen(v, U(1), U(0));
    return lsr(as_int(), bits-1);
  }
  
  inline
  boolvec<double,2> boolvec<double,2>::ifthen(boolvec_t x, boolvec_t y) const
  {
    return ifthen(x.as_int(), y.as_int()).as_bool();
  }
  
  inline
  intvec<double,2> boolvec<double,2>::ifthen(intvec_t x, intvec_t y) const
  {
    return ifthen(x.as_float(), y.as_float()).as_int();
  }
  
  inline
  realvec<double,2> boolvec<double,2>::ifthen(realvec_t x, realvec_t y) const
  {
#ifdef __SSE4_1__
    return _mm_blendv_pd(y.v, x.v, v);
#else
    return (( -convert_int() & x.as_int()) |
            (~-convert_int() & y.as_int())).as_float();
#endif
  }
  
  
  
  // intvec definitions
  
  inline realvec<double,2> intvec<double,2>::as_float() const
  {
    return _mm_castsi128_pd(v);
  }
  
  inline realvec<double,2> intvec<double,2>::convert_float() const
  {
    realvec_t r;
    r.set_elt(0, floatprops::convert_float((*this)[0]));
    r.set_elt(1, floatprops::convert_float((*this)[1]));
    return r;
  }
  
} // namespace vecmathlib

#endif  // #ifndef VEC_SSE_DOUBLE2_H
