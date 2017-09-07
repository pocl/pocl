// -*-C++-*-

#ifndef VEC_SSE_FLOAT4_H
#define VEC_SSE_FLOAT4_H

#include "floatprops.h"
#include "mathfuncs.h"
#include "vec_base.h"

#include <cmath>

// SSE2 intrinsics
#include <x86intrin.h>

namespace vecmathlib {

#define VECMATHLIB_HAVE_VEC_FLOAT_4
template <> struct boolvec<float, 4>;
template <> struct intvec<float, 4>;
template <> struct realvec<float, 4>;

template <> struct boolvec<float, 4> : floatprops<float> {
  static int const size = 4;
  typedef bool scalar_t;
  typedef __m128 bvector_t;
  static int const alignment = sizeof(bvector_t);

  static_assert(size * sizeof(real_t) == sizeof(bvector_t),
                "vector size is wrong");

private:
  // true values have the sign bit set, false values have it unset
  static uint_t from_bool(bool a) { return -int_t(a); }
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
  boolvec(bvector_t x) : v(x) {}
  boolvec(bool a) : v(_mm_castsi128_ps(_mm_set1_epi32(from_bool(a)))) {}
  boolvec(bool const *as)
      : v(_mm_castsi128_ps(_mm_set_epi32(from_bool(as[3]), from_bool(as[2]),
                                         from_bool(as[1]), from_bool(as[0])))) {
  }

  operator bvector_t() const { return v; }
  bool operator[](int n) const {
    return to_bool(vecmathlib::get_elt<BV, bvector_t, uint_t>(v, n));
  }
  boolvec_t &set_elt(int n, bool a) {
    return vecmathlib::set_elt<BV, bvector_t, uint_t>(v, n, from_bool(a)),
           *this;
  }

  intvec_t as_int() const;      // defined after intvec
  intvec_t convert_int() const; // defined after intvec

  boolvec_t operator!() const { return _mm_xor_ps(boolvec(true), v); }

  boolvec_t operator&&(boolvec_t x) const { return _mm_and_ps(v, x.v); }
  boolvec_t operator||(boolvec_t x) const { return _mm_or_ps(v, x.v); }
  boolvec_t operator==(boolvec_t x) const { return !(*this != x); }
  boolvec_t operator!=(boolvec_t x) const { return _mm_xor_ps(v, x.v); }

  bool all() const {
// return (*this)[0] && (*this)[1] && (*this)[2] && (*this)[3];
#if defined __AVX__
    return !(!*this).any();
#else
    boolvec_t x = *this;
    x = x && _mm_shuffle_ps(x.v, x.v, _MM_SHUFFLE(2, 3, 0, 1));
    return x[0] && x[2];
#endif
  }
  bool any() const {
// return (*this)[0] || (*this)[1] || (*this)[2] || (*this)[3];
#if defined __AVX__
    return !bool(_mm_testz_ps(v, v));
#else
    boolvec_t x = *this;
    x = x || _mm_shuffle_ps(x.v, x.v, _MM_SHUFFLE(2, 3, 0, 1));
    return x[0] || x[2];
#endif
  }

  // ifthen(condition, then-value, else-value)
  boolvec_t ifthen(boolvec_t x, boolvec_t y) const;
  intvec_t ifthen(intvec_t x, intvec_t y) const;    // defined after intvec
  realvec_t ifthen(realvec_t x, realvec_t y) const; // defined after realvec
};

template <> struct intvec<float, 4> : floatprops<float> {
  static int const size = 4;
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
  intvec(ivector_t x) : v(x) {}
  intvec(int_t a) : v(_mm_set1_epi32(a)) {}
  intvec(int_t const *as) : v(_mm_set_epi32(as[3], as[2], as[1], as[0])) {}
  static intvec_t iota() { return _mm_set_epi32(3, 2, 1, 0); }

  operator ivector_t() const { return v; }
  int_t operator[](int n) const {
    return vecmathlib::get_elt<IV, ivector_t, int_t>(v, n);
  }
  intvec_t &set_elt(int n, int_t a) {
    return vecmathlib::set_elt<IV, ivector_t, int_t>(v, n, a), *this;
  }

  boolvec_t as_bool() const { return _mm_castsi128_ps(v); }
  boolvec_t convert_bool() const {
    // Result: convert_bool(0)=false, convert_bool(else)=true
    return !IV(_mm_cmpeq_epi32(v, IV(0))).as_bool();
  }
  realvec_t as_float() const;      // defined after realvec
  realvec_t convert_float() const; // defined after realvec

  // Note: not all arithmetic operations are supported!

  intvec_t operator+() const { return *this; }
  intvec_t operator-() const { return IV(0) - *this; }

  intvec_t operator+(intvec_t x) const { return _mm_add_epi32(v, x.v); }
  intvec_t operator-(intvec_t x) const { return _mm_sub_epi32(v, x.v); }

  intvec_t &operator+=(intvec_t const &x) { return *this = *this + x; }
  intvec_t &operator-=(intvec_t const &x) { return *this = *this - x; }

  intvec_t operator~() const { return IV(~U(0)) ^ *this; }

  intvec_t operator&(intvec_t x) const {
    return _mm_castps_si128(
        _mm_and_ps(_mm_castsi128_ps(v), _mm_castsi128_ps(x.v)));
  }
  intvec_t operator|(intvec_t x) const {
    return _mm_castps_si128(
        _mm_or_ps(_mm_castsi128_ps(v), _mm_castsi128_ps(x.v)));
  }
  intvec_t operator^(intvec_t x) const {
    return _mm_castps_si128(
        _mm_xor_ps(_mm_castsi128_ps(v), _mm_castsi128_ps(x.v)));
  }

  intvec_t &operator&=(intvec_t const &x) { return *this = *this & x; }
  intvec_t &operator|=(intvec_t const &x) { return *this = *this | x; }
  intvec_t &operator^=(intvec_t const &x) { return *this = *this ^ x; }

  intvec_t bitifthen(intvec_t x, intvec_t y) const;

  intvec_t lsr(int_t n) const { return _mm_srli_epi32(v, n); }
  intvec_t rotate(int_t n) const;
  intvec_t operator>>(int_t n) const { return _mm_srai_epi32(v, n); }
  intvec_t operator<<(int_t n) const { return _mm_slli_epi32(v, n); }
  intvec_t &operator>>=(int_t n) { return *this = *this >> n; }
  intvec_t &operator<<=(int_t n) { return *this = *this << n; }

  intvec_t lsr(intvec_t n) const {
    intvec_t r;
    for (int i = 0; i < size; ++i) {
      r.set_elt(i, U((*this)[i]) >> U(n[i]));
    }
    return r;
  }
  intvec_t rotate(intvec_t n) const;
  intvec_t operator>>(intvec_t n) const {
    intvec_t r;
    for (int i = 0; i < size; ++i) {
      r.set_elt(i, (*this)[i] >> n[i]);
    }
    return r;
  }
  intvec_t operator<<(intvec_t n) const {
    intvec_t r;
    for (int i = 0; i < size; ++i) {
      r.set_elt(i, (*this)[i] << n[i]);
    }
    return r;
  }
  intvec_t &operator>>=(intvec_t n) { return *this = *this >> n; }
  intvec_t &operator<<=(intvec_t n) { return *this = *this << n; }

  intvec_t clz() const;
  intvec_t popcount() const;

  boolvec_t operator==(intvec_t const &x) const { return !(*this != x); }
  boolvec_t operator!=(intvec_t const &x) const {
    return (*this ^ x).convert_bool();
  }
  boolvec_t operator<(intvec_t const &x) const {
    // return (*this - x).as_bool();
    boolvec_t r;
    for (int i = 0; i < size; ++i) {
      r.set_elt(i, (*this)[i] < x[i]);
    }
    return r;
  }
  boolvec_t operator<=(intvec_t const &x) const { return !(*this > x); }
  boolvec_t operator>(intvec_t const &x) const { return x < *this; }
  boolvec_t operator>=(intvec_t const &x) const { return !(*this < x); }

  intvec_t abs() const;
  boolvec_t isignbit() const { return as_bool(); }
  intvec_t max(intvec_t x) const;
  intvec_t min(intvec_t x) const;
};

template <> struct realvec<float, 4> : floatprops<float> {
  static int const size = 4;
  typedef real_t scalar_t;
  typedef __m128 vector_t;
  static int const alignment = sizeof(vector_t);

  static char const *name() { return "<SSE2:4*float>"; }
  void barrier() { __asm__("" : "+x"(v)); }

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
  realvec(vector_t x) : v(x) {}
  realvec(real_t a) : v(_mm_set1_ps(a)) {}
  realvec(real_t const *as) : v(_mm_set_ps(as[3], as[2], as[1], as[0])) {}

  operator vector_t() const { return v; }
  real_t operator[](int n) const {
    return vecmathlib::get_elt<RV, vector_t, real_t>(v, n);
  }
  realvec_t &set_elt(int n, real_t a) {
    return vecmathlib::set_elt<RV, vector_t, real_t>(v, n, a), *this;
  }

  typedef vecmathlib::mask_t<realvec_t> mask_t;

  static realvec_t loada(real_t const *p) {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    return _mm_load_ps(p);
  }
  static realvec_t loadu(real_t const *p) { return _mm_loadu_ps(p); }
  static realvec_t loadu(real_t const *p, std::ptrdiff_t ioff) {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    if (ioff % realvec::size == 0)
      return loada(p + ioff);
    if (ioff == 0)
      return loada(p);
    return loadu(p + ioff);
  }
  realvec_t loada(real_t const *p, mask_t const &m) const {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    if (__builtin_expect(all(m.m), true)) {
      return loada(p);
    } else {
      return m.m.ifthen(loada(p), *this);
    }
  }
  realvec_t loadu(real_t const *p, mask_t const &m) const {
    if (__builtin_expect(m.all_m, true)) {
      return loadu(p);
    } else {
      return m.m.ifthen(loadu(p), *this);
    }
  }
  realvec_t loadu(real_t const *p, std::ptrdiff_t ioff, mask_t const &m) const {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    if (ioff % realvec::size == 0)
      return loada(p + ioff, m);
    return loadu(p + ioff, m);
  }

  void storea(real_t *p) const {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    _mm_store_ps(p, v);
  }
  void storeu(real_t *p) const { return _mm_storeu_ps(p, v); }
  void storeu(real_t *p, std::ptrdiff_t ioff) const {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    if (ioff % realvec::size == 0)
      return storea(p + ioff);
    storeu(p + ioff);
  }
  void storea(real_t *p, mask_t const &m) const {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    if (__builtin_expect(m.all_m, true)) {
      storea(p);
    } else {
#if defined __AVX__
      _mm_maskstore_ps(p, m.m.as_int(), v);
#else
      if (m.m[0])
        p[0] = (*this)[0];
      if (m.m[1])
        p[1] = (*this)[1];
      if (m.m[2])
        p[2] = (*this)[2];
      if (m.m[3])
        p[3] = (*this)[3];
#endif
    }
  }
  void storeu(real_t *p, mask_t const &m) const {
    if (__builtin_expect(m.all_m, true)) {
      storeu(p);
    } else {
      if (m.m[0])
        p[0] = (*this)[0];
      if (m.m[1])
        p[1] = (*this)[1];
      if (m.m[2])
        p[2] = (*this)[2];
      if (m.m[3])
        p[3] = (*this)[3];
    }
  }
  void storeu(real_t *p, std::ptrdiff_t ioff, mask_t const &m) const {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    if (ioff % realvec::size == 0)
      return storea(p + ioff, m);
    storeu(p + ioff, m);
  }

  intvec_t as_int() const { return _mm_castps_si128(v); }
  intvec_t convert_int() const { return _mm_cvttps_epi32(v); }

  realvec_t operator+() const { return *this; }
  realvec_t operator-() const { return RV(0.0) - *this; }

  realvec_t operator+(realvec_t x) const { return _mm_add_ps(v, x.v); }
  realvec_t operator-(realvec_t x) const { return _mm_sub_ps(v, x.v); }
  realvec_t operator*(realvec_t x) const { return _mm_mul_ps(v, x.v); }
  realvec_t operator/(realvec_t x) const { return _mm_div_ps(v, x.v); }

  realvec_t &operator+=(realvec_t const &x) { return *this = *this + x; }
  realvec_t &operator-=(realvec_t const &x) { return *this = *this - x; }
  realvec_t &operator*=(realvec_t const &x) { return *this = *this * x; }
  realvec_t &operator/=(realvec_t const &x) { return *this = *this / x; }

  real_t maxval() const {
    // return vml_std::fmax(vml_std::fmax((*this)[0], (*this)[1]),
    //                      vml_std::fmax((*this)[2], (*this)[3]));
    realvec_t x0123 = *this;
    realvec_t x1032 = _mm_shuffle_ps(x0123, x0123, 0b10110001);
    realvec_t y0022 = x0123.fmax(x1032);
    return vml_std::fmax(y0022[0], y0022[2]);
  }
  real_t minval() const {
    // return vml_std::fmin(vml_std::fmin((*this)[0], (*this)[1]),
    //                      vml_std::fmin((*this)[2], (*this)[3]));
    realvec_t x0123 = *this;
    realvec_t x1032 = _mm_shuffle_ps(x0123, x0123, 0b10110001);
    realvec_t y0022 = x0123.fmin(x1032);
    return vml_std::fmin(y0022[0], y0022[2]);
  }
  real_t prod() const {
    // return (*this)[0] * (*this)[1] * (*this)[2] * (*this)[3];
    realvec_t x0123 = *this;
    realvec_t x1032 = _mm_shuffle_ps(x0123, x0123, 0b10110001);
    realvec_t y0022 = x0123 * x1032;
    return y0022[0] * y0022[2];
  }
  real_t sum() const {
#ifdef __SSE3__
    realvec_t x = *this;
    x = _mm_hadd_ps(x.v, x.v);
    x = _mm_hadd_ps(x.v, x.v);
    return x[0];
#else
    // return (*this)[0] + (*this)[1] + (*this)[2] + (*this)[3];
    realvec_t x0123 = *this;
    realvec_t x1032 = _mm_shuffle_ps(x0123, x0123, 0b10110001);
    realvec_t y0022 = x0123 + x1032;
    return y0022[0] + y0022[2];
#endif
  }

  boolvec_t operator==(realvec_t const &x) const {
    return _mm_cmpeq_ps(v, x.v);
  }
  boolvec_t operator!=(realvec_t const &x) const {
    return _mm_cmpneq_ps(v, x.v);
  }
  boolvec_t operator<(realvec_t const &x) const { return _mm_cmplt_ps(v, x.v); }
  boolvec_t operator<=(realvec_t const &x) const {
    return _mm_cmple_ps(v, x.v);
  }
  boolvec_t operator>(realvec_t const &x) const { return _mm_cmpgt_ps(v, x.v); }
  boolvec_t operator>=(realvec_t const &x) const {
    return _mm_cmpge_ps(v, x.v);
  }

  realvec_t acos() const { return MF::vml_acos(*this); }
  realvec_t acosh() const { return MF::vml_acosh(*this); }
  realvec_t asin() const { return MF::vml_asin(*this); }
  realvec_t asinh() const { return MF::vml_asinh(*this); }
  realvec_t atan() const { return MF::vml_atan(*this); }
  realvec_t atan2(realvec_t y) const { return MF::vml_atan2(*this, y); }
  realvec_t atanh() const { return MF::vml_atanh(*this); }
  realvec_t cbrt() const { return MF::vml_cbrt(*this); }
  realvec_t ceil() const {
#ifdef __SSE4_1__
    return _mm_ceil_ps(v);
#else
    return MF::vml_ceil(*this);
#endif
  }
  realvec_t copysign(realvec_t y) const { return MF::vml_copysign(*this, y); }
  realvec_t cos() const { return MF::vml_cos(*this); }
  realvec_t cosh() const { return MF::vml_cosh(*this); }
  realvec_t exp() const { return MF::vml_exp(*this); }
  realvec_t exp10() const { return MF::vml_exp10(*this); }
  realvec_t exp2() const { return MF::vml_exp2(*this); }
  realvec_t expm1() const { return MF::vml_expm1(*this); }
  realvec_t fabs() const { return MF::vml_fabs(*this); }
  realvec_t fdim(realvec_t y) const { return MF::vml_fdim(*this, y); }
  realvec_t floor() const {
#ifdef __SSE4_1__
    return _mm_floor_ps(v);
#else
    return MF::vml_floor(*this);
#endif
  }
  realvec_t fma(realvec_t y, realvec_t z) const {
#if defined(__FMA4__)
    realvec_t x = *this;
    return _mm_macc_ps(x, y, z);
#elif defined(__FMA__)
    realvec_t x = *this;
    return _mm_fmadd_ps(x, y, z);
#else
    return MF::vml_fma(*this, y, z);
#endif
  }
  /* OpenCL spec: if one argument is NaN, return the second
   * instructions: if any argument is NaN, return the second
   * ... so we must take care of (x, NaN) arguments case
   */
  realvec_t fmax(realvec_t y) const {
    realvec_t res = _mm_max_ps(v, y.v);
#if defined VML_HAVE_NAN
    return y.isnan().ifthen(v, res);
#else
    return res;
#endif
  }
  realvec_t fmin(realvec_t y) const {
    realvec_t res = _mm_min_ps(v, y.v);
#if defined VML_HAVE_NAN
    return y.isnan().ifthen(v, res);
#else
    return res;
#endif
  }

  realvec_t fmod(realvec_t y) const { return MF::vml_fmod(*this, y); }
  realvec_t frexp(intvec_t *r) const { return MF::vml_frexp(*this, r); }
  realvec_t hypot(realvec_t y) const { return MF::vml_hypot(*this, y); }
  intvec_t ilogb() const { return MF::vml_ilogb(*this); }
  boolvec_t isfinite() const { return MF::vml_isfinite(*this); }
  boolvec_t isinf() const { return MF::vml_isinf(*this); }
  boolvec_t isnan() const {
#if defined VML_HAVE_NAN
    return _mm_cmpunord_ps(v, v);
#else
    return BV(false);
#endif
  }
  boolvec_t isnormal() const { return MF::vml_isnormal(*this); }
  realvec_t ldexp(int_t n) const { return MF::vml_ldexp(*this, n); }
  realvec_t ldexp(intvec_t n) const { return MF::vml_ldexp(*this, n); }
  realvec_t log() const { return MF::vml_log(*this); }
  realvec_t log10() const { return MF::vml_log10(*this); }
  realvec_t log1p() const { return MF::vml_log1p(*this); }
  realvec_t log2() const { return MF::vml_log2(*this); }
  realvec_t mad(realvec_t y, realvec_t z) const {
    return MF::vml_mad(*this, y, z);
  }
  realvec_t nextafter(realvec_t y) const { return MF::vml_nextafter(*this, y); }
  realvec_t pow(realvec_t y) const { return MF::vml_pow(*this, y); }
  realvec_t rcp() const {
    realvec_t x = *this;
    realvec_t r = _mm_rcp_ps(x); // this is only an approximation
    r *= RV(2.0) - r * x;        // one Newton iteration (see vml_rcp)
    return r;
  }
  realvec_t remainder(realvec_t y) const { return MF::vml_remainder(*this, y); }
  realvec_t rint() const {
#ifdef __SSE4_1__
    return _mm_round_ps(v, _MM_FROUND_TO_NEAREST_INT);
#else
    return MF::vml_rint(*this);
#endif
  }
  realvec_t round() const { return MF::vml_round(*this); }
  realvec_t rsqrt() const {
    realvec_t x = *this;
    realvec_t r = _mm_rsqrt_ps(x);      // this is only an approximation
    r *= RV(1.5) - RV(0.5) * x * r * r; // one Newton iteration (see vml_rsqrt)
    return r;
  }
  boolvec_t signbit() const { return v; }
  realvec_t sin() const { return MF::vml_sin(*this); }
  realvec_t sinh() const { return MF::vml_sinh(*this); }
  realvec_t sqrt() const { return _mm_sqrt_ps(v); }
  realvec_t tan() const { return MF::vml_tan(*this); }
  realvec_t tanh() const { return MF::vml_tanh(*this); }
  realvec_t trunc() const {
#ifdef __SSE4_1__
    return _mm_round_ps(v, _MM_FROUND_TO_ZERO);
#else
    return MF::vml_trunc(*this);
#endif
  }
};

// boolvec definitions

inline intvec<float, 4> boolvec<float, 4>::as_int() const {
  return _mm_castps_si128(v);
}

inline intvec<float, 4> boolvec<float, 4>::convert_int() const {
  return lsr(as_int(), bits - 1);
}

inline boolvec<float, 4> boolvec<float, 4>::ifthen(boolvec_t x,
                                                   boolvec_t y) const {
  return ifthen(x.as_int(), y.as_int()).as_bool();
}

inline intvec<float, 4> boolvec<float, 4>::ifthen(intvec_t x,
                                                  intvec_t y) const {
  return ifthen(x.as_float(), y.as_float()).as_int();
}

inline realvec<float, 4> boolvec<float, 4>::ifthen(realvec_t x,
                                                   realvec_t y) const {
#ifdef __SSE4_1__
  return _mm_blendv_ps(y.v, x.v, v);
#else
  return ((-convert_int() & x.as_int()) | (~ - convert_int() & y.as_int()))
      .as_float();
#endif
}

// intvec definitions

inline intvec<float, 4> intvec<float, 4>::abs() const {
#ifdef __SSSE3__
  return _mm_abs_epi32(v);
#else
  return MF::vml_abs(*this);
#endif
}

inline realvec<float, 4> intvec<float, 4>::as_float() const {
  return _mm_castsi128_ps(v);
}

inline intvec<float, 4> intvec<float, 4>::bitifthen(intvec_t x,
                                                    intvec_t y) const {
  return MF::vml_bitifthen(*this, x, y);
}

inline intvec<float, 4> intvec<float, 4>::clz() const {
  return MF::vml_clz(*this);
}

inline realvec<float, 4> intvec<float, 4>::convert_float() const {
  return _mm_cvtepi32_ps(v);
}

inline intvec<float, 4> intvec<float, 4>::max(intvec_t x) const {
#ifdef __SSE4_1__
  return _mm_max_epi32(v, x.v);
#else
  return MF::vml_max(*this, x);
#endif
}

inline intvec<float, 4> intvec<float, 4>::min(intvec_t x) const {
#ifdef __SSE4_1__
  return _mm_min_epi32(v, x.v);
#else
  return MF::vml_min(*this, x);
#endif
}

inline intvec<float, 4> intvec<float, 4>::popcount() const {
  return MF::vml_popcount(*this);
}

inline intvec<float, 4> intvec<float, 4>::rotate(int_t n) const {
  return MF::vml_rotate(*this, n);
}

inline intvec<float, 4> intvec<float, 4>::rotate(intvec_t n) const {
  return MF::vml_rotate(*this, n);
}

} // namespace vecmathlib

#endif // #ifndef VEC_SSE_FLOAT4_H
