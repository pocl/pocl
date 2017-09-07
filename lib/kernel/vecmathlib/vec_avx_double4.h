// -*-C++-*-

#ifndef VEC_AVX_DOUBLE4_H
#define VEC_AVX_DOUBLE4_H

#include "floatprops.h"
#include "mathfuncs.h"
#include "vec_base.h"

#include <cmath>

// AVX intrinsics
#include <x86intrin.h>

namespace vecmathlib {

#define VECMATHLIB_HAVE_VEC_DOUBLE_4
template <> struct boolvec<double, 4>;
template <> struct intvec<double, 4>;
template <> struct realvec<double, 4>;

template <> struct boolvec<double, 4> : floatprops<double> {
  static int const size = 4;
  typedef bool scalar_t;
  typedef __m256d bvector_t;
  static int const alignment = sizeof(bvector_t);

  static_assert(size * sizeof(real_t) == sizeof(bvector_t),
                "vector size is wrong");

private:
  // true values have the sign bit set, false values have it unset
  static uint_t from_bool(bool a) { return -uint_t(a); }
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
  boolvec(bool a) : v(_mm256_castsi256_pd(_mm256_set1_epi64x(from_bool(a)))) {}
  boolvec(bool const *as)
      : v(_mm256_castsi256_pd(
            _mm256_set_epi64x(from_bool(as[3]), from_bool(as[2]),
                              from_bool(as[1]), from_bool(as[0])))) {}

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

  boolvec_t operator!() const { return _mm256_xor_pd(boolvec(true), v); }

  boolvec_t operator&&(boolvec_t x) const { return _mm256_and_pd(v, x.v); }
  boolvec_t operator||(boolvec_t x) const { return _mm256_or_pd(v, x.v); }
  boolvec_t operator==(boolvec_t x) const { return !(*this != x); }
  boolvec_t operator!=(boolvec_t x) const { return _mm256_xor_pd(v, x.v); }

  bool all() const {
    // return (*this)[0] && (*this)[1] && (*this)[2] && (*this)[3];
    return !(!*this).any();
  }
  bool any() const {
    // return (*this)[0] || (*this)[1] || (*this)[2] || (*this)[3];
    return !bool(_mm256_testz_pd(v, v));
  }

  // ifthen(condition, then-value, else-value)
  boolvec_t ifthen(boolvec_t x, boolvec_t y) const;
  intvec_t ifthen(intvec_t x, intvec_t y) const;    // defined after intvec
  realvec_t ifthen(realvec_t x, realvec_t y) const; // defined after realvec
};

template <> struct intvec<double, 4> : floatprops<double> {
  static int const size = 4;
  typedef int_t scalar_t;
  typedef __m256i ivector_t;
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
  intvec(int_t a) : v(_mm256_set1_epi64x(a)) {}
  intvec(int_t const *as) : v(_mm256_set_epi64x(as[3], as[2], as[1], as[0])) {}
  static intvec_t iota() { return _mm256_set_epi64x(3, 2, 1, 0); }

  operator ivector_t() const { return v; }
  int_t operator[](int n) const {
    return vecmathlib::get_elt<IV, ivector_t, int_t>(v, n);
  }
  intvec_t &set_elt(int n, int_t a) {
    return vecmathlib::set_elt<IV, ivector_t, int_t>(v, n, a), *this;
  }

  boolvec_t as_bool() const { return _mm256_castsi256_pd(v); }
  boolvec_t convert_bool() const {
// Result: convert_bool(0)=false, convert_bool(else)=true
#ifdef __AVX2__
    return *this != IV(I(0));
#else
    // There is no intrinsic to compare to zero. Instead, we check
    // whether x is positive and x-1 is negative.
    intvec_t x = *this;
    // We know that boolvec_t values depend only on the sign bit
    // return (~(x-1) | x).as_bool();
    // return x.as_bool() || !(x-1).as_bool();
    return x.as_bool() || (x + (FP::signbit_mask - 1)).as_bool();
#endif
  }
  realvec_t as_float() const;      // defined after realvec
  realvec_t convert_float() const; // defined after realvec

  // Note: not all arithmetic operations are supported!

  intvec_t operator+() const { return *this; }
  intvec_t operator-() const { return IV(I(0)) - *this; }

  intvec_t operator+(intvec_t x) const {
#ifdef __AVX2__
    return _mm256_add_epi64(v, x.v);
#else
    __m128i vlo = _mm256_castsi256_si128(v);
    __m128i vhi = _mm256_extractf128_si256(v, 1);
    __m128i xvlo = _mm256_castsi256_si128(x.v);
    __m128i xvhi = _mm256_extractf128_si256(x.v, 1);
    vlo = _mm_add_epi64(vlo, xvlo);
    vhi = _mm_add_epi64(vhi, xvhi);
    return _mm256_insertf128_si256(_mm256_castsi128_si256(vlo), vhi, 1);
#endif
  }
  intvec_t operator-(intvec_t x) const {
#ifdef __AVX2__
    return _mm256_sub_epi64(v, x.v);
#else
    __m128i vlo = _mm256_castsi256_si128(v);
    __m128i vhi = _mm256_extractf128_si256(v, 1);
    __m128i xvlo = _mm256_castsi256_si128(x.v);
    __m128i xvhi = _mm256_extractf128_si256(x.v, 1);
    vlo = _mm_sub_epi64(vlo, xvlo);
    vhi = _mm_sub_epi64(vhi, xvhi);
    return _mm256_insertf128_si256(_mm256_castsi128_si256(vlo), vhi, 1);
#endif
  }

  intvec_t &operator+=(intvec_t const &x) { return *this = *this + x; }
  intvec_t &operator-=(intvec_t const &x) { return *this = *this - x; }

  intvec_t operator~() const { return IV(~U(0)) ^ *this; }

  intvec_t operator&(intvec_t x) const {
#ifdef __AVX2__
    return _mm256_and_si256(v, x.v);
#else
    return _mm256_castpd_si256(
        _mm256_and_pd(_mm256_castsi256_pd(v), _mm256_castsi256_pd(x.v)));
#endif
  }
  intvec_t operator|(intvec_t x) const {
#ifdef __AVX2__
    return _mm256_or_si256(v, x.v);
#else
    return _mm256_castpd_si256(
        _mm256_or_pd(_mm256_castsi256_pd(v), _mm256_castsi256_pd(x.v)));
#endif
  }
  intvec_t operator^(intvec_t x) const {
#ifdef __AVX2__
    return _mm256_xor_si256(v, x.v);
#else
    return _mm256_castpd_si256(
        _mm256_xor_pd(_mm256_castsi256_pd(v), _mm256_castsi256_pd(x.v)));
#endif
  }

  intvec_t &operator&=(intvec_t const &x) { return *this = *this & x; }
  intvec_t &operator|=(intvec_t const &x) { return *this = *this | x; }
  intvec_t &operator^=(intvec_t const &x) { return *this = *this ^ x; }

  intvec_t bitifthen(intvec_t x, intvec_t y) const;

  intvec_t lsr(int_t n) const {
#ifdef __AVX2__
    return _mm256_srli_epi64(v, n);
#else
    __m128i vlo = _mm256_castsi256_si128(v);
    __m128i vhi = _mm256_extractf128_si256(v, 1);
    vlo = _mm_srli_epi64(vlo, n);
    vhi = _mm_srli_epi64(vhi, n);
    return _mm256_insertf128_si256(_mm256_castsi128_si256(vlo), vhi, 1);
#endif
  }
  intvec_t rotate(int_t n) const;
  intvec_t operator>>(int_t n) const {
#ifdef __AVX2__
    // There is no _mm256_srai_epi64. To emulate it, add 0x80000000
    // before shifting, and subtract the shifted 0x80000000 after
    // shifting
    intvec_t offset = U(1) << (bits - 1);
    return (*this + offset).lsr(n) - offset.lsr(n);
#else
    __m128i vlo = _mm256_castsi256_si128(v);
    __m128i vhi = _mm256_extractf128_si256(v, 1);
// There is no _mm_srai_epi64. To emulate it, add 0x80000000
// before shifting, and subtract the shifted 0x80000000 after
// shifting
#if 0
      __m128i signmask01 = _mm_sub_epi64(_mm_set1_epi64x(0),
                                         _mm_srli_epi64(vlo, 63));
      __m128i signmask23 = _mm_sub_epi64(_mm_set1_epi64x(0),
                                         _mm_srli_epi64(vhi, 63));
      vlo = _mm_xor_si128(signmask01, vlo);
      vhi = _mm_xor_si128(signmask23, vhi);
      vlo = _mm_srli_epi64(vlo, n);
      vhi = _mm_srli_epi64(vhi, n);
      vlo = _mm_xor_si128(signmask01, vlo);
      vhi = _mm_xor_si128(signmask23, vhi);
#else
    // Convert signed to unsiged
    vlo = _mm_add_epi64(vlo, _mm_set1_epi64x(U(1) << (bits - 1)));
    vhi = _mm_add_epi64(vhi, _mm_set1_epi64x(U(1) << (bits - 1)));
    // Shift
    vlo = _mm_srli_epi64(vlo, n);
    vhi = _mm_srli_epi64(vhi, n);
    // Undo conversion
    vlo = _mm_sub_epi64(vlo, _mm_set1_epi64x(U(1) << (bits - 1 - n)));
    vhi = _mm_sub_epi64(vhi, _mm_set1_epi64x(U(1) << (bits - 1 - n)));
#endif
    return _mm256_insertf128_si256(_mm256_castsi128_si256(vlo), vhi, 1);
#endif
  }
  intvec_t operator<<(int_t n) const {
#ifdef __AVX2__
    return _mm256_slli_epi64(v, n);
#else
    __m128i vlo = _mm256_castsi256_si128(v);
    __m128i vhi = _mm256_extractf128_si256(v, 1);
    vlo = _mm_slli_epi64(vlo, n);
    vhi = _mm_slli_epi64(vhi, n);
    return _mm256_insertf128_si256(_mm256_castsi128_si256(vlo), vhi, 1);
#endif
  }
  intvec_t &operator>>=(int_t n) { return *this = *this >> n; }
  intvec_t &operator<<=(int_t n) { return *this = *this << n; }

  intvec_t lsr(intvec_t n) const {
#ifdef __AVX2__
    return _mm256_srlv_epi64(v, n.v);
#else
    intvec_t r;
    for (int i = 0; i < size; ++i) {
      r.set_elt(i, U((*this)[i]) >> U(n[i]));
    }
    return r;
#endif
  }
  intvec_t rotate(intvec_t n) const;
  intvec_t operator>>(intvec_t n) const {
#ifdef __AVX2__
    // See operator>> above
    intvec_t offset = U(1) << (bits - 1);
    return (*this + offset).lsr(n) - offset.lsr(n);
#else
    intvec_t r;
    for (int i = 0; i < size; ++i) {
      r.set_elt(i, (*this)[i] >> n[i]);
    }
    return r;
#endif
  }
  intvec_t operator<<(intvec_t n) const {
#ifdef __AVX2__
    return _mm256_sllv_epi64(v, n.v);
#else
    intvec_t r;
    for (int i = 0; i < size; ++i) {
      r.set_elt(i, (*this)[i] << n[i]);
    }
    return r;
#endif
  }
  intvec_t &operator>>=(intvec_t n) { return *this = *this >> n; }
  intvec_t &operator<<=(intvec_t n) { return *this = *this << n; }

  intvec_t clz() const;
  intvec_t popcount() const;

  boolvec_t operator==(intvec_t const &x) const {
#ifdef __AVX2__
    return _mm256_castsi256_pd(_mm256_cmpeq_epi64(v, x.v));
#else
    return !(*this != x);
#endif
  }
  boolvec_t operator!=(intvec_t const &x) const {
#ifdef __AVX2__
    return !(*this == x);
#else
    return (*this ^ x).convert_bool();
#endif
  }
  boolvec_t operator<(intvec_t const &x) const {
#ifdef __AVX2__
    return _mm256_castsi256_pd(_mm256_cmpgt_epi64(x.v, v));
#else
    // return (*this - x).as_bool();
    boolvec_t r;
    for (int i = 0; i < size; ++i) {
      r.set_elt(i, (*this)[i] < x[i]);
    }
    return r;
#endif
  }
  boolvec_t operator<=(intvec_t const &x) const { return !(*this > x); }
  boolvec_t operator>(intvec_t const &x) const { return x < *this; }
  boolvec_t operator>=(intvec_t const &x) const { return !(*this < x); }

  intvec_t abs() const;
  boolvec_t isignbit() const { return as_bool(); }
  intvec_t max(intvec_t x) const;
  intvec_t min(intvec_t x) const;
};

template <> struct realvec<double, 4> : floatprops<double> {
  static int const size = 4;
  typedef real_t scalar_t;
  typedef __m256d vector_t;
  static int const alignment = sizeof(vector_t);

  static char const *name() {
#ifdef __AVX2__
    return "<AVX2:4*double>";
#else
    return "<AVX:4*double>";
#endif
  }
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
  realvec(real_t a) : v(_mm256_set1_pd(a)) {}
  realvec(real_t const *as) : v(_mm256_set_pd(as[3], as[2], as[1], as[0])) {}

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
    return _mm256_load_pd(p);
  }
  static realvec_t loadu(real_t const *p) { return _mm256_loadu_pd(p); }
  static realvec_t loadu(real_t const *p, std::ptrdiff_t ioff) {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    if (ioff % realvec::size == 0)
      return loada(p + ioff);
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
    _mm256_store_pd(p, v);
  }
  void storeu(real_t *p) const { return _mm256_storeu_pd(p, v); }
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
      _mm256_maskstore_pd(p, m.m.as_int(), v);
    }
  }
  void storeu(real_t *p, mask_t const &m) const {
    if (__builtin_expect(m.all_m, true)) {
      storeu(p);
    } else {
      for (int d = 0; d < size; ++d) {
        if (m.m[d])
          p[d] = (*this)[d];
      }
    }
  }
  void storeu(real_t *p, std::ptrdiff_t ioff, mask_t const &m) const {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    if (ioff % realvec::size == 0)
      return storea(p + ioff, m);
    storeu(p + ioff, m);
  }

  intvec_t as_int() const { return _mm256_castpd_si256(v); }
  intvec_t convert_int() const {
    intvec_t r;
    for (int d = 0; d < size; ++d) {
      r.set_elt(d, floatprops::convert_int((*this)[d]));
    }
    return r;
  }

  realvec_t operator+() const { return *this; }
  realvec_t operator-() const { return RV(0.0) - *this; }

  realvec_t operator+(realvec_t x) const { return _mm256_add_pd(v, x.v); }
  realvec_t operator-(realvec_t x) const { return _mm256_sub_pd(v, x.v); }
  realvec_t operator*(realvec_t x) const { return _mm256_mul_pd(v, x.v); }
  realvec_t operator/(realvec_t x) const { return _mm256_div_pd(v, x.v); }

  realvec_t &operator+=(realvec_t const &x) { return *this = *this + x; }
  realvec_t &operator-=(realvec_t const &x) { return *this = *this - x; }
  realvec_t &operator*=(realvec_t const &x) { return *this = *this * x; }
  realvec_t &operator/=(realvec_t const &x) { return *this = *this / x; }

  real_t maxval() const {
    // return vml_std::fmax(vml_std::fmax((*this)[0], (*this)[1]),
    //                      vml_std::fmax((*this)[2], (*this)[3]));
    realvec_t x0123 = *this;
    realvec_t x1032 = _mm256_shuffle_pd(x0123, x0123, 0b0101);
    realvec_t y0022 = x0123.fmax(x1032);
    return vml_std::fmax(y0022[0], y0022[2]);
  }
  real_t minval() const {
    // return vml_std::fmin(vml_std::fmin((*this)[0], (*this)[1]),
    //                      vml_std::fmin((*this)[2], (*this)[3]));
    realvec_t x0123 = *this;
    realvec_t x1032 = _mm256_shuffle_pd(x0123, x0123, 0b0101);
    realvec_t y0022 = x0123.fmin(x1032);
    return vml_std::fmin(y0022[0], y0022[2]);
  }
  real_t prod() const {
    // return (*this)[0] * (*this)[1] * (*this)[2] * (*this)[3];
    realvec_t x0123 = *this;
    realvec_t x1032 = _mm256_shuffle_pd(x0123, x0123, 0b0101);
    realvec_t y0022 = x0123 * x1032;
    return y0022[0] * y0022[2];
  }
  real_t sum() const {
    // return (*this)[0] + (*this)[1] + (*this)[2] + (*this)[3];
    // __m256d x = _mm256_hadd_pd(v, v);
    // __m128d xlo = _mm256_extractf128_pd(x, 0);
    // __m128d xhi = _mm256_extractf128_pd(x, 1);
    realvec_t x = *this;
    x = _mm256_hadd_pd(x.v, x.v);
    return x[0] + x[2];
  }

  boolvec_t operator==(realvec_t const &x) const {
    return _mm256_cmp_pd(v, x.v, _CMP_EQ_OQ);
  }
  boolvec_t operator!=(realvec_t const &x) const {
    return _mm256_cmp_pd(v, x.v, _CMP_NEQ_UQ); // Note: _UQ here
  }
  boolvec_t operator<(realvec_t const &x) const {
    return _mm256_cmp_pd(v, x.v, _CMP_LT_OQ);
  }
  boolvec_t operator<=(realvec_t const &x) const {
    return _mm256_cmp_pd(v, x.v, _CMP_LE_OQ);
  }
  boolvec_t operator>(realvec_t const &x) const {
    return _mm256_cmp_pd(v, x.v, _CMP_GT_OQ);
  }
  boolvec_t operator>=(realvec_t const &x) const {
    return _mm256_cmp_pd(v, x.v, _CMP_GE_OQ);
  }

  realvec_t acos() const { return MF::vml_acos(*this); }
  realvec_t acosh() const { return MF::vml_acosh(*this); }
  realvec_t asin() const { return MF::vml_asin(*this); }
  realvec_t asinh() const { return MF::vml_asinh(*this); }
  realvec_t atan() const { return MF::vml_atan(*this); }
  realvec_t atan2(realvec_t y) const { return MF::vml_atan2(*this, y); }
  realvec_t atanh() const { return MF::vml_atanh(*this); }
  realvec_t cbrt() const { return MF::vml_cbrt(*this); }
  realvec_t ceil() const { return _mm256_ceil_pd(v); }
  realvec_t copysign(realvec_t y) const { return MF::vml_copysign(*this, y); }
  realvec_t cos() const { return MF::vml_cos(*this); }
  realvec_t cosh() const { return MF::vml_cosh(*this); }
  realvec_t exp() const { return MF::vml_exp(*this); }
  realvec_t exp10() const { return MF::vml_exp10(*this); }
  realvec_t exp2() const { return MF::vml_exp2(*this); }
  realvec_t expm1() const { return MF::vml_expm1(*this); }
  realvec_t fabs() const { return MF::vml_fabs(*this); }
  realvec_t fdim(realvec_t y) const { return MF::vml_fdim(*this, y); }
  realvec_t floor() const { return _mm256_floor_pd(v); }
  realvec_t fma(realvec_t y, realvec_t z) const {
#if defined(__FMA4__)
    realvec_t x = *this;
    return _mm256_macc_pd(x, y, z);
#elif defined(__FMA__)
    realvec_t x = *this;
    return _mm256_fmadd_pd(x, y, z);
#else
    return MF::vml_fma(*this, y, z);
#endif
  }
  /* OpenCL spec: if one argument is NaN, return the second
   * instructions: if any argument is NaN, return the second
   * ... so we must take care of (x, NaN) arguments case
   */
  realvec_t fmax(realvec_t y) const {
    realvec_t res = _mm256_max_pd(v, y.v);
#if defined VML_HAVE_NAN
    return y.isnan().ifthen(v, res);
#else
    return res;
#endif
  }
  realvec_t fmin(realvec_t y) const {
    realvec_t res = _mm256_min_pd(v, y.v);
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
#ifdef VML_HAVE_NAN
    return _mm256_cmp_pd(v, v, _CMP_UNORD_Q);
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
  realvec_t rcp() const { return _mm256_div_pd(_mm256_set1_pd(1.0), v); }
  realvec_t remainder(realvec_t y) const { return MF::vml_remainder(*this, y); }
  realvec_t rint() const {
    return _mm256_round_pd(v, _MM_FROUND_TO_NEAREST_INT);
  }
  realvec_t round() const { return MF::vml_round(*this); }
  realvec_t rsqrt() const { return MF::vml_rsqrt(*this); }
  boolvec_t signbit() const { return v; }
  realvec_t sin() const { return MF::vml_sin(*this); }
  realvec_t sinh() const { return MF::vml_sinh(*this); }
  realvec_t sqrt() const { return _mm256_sqrt_pd(v); }
  realvec_t tan() const { return MF::vml_tan(*this); }
  realvec_t tanh() const { return MF::vml_tanh(*this); }
  realvec_t trunc() const { return _mm256_round_pd(v, _MM_FROUND_TO_ZERO); }
};

// boolvec definitions

inline intvec<double, 4> boolvec<double, 4>::as_int() const {
  return _mm256_castpd_si256(v);
}

inline intvec<double, 4> boolvec<double, 4>::convert_int() const {
  // return ifthen(v, U(1), U(0));
  return lsr(as_int(), bits - 1);
}

inline boolvec<double, 4> boolvec<double, 4>::ifthen(boolvec_t x,
                                                     boolvec_t y) const {
  return ifthen(x.as_int(), y.as_int()).as_bool();
}

inline intvec<double, 4> boolvec<double, 4>::ifthen(intvec_t x,
                                                    intvec_t y) const {
  return ifthen(x.as_float(), y.as_float()).as_int();
}

inline realvec<double, 4> boolvec<double, 4>::ifthen(realvec_t x,
                                                     realvec_t y) const {
  return _mm256_blendv_pd(y.v, x.v, v);
}

// intvec definitions

inline intvec<double, 4> intvec<double, 4>::abs() const {
  return MF::vml_abs(*this);
}

inline intvec<double, 4> intvec<double, 4>::bitifthen(intvec_t x,
                                                      intvec_t y) const {
  return MF::vml_bitifthen(*this, x, y);
}

inline intvec<double, 4> intvec<double, 4>::clz() const {
  return MF::vml_clz(*this);
}

inline realvec<double, 4> intvec<double, 4>::as_float() const {
  return _mm256_castsi256_pd(v);
}

inline realvec<double, 4> intvec<double, 4>::convert_float() const {
  realvec_t r;
  for (int d = 0; d < size; ++d) {
    r.set_elt(d, floatprops::convert_float((*this)[d]));
  }
  return r;
}

inline intvec<double, 4> intvec<double, 4>::max(intvec_t x) const {
  return MF::vml_max(*this, x);
}

inline intvec<double, 4> intvec<double, 4>::min(intvec_t x) const {
  return MF::vml_min(*this, x);
}

inline intvec<double, 4> intvec<double, 4>::popcount() const {
  return MF::vml_popcount(*this);
}

inline intvec<double, 4> intvec<double, 4>::rotate(int_t n) const {
  return MF::vml_rotate(*this, n);
}

inline intvec<double, 4> intvec<double, 4>::rotate(intvec_t n) const {
  return MF::vml_rotate(*this, n);
}

} // namespace vecmathlib

#endif // #ifndef VEC_AVX_DOUBLE4_H
