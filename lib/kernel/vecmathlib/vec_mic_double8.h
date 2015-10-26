// -*-C++-*-

#ifndef VEC_MIC_DOUBLE8_H
#define VEC_MIC_DOUBLE8_H

#include "floatprops.h"
#include "mathfuncs.h"
#include "vec_base.h"

#include <cmath>

// MIC intrinsics
#include <immintrin.h>

namespace vecmathlib {

#define VECMATHLIB_HAVE_VEC_DOUBLE_8
template <> struct boolvec<double, 8>;
template <> struct intvec<double, 8>;
template <> struct realvec<double, 8>;

template <> struct boolvec<double, 8> : floatprops<double> {
  static const int size = 8;
  typedef bool scalar_t;
  typedef __mask8 bvector_t;
  static const int alignment = sizeof(bvector_t);

  // static_assert(size * sizeof(real_t) == sizeof(bvector_t),
  //               "vector size is wrong");

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
  // boolvec(const boolvec& x): v(x.v) {}
  // boolvec& operator=(const boolvec& x) { return v=x.v, *this; }
  boolvec(bvector_t x) : v(x) {}
  boolvec(bool a) : v(-bvector_t(a)) {}
  boolvec(const bool *as)
      : v((bvector_t(as[0]) << 0) | (bvector_t(as[1]) << 1) |
          (bvector_t(as[2]) << 2) | (bvector_t(as[3]) << 3) |
          (bvector_t(as[4]) << 4) | (bvector_t(as[5]) << 5) |
          (bvector_t(as[6]) << 6) | (bvector_t(as[7]) << 7)) {}

  operator bvector_t() const { return v; }
  bool operator[](int n) const { return (v >> n) & 1; }
  boolvec &set_elt(int n, bool a) {
    v &= ~(bvector_t(1) << n);
    v |= bvector_t(a) << n;
    return *this;
  }

  intvec_t as_int() const;      // defined after intvec
  intvec_t convert_int() const; // defined after intvec

  boolvec operator!() const { return _mm512_knot(v); }

  boolvec operator&&(boolvec x) const { return _mm512_kand(v, x.v); }
  boolvec operator||(boolvec x) const { return _mm512_kor(v, x.v); }
  boolvec operator==(boolvec x) const { return _mm512_kxnor(v, x.v); }
  boolvec operator!=(boolvec x) const { return _mm512_kxor(v, x.v); }

  bool all() const { return _mm512_kortestc(v, v); }
  bool any() const { return !bool(_mm512_kortestz(v, v)); }

  // ifthen(condition, then-value, else-value)
  boolvec_t ifthen(boolvec_t x, boolvec_t y) const;
  intvec_t ifthen(intvec_t x, intvec_t y) const;    // defined after intvec
  realvec_t ifthen(realvec_t x, realvec_t y) const; // defined after realvec
};

template <> struct intvec<double, 8> : floatprops<double> {
  static const int size = 8;
  typedef int_t scalar_t;
  typedef __m512i ivector_t;
  static const int alignment = sizeof(ivector_t);

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
  // intvec(const intvec& x): v(x.v) {}
  // intvec& operator=(const intvec& x) { return v=x.v, *this; }
  intvec(ivector_t x) : v(x) {}
  intvec(int_t a) : v(_mm512_set1_epi64(a)) {}
  intvec(const int_t *as) {
    v = _mm512_undefined_epi32();
    // v = _mm512_loadunpacklo_epi32(v, as);
    // v = _mm512_loadunpackhi_epi32(v, as+8);
    for (int n = 0; n < size; ++n)
      set_elt(n, as[n]);
  }
  static intvec iota() {
    intvec r;
    for (int n = 0; n < size; ++n)
      r.set_elt(n, n);
    return r;
  }

  operator ivector_t() const { return v; }
  int_t operator[](int n) const {
    return vecmathlib::get_elt<IV, ivector_t, int_t>(v, n);
  }
  intvec_t &set_elt(int n, int_t a) {
    return vecmathlib::set_elt<IV, ivector_t, int_t>(v, n, a), *this;
  }

private:
  static __mmask8 mask16tomask8(__mmask16 m16) {
    // combine 01
    m16 = ((m16 >> 1) | m16) & 0b0011001100110011;
    // combine 0123
    m16 = ((m16 >> 2) | m16) & 0b0000111100001111;
    // combine 01234567
    m16 = ((m16 >> 4) | m16) & 0b0000000011111111;
    return m16;
  }

public:
  boolvec_t as_bool() const { return convert_bool(); }
  boolvec_t convert_bool() const {
    // Result: convert_bool(0)=false, convert_bool(else)=true
    __mmask16 r16 = _mm512_test_epi32_mask(v, v);
    return mask16tomask8(r16);
  }
  realvec_t as_float() const;      // defined after realvec
  realvec_t convert_float() const; // defined after realvec

  // Note: not all arithmetic operations are supported!

  intvec operator+() const { return *this; }
  intvec operator-() const { return IV(I(0)) - *this; }
  intvec operator+(intvec x) const { return _mm512_add_epi64(v, x.v); }
  intvec operator-(intvec x) const { return _mm512_sub_epi64(v, x.v); }

  intvec &operator+=(const intvec &x) { return *this = *this + x; }
  intvec &operator-=(const intvec &x) { return *this = *this - x; }

  intvec operator~() const { return IV(~U(0)) ^ *this; }
  intvec operator&(intvec x) const { return _mm512_and_epi64(v, x.v); }
  intvec operator|(intvec x) const { return _mm512_or_epi64(v, x.v); }
  intvec operator^(intvec x) const { return _mm512_xor_epi64(v, x.v); }

  intvec &operator&=(const intvec &x) { return *this = *this & x; }
  intvec &operator|=(const intvec &x) { return *this = *this | x; }
  intvec &operator^=(const intvec &x) { return *this = *this ^ x; }

  intvec_t bitifthen(intvec_t x, intvec_t y) const;

  intvec lsr(int_t n) const {
    if (n < 32) {
      __m512i vlo = _mm512_srli_epi32(v, n);
      __m512i vhi = _mm512_slli_epi32(v, 32 - n);
      vhi = _mm512_swizzle_epi32(vhi, _MM_SWIZ_REG_CDAB);
      return _mm512_mask_or_epi32(vlo, 0xb0101010101010101, vhi, vlo);
    } else {
      __m512i vlo = _mm512_srli_epi32(v, n - 32);
      __m512i vhi = _mm512_setzero_epi32();
      return _mm512_mask_swizzle_epi32(vhi, 0xb0101010101010101, vlo);
    }
  }
  intvec_t rotate(int_t n) const;
  intvec operator>>(int_t n) const {
    if (n < 32) {
      __mm512i vlo = _mm512_srai_epi32(v, n);
      __mm512i vlo0 = _mm512_srli_epi32(v, n);
      __mm512i vhi = _mm512_slli_epi32(v, 32 - n);
      vhi = _mm512_swizzle_epi32(vhi, _MM_SWIZ_REG_CDAB);
      return _mm512_mask_or_epi32(vlo, 0xb0101010101010101, vhi, vlo0);
    } else {
      __m512i vlo = _mm512_srai_epi32(v, n - 32);
      __m512i vhi = _mm512_srai_epi32(v, 31);
      return _mm512_mask_swizzle_epi32(vhi, 0xb0101010101010101, vlo);
    }
  }
  intvec operator<<(int_t n) const {
    if (n < 32) {
      __m512i vlo = _mm512_srli_epi32(v, n);
      __m512i vhi = _mm512_slli_epi32(v, 32 - n);
      vlo = _mm512_swizzle_epi32(vlo, _MM_SWIZ_REG_CDAB);
      return _mm512_mask_or_epi32(vhi, 0xb1010101010101010, vhi, vlo);
    } else {
      __m512i vlo = _mm512_setzero_epi32();
      __m512i vhi = _mm512_slli_epi32(v, n - 32);
      return _mm512_mask_swizzle_epi32(vhi, 0xb1010101010101010, vlo);
    }
  }
  intvec &operator>>=(int_t n) { return *this = *this >> n; }
  intvec &operator<<=(int_t n) { return *this = *this << n; }

  intvec lsr(intvec n) const {
    // TODO: improve this
    intvec r;
    for (int i = 0; i < size; ++i) {
      r.set_elt(i, U((*this)[i]) >> U(n[i]));
    }
    return r;
  }
  intvec_t rotate(intvec_t n) const;
  intvec operator>>(intvec n) const {
    // TODO: improve this
    intvec r;
    for (int i = 0; i < size; ++i) {
      r.set_elt(i, (*this)[i] >> n[i]);
    }
    return r;
  }
  intvec operator<<(intvec n) const {
    // TODO: improve this
    intvec r;
    for (int i = 0; i < size; ++i) {
      r.set_elt(i, (*this)[i] << n[i]);
    }
    return r;
  }
  intvec &operator>>=(intvec n) { return *this = *this >> n; }
  intvec &operator<<=(intvec n) { return *this = *this << n; }

  intvec_t clz() const {
    // Return 8*sizeof(TYPE) when the input is 0
    intvec_t r;
    for (int i = 0; i < size; ++i) {
      // __lzcnt64
      r.set_elt(i, __builtin_clzll((*this)[i]));
    }
    return r;
  }
  intvec_t popcount() const {
    intvec_t r;
    for (int i = 0; i < size; ++i) {
      // _mm_popcnt_u64
      r.set_elt(i, __builtin_popcountll((*this)[i]));
    }
    return r;
  }

  boolvec_t operator==(const intvec &x) const {
    return mask16tomask8(_mm512_cmp_epi32_mask(v, x.v, _MM_CMPINT_EQ));
  }
  boolvec_t operator!=(const intvec &x) const {
    return mask16tomask8(_mm512_cmp_epi32_mask(v, x.v, _MM_CMPINT_NE));
  }
  boolvec_t operator<(const intvec &x) const {
    return mask16tomask8(_mm512_cmp_epi32_mask(v, x.v, _MM_CMPINT_LT));
  }
  boolvec_t operator<=(const intvec &x) const {
    return mask16tomask8(_mm512_cmp_epi32_mask(v, x.v, _MM_CMPINT_LE));
  }
  boolvec_t operator>(const intvec &x) const {
    return mask16tomask8(_mm512_cmp_epi32_mask(v, x.v, _MM_CMPINT_GT));
  }
  boolvec_t operator>=(const intvec &x) const {
    return mask16tomask8(_mm512_cmp_epi32_mask(v, x.v, _MM_CMPINT_GE));
  }

  intvec_t abs() const;
  boolvec_t isignbit() const;
  intvec_t max(intvec_t x) const;
  intvec_t min(intvec_t x) const;
};

template <> struct realvec<double, 8> : floatprops<double> {
  static const int size = 8;
  typedef real_t scalar_t;
  typedef __m512d vector_t;
  static const int alignment = sizeof(vector_t);

  static const char *name() { return "<MIC:8*double>"; }
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
  // realvec(const realvec& x): v(x.v) {}
  // realvec& operator=(const realvec& x) { return v=x.v, *this; }
  realvec(vector_t x) : v(x) {}
  realvec(real_t a) : v(_mm512_set1_pd(a)) {}
  realvec(const real_t *as) {
    v = _mm512_undefined_pd();
    // v = _mm512_loadunpacklo_pd(v, as);
    // v = _mm512_loadunpackhi_pd(v, as+8);
    for (int n = 0; n < size; ++n)
      set_elt(n, as[n]);
  }

  operator vector_t() const { return v; }
  real_t operator[](int n) const {
    return vecmathlib::get_elt<RV, vector_t, real_t>(v, n);
  }
  realvec_t &set_elt(int n, real_t a) {
    return vecmathlib::set_elt<RV, vector_t, real_t>(v, n, a), *this;
  }

  typedef vecmathlib::mask_t<realvec_t> mask_t;

  static realvec_t loada(const real_t *p) {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    return _mm512_load_pd(p);
  }
  static realvec_t loadu(const real_t *p) {
    realvec_t r(_mm512_undefined_pd());
    r.v = _mm512_loadunpacklo_pd(r.v, p);
    r.v = _mm512_loadunpackhi_pd(r.v, p + 8);
    return r.v;
  }
  static realvec_t loadu(const real_t *p, std::ptrdiff_t ioff) {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    if (ioff % realvec::size == 0)
      return loada(p + ioff);
    return loadu(p + ioff);
  }
  realvec_t loada(const real_t *p, const mask_t &m) const {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    return _mm512_mask_load_pd(v, m.m.v, p);
  }
  realvec_t loadu(const real_t *p, const mask_t &m) const {
    if (__builtin_expect(m.all_m, true)) {
      return loadu(p);
    } else {
      return m.m.ifthen(loadu(p), *this);
    }
  }
  realvec_t loadu(const real_t *p, std::ptrdiff_t ioff, const mask_t &m) const {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    if (ioff % realvec::size == 0)
      return loada(p + ioff, m);
    return loadu(p + ioff, m);
  }

  void storea(real_t *p) const {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    _mm512_store_pd(p, v);
  }
  void storeu(real_t *p) const {
    _mm512_packstorelo_pd(p, v);
    _mm512_packstorehi_pd(p + 8, v);
  }
  void storeu(real_t *p, std::ptrdiff_t ioff) const {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    if (ioff % realvec::size == 0)
      return storea(p + ioff);
    storeu(p + ioff);
  }
  void storea(real_t *p, const mask_t &m) const {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    _mm512_mask_store_pd(p, m.m.v, v);
  }
  void storeu(real_t *p, const mask_t &m) const {
    if (__builtin_expect(m.all_m, true)) {
      storeu(p);
    } else {
      for (int n = 0; n < size; ++n) {
        if (m.m[n])
          p[n] = (*this)[n];
      }
    }
  }
  void storeu(real_t *p, std::ptrdiff_t ioff, const mask_t &m) const {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    if (ioff % realvec::size == 0)
      return storea(p + ioff, m);
    storeu(p + ioff, m);
  }

  intvec_t as_int() const { return _mm512_castpd_si512(v); }
  intvec_t convert_int() const {
    intvec_t r(_mm512_undefined_epi32());
    for (int n = 0; n < size; ++n) {
      r.set_elt(n, floatprops::convert_int((*this)[n]));
    }
    return r;
  }

  realvec operator+() const { return *this; }
  realvec operator-() const { return RV(0.0) - *this; }

  realvec operator+(realvec x) const { return _mm512_add_pd(v, x.v); }
  realvec operator-(realvec x) const { return _mm512_sub_pd(v, x.v); }
  realvec operator*(realvec x) const { return _mm512_mul_pd(v, x.v); }
  realvec operator/(realvec x) const { return _mm512_div_pd(v, x.v); }

  realvec &operator+=(const realvec &x) { return *this = *this + x; }
  realvec &operator-=(const realvec &x) { return *this = *this - x; }
  realvec &operator*=(const realvec &x) { return *this = *this * x; }
  realvec &operator/=(const realvec &x) { return *this = *this / x; }

  real_t maxval() const { returm _mm512_reduce_gmax_pd(v); }
  real_t minval() const { returm _mm512_reduce_gmin_pd(v); }
  real_t prod() const { returm _mm512_reduce_mul_pd(v); }
  real_t sum() const { returm _mm512_reduce_add_pd(v); }

  boolvec_t operator==(const realvec &x) const {
    return _mm512_cmp_pd(v, x.v, _CMP_EQ_OQ);
  }
  boolvec_t operator!=(const realvec &x) const {
    return _mm512_cmp_pd(v, x.v, _CMP_NEQ_UQ); // Note: _UQ here
  }
  boolvec_t operator<(const realvec &x) const {
    return _mm512_cmp_pd(v, x.v, _CMP_LT_OQ);
  }
  boolvec_t operator<=(const realvec &x) const {
    return _mm512_cmp_pd(v, x.v, _CMP_LE_OQ);
  }
  boolvec_t operator>(const realvec &x) const {
    return _mm512_cmp_pd(v, x.v, _CMP_GT_OQ);
  }
  boolvec_t operator>=(const realvec &x) const {
    return _mm512_cmp_pd(v, x.v, _CMP_GE_OQ);
  }

  realvec acos() const { return MF::vml_acos(*this); }
  realvec acosh() const { return MF::vml_acosh(*this); }
  realvec asin() const { return MF::vml_asin(*this); }
  realvec asinh() const { return MF::vml_asinh(*this); }
  realvec atan() const { return MF::vml_atan(*this); }
  realvec atan2(realvec y) const { return MF::vml_atan2(*this, y); }
  realvec atanh() const { return MF::vml_atanh(*this); }
  realvec cbrt() const { return MF::vml_cbrt(*this); }
  realvec ceil() const { return _mm512_ceil_pd(v); }
  realvec copysign(realvec y) const { return MF::vml_copysign(*this, y); }
  realvec cos() const { return MF::vml_cos(*this); }
  realvec cosh() const { return MF::vml_cosh(*this); }
  realvec exp() const { return MF::vml_exp(*this); }
  realvec exp10() const { return MF::vml_exp10(*this); }
  realvec exp2() const { return MF::vml_exp2(*this); }
  realvec expm1() const { return MF::vml_expm1(*this); }
  realvec fabs() const { return MF::vml_fabs(*this); }
  realvec fdim(realvec y) const { return MF::vml_fdim(*this, y); }
  realvec floor() const { return _mm512_floor_pd(v); }
  realvec fma(realvec y, realvec z) const {
    return _mm512_fmadd_pd(v, x.v, y.v);
  }
  realvec fmax(realvec y) const { return _mm512_gmax_pd(v, y.v); }
  realvec fmin(realvec y) const { return _mm512_gmin_pd(v, y.v); }
  realvec fmod(realvec y) const { return MF::vml_fmod(*this, y); }
  realvec frexp(intvec_t *r) const { return MF::vml_frexp(*this, r); }
  realvec hypot(realvec y) const { return MF::vml_hypot(*this, y); }
  intvec_t ilogb() const { return MF::vml_ilogb(*this); }
  boolvec_t isfinite() const { return MF::vml_isfinite(*this); }
  boolvec_t isinf() const { return MF::vml_isinf(*this); }
  boolvec_t isnan() const {
#ifdef VML_HAVE_NAN
    return _mm512_cmp_pd(v, v, _CMP_UNORD_Q);
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
  realvec_t mad(realvec_t y, realvec_t z) const {
    return _mm512_fmadd_pd(v, x.v, y.v);
  }
  realvec nextafter(realvec y) const { return MF::vml_nextafter(*this, y); }
  realvec pow(realvec y) const { return MF::vml_pow(*this, y); }
  realvec rcp() const { return _mm512_div_pd(_mm512_set1_pd(1.0), v); }
  realvec remainder(realvec y) const { return MF::vml_remainder(*this, y); }
  realvec rint() const { return _mm512_round_pd(v, _MM_FROUND_TO_NEAREST_INT); }
  realvec round() const { return MF::vml_round(*this); }
  realvec rsqrt() const { return MF::vml_rsqrt(*this); }
  boolvec_t signbit() const { return as_int().signbit(); }
  realvec sin() const { return MF::vml_sin(*this); }
  realvec sinh() const { return MF::vml_sinh(*this); }
  realvec sqrt() const { return _mm512_sqrt_pd(v); }
  realvec tan() const { return MF::vml_tan(*this); }
  realvec tanh() const { return MF::vml_tanh(*this); }
  realvec trunc() const { return _mm512_round_pd(v, _MM_FROUND_TO_ZERO); }
};

// boolvec definitions

inline intvec<double, 4> boolvec<double, 4>::as_int() const {
  return _mm512_castpd_si512(v);
}

inline intvec<double, 4> boolvec<double, 4>::convert_int() const {
  return ifthen(v, IV(I(1)), IV(I(0)));
}

inline boolvec<double, 4> boolvec<double, 4>::ifthen(boolvec_t x,
                                                     boolvec_t y) const {
  return (v & x.v) | (~v & y.v);
}

inline intvec<double, 4> boolvec<double, 4>::ifthen(intvec_t x,
                                                    intvec_t y) const {
  return _mm512_blend_epi64(v, y.v, x.v)
}

inline realvec<double, 4> boolvec<double, 4>::ifthen(realvec_t x,
                                                     realvec_t y) const {
  return _mm512_blend_pd(v, y.v, x.v)
}

// intvec definitions

inline realvec<double, 4> intvec<double, 4>::as_float() const {
  return _mm512_castsi512_pd(v);
}

inline realvec<double, 4> intvec<double, 4>::convert_float() const {
  intvec_t r(_mm512_undefined_pd());
  for (int n = 0; n < size; ++n) {
    r.set_elt(n, floatprops::convert_float((*this)[n]));
  }
  return r;
}

inline intvec<double, 8> intvec<double, 8>::abs() const {
  return MF::vml_abs(*this);
}

inline intvec<double, 8> intvec<double, 8>::bitifthen(intvec_t x,
                                                      intvec_t y) const {
  return MF::vml_bitifthen(*this, x, y);
}

inline boolvec<double, 8> intvec<double, 8>::isignbit() const {
  return MF::vml_isignbit(*this);
}

inline intvec<double, 8> intvec<double, 8>::max(intvec_t x) const {
  return MF::vml_max(*this, x);
}

inline intvec<double, 8> intvec<double, 8>::min(intvec_t x) const {
  return MF::vml_min(*this, x);
}

inline intvec<double, 8> intvec<double, 8>::rotate(int_t n) const {
  return MF::vml_rotate(*this, n);
}

inline intvec<double, 8> intvec<double, 8>::rotate(intvec_t n) const {
  return MF::vml_rotate(*this, n);
}

} // namespace vecmathlib

#endif // #ifndef VEC_MIC_DOUBLE8_H
