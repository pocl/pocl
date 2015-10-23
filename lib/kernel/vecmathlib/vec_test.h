// -*-C++-*-

#ifndef VEC_TEST_H
#define VEC_TEST_H

#include "floatprops.h"
#include "mathfuncs.h"
#include "vec_base.h"

#include <cmath>
#ifndef VML_NO_IOSTREAM
#include <sstream>
#endif

namespace vecmathlib {

template <typename T, int N> struct booltestvec;
template <typename T, int N> struct inttestvec;
template <typename T, int N> struct realtestvec;

template <typename T, int N> struct booltestvec : floatprops<T> {
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
  // booltestvec(vector_t x): v(x) {}
  booltestvec(bool a) {
    for (int d = 0; d < size; ++d)
      v[d] = a;
  }
  booltestvec(bool const *as) {
    for (int d = 0; d < size; ++d)
      v[d] = as[d];
  }

  bool operator[](int n) const { return v[n]; }
  boolvec_t &set_elt(int n, bool a) { return v[n] = a, *this; }

  intvec_t as_int() const;      // defined after inttestvec
  intvec_t convert_int() const; // defined after inttestvec

  boolvec_t operator!() const {
    boolvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = !v[d];
    return res;
  }

  boolvec_t operator&&(boolvec_t x) const {
    boolvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = v[d] && x.v[d];
    return res;
  }
  boolvec_t operator||(boolvec_t x) const {
    boolvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = v[d] || x.v[d];
    return res;
  }
  boolvec_t operator==(boolvec_t x) const {
    boolvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = v[d] == x.v[d];
    return res;
  }
  boolvec_t operator!=(boolvec_t x) const {
    boolvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = v[d] != x.v[d];
    return res;
  }

  bool all() const {
    bool res = v[0];
    for (int d = 1; d < size; ++d)
      res = res && v[d];
    return res;
  }
  bool any() const {
    bool res = v[0];
    for (int d = 1; d < size; ++d)
      res = res || v[d];
    return res;
  }

  // ifthen(condition, then-value, else-value)
  boolvec_t ifthen(boolvec_t x, boolvec_t y) const;
  intvec_t ifthen(intvec_t x, intvec_t y) const;    // defined after inttestvec
  realvec_t ifthen(realvec_t x, realvec_t y) const; // defined after realtestvec
};

template <typename T, int N> struct inttestvec : floatprops<T> {
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
  // inttestvec(vector_t x): v(x) {}
  inttestvec(int_t a) {
    for (int d = 0; d < size; ++d)
      v[d] = a;
  }
  inttestvec(int_t const *as) {
    for (int d = 0; d < size; ++d)
      v[d] = as[d];
  }
  static intvec_t iota() {
    intvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = d;
    return res;
  }

  int_t operator[](int n) const { return v[n]; }
  intvec_t &set_elt(int n, int_t a) { return v[n] = a, *this; }

  boolvec_t as_bool() const { return convert_bool(); }
  boolvec_t convert_bool() const {
    // result: convert_bool(0)=false, convert_bool(else)=true
    boolvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = v[d];
    return res;
  }
  realvec_t as_float() const;      // defined after realtestvec
  realvec_t convert_float() const; // defined after realtestvec

  intvec_t operator+() const {
    intvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = +v[d];
    return res;
  }
  intvec_t operator-() const {
    intvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = -v[d];
    return res;
  }

  intvec_t &operator+=(intvec_t const &x) {
    for (int d = 0; d < size; ++d)
      v[d] += x.v[d];
    return *this;
  }
  intvec_t &operator-=(intvec_t const &x) {
    for (int d = 0; d < size; ++d)
      v[d] -= x.v[d];
    return *this;
  }
  intvec_t &operator*=(intvec_t const &x) {
    for (int d = 0; d < size; ++d)
      v[d] *= x.v[d];
    return *this;
  }
  intvec_t &operator/=(intvec_t const &x) {
    for (int d = 0; d < size; ++d)
      v[d] /= x.v[d];
    return *this;
  }
  intvec_t &operator%=(intvec_t const &x) {
    for (int d = 0; d < size; ++d)
      v[d] %= x.v[d];
    return *this;
  }

  intvec_t operator+(intvec_t x) const {
    intvec_t res = *this;
    return res += x;
  }
  intvec_t operator-(intvec_t x) const {
    intvec_t res = *this;
    return res -= x;
  }
  intvec_t operator*(intvec_t x) const {
    intvec_t res = *this;
    return res *= x;
  }
  intvec_t operator/(intvec_t x) const {
    intvec_t res = *this;
    return res /= x;
  }
  intvec_t operator%(intvec_t x) const {
    intvec_t res = *this;
    return res %= x;
  }

  intvec_t operator~() const {
    intvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = ~v[d];
    return res;
  }

  intvec_t &operator&=(intvec_t const &x) {
    for (int d = 0; d < size; ++d)
      v[d] &= x.v[d];
    return *this;
  }
  intvec_t &operator|=(intvec_t const &x) {
    for (int d = 0; d < size; ++d)
      v[d] |= x.v[d];
    return *this;
  }
  intvec_t &operator^=(intvec_t const &x) {
    for (int d = 0; d < size; ++d)
      v[d] ^= x.v[d];
    return *this;
  }

  intvec_t operator&(intvec_t x) const {
    intvec_t res = *this;
    return res &= x;
  }
  intvec_t operator|(intvec_t x) const {
    intvec_t res = *this;
    return res |= x;
  }
  intvec_t operator^(intvec_t x) const {
    intvec_t res = *this;
    return res ^= x;
  }

  intvec_t bitifthen(intvec_t x, intvec_t y) const {
    return MF::vml_bitifthen(*this, x, y);
  }

  intvec_t lsr(int_t n) const {
    intvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = I(U(v[d]) >> U(n));
    return res;
  }
  intvec_t rotate(int_t n) const { return MF::vml_rotate(*this, n); }
  intvec_t &operator>>=(int_t n) {
    for (int d = 0; d < size; ++d)
      v[d] >>= n;
    return *this;
  }
  intvec_t &operator<<=(int_t n) {
    for (int d = 0; d < size; ++d)
      v[d] <<= n;
    return *this;
  }
  intvec_t operator>>(int_t n) const {
    intvec_t res = *this;
    return res >>= n;
  }
  intvec_t operator<<(int_t n) const {
    intvec_t res = *this;
    return res <<= n;
  }

  intvec_t lsr(intvec_t n) const {
    intvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = I(U(v[d]) >> U(n.v[d]));
    return res;
  }
  intvec_t rotate(intvec_t n) const { return MF::vml_rotate(*this, n); }
  intvec_t &operator>>=(intvec_t n) {
    for (int d = 0; d < size; ++d)
      v[d] >>= n.v[d];
    return *this;
  }
  intvec_t &operator<<=(intvec_t n) {
    for (int d = 0; d < size; ++d)
      v[d] <<= n.v[d];
    return *this;
  }
  intvec_t operator>>(intvec_t n) const {
    intvec_t res = *this;
    return res >>= n;
  }
  intvec_t operator<<(intvec_t n) const {
    intvec_t res = *this;
    return res <<= n;
  }

  intvec_t clz() const { return MF::vml_clz(*this); }
  intvec_t popcount() const { return MF::vml_popcount(*this); }

  boolvec_t operator==(intvec_t const &x) const {
    boolvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = v[d] == x.v[d];
    return res;
  }
  boolvec_t operator!=(intvec_t const &x) const {
    boolvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = v[d] != x.v[d];
    return res;
  }
  boolvec_t operator<(intvec_t const &x) const {
    boolvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = v[d] < x.v[d];
    return res;
  }
  boolvec_t operator<=(intvec_t const &x) const {
    boolvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = v[d] <= x.v[d];
    return res;
  }
  boolvec_t operator>(intvec_t const &x) const {
    boolvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = v[d] > x.v[d];
    return res;
  }
  boolvec_t operator>=(intvec_t const &x) const {
    boolvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = v[d] >= x.v[d];
    return res;
  }

  intvec_t abs() const { return MF::vml_abs(*this); }
  boolvec_t isignbit() const { return MF::vml_isignbit(*this); }
  intvec_t max(intvec_t x) const { return MF::vml_max(*this, x); }
  intvec_t min(intvec_t x) const { return MF::vml_min(*this, x); }
};

template <typename T, int N> struct realtestvec : floatprops<T> {
  typedef typename floatprops<T>::int_t int_t;
  typedef typename floatprops<T>::uint_t uint_t;
  typedef typename floatprops<T>::real_t real_t;

  static int const size = N;
  typedef real_t scalar_t;
  typedef real_t vector_t[size];
  static int const alignment = sizeof(real_t);

#ifndef VML_NO_IOSTREAM
  static char const *name() {
    static std::string name_;
    if (name_.empty()) {
      std::stringstream buf;
      buf << "<VML:" << N << "*" << FP::name() << ">";
      name_ = buf.str();
    }
    return name_.c_str();
  }
#endif
  void barrier() {
#if defined __GNUC__ && !defined __clang__ && !defined __ICC
// GCC crashes when +X is used as constraint
#if defined __SSE2__
    for (int d = 0; d < size; ++d)
      __asm__("" : "+x"(v[d]));
#elif defined __PPC64__ // maybe also __PPC__
    for (int d = 0; d < size; ++d)
      __asm__("" : "+f"(v[d]));
#elif defined __arm__
    for (int d = 0; d < size; ++d)
      __asm__("" : "+w"(v[d]));
#else
#error "Floating point barrier undefined on this architecture"
#endif
#elif defined __clang__
    for (int d = 0; d < size; ++d)
      __asm__("" : "+X"(v[d]));
#elif defined __ICC
    for (int d = 0; d < size; ++d) {
      real_t tmp = v[d];
      __asm__("" : "+X"(tmp));
      v[d] = tmp;
    }
#elif defined __IBMCPP__
    for (int d = 0; d < size; ++d)
      __asm__("" : "+f"(v[d]));
#else
#error "Floating point barrier undefined on this architecture"
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
  // realtestvec(vector_t x): v(x) {}
  realtestvec(real_t a) {
    for (int d = 0; d < size; ++d)
      v[d] = a;
  }
  realtestvec(real_t const *as) {
    for (int d = 0; d < size; ++d)
      v[d] = as[d];
  }

  real_t operator[](int n) const { return v[n]; }
  realvec_t &set_elt(int n, real_t a) { return v[n] = a, *this; }

  typedef vecmathlib::mask_t<realvec_t> mask_t;

  static realvec_t loada(real_t const *p) {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    return loadu(p);
  }
  static realvec_t loadu(real_t const *p) {
    realvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = p[d];
    return res;
  }
  static realvec_t loadu(real_t const *p, size_t ioff) {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    return loadu(p + ioff);
  }
  realvec_t loada(real_t const *p, mask_t const &m) const {
    return m.m.ifthen(loada(p), *this);
  }
  realvec_t loadu(real_t const *p, mask_t const &m) const {
    return m.m.ifthen(loadu(p), *this);
  }
  realvec_t loadu(real_t const *p, size_t ioff, mask_t const &m) const {
    return m.m.ifthen(loadu(p, ioff), *this);
  }

  void storea(real_t *p) const {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    storeu(p);
  }
  void storeu(real_t *p) const {
    for (int d = 0; d < size; ++d)
      p[d] = v[d];
  }
  void storeu(real_t *p, size_t ioff) const {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    storeu(p + ioff);
  }
  void storea(real_t *p, mask_t const &m) const {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    storeu(p, m);
  }
  void storeu(real_t *p, mask_t const &m) const {
    for (int d = 0; d < size; ++d)
      if (m.m[d])
        p[d] = v[d];
  }
  void storeu(real_t *p, size_t ioff, mask_t const &m) const {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    storeu(p + ioff, m);
  }

  intvec_t as_int() const {
    intvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = FP::as_int(v[d]);
    return res;
  }
  intvec_t convert_int() const { return MF::vml_convert_int(*this); }

  realvec_t operator+() const {
    realvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = +v[d];
    return res;
  }
  realvec_t operator-() const {
    realvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = -v[d];
    return res;
  }

  realvec_t &operator+=(realvec_t const &x) {
    for (int d = 0; d < size; ++d)
      v[d] += x.v[d];
    return *this;
  }
  realvec_t &operator-=(realvec_t const &x) {
    for (int d = 0; d < size; ++d)
      v[d] -= x.v[d];
    return *this;
  }
  realvec_t &operator*=(realvec_t const &x) {
    for (int d = 0; d < size; ++d)
      v[d] *= x.v[d];
    return *this;
  }
  realvec_t &operator/=(realvec_t const &x) {
    for (int d = 0; d < size; ++d)
      v[d] /= x.v[d];
    return *this;
  }

  realvec_t operator+(realvec_t x) const {
    realvec_t res = *this;
    return res += x;
  }
  realvec_t operator-(realvec_t x) const {
    realvec_t res = *this;
    return res -= x;
  }
  realvec_t operator*(realvec_t x) const {
    realvec_t res = *this;
    return res *= x;
  }
  realvec_t operator/(realvec_t x) const {
    realvec_t res = *this;
    return res /= x;
  }

  real_t maxval() const {
    real_t res = v[0];
    for (int d = 1; d < size; ++d)
      res = vml_std::fmax(res, v[d]);
    return res;
  }
  real_t minval() const {
    real_t res = v[0];
    for (int d = 1; d < size; ++d)
      res = vml_std::fmin(res, v[d]);
    return res;
  }
  real_t prod() const {
    real_t res = v[0];
    for (int d = 1; d < size; ++d)
      res *= v[d];
    return res;
  }
  real_t sum() const {
    real_t res = v[0];
    for (int d = 1; d < size; ++d)
      res += v[d];
    return res;
  }

  boolvec_t operator==(realvec_t const &x) const {
    boolvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = v[d] == x.v[d];
    return res;
  }
  boolvec_t operator!=(realvec_t const &x) const {
    boolvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = v[d] != x.v[d];
    return res;
  }
  boolvec_t operator<(realvec_t const &x) const {
    boolvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = v[d] < x.v[d];
    return res;
  }
  boolvec_t operator<=(realvec_t const &x) const {
    boolvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = v[d] <= x.v[d];
    return res;
  }
  boolvec_t operator>(realvec_t const &x) const {
    boolvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = v[d] > x.v[d];
    return res;
  }
  boolvec_t operator>=(realvec_t const &x) const {
    boolvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = v[d] >= x.v[d];
    return res;
  }

  realvec_t acos() const { return MF::vml_acos(*this); }
  realvec_t acosh() const { return MF::vml_acosh(*this); }
  realvec_t asin() const { return MF::vml_asin(*this); }
  realvec_t asinh() const { return MF::vml_asinh(*this); }
  realvec_t atan() const { return MF::vml_atan(*this); }
  realvec_t atan2(realvec_t y) const { return MF::vml_atan2(*this, y); }
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
  realvec_t fma(realvec_t y, realvec_t z) const {
    return MF::vml_fma(*this, y, z);
  }
  realvec_t fmax(realvec_t y) const { return MF::vml_fmax(*this, y); }
  realvec_t fmin(realvec_t y) const { return MF::vml_fmin(*this, y); }
  realvec_t fmod(realvec_t y) const { return MF::vml_fmod(*this, y); }
  realvec_t frexp(intvec_t *r) const { return MF::vml_frexp(*this, r); }
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
  intvec_t lrint() const { return MF::vml_lrint(*this); }
  realvec_t mad(realvec_t y, realvec_t z) const {
    return MF::vml_mad(*this, y, z);
  }
  realvec_t nextafter(realvec_t y) const { return MF::vml_nextafter(*this, y); }
  realvec_t pow(realvec_t y) const { return MF::vml_pow(*this, y); }
  realvec_t rcp() const { return MF::vml_rcp(*this); }
  realvec_t remainder(realvec_t y) const { return MF::vml_remainder(*this, y); }
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

// booltestvec definitions

template <typename T, int N>
inline typename booltestvec<T, N>::intvec_t booltestvec<T, N>::as_int() const {
  return convert_int();
}

template <typename T, int N>
inline typename booltestvec<T, N>::intvec_t
booltestvec<T, N>::convert_int() const {
  intvec_t res;
  for (int d = 0; d < size; ++d)
    res.v[d] = v[d];
  return res;
}

template <typename T, int N>
inline typename booltestvec<T, N>::boolvec_t
booltestvec<T, N>::ifthen(boolvec_t x, boolvec_t y) const {
  boolvec_t res;
  for (int d = 0; d < size; ++d)
    res.v[d] = v[d] ? x.v[d] : y.v[d];
  return res;
}

template <typename T, int N>
inline typename booltestvec<T, N>::intvec_t
booltestvec<T, N>::ifthen(intvec_t x, intvec_t y) const {
  intvec_t res;
  for (int d = 0; d < size; ++d)
    res.v[d] = v[d] ? x.v[d] : y.v[d];
  return res;
}

template <typename T, int N>
inline typename booltestvec<T, N>::realvec_t
booltestvec<T, N>::ifthen(realvec_t x, realvec_t y) const {
  realvec_t res;
  for (int d = 0; d < size; ++d)
    res.v[d] = v[d] ? x.v[d] : y.v[d];
  return res;
}

// inttestvec definitions

template <typename T, int N>
inline typename inttestvec<T, N>::realvec_t inttestvec<T, N>::as_float() const {
  realvec_t res;
  for (int d = 0; d < size; ++d)
    res.v[d] = FP::as_float(v[d]);
  return res;
}

template <typename T, int N>
inline typename inttestvec<T, N>::realvec_t
inttestvec<T, N>::convert_float() const {
  return MF::vml_convert_float(*this);
}

// Wrappers

// booltestvec wrappers

template <typename real_t, int size>
inline inttestvec<real_t, size> as_int(booltestvec<real_t, size> x) {
  return x.as_int();
}

template <typename real_t, int size>
inline inttestvec<real_t, size> convert_int(booltestvec<real_t, size> x) {
  return x.convert_int();
}

template <typename real_t, int size>
inline bool all(booltestvec<real_t, size> x) {
  return x.all();
}

template <typename real_t, int size>
inline bool any(booltestvec<real_t, size> x) {
  return x.any();
}

template <typename real_t, int size>
inline booltestvec<real_t, size> ifthen(booltestvec<real_t, size> c,
                                        booltestvec<real_t, size> x,
                                        booltestvec<real_t, size> y) {
  return c.ifthen(x, y);
}

template <typename real_t, int size>
inline inttestvec<real_t, size> ifthen(booltestvec<real_t, size> c,
                                       inttestvec<real_t, size> x,
                                       inttestvec<real_t, size> y) {
  return c.ifthen(x, y);
}

template <typename real_t, int size>
inline realtestvec<real_t, size> ifthen(booltestvec<real_t, size> c,
                                        realtestvec<real_t, size> x,
                                        realtestvec<real_t, size> y) {
  return c.ifthen(x, y);
}

// inttestvec wrappers

template <typename real_t, int size>
inline inttestvec<real_t, size> abs(inttestvec<real_t, size> x) {
  return x.abs();
}

template <typename real_t, int size>
inline booltestvec<real_t, size> as_bool(inttestvec<real_t, size> x) {
  return x.as_bool();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> as_float(inttestvec<real_t, size> x) {
  return x.as_float();
}

template <typename real_t, int size>
inline inttestvec<real_t, size> bitifthen(inttestvec<real_t, size> x,
                                          inttestvec<real_t, size> y,
                                          inttestvec<real_t, size> z) {
  return x.bitifthen(y, z);
}

template <typename real_t, int size>
inline inttestvec<real_t, size> clz(inttestvec<real_t, size> x) {
  return x.clz();
}

template <typename real_t, int size>
inline booltestvec<real_t, size> convert_bool(inttestvec<real_t, size> x) {
  return x.convert_bool();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> convert_float(inttestvec<real_t, size> x) {
  return x.convert_float();
}

template <typename real_t, int size>
inline booltestvec<real_t, size> isignbit(inttestvec<real_t, size> x) {
  return x.isignbit();
}

template <typename real_t, int size>
inline inttestvec<real_t, size>
lsr(inttestvec<real_t, size> x, typename inttestvec<real_t, size>::int_t n) {
  return x.lsr(n);
}

template <typename real_t, int size>
inline inttestvec<real_t, size> lsr(inttestvec<real_t, size> x,
                                    inttestvec<real_t, size> n) {
  return x.lsr(n);
}

template <typename real_t, int size>
inline inttestvec<real_t, size> max(inttestvec<real_t, size> x,
                                    inttestvec<real_t, size> y) {
  return x.max(y);
}

template <typename real_t, int size>
inline inttestvec<real_t, size> min(inttestvec<real_t, size> x,
                                    inttestvec<real_t, size> y) {
  return x.min(y);
}

template <typename real_t, int size>
inline inttestvec<real_t, size> popcount(inttestvec<real_t, size> x) {
  return x.popcount();
}

template <typename real_t, int size>
inline inttestvec<real_t, size>
rotate(inttestvec<real_t, size> x, typename inttestvec<real_t, size>::int_t n) {
  return x.rotate(n);
}

template <typename real_t, int size>
inline inttestvec<real_t, size> rotate(inttestvec<real_t, size> x,
                                       inttestvec<real_t, size> n) {
  return x.rotate(n);
}

// realtestvec wrappers

template <typename real_t, int size>
inline realtestvec<real_t, size>
loada(real_t const *p, realtestvec<real_t, size> x,
      typename realtestvec<real_t, size>::mask_t const &m) {
  return x.loada(p, m);
}

template <typename real_t, int size>
inline realtestvec<real_t, size>
loadu(real_t const *p, realtestvec<real_t, size> x,
      typename realtestvec<real_t, size>::mask_t const &m) {
  return x.loadu(p, m);
}

template <typename real_t, int size>
inline realtestvec<real_t, size>
loadu(real_t const *p, size_t ioff, realtestvec<real_t, size> x,
      typename realtestvec<real_t, size>::mask_t const &m) {
  return x.loadu(p, ioff, m);
}

template <typename real_t, int size>
inline void storea(realtestvec<real_t, size> x, real_t *p) {
  return x.storea(p);
}

template <typename real_t, int size>
inline void storeu(realtestvec<real_t, size> x, real_t *p) {
  return x.storeu(p);
}

template <typename real_t, int size>
inline void storeu(realtestvec<real_t, size> x, real_t *p, size_t ioff) {
  return x.storeu(p, ioff);
}

template <typename real_t, int size>
inline void storea(realtestvec<real_t, size> x, real_t *p,
                   typename realtestvec<real_t, size>::mask_t const &m) {
  return x.storea(p, m);
}

template <typename real_t, int size>
inline void storeu(realtestvec<real_t, size> x, real_t *p,
                   typename realtestvec<real_t, size>::mask_t const &m) {
  return x.storeu(p, m);
}

template <typename real_t, int size>
inline void storeu(realtestvec<real_t, size> x, real_t *p, size_t ioff,
                   typename realtestvec<real_t, size>::mask_t const &m) {
  return x.storeu(p, ioff, m);
}

template <typename real_t, int size>
inline inttestvec<real_t, size> as_int(realtestvec<real_t, size> x) {
  return x.as_int();
}

template <typename real_t, int size>
inline inttestvec<real_t, size> convert_int(realtestvec<real_t, size> x) {
  return x.convert_int();
}

template <typename real_t, int size>
inline real_t maxval(realtestvec<real_t, size> x) {
  return x.maxval();
}

template <typename real_t, int size>
inline real_t minval(realtestvec<real_t, size> x) {
  return x.minval();
}

template <typename real_t, int size>
inline real_t prod(realtestvec<real_t, size> x) {
  return x.prod();
}

template <typename real_t, int size>
inline real_t sum(realtestvec<real_t, size> x) {
  return x.sum();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> acos(realtestvec<real_t, size> x) {
  return x.acos();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> acosh(realtestvec<real_t, size> x) {
  return x.acosh();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> asin(realtestvec<real_t, size> x) {
  return x.asin();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> asinh(realtestvec<real_t, size> x) {
  return x.asinh();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> atan(realtestvec<real_t, size> x) {
  return x.atan();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> atan2(realtestvec<real_t, size> x,
                                       realtestvec<real_t, size> y) {
  return x.atan2(y);
}

template <typename real_t, int size>
inline realtestvec<real_t, size> atanh(realtestvec<real_t, size> x) {
  return x.atanh();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> cbrt(realtestvec<real_t, size> x) {
  return x.cbrt();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> ceil(realtestvec<real_t, size> x) {
  return x.ceil();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> copysign(realtestvec<real_t, size> x,
                                          realtestvec<real_t, size> y) {
  return x.copysign(y);
}

template <typename real_t, int size>
inline realtestvec<real_t, size> cos(realtestvec<real_t, size> x) {
  return x.cos();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> cosh(realtestvec<real_t, size> x) {
  return x.cosh();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> exp(realtestvec<real_t, size> x) {
  return x.exp();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> exp10(realtestvec<real_t, size> x) {
  return x.exp10();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> exp2(realtestvec<real_t, size> x) {
  return x.exp2();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> expm1(realtestvec<real_t, size> x) {
  return x.expm1();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> fabs(realtestvec<real_t, size> x) {
  return x.fabs();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> floor(realtestvec<real_t, size> x) {
  return x.floor();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> fdim(realtestvec<real_t, size> x,
                                      realtestvec<real_t, size> y) {
  return x.fdim(y);
}

template <typename real_t, int size>
inline realtestvec<real_t, size> fma(realtestvec<real_t, size> x,
                                     realtestvec<real_t, size> y,
                                     realtestvec<real_t, size> z) {
  return x.fma(y, z);
}

template <typename real_t, int size>
inline realtestvec<real_t, size> fmax(realtestvec<real_t, size> x,
                                      realtestvec<real_t, size> y) {
  return x.fmax(y);
}

template <typename real_t, int size>
inline realtestvec<real_t, size> fmin(realtestvec<real_t, size> x,
                                      realtestvec<real_t, size> y) {
  return x.fmin(y);
}

template <typename real_t, int size>
inline realtestvec<real_t, size> fmod(realtestvec<real_t, size> x,
                                      realtestvec<real_t, size> y) {
  return x.fmod(y);
}

template <typename real_t, int size>
inline realtestvec<real_t, size> frexp(realtestvec<real_t, size> x,
                                       inttestvec<real_t, size> *r) {
  return x.frexp(r);
}

template <typename real_t, int size>
inline realtestvec<real_t, size> hypot(realtestvec<real_t, size> x,
                                       realtestvec<real_t, size> y) {
  return x.hypot(y);
}

template <typename real_t, int size>
inline inttestvec<real_t, size> ilogb(realtestvec<real_t, size> x) {
  return x.ilogb();
}

template <typename real_t, int size>
inline booltestvec<real_t, size> isfinite(realtestvec<real_t, size> x) {
  return x.isfinite();
}

template <typename real_t, int size>
inline booltestvec<real_t, size> isinf(realtestvec<real_t, size> x) {
  return x.isinf();
}

template <typename real_t, int size>
inline booltestvec<real_t, size> isnan(realtestvec<real_t, size> x) {
  return x.isnan();
}

template <typename real_t, int size>
inline booltestvec<real_t, size> isnormal(realtestvec<real_t, size> x) {
  return x.isnormal();
}

template <typename real_t, int size>
inline realtestvec<real_t, size>
ldexp(realtestvec<real_t, size> x, typename inttestvec<real_t, size>::int_t n) {
  return x.ldexp(n);
}

template <typename real_t, int size>
inline realtestvec<real_t, size> ldexp(realtestvec<real_t, size> x,
                                       inttestvec<real_t, size> n) {
  return x.ldexp(n);
}

template <typename real_t, int size>
inline realtestvec<real_t, size> log(realtestvec<real_t, size> x) {
  return x.log();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> log10(realtestvec<real_t, size> x) {
  return x.log10();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> log1p(realtestvec<real_t, size> x) {
  return x.log1p();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> log2(realtestvec<real_t, size> x) {
  return x.log2();
}

template <typename real_t, int size>
inline inttestvec<real_t, size> lrint(realtestvec<real_t, size> x) {
  return x.lrint();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> mad(realtestvec<real_t, size> x,
                                     realtestvec<real_t, size> y,
                                     realtestvec<real_t, size> z) {
  return x.mad(y, z);
}

template <typename real_t, int size>
inline realtestvec<real_t, size> nextafter(realtestvec<real_t, size> x,
                                           realtestvec<real_t, size> y) {
  return x.nextafter(y);
}

template <typename real_t, int size>
inline realtestvec<real_t, size> pow(realtestvec<real_t, size> x,
                                     realtestvec<real_t, size> y) {
  return x.pow(y);
}

template <typename real_t, int size>
inline realtestvec<real_t, size> rcp(realtestvec<real_t, size> x) {
  return x.rcp();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> remainder(realtestvec<real_t, size> x,
                                           realtestvec<real_t, size> y) {
  return x.remainder(y);
}

template <typename real_t, int size>
inline realtestvec<real_t, size> rint(realtestvec<real_t, size> x) {
  return x.rint();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> round(realtestvec<real_t, size> x) {
  return x.round();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> rsqrt(realtestvec<real_t, size> x) {
  return x.rsqrt();
}

template <typename real_t, int size>
inline booltestvec<real_t, size> signbit(realtestvec<real_t, size> x) {
  return x.signbit();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> sin(realtestvec<real_t, size> x) {
  return x.sin();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> sinh(realtestvec<real_t, size> x) {
  return x.sinh();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> sqrt(realtestvec<real_t, size> x) {
  return x.sqrt();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> tan(realtestvec<real_t, size> x) {
  return x.tan();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> tanh(realtestvec<real_t, size> x) {
  return x.tanh();
}

template <typename real_t, int size>
inline realtestvec<real_t, size> trunc(realtestvec<real_t, size> x) {
  return x.trunc();
}

#ifndef VML_NO_IOSTREAM
template <typename real_t, int size>
std::ostream &operator<<(std::ostream &os, booltestvec<real_t, size> const &x) {
  os << "[";
  for (int i = 0; i < size; ++i) {
    if (i != 0)
      os << ",";
    os << x[i];
  }
  os << "]";
  return os;
}

template <typename real_t, int size>
std::ostream &operator<<(std::ostream &os, inttestvec<real_t, size> const &x) {
  os << "[";
  for (int i = 0; i < size; ++i) {
    if (i != 0)
      os << ",";
    os << x[i];
  }
  os << "]";
  return os;
}

template <typename real_t, int size>
std::ostream &operator<<(std::ostream &os, realtestvec<real_t, size> const &x) {
  os << "[";
  for (int i = 0; i < size; ++i) {
    if (i != 0)
      os << ",";
    os << x[i];
  }
  os << "]";
  return os;
}
#endif

} // namespace vecmathlib

#endif // #ifndef VEC_TEST_H
