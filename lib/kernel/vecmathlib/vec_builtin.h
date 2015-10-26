// -*-C++-*-

#ifndef VEC_BUILTIN_H
#define VEC_BUILTIN_H

#include "floatprops.h"
#include "floatbuiltins.h"
#include "mathfuncs.h"
#include "vec_base.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#ifndef VML_NO_IOSTREAM
#include <sstream>
#endif
#include <string>

namespace vecmathlib {

template <typename T, int N> struct boolbuiltinvec;
template <typename T, int N> struct intbuiltinvec;
template <typename T, int N> struct realbuiltinvec;

template <typename T, int N> struct boolbuiltinvec : floatprops<T> {
  typedef typename floatprops<T>::int_t int_t;
  typedef typename floatprops<T>::uint_t uint_t;
  typedef typename floatprops<T>::real_t real_t;

  static const int size = N;
  typedef bool scalar_t;
  typedef int_t bvector_t __attribute__((__ext_vector_type__(N)));
  static const int alignment = sizeof(bvector_t);

  static_assert(size * sizeof(real_t) == sizeof(bvector_t),
                "vector size is wrong");

private:
  // true is -1, false is 0
  static int_t from_bool(bool a) { return -uint_t(a); }
  static bool to_bool(int_t a) { return a; }

public:
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

  bvector_t v;

  boolbuiltinvec() {}
  // Can't have a non-trivial copy constructor; if so, objects won't
  // be passed in registers
  // boolbuiltinvec(const boolbuiltinvec& x): v(x.v) {}
  // boolbuiltinvec& operator=(const boolbuiltinvec& x) { return v=x.v, *this; }
  // Can't have a constructor from bvector_t, since this would
  // conflict with the constructor from bool
  // boolbuiltinvec(bvector_t x): v(x) {}
  static boolvec_t mkvec(bvector_t x) {
    boolvec_t res;
    res.v = x;
    return res;
  }
  boolbuiltinvec(bool a) : v(from_bool(a)) {}
  boolbuiltinvec(const bool *as) {
    for (int d = 0; d < size; ++d)
      set_elt(d, as[d]);
  }

  operator bvector_t() const { return v; }
  bool operator[](int n) const { return to_bool(v[n]); }
  boolvec_t &set_elt(int n, bool a) { return v[n] = from_bool(a), *this; }

  intvec_t as_int() const;      // defined after intbuiltinvec
  intvec_t convert_int() const; // defined after intbuiltinvec

  boolvec_t operator!() const { return mkvec(!v); }

  boolvec_t operator&&(boolvec_t x) const { return mkvec(v && x.v); }
  boolvec_t operator||(boolvec_t x) const { return mkvec(v || x.v); }
  boolvec_t operator==(boolvec_t x) const { return mkvec(v == x.v); }
  boolvec_t operator!=(boolvec_t x) const { return mkvec(v != x.v); }

  bool all() const {
    bool res = (*this)[0];
    for (int d = 1; d < size; ++d)
      res = res && (*this)[d];
    return res;
  }
  bool any() const {
    bool res = (*this)[0];
    for (int d = 1; d < size; ++d)
      res = res || (*this)[d];
    return res;
  }

  // ifthen(condition, then-value, else-value)
  boolvec_t ifthen(boolvec_t x, boolvec_t y) const;
  intvec_t ifthen(intvec_t x, intvec_t y) const; // defined after intbuiltinvec
  realvec_t ifthen(realvec_t x,
                   realvec_t y) const; // defined after realbuiltinvec
};

template <typename T, int N> struct intbuiltinvec : floatprops<T> {
  typedef typename floatprops<T>::int_t int_t;
  typedef typename floatprops<T>::uint_t uint_t;
  typedef typename floatprops<T>::real_t real_t;

  static const int size = N;
  typedef int_t scalar_t;
  typedef int_t ivector_t __attribute__((__ext_vector_type__(N)));
  typedef uint_t uvector_t __attribute__((__ext_vector_type__(N)));
  static const int alignment = sizeof(ivector_t);

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

  ivector_t v;

  intbuiltinvec() {}
  // Can't have a non-trivial copy constructor; if so, objects won't
  // be passed in registers
  // intbuiltinvec(const intbuiltinvec& x): v(x.v) {}
  // intbuiltinvec& operator=(const intbuiltinvec& x) { return v=x.v, *this; }
  // Can't have a constructor from ivector_t, since this would
  // conflict with the constructor from int_t
  // intbuiltinvec(ivector_t x): v(x) {}
  static intvec_t mkvec(ivector_t x) {
    intvec_t res;
    res.v = x;
    return res;
  }
  intbuiltinvec(int_t a) : v(a) {}
  intbuiltinvec(const int_t *as) { std::memcpy(&v, as, sizeof v); }
  static intvec_t iota() {
    intvec_t res;
    for (int d = 0; d < size; ++d)
      res.set_elt(d, d);
    return res;
  }

  int_t operator[](int n) const { return v[n]; }
  intvec_t &set_elt(int n, int_t a) { return v[n] = a, *this; }

  boolvec_t as_bool() const {
    boolvec_t res;
    std::memcpy(&res.v, &v, sizeof res.v);
    return res;
  }
  boolvec_t convert_bool() const { return *this != IV(I(0)); }
  realvec_t as_float() const;      // defined after realbuiltinvec
  realvec_t convert_float() const; // defined after realbuiltinvec

  intvec_t operator+() const { return mkvec(+v); }
  intvec_t operator-() const { return mkvec(-v); }

  intvec_t operator+(intvec_t x) const { return mkvec(v + x.v); }
  intvec_t operator-(intvec_t x) const { return mkvec(v - x.v); }
  intvec_t operator*(intvec_t x) const { return mkvec(v * x.v); }
  intvec_t operator/(intvec_t x) const { return mkvec(v / x.v); }
  intvec_t operator%(intvec_t x) const { return mkvec(v % x.v); }

  intvec_t &operator+=(const intvec_t &x) { return *this = *this + x; }
  intvec_t &operator-=(const intvec_t &x) { return *this = *this - x; }
  intvec_t &operator*=(const intvec_t &x) { return *this = *this * x; }
  intvec_t &operator/=(const intvec_t &x) { return *this = *this / x; }
  intvec_t &operator%=(const intvec_t &x) { return *this = *this % x; }

  intvec_t operator~() const { return mkvec(~v); }

  intvec_t operator&(intvec_t x) const { return mkvec(v & x.v); }
  intvec_t operator|(intvec_t x) const { return mkvec(v | x.v); }
  intvec_t operator^(intvec_t x) const { return mkvec(v ^ x.v); }

  intvec_t &operator&=(const intvec_t &x) { return *this = *this & x; }
  intvec_t &operator|=(const intvec_t &x) { return *this = *this | x; }
  intvec_t &operator^=(const intvec_t &x) { return *this = *this ^ x; }

  intvec_t bitifthen(intvec_t x, intvec_t y) const {
    return MF::vml_bitifthen(*this, x, y);
  }

  intvec_t lsr(int_t n) const { return mkvec(ivector_t(uvector_t(v) >> U(n))); }
  intvec_t rotate(int_t n) const { return MF::vml_rotate(*this, n); }
  intvec_t operator>>(int_t n) const { return mkvec(v >> n); }
  intvec_t operator<<(int_t n) const { return mkvec(v << n); }

  intvec_t &operator>>=(int_t n) { return *this = *this >> n; }
  intvec_t &operator<<=(int_t n) { return *this = *this << n; }

  intvec_t lsr(intvec_t n) const {
    return mkvec(ivector_t(uvector_t(v) >> uvector_t(n.v)));
  }
  intvec_t rotate(intvec_t n) const { return MF::vml_rotate(*this, n); }
  intvec_t operator>>(intvec_t n) const { return mkvec(v >> n.v); }
  intvec_t operator<<(intvec_t n) const { return mkvec(v << n.v); }

  intvec_t &operator>>=(intvec_t n) { return *this = *this >> n; }
  intvec_t &operator<<=(intvec_t n) { return *this = *this << n; }

  intvec_t clz() const {
    intvec_t res;
    for (int d = 0; d < size; ++d) {
      int_t val = (*this)[d];
      int_t cnt = val == 0 ? CHAR_BIT * sizeof val : builtin_clz(U(val));
      res.set_elt(d, cnt);
    }
    return res;
  }
  intvec_t popcount() const {
    intvec_t res;
    for (int d = 0; d < size; ++d) {
      res.set_elt(d, builtin_popcount(U((*this)[d])));
    }
    return res;
  }

  boolvec_t operator==(const intvec_t &x) const {
    return boolvec_t::mkvec(v == x.v);
  }
  boolvec_t operator!=(const intvec_t &x) const {
    return boolvec_t::mkvec(v != x.v);
  }
  boolvec_t operator<(const intvec_t &x) const {
    return boolvec_t::mkvec(v < x.v);
  }
  boolvec_t operator<=(const intvec_t &x) const {
    return boolvec_t::mkvec(v <= x.v);
  }
  boolvec_t operator>(const intvec_t &x) const {
    return boolvec_t::mkvec(v > x.v);
  }
  boolvec_t operator>=(const intvec_t &x) const {
    return boolvec_t::mkvec(v >= x.v);
  }

  intvec_t abs() const {
    intvec_t res;
    for (int d = 0; d < size; ++d)
      res.set_elt(d, builtin_abs((*this)[d]));
    return res;
  }

  boolvec_t isignbit() const { return MF::vml_isignbit(*this); }

  intvec_t max(intvec_t x) const { return MF::vml_max(*this, x); }
  intvec_t min(intvec_t x) const { return MF::vml_min(*this, x); }
};

template <typename T, int N> struct realbuiltinvec : floatprops<T> {
  typedef typename floatprops<T>::int_t int_t;
  typedef typename floatprops<T>::uint_t uint_t;
  typedef typename floatprops<T>::real_t real_t;

  static const int size = N;
  typedef real_t scalar_t;
  typedef real_t vector_t __attribute__((__ext_vector_type__(N)));
  static const int alignment = sizeof(vector_t);

  static_assert(size * sizeof(real_t) == sizeof(vector_t),
                "vector size is wrong");

#ifndef VML_NO_IOSTREAM
  static const char *name() {
    static std::string name_;
    if (name_.empty()) {
      std::stringstream buf;
      buf << "<builtin:" << N << "*" << FP::name() << ">";
      name_ = buf.str();
    }
    return name_.c_str();
  }
#endif
  void barrier() { volatile vector_t x __attribute__((__unused__)) = v; }

  typedef boolbuiltinvec<real_t, size> boolvec_t;
  typedef intbuiltinvec<real_t, size> intvec_t;
  typedef realbuiltinvec realvec_t;

private:
  boolvec_t mapb(bool f(real_t)) const {
    boolvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = f(v[d]);
    return res;
  }
  intvec_t map(int_t f(real_t)) const {
    intvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = f(v[d]);
    return res;
  }
  realvec_t map(real_t f(real_t)) const {
    realvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = f(v[d]);
    return res;
  }
  realvec_t map(real_t f(real_t, int_t), intvec_t x) const {
    realvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = f(v[d], x.v[d]);
    return res;
  }
  realvec_t map(real_t f(real_t, int_t *), intvec_t *x) const {
    realvec_t res;
    for (int d = 0; d < size; ++d) {
      int_t ix;
      res.v[d] = f(v[d], &ix);
      x->set_elt(d, ix);
    }
    return res;
  }
  realvec_t map(real_t f(real_t, real_t), realvec_t x) const {
    realvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = f(v[d], x.v[d]);
    return res;
  }
  realvec_t map(real_t f(real_t, real_t, real_t), realvec_t x,
                realvec_t y) const {
    realvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = f(v[d], x.v[d], y.v[d]);
    return res;
  }

public:
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

  realbuiltinvec() {}
  // Can't have a non-trivial copy constructor; if so, objects won't
  // be passed in registers
  // realbuiltinvec(const realbuiltinvec& x): v(x.v) {}
  // realbuiltinvec& operator=(const realbuiltinvec& x) { return v=x.v, *this; }
  // Can't have a constructor from vector_t, since this would
  // conflict with the constructor from real_t
  // realbuiltinvec(vector_t x): v(x) {}
  static realvec_t mkvec(vector_t x) {
    realvec_t res;
    res.v = x;
    return res;
  }
  realbuiltinvec(real_t a) : v(a) {}
  realbuiltinvec(const real_t *as) { std::memcpy(&v, as, sizeof v); }

  real_t operator[](int n) const { return v[n]; }
  realvec_t &set_elt(int n, real_t a) { return v[n] = a, *this; }

  typedef vecmathlib::mask_t<realvec_t> mask_t;

  static realvec_t loada(const real_t *p) {
    VML_ASSERT(intptr_t(p) % alignment == 0);
#if __has_builtin(__builtin_assume_aligned)
    p = (const real_t *)__builtin_assume_aligned(p, sizeof(realvec_t));
#endif
    return mkvec(*(const vector_t *)p);
  }
  static realvec_t loadu(const real_t *p) {
    // return mkvec(*(const vector_t*)p);
    realvec_t res;
    for (int d = 0; d < size; ++d)
      res.set_elt(d, p[d]);
    return res;
    // realvec_t res;
    // memcpy(&res.v, p, sizeof res.v);
    // return res;
  }
  static realvec_t loadu(const real_t *p, size_t ioff) {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    return loadu(p + ioff);
  }
  realvec_t loada(const real_t *p, const mask_t &m) const {
    return m.m.ifthen(loada(p), *this);
  }
  realvec_t loadu(const real_t *p, const mask_t &m) const {
    return m.m.ifthen(loadu(p), *this);
  }
  realvec_t loadu(const real_t *p, size_t ioff, const mask_t &m) const {
    return m.m.ifthen(loadu(p, ioff), *this);
  }

  void storea(real_t *p) const {
    VML_ASSERT(intptr_t(p) % alignment == 0);
#if __has_builtin(__builtin_assume_aligned)
    p = (real_t *)__builtin_assume_aligned(p, sizeof(realvec_t));
#endif
    *(vector_t *)p = v;
  }
  void storeu(real_t *p) const {
    // *(vector_t*)p = v;
    for (int d = 0; d < size; ++d)
      p[d] = (*this)[d];
    // memcpy(p, &v, sizeof res.v);
  }
  void storeu(real_t *p, size_t ioff) const {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    storeu(p + ioff);
  }
  void storea(real_t *p, const mask_t &m) const {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    storeu(p, m);
  }
  void storeu(real_t *p, const mask_t &m) const {
    for (int d = 0; d < size; ++d)
      if (m.m[d])
        p[d] = (*this)[d];
  }
  void storeu(real_t *p, size_t ioff, const mask_t &m) const {
    VML_ASSERT(intptr_t(p) % alignment == 0);
    storeu(p + ioff, m);
  }

  intvec_t as_int() const {
    intvec_t res;
    std::memcpy(&res.v, &v, sizeof res.v);
    return res;
  }
  intvec_t convert_int() const {
    intvec_t res;
    for (int d = 0; d < size; ++d)
      res.set_elt(d, int_t((*this)[d]));
    return res;
  }

  realvec_t operator+() const { return mkvec(+v); }
  realvec_t operator-() const { return mkvec(-v); }

  realvec_t operator+(realvec_t x) const { return mkvec(v + x.v); }
  realvec_t operator-(realvec_t x) const { return mkvec(v - x.v); }
  realvec_t operator*(realvec_t x) const { return mkvec(v * x.v); }
  realvec_t operator/(realvec_t x) const { return mkvec(v / x.v); }

  realvec_t &operator+=(const realvec_t &x) { return *this = *this + x; }
  realvec_t &operator-=(const realvec_t &x) { return *this = *this - x; }
  realvec_t &operator*=(const realvec_t &x) { return *this = *this * x; }
  realvec_t &operator/=(const realvec_t &x) { return *this = *this / x; }

  real_t maxval() const {
    real_t res = v[0];
    for (int d = 1; d < size; ++d) {
      res = builtin_fmax(res, (*this)[d]);
    }
    return res;
  }
  real_t minval() const {
    real_t res = v[0];
    for (int d = 1; d < size; ++d) {
      res = builtin_fmin(res, (*this)[d]);
    }
    return res;
  }
  real_t prod() const {
    real_t res = (*this)[0];
    for (int d = 1; d < size; ++d)
      res *= (*this)[d];
    return res;
  }
  real_t sum() const {
    real_t res = (*this)[0];
    for (int d = 1; d < size; ++d)
      res += (*this)[d];
    return res;
  }

  boolvec_t operator==(const realvec_t &x) const {
    return boolvec_t::mkvec(v == x.v);
  }
  boolvec_t operator!=(const realvec_t &x) const {
    return boolvec_t::mkvec(v != x.v);
  }
  boolvec_t operator<(const realvec_t &x) const {
    return boolvec_t::mkvec(v < x.v);
  }
  boolvec_t operator<=(const realvec_t &x) const {
    return boolvec_t::mkvec(v <= x.v);
  }
  boolvec_t operator>(const realvec_t &x) const {
    return boolvec_t::mkvec(v > x.v);
  }
  boolvec_t operator>=(const realvec_t &x) const {
    return boolvec_t::mkvec(v >= x.v);
  }

  realvec_t acos() const { return map(builtin_acos); }
  realvec_t acosh() const { return map(builtin_acosh); }
  realvec_t asin() const { return map(builtin_asin); }
  realvec_t asinh() const { return map(builtin_asinh); }
  realvec_t atan() const { return map(builtin_atan); }
  realvec_t atan2(realvec_t y) const { return map(builtin_atan2, y); }
  realvec_t atanh() const { return map(builtin_atanh); }
  realvec_t cbrt() const { return map(builtin_cbrt); }
  realvec_t ceil() const { return map(builtin_ceil); }
  realvec_t copysign(realvec_t y) const { return map(builtin_copysign, y); }
  realvec_t cos() const { return map(builtin_cos); }
  realvec_t cosh() const { return map(builtin_cosh); }
  realvec_t exp() const { return map(builtin_exp); }
  realvec_t exp10() const { return MF::vml_exp10(*this); }
  realvec_t exp2() const { return map(builtin_exp2); }
  realvec_t expm1() const { return map(builtin_expm1); }
  realvec_t fabs() const { return map(builtin_fabs); }
  realvec_t fdim(realvec_t y) const { return map(builtin_fdim, y); }
  realvec_t floor() const { return map(builtin_floor); }
  realvec_t fma(realvec_t y, realvec_t z) const {
    return map(builtin_fma, y, z);
  }
  realvec_t fmax(realvec_t y) const { return map(builtin_fmax, y); }
  realvec_t fmin(realvec_t y) const { return map(builtin_fmin, y); }
  realvec_t fmod(realvec_t y) const { return map(builtin_fmod, y); }
  realvec_t frexp(intvec_t *r) const {
    realvec_t res;
    intvec_t exp;
    for (int d = 0; d < size; ++d) {
      real_t val = (*this)[d];
      int iexp;
      res.set_elt(d, __builtin_frexp(val, &iexp));
      int_t jexp = int_t(iexp);
      if (__builtin_isinf(val))
        jexp = std::numeric_limits<int_t>::max();
      if (__builtin_isnan(val))
        jexp = std::numeric_limits<int_t>::min();
      exp.set_elt(d, jexp);
    }
    *r = exp;
    return res;
  }
  realvec_t hypot(realvec_t y) const { return map(builtin_hypot, y); }
  intvec_t ilogb() const {
    intvec_t res;
    for (int d = 0; d < size; ++d) {
      real_t val = (*this)[d];
      int iexp = __builtin_ilogb(val);
      int_t jexp = int_t(iexp);
      if (val == R(0.0))
        jexp = std::numeric_limits<int_t>::min();
      if (__builtin_isinf(val))
        jexp = std::numeric_limits<int_t>::max();
      if (__builtin_isnan(val))
        jexp = std::numeric_limits<int_t>::min();
      res.set_elt(d, jexp);
    }
    return res;
  }
  boolvec_t isfinite() const {
    boolvec_t res;
    for (int d = 0; d < size; ++d) {
      res.set_elt(d, builtin_isfinite((*this)[d]) != 0);
    }
    return res;
  }
  boolvec_t isinf() const {
    boolvec_t res;
    for (int d = 0; d < size; ++d) {
      res.set_elt(d, builtin_isinf((*this)[d]) != 0);
    }
    return res;
  }
  boolvec_t isnan() const {
    boolvec_t res;
    for (int d = 0; d < size; ++d) {
      res.set_elt(d, builtin_isnan((*this)[d]) != 0);
    }
    return res;
  }
  boolvec_t isnormal() const {
    boolvec_t res;
    for (int d = 0; d < size; ++d) {
      res.set_elt(d, builtin_isnormal((*this)[d]) != 0);
    }
    return res;
  }
  realvec_t ldexp(int_t n) const {
    realvec_t res;
    for (int d = 0; d < size; ++d) {
      res.set_elt(d, builtin_ldexp((*this)[d], int(n)));
    }
    return res;
  }
  realvec_t ldexp(intvec_t n) const {
    realvec_t res;
    for (int d = 0; d < size; ++d) {
      res.set_elt(d, builtin_ldexp((*this)[d], int(n[d])));
    }
    return res;
  }
  realvec_t log() const { return map(builtin_log); }
  realvec_t log10() const { return map(builtin_log10); }
  realvec_t log1p() const { return map(builtin_log1p); }
  realvec_t log2() const { return map(builtin_log2); }
  intvec_t lrint() const {
    if (sizeof(int_t) <= sizeof(long)) {
      return map(builtin_lrint);
    } else if (sizeof(int_t) <= sizeof(long long)) {
      return map(builtin_llrint);
    }
    __builtin_unreachable();
  }
  realvec_t mad(realvec_t y, realvec_t z) const {
    return MF::vml_mad(*this, y, z);
  }
  realvec_t nextafter(realvec_t y) const { return map(builtin_nextafter, y); }
  realvec_t pow(realvec_t y) const { return map(builtin_pow, y); }
  realvec_t rcp() const { return RV(1.0) / *this; }
  realvec_t remainder(realvec_t y) const { return map(builtin_remainder, y); }
  realvec_t rint() const { return map(builtin_rint); }
  realvec_t round() const { return map(builtin_round); }
  realvec_t rsqrt() const { return RV(1.0) / sqrt(); }
  boolvec_t signbit() const {
    boolvec_t res;
    for (int d = 0; d < size; ++d) {
      res.set_elt(d, builtin_signbit((*this)[d]) != 0);
    }
    return res;
  }
  realvec_t sin() const { return map(builtin_sin); }
  realvec_t sinh() const { return map(builtin_sinh); }
  realvec_t sqrt() const { return map(builtin_sqrt); }
  realvec_t tan() const { return map(builtin_tan); }
  realvec_t tanh() const { return map(builtin_tanh); }
  realvec_t trunc() const { return map(builtin_trunc); }
};

// boolbuiltinvec definitions

template <typename T, int N>
inline typename boolbuiltinvec<T, N>::intvec_t
boolbuiltinvec<T, N>::as_int() const {
  intvec_t res;
  std::memcpy(&res.v, &v, sizeof res.v);
  return res;
}

template <typename T, int N>
inline typename boolbuiltinvec<T, N>::intvec_t
boolbuiltinvec<T, N>::convert_int() const {
  return -as_int();
}

template <typename T, int N>
inline typename boolbuiltinvec<T, N>::boolvec_t
boolbuiltinvec<T, N>::ifthen(boolvec_t x, boolvec_t y) const {
  // return v ? x.v : y.v;
  boolvec_t res;
  for (int d = 0; d < size; ++d)
    res.set_elt(d, (*this)[d] ? x[d] : y[d]);
  return res;
}

template <typename T, int N>
inline typename boolbuiltinvec<T, N>::intvec_t
boolbuiltinvec<T, N>::ifthen(intvec_t x, intvec_t y) const {
  // return v ? x.v : y.v;
  intvec_t res;
  for (int d = 0; d < size; ++d)
    res.set_elt(d, (*this)[d] ? x[d] : y[d]);
  return res;
}

template <typename T, int N>
inline typename boolbuiltinvec<T, N>::realvec_t
boolbuiltinvec<T, N>::ifthen(realvec_t x, realvec_t y) const {
  // return v ? x.v : y.v;
  realvec_t res;
  for (int d = 0; d < size; ++d)
    res.set_elt(d, (*this)[d] ? x[d] : y[d]);
  return res;
}

// intbuiltinvec definitions

template <typename T, int N>
inline typename intbuiltinvec<T, N>::realvec_t
intbuiltinvec<T, N>::as_float() const {
  realvec_t res;
  std::memcpy(&res.v, &v, sizeof res.v);
  return res;
}

template <typename T, int N>
inline typename intbuiltinvec<T, N>::realvec_t
intbuiltinvec<T, N>::convert_float() const {
  realvec_t res;
  for (int d = 0; d < size; ++d)
    res.set_elt(d, real_t((*this)[d]));
  return res;
}

// Wrappers

// boolbuiltinvec wrappers

template <typename real_t, int size>
inline intbuiltinvec<real_t, size> as_int(boolbuiltinvec<real_t, size> x) {
  return x.as_int();
}

template <typename real_t, int size>
inline intbuiltinvec<real_t, size> convert_int(boolbuiltinvec<real_t, size> x) {
  return x.convert_int();
}

template <typename real_t, int size>
inline bool all(boolbuiltinvec<real_t, size> x) {
  return x.all();
}

template <typename real_t, int size>
inline bool any(boolbuiltinvec<real_t, size> x) {
  return x.any();
}

template <typename real_t, int size>
inline boolbuiltinvec<real_t, size> ifthen(boolbuiltinvec<real_t, size> c,
                                           boolbuiltinvec<real_t, size> x,
                                           boolbuiltinvec<real_t, size> y) {
  return c.ifthen(x, y);
}

template <typename real_t, int size>
inline intbuiltinvec<real_t, size> ifthen(boolbuiltinvec<real_t, size> c,
                                          intbuiltinvec<real_t, size> x,
                                          intbuiltinvec<real_t, size> y) {
  return c.ifthen(x, y);
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> ifthen(boolbuiltinvec<real_t, size> c,
                                           realbuiltinvec<real_t, size> x,
                                           realbuiltinvec<real_t, size> y) {
  return c.ifthen(x, y);
}

// intbuiltinvec wrappers

template <typename real_t, int size>
inline intbuiltinvec<real_t, size> abs(intbuiltinvec<real_t, size> x) {
  return x.abs();
}

template <typename real_t, int size>
inline boolbuiltinvec<real_t, size> as_bool(intbuiltinvec<real_t, size> x) {
  return x.as_bool();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> as_float(intbuiltinvec<real_t, size> x) {
  return x.as_float();
}

template <typename real_t, int size>
inline intbuiltinvec<real_t, size> bitifthen(intbuiltinvec<real_t, size> x,
                                             intbuiltinvec<real_t, size> y,
                                             intbuiltinvec<real_t, size> z) {
  return x.bitifthen(y, z);
}

template <typename real_t, int size>
inline intbuiltinvec<real_t, size> clz(intbuiltinvec<real_t, size> x) {
  return x.clz();
}

template <typename real_t, int size>
inline boolbuiltinvec<real_t, size>
convert_bool(intbuiltinvec<real_t, size> x) {
  return x.convert_bool();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size>
convert_float(intbuiltinvec<real_t, size> x) {
  return x.convert_float();
}

template <typename real_t, int size>
inline boolbuiltinvec<real_t, size> isignbit(intbuiltinvec<real_t, size> x) {
  return x.isignbit();
}

template <typename real_t, int size>
inline intbuiltinvec<real_t, size>
lsr(intbuiltinvec<real_t, size> x,
    typename intbuiltinvec<real_t, size>::int_t n) {
  return x.lsr(n);
}

template <typename real_t, int size>
inline intbuiltinvec<real_t, size> lsr(intbuiltinvec<real_t, size> x,
                                       intbuiltinvec<real_t, size> n) {
  return x.lsr(n);
}

template <typename real_t, int size>
inline intbuiltinvec<real_t, size> max(intbuiltinvec<real_t, size> x,
                                       intbuiltinvec<real_t, size> y) {
  return x.max(y);
}

template <typename real_t, int size>
inline intbuiltinvec<real_t, size> min(intbuiltinvec<real_t, size> x,
                                       intbuiltinvec<real_t, size> y) {
  return x.min(y);
}

template <typename real_t, int size>
inline intbuiltinvec<real_t, size> popcount(intbuiltinvec<real_t, size> x) {
  return x.popcount();
}

template <typename real_t, int size>
inline intbuiltinvec<real_t, size>
rotate(intbuiltinvec<real_t, size> x,
       typename intbuiltinvec<real_t, size>::int_t n) {
  return x.rotate(n);
}

template <typename real_t, int size>
inline intbuiltinvec<real_t, size> rotate(intbuiltinvec<real_t, size> x,
                                          intbuiltinvec<real_t, size> n) {
  return x.rotate(n);
}

// realbuiltinvec wrappers

template <typename real_t, int size>
inline realbuiltinvec<real_t, size>
loada(real_t const *p, realbuiltinvec<real_t, size> x,
      typename realbuiltinvec<real_t, size>::mask_t const &m) {
  return x.loada(p, m);
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size>
loadu(real_t const *p, realbuiltinvec<real_t, size> x,
      typename realbuiltinvec<real_t, size>::mask_t const &m) {
  return x.loadu(p, m);
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size>
loadu(real_t const *p, size_t ioff, realbuiltinvec<real_t, size> x,
      typename realbuiltinvec<real_t, size>::mask_t const &m) {
  return x.loadu(p, ioff, m);
}

template <typename real_t, int size>
inline void storea(realbuiltinvec<real_t, size> x, real_t *p) {
  return x.storea(p);
}

template <typename real_t, int size>
inline void storeu(realbuiltinvec<real_t, size> x, real_t *p) {
  return x.storeu(p);
}

template <typename real_t, int size>
inline void storeu(realbuiltinvec<real_t, size> x, real_t *p, size_t ioff) {
  return x.storeu(p, ioff);
}

template <typename real_t, int size>
inline void storea(realbuiltinvec<real_t, size> x, real_t *p,
                   typename realbuiltinvec<real_t, size>::mask_t const &m) {
  return x.storea(p, m);
}

template <typename real_t, int size>
inline void storeu(realbuiltinvec<real_t, size> x, real_t *p,
                   typename realbuiltinvec<real_t, size>::mask_t const &m) {
  return x.storeu(p, m);
}

template <typename real_t, int size>
inline void storeu(realbuiltinvec<real_t, size> x, real_t *p, size_t ioff,
                   typename realbuiltinvec<real_t, size>::mask_t const &m) {
  return x.storeu(p, ioff, m);
}

template <typename real_t, int size>
inline intbuiltinvec<real_t, size> as_int(realbuiltinvec<real_t, size> x) {
  return x.as_int();
}

template <typename real_t, int size>
inline intbuiltinvec<real_t, size> convert_int(realbuiltinvec<real_t, size> x) {
  return x.convert_int();
}

template <typename real_t, int size>
inline real_t maxval(realbuiltinvec<real_t, size> x) {
  return x.maxval();
}

template <typename real_t, int size>
inline real_t minval(realbuiltinvec<real_t, size> x) {
  return x.minval();
}

template <typename real_t, int size>
inline real_t prod(realbuiltinvec<real_t, size> x) {
  return x.prod();
}

template <typename real_t, int size>
inline real_t sum(realbuiltinvec<real_t, size> x) {
  return x.sum();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> acos(realbuiltinvec<real_t, size> x) {
  return x.acos();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> acosh(realbuiltinvec<real_t, size> x) {
  return x.acosh();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> asin(realbuiltinvec<real_t, size> x) {
  return x.asin();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> asinh(realbuiltinvec<real_t, size> x) {
  return x.asinh();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> atan(realbuiltinvec<real_t, size> x) {
  return x.atan();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> atan2(realbuiltinvec<real_t, size> x,
                                          realbuiltinvec<real_t, size> y) {
  return x.atan2(y);
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> atanh(realbuiltinvec<real_t, size> x) {
  return x.atanh();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> cbrt(realbuiltinvec<real_t, size> x) {
  return x.cbrt();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> ceil(realbuiltinvec<real_t, size> x) {
  return x.ceil();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> copysign(realbuiltinvec<real_t, size> x,
                                             realbuiltinvec<real_t, size> y) {
  return x.copysign(y);
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> cos(realbuiltinvec<real_t, size> x) {
  return x.cos();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> cosh(realbuiltinvec<real_t, size> x) {
  return x.cosh();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> exp(realbuiltinvec<real_t, size> x) {
  return x.exp();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> exp10(realbuiltinvec<real_t, size> x) {
  return x.exp10();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> exp2(realbuiltinvec<real_t, size> x) {
  return x.exp2();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> expm1(realbuiltinvec<real_t, size> x) {
  return x.expm1();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> fabs(realbuiltinvec<real_t, size> x) {
  return x.fabs();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> floor(realbuiltinvec<real_t, size> x) {
  return x.floor();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> fdim(realbuiltinvec<real_t, size> x,
                                         realbuiltinvec<real_t, size> y) {
  return x.fdim(y);
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> fma(realbuiltinvec<real_t, size> x,
                                        realbuiltinvec<real_t, size> y,
                                        realbuiltinvec<real_t, size> z) {
  return x.fma(y, z);
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> fmax(realbuiltinvec<real_t, size> x,
                                         realbuiltinvec<real_t, size> y) {
  return x.fmax(y);
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> fmin(realbuiltinvec<real_t, size> x,
                                         realbuiltinvec<real_t, size> y) {
  return x.fmin(y);
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> fmod(realbuiltinvec<real_t, size> x,
                                         realbuiltinvec<real_t, size> y) {
  return x.fmod(y);
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> frexp(realbuiltinvec<real_t, size> x,
                                          intbuiltinvec<real_t, size> *r) {
  return x.frexp(r);
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> hypot(realbuiltinvec<real_t, size> x,
                                          realbuiltinvec<real_t, size> y) {
  return x.hypot(y);
}

template <typename real_t, int size>
inline intbuiltinvec<real_t, size> ilogb(realbuiltinvec<real_t, size> x) {
  return x.ilogb();
}

template <typename real_t, int size>
inline boolbuiltinvec<real_t, size> isfinite(realbuiltinvec<real_t, size> x) {
  return x.isfinite();
}

template <typename real_t, int size>
inline boolbuiltinvec<real_t, size> isinf(realbuiltinvec<real_t, size> x) {
  return x.isinf();
}

template <typename real_t, int size>
inline boolbuiltinvec<real_t, size> isnan(realbuiltinvec<real_t, size> x) {
  return x.isnan();
}

template <typename real_t, int size>
inline boolbuiltinvec<real_t, size> isnormal(realbuiltinvec<real_t, size> x) {
  return x.isnormal();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size>
ldexp(realbuiltinvec<real_t, size> x,
      typename intbuiltinvec<real_t, size>::int_t n) {
  return x.ldexp(n);
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> ldexp(realbuiltinvec<real_t, size> x,
                                          intbuiltinvec<real_t, size> n) {
  return x.ldexp(n);
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> log(realbuiltinvec<real_t, size> x) {
  return x.log();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> log10(realbuiltinvec<real_t, size> x) {
  return x.log10();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> log1p(realbuiltinvec<real_t, size> x) {
  return x.log1p();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> log2(realbuiltinvec<real_t, size> x) {
  return x.log2();
}

template <typename real_t, int size>
inline intbuiltinvec<real_t, size> lrint(realbuiltinvec<real_t, size> x) {
  return x.lrint();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> mad(realbuiltinvec<real_t, size> x,
                                        realbuiltinvec<real_t, size> y,
                                        realbuiltinvec<real_t, size> z) {
  return x.mad(y, z);
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> nextafter(realbuiltinvec<real_t, size> x,
                                              realbuiltinvec<real_t, size> y) {
  return x.nextafter(y);
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> pow(realbuiltinvec<real_t, size> x,
                                        realbuiltinvec<real_t, size> y) {
  return x.pow(y);
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> rcp(realbuiltinvec<real_t, size> x) {
  return x.rcp();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> remainder(realbuiltinvec<real_t, size> x,
                                              realbuiltinvec<real_t, size> y) {
  return x.remainder(y);
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> rint(realbuiltinvec<real_t, size> x) {
  return x.rint();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> round(realbuiltinvec<real_t, size> x) {
  return x.round();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> rsqrt(realbuiltinvec<real_t, size> x) {
  return x.rsqrt();
}

template <typename real_t, int size>
inline boolbuiltinvec<real_t, size> signbit(realbuiltinvec<real_t, size> x) {
  return x.signbit();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> sin(realbuiltinvec<real_t, size> x) {
  return x.sin();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> sinh(realbuiltinvec<real_t, size> x) {
  return x.sinh();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> sqrt(realbuiltinvec<real_t, size> x) {
  return x.sqrt();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> tan(realbuiltinvec<real_t, size> x) {
  return x.tan();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> tanh(realbuiltinvec<real_t, size> x) {
  return x.tanh();
}

template <typename real_t, int size>
inline realbuiltinvec<real_t, size> trunc(realbuiltinvec<real_t, size> x) {
  return x.trunc();
}

#ifndef VML_NO_IOSTREAM
template <typename real_t, int size>
std::ostream &operator<<(std::ostream &os,
                         boolbuiltinvec<real_t, size> const &x) {
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
std::ostream &operator<<(std::ostream &os,
                         intbuiltinvec<real_t, size> const &x) {
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
std::ostream &operator<<(std::ostream &os,
                         realbuiltinvec<real_t, size> const &x) {
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

#endif // #ifndef VEC_BUILTIN_H
