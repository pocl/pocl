// -*-C++-*-

#ifndef VEC_PSEUDO_H
#define VEC_PSEUDO_H

#include "floatprops.h"
#include "mathfuncs.h"
#include "vec_base.h"

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdlib>
#ifndef VML_NO_IOSTREAM
#include <sstream>
#endif
#include <string>

namespace vecmathlib {

template <typename T, int N> struct boolpseudovec;
template <typename T, int N> struct intpseudovec;
template <typename T, int N> struct realpseudovec;

template <typename T, int N> struct boolpseudovec : floatprops<T> {
  typedef typename floatprops<T>::int_t int_t;
  typedef typename floatprops<T>::uint_t uint_t;
  typedef typename floatprops<T>::real_t real_t;

  static int const size = N;
  typedef bool scalar_t;
  typedef bool bvector_t[size];
  static int const alignment = sizeof(bool);

  typedef boolpseudovec boolvec_t;
  typedef intpseudovec<real_t, size> intvec_t;
  typedef realpseudovec<real_t, size> realvec_t;

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

  boolpseudovec() {}
  // Can't have a non-trivial copy constructor; if so, objects won't
  // be passed in registers
  // boolpseudovec(boolpseudovec const& x): v(x.v) {}
  // boolpseudovec& operator=(boolpseudovec const& x) { return v=x.v, *this; }
  boolpseudovec(bool a) {
    for (int d = 0; d < size; ++d)
      v[d] = a;
  }
  boolpseudovec(bool const *as) {
    for (int d = 0; d < size; ++d)
      v[d] = as[d];
  }

  bool operator[](int n) const { return v[n]; }
  boolvec_t &set_elt(int n, bool a) { return v[n] = a, *this; }

  intvec_t as_int() const;      // defined after intpseudovec
  intvec_t convert_int() const; // defined after intpseudovec

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
  intvec_t ifthen(intvec_t x, intvec_t y) const; // defined after intpseudovec
  realvec_t ifthen(realvec_t x,
                   realvec_t y) const; // defined after realpseudovec
};

template <typename T, int N> struct intpseudovec : floatprops<T> {
  typedef typename floatprops<T>::int_t int_t;
  typedef typename floatprops<T>::uint_t uint_t;
  typedef typename floatprops<T>::real_t real_t;

  static int const size = N;
  typedef int_t scalar_t;
  typedef int_t ivector_t[size];
  static int const alignment = sizeof(int_t);

  typedef boolpseudovec<real_t, size> boolvec_t;
  typedef intpseudovec intvec_t;
  typedef realpseudovec<real_t, size> realvec_t;

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

  intpseudovec() {}
  // Can't have a non-trivial copy constructor; if so, objects won't
  // be passed in registers
  // intpseudovec(intpseudovec const& x): v(x.v) {}
  // intpseudovec& operator=(intpseudovec const& x) { return v=x.v, *this; }
  intpseudovec(int_t a) {
    for (int d = 0; d < size; ++d)
      v[d] = a;
  }
  intpseudovec(int_t const *as) {
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

  boolvec_t as_bool() const {
    boolvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = v[d];
    return res;
  }
  boolvec_t convert_bool() const {
    // Result: convert_bool(0)=false, convert_bool(else)=true
    boolvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = v[d];
    return res;
  }
  realvec_t as_float() const;      // defined after realpseudovec
  realvec_t convert_float() const; // defined after realpseudovec

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

  intvec_t bitifthen(intvec_t x, intvec_t y) const;

  intvec_t lsr(int_t n) const {
    intvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = I(U(v[d]) >> U(n));
    return res;
  }
  intvec_t rotate(int_t n) const;
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
  intvec_t rotate(intvec_t n) const;
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

  intvec_t clz() const {
    intvec_t res;
#if defined __clang__ || defined __gcc__
    for (int d = 0; d < size; ++d) {
      if (v[d] == 0) {
        res.v[d] = CHAR_BIT * sizeof v[d];
      } else {
        if (sizeof v[d] == sizeof(long long)) {
          res.v[d] = __builtin_clzll(v[d]);
        } else if (sizeof v[d] == sizeof(long)) {
          res.v[d] = __builtin_clzl(v[d]);
        } else if (sizeof v[d] == sizeof(int)) {
          res.v[d] = __builtin_clz(v[d]);
        } else if (sizeof v[d] == sizeof(short)) {
          res.v[d] = __builtin_clzs(v[d]);
        } else if (sizeof v[d] == sizeof(char)) {
          res.v[d] = __builtin_clzs((unsigned short)(unsigned char)v[d]) -
                     CHAR_BIT * (sizeof(short) - sizeof(char));
        } else {
          __builtin_unreachable();
        }
      }
    }
#else
    res = MF::vml_clz(*this);
#endif
    return res;
  }
  intvec_t popcount() const {
    intvec_t res;
#if defined __clang__ || defined __gcc__
    if (sizeof(int_t) == sizeof(long long)) {
      for (int d = 0; d < size; ++d)
        res.v[d] = __builtin_popcountll(v[d]);
    } else if (sizeof(int_t) == sizeof(long)) {
      for (int d = 0; d < size; ++d)
        res.v[d] = __builtin_popcountl(v[d]);
    } else if (sizeof(int_t) <= sizeof(int)) {
      for (int d = 0; d < size; ++d)
        res.v[d] = __builtin_popcount(v[d]);
    } else {
      __builtin_unreachable();
    }
#else
    res = MF::vml_popcount(*this);
#endif
    return res;
  }

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

  intvec_t abs() const {
    intvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = std::abs(v[d]);
    return res;
  }

  boolvec_t isignbit() const {
    boolvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = v[d] < 0;
    return res;
  }

  intvec_t max(intvec_t x) const {
    intvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = std::max(v[d], x.v[d]);
    return res;
  }

  intvec_t min(intvec_t x) const {
    intvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = std::min(v[d], x.v[d]);
    return res;
  }
};

template <typename T, int N> struct realpseudovec : floatprops<T> {
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
      buf << "<libm:" << N << "*" << FP::name() << ">";
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

  typedef boolpseudovec<real_t, size> boolvec_t;
  typedef intpseudovec<real_t, size> intvec_t;
  typedef realpseudovec realvec_t;

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

  realpseudovec() {}
  // Can't have a non-trivial copy constructor; if so, objects won't
  // be passed in registers
  // realpseudovec(realpseudovec const& x): v(x.v) {}
  // realpseudovec& operator=(realpseudovec const& x) { return v=x.v, *this; }
  realpseudovec(real_t a) {
    for (int d = 0; d < size; ++d)
      v[d] = a;
  }
  realpseudovec(real_t const *as) {
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
  intvec_t convert_int() const {
    intvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = FP::convert_int(v[d]);
    return res;
  }

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

  realvec_t acos() const { return map(vml_std::acos); }
  realvec_t acosh() const { return map(vml_std::acosh); }
  realvec_t asin() const { return map(vml_std::asin); }
  realvec_t asinh() const { return map(vml_std::asinh); }
  realvec_t atan() const { return map(vml_std::atan); }
  realvec_t atan2(realvec_t y) const { return MF::vml_atan2(*this, y); }
  realvec_t atanh() const { return map(vml_std::atanh); }
  realvec_t cbrt() const { return map(vml_std::cbrt); }
  realvec_t ceil() const { return map(vml_std::ceil); }
  realvec_t copysign(realvec_t y) const { return map(vml_std::copysign, y); }
  realvec_t cos() const { return map(vml_std::cos); }
  realvec_t cosh() const { return map(vml_std::cosh); }
  realvec_t exp() const { return map(vml_std::exp); }
  realvec_t exp10() const {
    realvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = vml_std::exp(R(M_LN10) * v[d]);
    return res;
  }
  realvec_t exp2() const { return map(vml_std::exp2); }
  realvec_t expm1() const { return map(vml_std::expm1); }
  realvec_t fabs() const { return map(vml_std::fabs); }
  realvec_t fdim(realvec_t y) const { return map(vml_std::fdim, y); }
  realvec_t floor() const { return map(vml_std::floor); }
  realvec_t fma(realvec_t y, realvec_t z) const {
    return map(vml_std::fma, y, z);
  }
  realvec_t fmax(realvec_t y) const { return map(vml_std::fmax, y); }
  realvec_t fmin(realvec_t y) const { return map(vml_std::fmin, y); }
  realvec_t fmod(realvec_t y) const { return map(vml_std::fmod, y); }
  realvec_t frexp(intvec_t *ires) const {
    realvec_t res;
    for (int d = 0; d < size; ++d) {
      int iri;
      real_t r = vml_std::frexp(v[d], &iri);
      int_t ir = iri;
#if defined VML_HAVE_INF
      if (vml_std::isinf(v[d]))
        ir = std::numeric_limits<int_t>::max();
#endif
#if defined VML_HAVE_NAN
      if (vml_std::isnan(v[d]))
        ir = std::numeric_limits<int_t>::min();
#endif
      res.v[d] = r;
      ires->v[d] = ir;
    }
    return res;
  }
  realvec_t hypot(realvec_t y) const { return map(vml_std::hypot, y); }
  intvec_t ilogb() const {
    intvec_t res;
    for (int d = 0; d < size; ++d) {
      int_t r = vml_std::ilogb(v[d]);
      typedef std::numeric_limits<int_t> NL;
      if (FP_ILOGB0 != NL::min() and v[d] == R(0.0)) {
        r = NL::min();
#if defined VML_HAVE_INF
      } else if (INT_MAX != NL::max() and vml_std::isinf(v[d])) {
        r = NL::max();
#endif
#if defined VML_HAVE_NAN
      } else if (FP_ILOGBNAN != NL::min() and vml_std::isnan(v[d])) {
        r = NL::min();
#endif
      }
      res.v[d] = r;
    }
    return res;
  }
  boolvec_t isfinite() const { return mapb(vml_std::isfinite); }
  boolvec_t isinf() const { return mapb(vml_std::isinf); }
  boolvec_t isnan() const { return mapb(vml_std::isnan); }
  boolvec_t isnormal() const { return mapb(vml_std::isnormal); }
  realvec_t ldexp(int_t n) const {
    realvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = vml_std::ldexp(v[d], n);
    return res;
  }
  realvec_t ldexp(intvec_t n) const {
    realvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = vml_std::ldexp(v[d], n.v[d]);
    return res;
  }
  realvec_t log() const { return map(vml_std::log); }
  realvec_t log10() const { return map(vml_std::log10); }
  realvec_t log1p() const { return map(vml_std::log1p); }
  realvec_t log2() const { return map(vml_std::log2); }
  intvec_t lrint() const {
    realvec_t res;
    if (sizeof(int_t) <= sizeof(long)) {
      for (int d = 0; d < size; ++d)
        res.v[d] = vml_std::lrint(v[d]);
    } else if (sizeof(int_t) <= sizeof(long long)) {
      for (int d = 0; d < size; ++d)
        res.v[d] = vml_std::llrint(v[d]);
    } else {
      __builtin_unreachable();
    }
    return res;
  }
  realvec_t mad(realvec_t y, realvec_t z) const {
    return MF::vml_mad(*this, y, z);
  }
  realvec_t nextafter(realvec_t y) const { return map(vml_std::nextafter, y); }
  realvec_t pow(realvec_t y) const { return map(vml_std::pow, y); }
  realvec_t rcp() const {
    realvec_t res;
    for (int d = 0; d < size; ++d)
      res.v[d] = R(1.0) / v[d];
    return res;
  }
  realvec_t remainder(realvec_t y) const { return map(vml_std::remainder, y); }
  realvec_t rint() const { return map(vml_std::rint); }
  realvec_t round() const { return map(vml_std::round); }
  realvec_t rsqrt() const { return sqrt().rcp(); }
  boolvec_t signbit() const { return mapb(vml_std::signbit); }
  realvec_t sin() const { return map(vml_std::sin); }
  realvec_t sinh() const { return map(vml_std::sinh); }
  realvec_t sqrt() const { return map(vml_std::sqrt); }
  realvec_t tan() const { return map(vml_std::tan); }
  realvec_t tanh() const { return map(vml_std::tanh); }
  realvec_t trunc() const { return map(vml_std::trunc); }
};

// boolpseudovec definitions

template <typename T, int N>
inline typename boolpseudovec<T, N>::intvec_t
boolpseudovec<T, N>::as_int() const {
  return convert_int();
}

template <typename T, int N>
inline typename boolpseudovec<T, N>::intvec_t
boolpseudovec<T, N>::convert_int() const {
  intvec_t res;
  for (int d = 0; d < size; ++d)
    res.v[d] = v[d];
  return res;
}

template <typename T, int N>
inline typename boolpseudovec<T, N>::boolvec_t
boolpseudovec<T, N>::ifthen(boolvec_t x, boolvec_t y) const {
  boolvec_t res;
  for (int d = 0; d < size; ++d)
    res.v[d] = v[d] ? x.v[d] : y.v[d];
  return res;
}

template <typename T, int N>
inline typename boolpseudovec<T, N>::intvec_t
boolpseudovec<T, N>::ifthen(intvec_t x, intvec_t y) const {
  intvec_t res;
  for (int d = 0; d < size; ++d)
    res.v[d] = v[d] ? x.v[d] : y.v[d];
  return res;
}

template <typename T, int N>
inline typename boolpseudovec<T, N>::realvec_t
boolpseudovec<T, N>::ifthen(realvec_t x, realvec_t y) const {
  realvec_t res;
  for (int d = 0; d < size; ++d)
    res.v[d] = v[d] ? x.v[d] : y.v[d];
  return res;
}

// intpseudovec definitions

template <typename T, int N>
inline typename intpseudovec<T, N>::realvec_t
intpseudovec<T, N>::as_float() const {
  realvec_t res;
  for (int d = 0; d < size; ++d)
    res.v[d] = FP::as_float(v[d]);
  return res;
}

template <typename T, int N>
inline intpseudovec<T, N> intpseudovec<T, N>::bitifthen(intvec_t x,
                                                        intvec_t y) const {
  return MF::vml_bitifthen(*this, x, y);
}

template <typename T, int N>
inline typename intpseudovec<T, N>::realvec_t
intpseudovec<T, N>::convert_float() const {
  realvec_t res;
  for (int d = 0; d < size; ++d)
    res.v[d] = FP::convert_float(v[d]);
  return res;
}

template <typename T, int N>
inline intpseudovec<T, N> intpseudovec<T, N>::rotate(int_t n) const {
  return MF::vml_rotate(*this, n);
}

template <typename T, int N>
inline intpseudovec<T, N> intpseudovec<T, N>::rotate(intvec_t n) const {
  return MF::vml_rotate(*this, n);
}

// Wrappers

// boolpseudovec wrappers

template <typename real_t, int size>
inline intpseudovec<real_t, size> as_int(boolpseudovec<real_t, size> x) {
  return x.as_int();
}

template <typename real_t, int size>
inline intpseudovec<real_t, size> convert_int(boolpseudovec<real_t, size> x) {
  return x.convert_int();
}

template <typename real_t, int size>
inline bool all(boolpseudovec<real_t, size> x) {
  return x.all();
}

template <typename real_t, int size>
inline bool any(boolpseudovec<real_t, size> x) {
  return x.any();
}

template <typename real_t, int size>
inline boolpseudovec<real_t, size> ifthen(boolpseudovec<real_t, size> c,
                                          boolpseudovec<real_t, size> x,
                                          boolpseudovec<real_t, size> y) {
  return c.ifthen(x, y);
}

template <typename real_t, int size>
inline intpseudovec<real_t, size> ifthen(boolpseudovec<real_t, size> c,
                                         intpseudovec<real_t, size> x,
                                         intpseudovec<real_t, size> y) {
  return c.ifthen(x, y);
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> ifthen(boolpseudovec<real_t, size> c,
                                          realpseudovec<real_t, size> x,
                                          realpseudovec<real_t, size> y) {
  return c.ifthen(x, y);
}

// intpseudovec wrappers

template <typename real_t, int size>
inline intpseudovec<real_t, size> abs(intpseudovec<real_t, size> x) {
  return x.abs();
}

template <typename real_t, int size>
inline boolpseudovec<real_t, size> as_bool(intpseudovec<real_t, size> x) {
  return x.as_bool();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> as_float(intpseudovec<real_t, size> x) {
  return x.as_float();
}

template <typename real_t, int size>
inline intpseudovec<real_t, size> bitifthen(intpseudovec<real_t, size> x,
                                            intpseudovec<real_t, size> y,
                                            intpseudovec<real_t, size> z) {
  return x.bitifthen(y, z);
}

template <typename real_t, int size>
inline intpseudovec<real_t, size> clz(intpseudovec<real_t, size> x) {
  return x.clz();
}

template <typename real_t, int size>
inline boolpseudovec<real_t, size> convert_bool(intpseudovec<real_t, size> x) {
  return x.convert_bool();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> convert_float(intpseudovec<real_t, size> x) {
  return x.convert_float();
}

template <typename real_t, int size>
inline boolpseudovec<real_t, size> isignbit(intpseudovec<real_t, size> x) {
  return x.isignbit();
}

template <typename real_t, int size>
inline intpseudovec<real_t, size>
lsr(intpseudovec<real_t, size> x,
    typename intpseudovec<real_t, size>::int_t n) {
  return x.lsr(n);
}

template <typename real_t, int size>
inline intpseudovec<real_t, size> lsr(intpseudovec<real_t, size> x,
                                      intpseudovec<real_t, size> n) {
  return x.lsr(n);
}

template <typename real_t, int size>
inline intpseudovec<real_t, size> max(intpseudovec<real_t, size> x,
                                      intpseudovec<real_t, size> y) {
  return x.max(y);
}

template <typename real_t, int size>
inline intpseudovec<real_t, size> min(intpseudovec<real_t, size> x,
                                      intpseudovec<real_t, size> y) {
  return x.min(y);
}

template <typename real_t, int size>
inline intpseudovec<real_t, size> popcount(intpseudovec<real_t, size> x) {
  return x.popcount();
}

template <typename real_t, int size>
inline intpseudovec<real_t, size>
rotate(intpseudovec<real_t, size> x,
       typename intpseudovec<real_t, size>::int_t n) {
  return x.rotate(n);
}

template <typename real_t, int size>
inline intpseudovec<real_t, size> rotate(intpseudovec<real_t, size> x,
                                         intpseudovec<real_t, size> n) {
  return x.rotate(n);
}

// realpseudovec wrappers

template <typename real_t, int size>
inline realpseudovec<real_t, size>
loada(real_t const *p, realpseudovec<real_t, size> x,
      typename realpseudovec<real_t, size>::mask_t const &m) {
  return x.loada(p, m);
}

template <typename real_t, int size>
inline realpseudovec<real_t, size>
loadu(real_t const *p, realpseudovec<real_t, size> x,
      typename realpseudovec<real_t, size>::mask_t const &m) {
  return x.loadu(p, m);
}

template <typename real_t, int size>
inline realpseudovec<real_t, size>
loadu(real_t const *p, size_t ioff, realpseudovec<real_t, size> x,
      typename realpseudovec<real_t, size>::mask_t const &m) {
  return x.loadu(p, ioff, m);
}

template <typename real_t, int size>
inline void storea(realpseudovec<real_t, size> x, real_t *p) {
  return x.storea(p);
}

template <typename real_t, int size>
inline void storeu(realpseudovec<real_t, size> x, real_t *p) {
  return x.storeu(p);
}

template <typename real_t, int size>
inline void storeu(realpseudovec<real_t, size> x, real_t *p, size_t ioff) {
  return x.storeu(p, ioff);
}

template <typename real_t, int size>
inline void storea(realpseudovec<real_t, size> x, real_t *p,
                   typename realpseudovec<real_t, size>::mask_t const &m) {
  return x.storea(p, m);
}

template <typename real_t, int size>
inline void storeu(realpseudovec<real_t, size> x, real_t *p,
                   typename realpseudovec<real_t, size>::mask_t const &m) {
  return x.storeu(p, m);
}

template <typename real_t, int size>
inline void storeu(realpseudovec<real_t, size> x, real_t *p, size_t ioff,
                   typename realpseudovec<real_t, size>::mask_t const &m) {
  return x.storeu(p, ioff, m);
}

template <typename real_t, int size>
inline intpseudovec<real_t, size> as_int(realpseudovec<real_t, size> x) {
  return x.as_int();
}

template <typename real_t, int size>
inline intpseudovec<real_t, size> convert_int(realpseudovec<real_t, size> x) {
  return x.convert_int();
}

template <typename real_t, int size>
inline real_t maxval(realpseudovec<real_t, size> x) {
  return x.maxval();
}

template <typename real_t, int size>
inline real_t minval(realpseudovec<real_t, size> x) {
  return x.minval();
}

template <typename real_t, int size>
inline real_t prod(realpseudovec<real_t, size> x) {
  return x.prod();
}

template <typename real_t, int size>
inline real_t sum(realpseudovec<real_t, size> x) {
  return x.sum();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> acos(realpseudovec<real_t, size> x) {
  return x.acos();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> acosh(realpseudovec<real_t, size> x) {
  return x.acosh();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> asin(realpseudovec<real_t, size> x) {
  return x.asin();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> asinh(realpseudovec<real_t, size> x) {
  return x.asinh();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> atan(realpseudovec<real_t, size> x) {
  return x.atan();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> atan2(realpseudovec<real_t, size> x,
                                         realpseudovec<real_t, size> y) {
  return x.atan2(y);
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> atanh(realpseudovec<real_t, size> x) {
  return x.atanh();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> cbrt(realpseudovec<real_t, size> x) {
  return x.cbrt();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> ceil(realpseudovec<real_t, size> x) {
  return x.ceil();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> copysign(realpseudovec<real_t, size> x,
                                            realpseudovec<real_t, size> y) {
  return x.copysign(y);
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> cos(realpseudovec<real_t, size> x) {
  return x.cos();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> cosh(realpseudovec<real_t, size> x) {
  return x.cosh();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> exp(realpseudovec<real_t, size> x) {
  return x.exp();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> exp10(realpseudovec<real_t, size> x) {
  return x.exp10();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> exp2(realpseudovec<real_t, size> x) {
  return x.exp2();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> expm1(realpseudovec<real_t, size> x) {
  return x.expm1();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> fabs(realpseudovec<real_t, size> x) {
  return x.fabs();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> floor(realpseudovec<real_t, size> x) {
  return x.floor();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> fdim(realpseudovec<real_t, size> x,
                                        realpseudovec<real_t, size> y) {
  return x.fdim(y);
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> fma(realpseudovec<real_t, size> x,
                                       realpseudovec<real_t, size> y,
                                       realpseudovec<real_t, size> z) {
  return x.fma(y, z);
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> fmax(realpseudovec<real_t, size> x,
                                        realpseudovec<real_t, size> y) {
  return x.fmax(y);
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> fmin(realpseudovec<real_t, size> x,
                                        realpseudovec<real_t, size> y) {
  return x.fmin(y);
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> fmod(realpseudovec<real_t, size> x,
                                        realpseudovec<real_t, size> y) {
  return x.fmod(y);
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> frexp(realpseudovec<real_t, size> x,
                                         intpseudovec<real_t, size> *r) {
  return x.frexp(r);
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> hypot(realpseudovec<real_t, size> x,
                                         realpseudovec<real_t, size> y) {
  return x.hypot(y);
}

template <typename real_t, int size>
inline intpseudovec<real_t, size> ilogb(realpseudovec<real_t, size> x) {
  return x.ilogb();
}

template <typename real_t, int size>
inline boolpseudovec<real_t, size> isfinite(realpseudovec<real_t, size> x) {
  return x.isfinite();
}

template <typename real_t, int size>
inline boolpseudovec<real_t, size> isinf(realpseudovec<real_t, size> x) {
  return x.isinf();
}

template <typename real_t, int size>
inline boolpseudovec<real_t, size> isnan(realpseudovec<real_t, size> x) {
  return x.isnan();
}

template <typename real_t, int size>
inline boolpseudovec<real_t, size> isnormal(realpseudovec<real_t, size> x) {
  return x.isnormal();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size>
ldexp(realpseudovec<real_t, size> x,
      typename intpseudovec<real_t, size>::int_t n) {
  return x.ldexp(n);
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> ldexp(realpseudovec<real_t, size> x,
                                         intpseudovec<real_t, size> n) {
  return x.ldexp(n);
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> log(realpseudovec<real_t, size> x) {
  return x.log();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> log10(realpseudovec<real_t, size> x) {
  return x.log10();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> log1p(realpseudovec<real_t, size> x) {
  return x.log1p();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> log2(realpseudovec<real_t, size> x) {
  return x.log2();
}

template <typename real_t, int size>
inline intpseudovec<real_t, size> lrint(realpseudovec<real_t, size> x) {
  return x.lrint();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> mad(realpseudovec<real_t, size> x,
                                       realpseudovec<real_t, size> y,
                                       realpseudovec<real_t, size> z) {
  return x.mad(y, z);
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> nextafter(realpseudovec<real_t, size> x,
                                             realpseudovec<real_t, size> y) {
  return x.nextafter(y);
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> pow(realpseudovec<real_t, size> x,
                                       realpseudovec<real_t, size> y) {
  return x.pow(y);
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> rcp(realpseudovec<real_t, size> x) {
  return x.rcp();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> remainder(realpseudovec<real_t, size> x,
                                             realpseudovec<real_t, size> y) {
  return x.remainder(y);
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> rint(realpseudovec<real_t, size> x) {
  return x.rint();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> round(realpseudovec<real_t, size> x) {
  return x.round();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> rsqrt(realpseudovec<real_t, size> x) {
  return x.rsqrt();
}

template <typename real_t, int size>
inline boolpseudovec<real_t, size> signbit(realpseudovec<real_t, size> x) {
  return x.signbit();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> sin(realpseudovec<real_t, size> x) {
  return x.sin();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> sinh(realpseudovec<real_t, size> x) {
  return x.sinh();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> sqrt(realpseudovec<real_t, size> x) {
  return x.sqrt();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> tan(realpseudovec<real_t, size> x) {
  return x.tan();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> tanh(realpseudovec<real_t, size> x) {
  return x.tanh();
}

template <typename real_t, int size>
inline realpseudovec<real_t, size> trunc(realpseudovec<real_t, size> x) {
  return x.trunc();
}

#ifndef VML_NO_IOSTREAM
template <typename real_t, int size>
std::ostream &operator<<(std::ostream &os,
                         boolpseudovec<real_t, size> const &x) {
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
                         intpseudovec<real_t, size> const &x) {
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
                         realpseudovec<real_t, size> const &x) {
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

#endif // #ifndef VEC_PSEUDO_H
