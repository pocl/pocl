// -*-C++-*-

#ifndef FLOATTYPES_H
#define FLOATTYPES_H



#include <cassert>
#include <cstdlib>



#if ! (defined __clang__ || defined __gcc__)
#  define __builtin_unreachable() (assert(0))
#  define __builtin_expect(expr, val) (expr)
#endif



// We expect either 199711L or 201103L
#if __cplusplus >= 201103L
// C++11 is supported, use it

#include <cmath>
#include <cstdint>

namespace vml_std {
  using namespace std;
  
  // template<typename T> inline bool vml_isfinite(T x) { return isfinite(x); }
  // template<typename T> inline bool vml_isinf(T x) { return isinf(x); }
  // template<typename T> inline bool vml_isnan(T x) { return isnan(x); }
  // template<typename T> inline bool vml_isnormal(T x) { return isnormal(x); }
  // template<typename T> inline bool vml_signbit(T x) { return signbit(x); }
}



#else
// C++11 is not supported, work around the missing pieces

// <cmath> does not work with all compilers; e.g. IBM's xlC on a Blue
// Gene/Q then does not provide macros for isnan, but provides
// functions instead.
#include <math.h>
#include <stdint.h>

#ifndef static_assert
#  define static_assert(cond, msg)
#endif



// Capture libc macros, then undefine them
#ifndef isfinite
#  error "isfinite is not a macro"
#endif
#ifndef isinf
#  error "isinf is not a macro"
#endif
#ifndef isnan
#  error "isnan is not a macro"
#endif
#ifndef isnormal
#  error "isnormal is not a macro"
#endif
#ifndef signbit
#  error "signbit is not a macro"
#endif

namespace {
  template<typename T> inline int libc_isfinite(T x) { return isfinite(x); }
  template<typename T> inline int libc_isinf(T x) { return isinf(x); }
  template<typename T> inline int libc_isnan(T x) { return isnan(x); }
  template<typename T> inline int libc_isnormal(T x) { return isnormal(x); }
  template<typename T> inline int libc_signbit(T x) { return signbit(x); }
}

// Include this before undefining the macros below
#include <cmath>

#undef isfinite
#undef isinf
#undef isnon
#undef isnormal
#undef signbit



namespace vml_std {
  
  // Make some type definitions from stdint.h available in std
  typedef ::uint8_t uint8_t;
  typedef ::int8_t int8_t;
  typedef ::uint16_t uint16_t;
  typedef ::int16_t int16_t;
  typedef ::uint32_t uint32_t;
  typedef ::int32_t int32_t;
#if __SIZEOF_LONG__ == 8
  // Even if both "long" and "long long" have the same size, they are
  // still different types. In many cases, it is then preferable to
  // use "long" instead of "long long".
  typedef unsigned long uint64_t;
  typedef long int64_t;
#else
  typedef ::uint64_t uint64_t;
  typedef ::int64_t int64_t;
#endif
  
  
  
  // Make math functions from math.h available in vml_std
  // (We could instead take some of them -- but not all -- from std.)
  
  inline float acos(float x) { return ::acosf(x); }
  inline float acosh(float x) { return ::acoshf(x); }
  inline float asin(float x) { return ::asinf(x); }
  inline float asinh(float x) { return ::asinhf(x); }
  inline float atan(float x) { return ::atanf(x); }
  inline float atan2(float x, float y) { return ::atan2f(x, y); }
  inline float atanh(float x) { return ::atanhf(x); }
  inline float cbrt(float x) { return ::cbrtf(x); }
  inline float ceil(float x) { return ::ceilf(x); }
  inline float cos(float x) { return ::cosf(x); }
  inline float cosh(float x) { return ::coshf(x); }
  inline float copysign(float x, float y) { return ::copysignf(x, y); }
  inline float exp(float x) { return ::expf(x); }
  inline float exp2(float x) { return ::exp2f(x); }
  inline float expm1(float x) { return ::expm1f(x); }
  inline float fabs(float x) { return ::fabsf(x); }
  inline float fdim(float x, float y) { return ::fdimf(x, y); }
  inline float floor(float x) { return ::floorf(x); }
  inline float fma(float x, float y, float z) { return ::fmaf(x, y, z); }
  inline float fmax(float x, float y) { return ::fmaxf(x, y); }
  inline float fmin(float x, float y) { return ::fminf(x, y); }
  inline float fmod(float x, float y) { return ::fmodf(x, y); }
  inline float frexp(float x, int* r) { return ::frexpf(x, r); }
  inline float hypot(float x, float y) { return ::hypotf(x, y); }
  inline int ilogb(float x) { return ::ilogbf(x); }
  inline bool isfinite(float x) { return libc_isfinite(x); }
  inline bool isinf(float x) { return libc_isinf(x); }
  inline bool isnan(float x) { return libc_isnan(x); }
  inline bool isnormal(float x) { return libc_isnormal(x); }
  inline float ldexp(float x, int n) { return ::ldexpf(x, n); }
  inline float log(float x) { return ::logf(x); }
  inline float log10(float x) { return ::log10f(x); }
  inline float log1p(float x) { return ::log1pf(x); }
  inline float log2(float x) { return ::log2f(x); }
  inline float nextafter(float x, float y) { return ::nextafterf(x, y); }
  inline float pow(float x, float y) { return ::powf(x, y); }
  inline float remainder(float x, float y) { return ::remainderf(x, y); }
  inline float rint(float x) { return ::rintf(x); }
  inline float round(float x) { return ::roundf(x); }
  inline bool signbit(float x) { return libc_signbit(x); }
  inline float sin(float x) { return ::sinf(x); }
  inline float sinh(float x) { return ::sinhf(x); }
  inline float sqrt(float x) { return ::sqrtf(x); }
  inline float tan(float x) { return ::tanf(x); }
  inline float tanh(float x) { return ::tanhf(x); }
  inline float trunc(float x) { return ::truncf(x); }
  
  inline double acos(double x) { return ::acos(x); }
  inline double acosh(double x) { return ::acosh(x); }
  inline double asin(double x) { return ::asin(x); }
  inline double asinh(double x) { return ::asinh(x); }
  inline double atan(double x) { return ::atan(x); }
  inline double atan2(double x, double y) { return ::atan2(x, y); }
  inline double atanh(double x) { return ::atanh(x); }
  inline double cbrt(double x) { return ::cbrt(x); }
  inline double ceil(double x) { return ::ceil(x); }
  inline double cos(double x) { return ::cos(x); }
  inline double cosh(double x) { return ::cosh(x); }
  inline double copysign(double x, double y) { return ::copysign(x, y); }
  inline double exp(double x) { return ::exp(x); }
  inline double exp2(double x) { return ::exp2(x); }
  inline double expm1(double x) { return ::expm1(x); }
  inline double fabs(double x) { return ::fabs(x); }
  inline double fdim(double x, double y) { return ::fdim(x, y); }
  inline double floor(double x) { return ::floor(x); }
  inline double fma(double x, double y, double z) { return ::fma(x, y, z); }
  inline double fmax(double x, double y) { return ::fmax(x, y); }
  inline double fmin(double x, double y) { return ::fmin(x, y); }
  inline double fmod(double x, double y) { return ::fmod(x, y); }
  inline double frexp(double x, int* r) { return ::frexp(x, r); }
  inline double hypot(double x, double y) { return ::hypot(x, y); }
  inline int ilogb(double x) { return ::ilogb(x); }
  inline bool isfinite(double x) { return libc_isfinite(x); }
  inline bool isinf(double x) { return libc_isinf(x); }
  inline bool isnan(double x) { return libc_isnan(x); }
  inline bool isnormal(double x) { return libc_isnormal(x); }
  inline double ldexp(double x, int n) { return ::ldexp(x, n); }
  inline double log(double x) { return ::log(x); }
  inline double log10(double x) { return ::log10(x); }
  inline double log1p(double x) { return ::log1p(x); }
  inline double log2(double x) { return ::log2(x); }
  inline double nextafter(double x, double y) { return ::nextafter(x, y); }
  inline double pow(double x, double y) { return ::pow(x, y); }
  inline double remainder(double x, double y) { return ::remainder(x, y); }
  inline double rint(double x) { return ::rint(x); }
  inline double round(double x) { return ::round(x); }
  inline bool signbit(double x) { return libc_signbit(x); }
  inline double sin(double x) { return ::sin(x); }
  inline double sinh(double x) { return ::sinh(x); }
  inline double sqrt(double x) { return ::sqrt(x); }
  inline double tan(double x) { return ::tan(x); }
  inline double tanh(double x) { return ::tanh(x); }
  inline double trunc(double x) { return ::trunc(x); }
  
}

#endif



namespace vecmathlib {
  
  struct fp8 {
    // 1 bit sign, 4 bits exponent, 3 bits mantissa
    vml_std::uint8_t val;
  };
  
  struct fp16 {
    // 1 bit sign, 5 bits exponent, 10 bits mantissa
    vml_std::uint16_t val;
  };
  
} // namespace vecmathlib

#endif  // #ifndef FLOATTYPES_H
