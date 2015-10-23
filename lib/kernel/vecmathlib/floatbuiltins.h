// -*-C++-*-

#ifndef FLOATBUILTINS_H
#define FLOATBUILTINS_H

#if defined __clang__

namespace vecmathlib {

inline char builtin_abs(char x) { return __builtin_abs(x); }
inline short builtin_abs(short x) { return __builtin_abs(x); }
inline int builtin_abs(int x) { return __builtin_abs(x); }
inline long builtin_abs(long x) { return __builtin_labs(x); }
#if __SIZEOF_LONG_LONG__
inline long long builtin_abs(long long x) { return __builtin_llabs(x); }
#endif

inline unsigned char builtin_clz(unsigned char x) {
  return __builtin_clzs(x) -
         CHAR_BIT * (sizeof(unsigned short) - sizeof(unsigned char));
}
inline unsigned short builtin_clz(unsigned short x) {
  return __builtin_clzs(x);
}
inline unsigned int builtin_clz(unsigned int x) { return __builtin_clz(x); }
inline unsigned long builtin_clz(unsigned long x) { return __builtin_clzl(x); }
#if __SIZEOF_LONG_LONG__
inline unsigned long long builtin_clz(unsigned long long x) {
  return __builtin_clzll(x);
}
#endif

inline unsigned char builtin_popcount(unsigned char x) {
  return __builtin_popcount(x);
}
inline unsigned short builtin_popcount(unsigned short x) {
  return __builtin_popcount(x);
}
inline unsigned int builtin_popcount(unsigned int x) {
  return __builtin_popcount(x);
}
inline unsigned long builtin_popcount(unsigned long x) {
  return __builtin_popcountl(x);
}
#if __SIZEOF_LONG_LONG__
inline unsigned long long builtin_popcount(unsigned long long x) {
  return __builtin_popcountll(x);
}
#endif

inline float builtin_acos(float x) { return __builtin_acosf(x); }
inline double builtin_acos(double x) { return __builtin_acos(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_acos(long double x) { return __builtin_acosl(x); }
#endif

inline float builtin_acosh(float x) { return __builtin_acoshf(x); }
inline double builtin_acosh(double x) { return __builtin_acosh(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_acosh(long double x) { return __builtin_acoshl(x); }
#endif

inline float builtin_asin(float x) { return __builtin_asinf(x); }
inline double builtin_asin(double x) { return __builtin_asin(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_asin(long double x) { return __builtin_asinl(x); }
#endif

inline float builtin_asinh(float x) { return __builtin_asinhf(x); }
inline double builtin_asinh(double x) { return __builtin_asinh(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_asinh(long double x) { return __builtin_asinhl(x); }
#endif

inline float builtin_atan(float x) { return __builtin_atanf(x); }
inline double builtin_atan(double x) { return __builtin_atan(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_atan(long double x) { return __builtin_atanl(x); }
#endif

inline float builtin_atan2(float x, float y) { return __builtin_atan2f(x, y); }
inline double builtin_atan2(double x, double y) {
  return __builtin_atan2(x, y);
}
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_atan2(long double x, long double y) {
  return __builtin_atan2l(x, y);
}
#endif

inline float builtin_atanh(float x) { return __builtin_atanhf(x); }
inline double builtin_atanh(double x) { return __builtin_atanh(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_atanh(long double x) { return __builtin_atanhl(x); }
#endif

inline float builtin_cbrt(float x) { return __builtin_cbrtf(x); }
inline double builtin_cbrt(double x) { return __builtin_cbrt(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_cbrt(long double x) { return __builtin_cbrtl(x); }
#endif

inline float builtin_ceil(float x) { return __builtin_ceilf(x); }
inline double builtin_ceil(double x) { return __builtin_ceil(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_ceil(long double x) { return __builtin_ceill(x); }
#endif

inline float builtin_copysign(float x, float y) {
  return __builtin_copysignf(x, y);
}
inline double builtin_copysign(double x, double y) {
  return __builtin_copysign(x, y);
}
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_copysign(long double x, long double y) {
  return __builtin_copysignl(x, y);
}
#endif

inline float builtin_cos(float x) { return __builtin_cosf(x); }
inline double builtin_cos(double x) { return __builtin_cos(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_cos(long double x) { return __builtin_cosl(x); }
#endif

inline float builtin_cosh(float x) { return __builtin_coshf(x); }
inline double builtin_cosh(double x) { return __builtin_cosh(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_cosh(long double x) { return __builtin_coshl(x); }
#endif

inline float builtin_exp(float x) { return __builtin_expf(x); }
inline double builtin_exp(double x) { return __builtin_exp(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_exp(long double x) { return __builtin_expl(x); }
#endif

inline float builtin_exp2(float x) { return __builtin_exp2f(x); }
inline double builtin_exp2(double x) { return __builtin_exp2(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_exp2(long double x) { return __builtin_exp2l(x); }
#endif

inline float builtin_expm1(float x) { return __builtin_expm1f(x); }
inline double builtin_expm1(double x) { return __builtin_expm1(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_expm1(long double x) { return __builtin_expm1l(x); }
#endif

inline float builtin_fabs(float x) { return __builtin_fabsf(x); }
inline double builtin_fabs(double x) { return __builtin_fabs(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_fabs(long double x) { return __builtin_fabsl(x); }
#endif

inline float builtin_fdim(float x, float y) { return __builtin_fdimf(x, y); }
inline double builtin_fdim(double x, double y) { return __builtin_fdim(x, y); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_fdim(long double x, long double y) {
  return __builtin_fdiml(x, y);
}
#endif

inline float builtin_floor(float x) { return __builtin_floorf(x); }
inline double builtin_floor(double x) { return __builtin_floor(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_floor(long double x) { return __builtin_floorl(x); }
#endif

inline float builtin_fma(float x, float y, float z) {
  return __builtin_fmaf(x, y, z);
}
inline double builtin_fma(double x, double y, double z) {
  return __builtin_fma(x, y, z);
}
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_fma(long double x, long double y, long double z) {
  return __builtin_fmal(x, y, z);
}
#endif

inline float builtin_fmax(float x, float y) { return __builtin_fmaxf(x, y); }
inline double builtin_fmax(double x, double y) { return __builtin_fmax(x, y); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_fmax(long double x, long double y) {
  return __builtin_fmaxl(x, y);
}
#endif

inline float builtin_fmin(float x, float y) { return __builtin_fminf(x, y); }
inline double builtin_fmin(double x, double y) { return __builtin_fmin(x, y); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_fmin(long double x, long double y) {
  return __builtin_fminl(x, y);
}
#endif

inline float builtin_fmod(float x, float y) { return __builtin_fmodf(x, y); }
inline double builtin_fmod(double x, double y) { return __builtin_fmod(x, y); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_fmod(long double x, long double y) {
  return __builtin_fmodl(x, y);
}
#endif

inline float builtin_frexp(float x, int *r) { return __builtin_frexpf(x, r); }
inline double builtin_frexp(double x, int *r) { return __builtin_frexp(x, r); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_frexp(long double x, int *r) {
  return __builtin_frexpl(x, r);
}
#endif

inline float builtin_hypot(float x, float y) { return __builtin_hypotf(x, y); }
inline double builtin_hypot(double x, double y) {
  return __builtin_hypot(x, y);
}
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_hypot(long double x, long double y) {
  return __builtin_hypotl(x, y);
}
#endif

inline int builtin_ilogb(float x) { return __builtin_ilogbf(x); }
inline int builtin_ilogb(double x) { return __builtin_ilogb(x); }
#if __SIZEOF_LONG_DOUBLE__
inline int builtin_ilogb(long double x) { return __builtin_ilogbl(x); }
#endif

inline int builtin_isfinite(float x) { return __builtin_isfinite(x); }
inline int builtin_isfinite(double x) { return __builtin_isfinite(x); }
#if __SIZEOF_LONG_DOUBLE__
inline int builtin_isfinite(long double x) { return __builtin_isfinite(x); }
#endif

inline int builtin_isinf(float x) { return __builtin_isinf(x); }
inline int builtin_isinf(double x) { return __builtin_isinf(x); }
#if __SIZEOF_LONG_DOUBLE__
inline int builtin_isinf(long double x) { return __builtin_isinf(x); }
#endif

inline int builtin_isnan(float x) { return __builtin_isnan(x); }
inline int builtin_isnan(double x) { return __builtin_isnan(x); }
#if __SIZEOF_LONG_DOUBLE__
inline int builtin_isnan(long double x) { return __builtin_isnan(x); }
#endif

inline int builtin_isnormal(float x) { return __builtin_isnormal(x); }
inline int builtin_isnormal(double x) { return __builtin_isnormal(x); }
#if __SIZEOF_LONG_DOUBLE__
inline int builtin_isnormal(long double x) { return __builtin_isnormal(x); }
#endif

inline float builtin_ldexp(float x, int y) { return __builtin_ldexpf(x, y); }
inline double builtin_ldexp(double x, int y) { return __builtin_ldexp(x, y); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_ldexp(long double x, int y) {
  return __builtin_ldexpl(x, y);
}
#endif

inline long long builtin_llrint(float x) { return __builtin_llrintf(x); }
inline long long builtin_llrint(double x) { return __builtin_llrint(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long long builtin_llrint(long double x) { return __builtin_llrintl(x); }
#endif

inline float builtin_log(float x) { return __builtin_logf(x); }
inline double builtin_log(double x) { return __builtin_log(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_log(long double x) { return __builtin_logl(x); }
#endif

inline float builtin_log10(float x) { return __builtin_log10f(x); }
inline double builtin_log10(double x) { return __builtin_log10(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_log10(long double x) { return __builtin_log10l(x); }
#endif

inline float builtin_log1p(float x) { return __builtin_log1pf(x); }
inline double builtin_log1p(double x) { return __builtin_log1p(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_log1p(long double x) { return __builtin_log1pl(x); }
#endif

inline float builtin_log2(float x) { return __builtin_log2f(x); }
inline double builtin_log2(double x) { return __builtin_log2(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_log2(long double x) { return __builtin_log2l(x); }
#endif

inline long builtin_lrint(float x) { return __builtin_lrintf(x); }
inline long builtin_lrint(double x) { return __builtin_lrint(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long builtin_lrint(long double x) { return __builtin_lrintl(x); }
#endif

inline float builtin_nextafter(float x, float y) {
  return __builtin_nextafterf(x, y);
}
inline double builtin_nextafter(double x, double y) {
  return __builtin_nextafter(x, y);
}
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_nextafter(long double x, long double y) {
  return __builtin_nextafterl(x, y);
}
#endif

inline float builtin_pow(float x, float y) { return __builtin_powf(x, y); }
inline double builtin_pow(double x, double y) { return __builtin_pow(x, y); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_pow(long double x, long double y) {
  return __builtin_powl(x, y);
}
#endif

inline float builtin_remainder(float x, float y) {
  return __builtin_remainderf(x, y);
}
inline double builtin_remainder(double x, double y) {
  return __builtin_remainder(x, y);
}
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_remainder(long double x, long double y) {
  return __builtin_remainderl(x, y);
}
#endif

inline float builtin_rint(float x) { return __builtin_rintf(x); }
inline double builtin_rint(double x) { return __builtin_rint(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_rint(long double x) { return __builtin_rintl(x); }
#endif

inline float builtin_round(float x) { return __builtin_roundf(x); }
inline double builtin_round(double x) { return __builtin_round(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_round(long double x) { return __builtin_roundl(x); }
#endif

inline int builtin_signbit(float x) { return __builtin_signbitf(x); }
inline int builtin_signbit(double x) { return __builtin_signbit(x); }
#if __SIZEOF_LONG_DOUBLE__
inline int builtin_signbit(long double x) { return __builtin_signbitl(x); }
#endif

inline float builtin_sin(float x) { return __builtin_sinf(x); }
inline double builtin_sin(double x) { return __builtin_sin(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_sin(long double x) { return __builtin_sinl(x); }
#endif

inline float builtin_sinh(float x) { return __builtin_sinhf(x); }
inline double builtin_sinh(double x) { return __builtin_sinh(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_sinh(long double x) { return __builtin_sinhl(x); }
#endif

inline float builtin_sqrt(float x) { return __builtin_sqrtf(x); }
inline double builtin_sqrt(double x) { return __builtin_sqrt(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_sqrt(long double x) { return __builtin_sqrtl(x); }
#endif

inline float builtin_tan(float x) { return __builtin_tanf(x); }
inline double builtin_tan(double x) { return __builtin_tan(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_tan(long double x) { return __builtin_tanl(x); }
#endif

inline float builtin_tanh(float x) { return __builtin_tanhf(x); }
inline double builtin_tanh(double x) { return __builtin_tanh(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_tanh(long double x) { return __builtin_tanhl(x); }
#endif

inline float builtin_trunc(float x) { return __builtin_truncf(x); }
inline double builtin_trunc(double x) { return __builtin_trunc(x); }
#if __SIZEOF_LONG_DOUBLE__
inline long double builtin_trunc(long double x) { return __builtin_truncl(x); }
#endif
}

#endif

#endif // #ifndef FLOATBUILTINS_H
