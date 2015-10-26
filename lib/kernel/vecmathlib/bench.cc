// -*-C++-*-

#define VML_NODEBUG

#include "vecmathlib.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include <sys/time.h>
#include <time.h>

using namespace std;
using namespace vecmathlib;

#ifndef __has_builtin
#define __has_builtin(x) 0 // Compatibility with non-clang compilers
#endif

typedef unsigned long long ticks;
inline ticks getticks() {
#if __has_builtin(__builtin_readcyclecounter)
  return __builtin_readcyclecounter();
#elif defined __x86_64__
  ticks a, d;
  asm volatile("rdtsc" : "=a"(a), "=d"(d));
  return a | (d << 32);
#elif defined __powerpc__
  unsigned int tbl, tbu, tbu1;
  do {
    asm volatile("mftbu %0" : "=r"(tbu));
    asm volatile("mftb %0" : "=r"(tbl));
    asm volatile("mftbu %0" : "=r"(tbu1));
  } while (tbu != tbu1);
  return ((unsigned long long)tbu << 32) | tbl;
#else
  timeval tv;
  gettimeofday(&tv, NULL);
  return 1000000ULL * tv.tv_sec + tv.tv_usec;
// timespec ts;
// clock_gettime(CLOCK_REALTIME, &ts);
// return 1000000000ULL * ts.tv_sec + ts.tv_nsec;
#endif
}
inline double elapsed(ticks t1, ticks t0) { return t1 - t0; }

double get_sys_time() {
  timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + 1.0e-6 * tv.tv_usec;
  // timespec ts;
  // clock_gettime(CLOCK_REALTIME, &ts);
  // return ts.tv_sec + 1.0e-9 * ts.tv_nsec;
}

double measure_tick() {
  ticks const rstart = getticks();
  double const wstart = get_sys_time();
  while (get_sys_time() - wstart < 0.1) {
    // do nothing, just wait
  }
  ticks const rend = getticks();
  double const wend = get_sys_time();
  assert(wend - wstart >= 0.09);
  return (wend - wstart) / elapsed(rend, rstart);
}

double global_result = 0.0;
template <typename realvec_t> void save_result(realvec_t result) {
  for (int i = 0; i < realvec_t::size; ++i) {
    global_result += result[i];
  }
  // Check global accumulator to prevent optimisation
  if (!vml_std::isfinite(global_result)) {
    cout << "\n"
         << "WARNING: Global accumulator is not finite\n";
  }
}

template <typename T> inline T nop(T x) { return x; }

template <typename T> inline T fneg(T x) { return -x; }

template <typename T> inline T fadd(T x, T y) { return x + y; }
template <typename T> inline T fsub(T x, T y) { return x - y; }
template <typename T> inline T fmul(T x, T y) { return x * y; }
template <typename T> inline T fdiv(T x, T y) { return x / y; }

template <typename T> inline T frexp0(T x) {
  typename T::intvec_t ir;
  return frexp(x, &ir);
}
template <typename T> inline typename T::intvec_t frexp1(T x) {
  typename T::intvec_t ir;
  frexp(x, &ir);
  return ir;
}

template <typename T> inline T ldexps(T x, T y) {
  typename T::intvec_t iy = convert_int(y);
  return ldexp(x, iy[0]);
}
template <typename T> inline T ldexpv(T x, T y) {
  typename T::intvec_t iy = convert_int(y);
  return ldexp(x, iy);
}

#define DECLARE_FUNCTOR(FUNC, XMIN, XMAX)                                      \
  template <typename T> struct functor_##FUNC {                                \
    static typename T::real_t get_xmin() { return XMIN; }                      \
    static typename T::real_t get_xmax() { return XMAX; }                      \
    static const char *name() { return #FUNC; }                                \
    T operator()(T x) { return FUNC(x); }                                      \
  }

#define DECLARE_BFUNCTOR(FUNC, XMIN, XMAX)                                     \
  template <typename T> struct functor_##FUNC {                                \
    static typename T::real_t get_xmin() { return XMIN; }                      \
    static typename T::real_t get_xmax() { return XMAX; }                      \
    static const char *name() { return #FUNC; }                                \
    T operator()(T x) {                                                        \
      typename T::boolvec_t res = FUNC(x);                                     \
      return convert_float(convert_int(res));                                  \
    }                                                                          \
  }

#define DECLARE_IFUNCTOR(FUNC, XMIN, XMAX)                                     \
  template <typename T> struct functor_##FUNC {                                \
    static typename T::real_t get_xmin() { return XMIN; }                      \
    static typename T::real_t get_xmax() { return XMAX; }                      \
    static const char *name() { return #FUNC; }                                \
    T operator()(T x) {                                                        \
      typename T::intvec_t res = FUNC(x);                                      \
      return convert_float(res);                                               \
    }                                                                          \
  }

#define DECLARE_FUNCTOR2(FUNC, XMIN, XMAX, YOFFSET)                            \
  template <typename T> struct functor_##FUNC {                                \
    static typename T::real_t get_xmin() { return XMIN; }                      \
    static typename T::real_t get_xmax() { return XMAX; }                      \
    static const char *name() { return #FUNC; }                                \
    T operator()(T x) {                                                        \
      const typename T::real_t yoffset = YOFFSET;                              \
      return FUNC(x, x + T(yoffset));                                          \
    }                                                                          \
  }

#define DECLARE_FUNCTOR3(FUNC, XMIN, XMAX, YOFFSET, ZOFFSET)                   \
  template <typename T> struct functor_##FUNC {                                \
    static typename T::real_t get_xmin() { return XMIN; }                      \
    static typename T::real_t get_xmax() { return XMAX; }                      \
    static const char *name() { return #FUNC; }                                \
    T operator()(T x) {                                                        \
      const typename T::real_t yoffset = YOFFSET;                              \
      const typename T::real_t zoffset = ZOFFSET;                              \
      return FUNC(x, x + T(yoffset), x + T(zoffset));                          \
    }                                                                          \
  }

DECLARE_FUNCTOR(nop, 0.0, 1.0);

DECLARE_FUNCTOR(fneg, 0.0, 1.0);
DECLARE_FUNCTOR2(fadd, 0.0, 1.0, 2.0);
DECLARE_FUNCTOR2(fsub, 0.0, 1.0, 2.0);
DECLARE_FUNCTOR2(fmul, 0.0, 1.0, 2.0);
DECLARE_FUNCTOR2(fdiv, 0.0, 1.0, 2.0);

DECLARE_FUNCTOR(acos, -0.5, +0.5);
DECLARE_FUNCTOR(acosh, 1.0, 2.0);
DECLARE_FUNCTOR(asin, -0.5, +0.5);
DECLARE_FUNCTOR(asinh, -1.0, +1.0);
DECLARE_FUNCTOR(atan, -1.0, +1.0);
DECLARE_FUNCTOR2(atan2, 0.0, 1.0, 2.0);
DECLARE_FUNCTOR(atanh, -0.5, +0.5);
DECLARE_FUNCTOR(cbrt, -1.0, 1.0);
DECLARE_FUNCTOR(ceil, -1.0, +1.0);
DECLARE_FUNCTOR2(copysign, -1.0, +1.0, 2.0);
DECLARE_FUNCTOR(cos, 0.0, 1.0);
DECLARE_FUNCTOR(cosh, 0.0, 1.0);
DECLARE_FUNCTOR(exp, 0.0, 1.0);
DECLARE_FUNCTOR(exp10, 0.0, 1.0);
DECLARE_FUNCTOR(exp2, 0.0, 1.0);
DECLARE_FUNCTOR(expm1, 0.0, 1.0);
DECLARE_FUNCTOR(fabs, -1.0, 1.0);
DECLARE_FUNCTOR(floor, -1.0, +1.0);
DECLARE_FUNCTOR2(fdim, 0.0, 1.0, 2.0);
DECLARE_FUNCTOR3(fma, 0.0, 1.0, 2.0, 3.0);
DECLARE_FUNCTOR2(fmax, 0.0, 1.0, 2.0);
DECLARE_FUNCTOR2(fmin, 0.0, 1.0, 2.0);
DECLARE_FUNCTOR2(fmod, 0.0, 1.0, 2.0);
DECLARE_FUNCTOR(frexp0, 1.0, 100.0);
DECLARE_IFUNCTOR(frexp1, 1.0, 100.0);
DECLARE_FUNCTOR2(hypot, 0.0, 1.0, 2.0);
DECLARE_IFUNCTOR(ilogb, 1.0, 100.0);
DECLARE_BFUNCTOR(isfinite, 0.0, 1.0);
DECLARE_BFUNCTOR(isinf, 0.0, 1.0);
DECLARE_BFUNCTOR(isnan, 0.0, 1.0);
DECLARE_BFUNCTOR(isnormal, 0.0, 1.0);
DECLARE_FUNCTOR2(ldexps, 1.0, 20.0, -10.0);
DECLARE_FUNCTOR2(ldexpv, 1.0, 20.0, -10.0);
DECLARE_FUNCTOR(log, 1.0, 2.0);
DECLARE_FUNCTOR(log10, 1.0, 2.0);
DECLARE_FUNCTOR(log1p, 0.0, 1.0);
DECLARE_FUNCTOR(log2, 1.0, 2.0);
DECLARE_FUNCTOR2(nextafter, -1.0, +1.0, 0.0);
DECLARE_FUNCTOR2(pow, 0.0, 1.0, 2.0);
DECLARE_FUNCTOR(rcp, 1.0, 2.0);
DECLARE_FUNCTOR2(remainder, 0.0, 1.0, 2.0);
DECLARE_FUNCTOR(rint, -1.0, +1.0);
DECLARE_FUNCTOR(round, -1.0, +1.0);
DECLARE_FUNCTOR(rsqrt, 1.0, 2.0);
DECLARE_BFUNCTOR(signbit, -1.0, +1.0);
DECLARE_FUNCTOR(sin, 0.0, 1.0);
DECLARE_FUNCTOR(sinh, -1.0, +1.0);
DECLARE_FUNCTOR(sqrt, 0.0, 1.0);
DECLARE_FUNCTOR(tan, 0.0, 1.0);
DECLARE_FUNCTOR(tanh, -1.0, +1.0);
DECLARE_FUNCTOR(trunc, -1.0, +1.0);

template <typename realvec_t, template <typename> class func_t>
double run_bench() {
  const int numiters = 1000000;

  typedef typename realvec_t::real_t real_t;
  const real_t xmin = func_t<realvec_t>::get_xmin();
  const real_t xmax = func_t<realvec_t>::get_xmax();
  realvec_t x0, dx;
  for (int i = 0; i < realvec_t::size; ++i) {
    x0.set_elt(i, xmin + (xmax - xmin) / numiters * i / realvec_t::size);
    dx.set_elt(i, (xmax - xmin) / numiters);
  }
  realvec_t x, y;
  ticks t0, t1;
  double const cycles_per_tick = 1.0; // measure_tick();

  func_t<realvec_t> func;
  t0 = getticks();
  x = y = x0;
  for (int n = 0; n < numiters; ++n) {
    y += func(x);
    x += dx;
  }
  t1 = getticks();
  save_result(y);

  return cycles_per_tick * elapsed(t1, t0) * realvec_t::size / numiters;
}

template <typename realvec_t, template <typename> class func_t>
void bench_type_func() {
  cout << "   " << setw(-5) << func_t<realvec_t>::name() << " " << setw(18)
       << realvec_t::name() << ": " << flush;
  double const cycles = run_bench<realvec_t, func_t>();
  cout << cycles << " cycles\n" << flush;
}

template <template <typename> class func_t> void bench_func() {
  cout << "\n"
       << "Benchmarking " << func_t<float32_vec>().name() << ":\n";

  // Note: We benchmark neither testvec (since this is known to be
  // slow), nor builtinvec (since this has about the same performance
  // as pseudovec, and is also not very efficient).

  bench_type_func<realpseudovec<float, 1>, func_t>();
#ifdef __clang__
  bench_type_func<realbuiltinvec<float, 1>, func_t>();
#endif
  bench_type_func<realtestvec<float, 1>, func_t>();
#ifdef VECMATHLIB_HAVE_VEC_FLOAT_1
  bench_type_func<realvec<float, 1>, func_t>();
#endif
#ifdef VECMATHLIB_HAVE_VEC_FLOAT_2
  bench_type_func<realpseudovec<float, 2>, func_t>();
#ifdef __clang__
  bench_type_func<realbuiltinvec<float, 2>, func_t>();
#endif
  // bench_type_func<realtestvec<float,2>, func_t>();
  bench_type_func<realvec<float, 2>, func_t>();
#endif
#ifdef VECMATHLIB_HAVE_VEC_FLOAT_4
  bench_type_func<realpseudovec<float, 4>, func_t>();
#ifdef __clang__
  bench_type_func<realbuiltinvec<float, 4>, func_t>();
#endif
  // bench_type_func<realtestvec<float,4>, func_t>();
  bench_type_func<realvec<float, 4>, func_t>();
#endif
#ifdef VECMATHLIB_HAVE_VEC_FLOAT_8
  bench_type_func<realpseudovec<float, 8>, func_t>();
#ifdef __clang__
  bench_type_func<realbuiltinvec<float, 8>, func_t>();
#endif
  // bench_type_func<realtestvec<float,8>, func_t>();
  bench_type_func<realvec<float, 8>, func_t>();
#endif

  bench_type_func<realpseudovec<double, 1>, func_t>();
#ifdef __clang__
  bench_type_func<realbuiltinvec<double, 1>, func_t>();
#endif
  bench_type_func<realtestvec<double, 1>, func_t>();
#ifdef VECMATHLIB_HAVE_VEC_DOUBLE_1
  bench_type_func<realvec<double, 1>, func_t>();
#endif
#ifdef VECMATHLIB_HAVE_VEC_DOUBLE_2
  bench_type_func<realpseudovec<double, 2>, func_t>();
#ifdef __clang__
  bench_type_func<realbuiltinvec<double, 2>, func_t>();
#endif
  // bench_type_func<realtestvec<double,2>, func_t>();
  bench_type_func<realvec<double, 2>, func_t>();
#endif
#ifdef VECMATHLIB_HAVE_VEC_DOUBLE_4
  bench_type_func<realpseudovec<double, 4>, func_t>();
#ifdef __clang__
  bench_type_func<realbuiltinvec<double, 4>, func_t>();
#endif
  // bench_type_func<realtestvec<double,4>, func_t>();
  bench_type_func<realvec<double, 4>, func_t>();
#endif
#ifdef VECMATHLIB_HAVE_VEC_DOUBLE_8
  bench_type_func<realpseudovec<double, 8>, func_t>();
#ifdef __clang__
  bench_type_func<realbuiltinvec<double, 8>, func_t>();
#endif
  // bench_type_func<realtestvec<double,8>, func_t>();
  bench_type_func<realvec<double, 8>, func_t>();
#endif
}

void bench() {
  bench_func<functor_nop>();

  bench_func<functor_fneg>();
  bench_func<functor_fadd>();
  bench_func<functor_fsub>();
  bench_func<functor_fmul>();
  bench_func<functor_fdiv>();

  bench_func<functor_acos>();
  bench_func<functor_acosh>();
  bench_func<functor_asin>();
  bench_func<functor_asinh>();
  bench_func<functor_atan>();
  bench_func<functor_atan2>();
  bench_func<functor_atanh>();
  bench_func<functor_cbrt>();
  bench_func<functor_ceil>();
  bench_func<functor_copysign>();
  bench_func<functor_cos>();
  bench_func<functor_cosh>();
  bench_func<functor_exp>();
  bench_func<functor_exp10>();
  bench_func<functor_exp2>();
  bench_func<functor_expm1>();
  bench_func<functor_fabs>();
  bench_func<functor_floor>();
  bench_func<functor_fdim>();
  bench_func<functor_fma>();
  bench_func<functor_fmax>();
  bench_func<functor_fmin>();
  bench_func<functor_fmod>();
  bench_func<functor_frexp0>();
  bench_func<functor_frexp1>();
  bench_func<functor_hypot>();
  bench_func<functor_ilogb>();
  bench_func<functor_isfinite>();
  bench_func<functor_isinf>();
  bench_func<functor_isnan>();
  bench_func<functor_isnormal>();
  bench_func<functor_ldexps>();
  bench_func<functor_ldexpv>();
  bench_func<functor_log>();
  bench_func<functor_log10>();
  bench_func<functor_log1p>();
  bench_func<functor_log2>();
  bench_func<functor_nextafter>();
  bench_func<functor_pow>();
  bench_func<functor_rcp>();
  bench_func<functor_remainder>();
  bench_func<functor_rint>();
  bench_func<functor_round>();
  bench_func<functor_rsqrt>();
  bench_func<functor_signbit>();
  bench_func<functor_sin>();
  bench_func<functor_sinh>();
  bench_func<functor_sqrt>();
  bench_func<functor_tan>();
  bench_func<functor_tanh>();
  bench_func<functor_trunc>();
}

int main(int argc, char **argv) {
  cout << "Benchmarking math functions:\n";
  bench();
  return 0;
}
