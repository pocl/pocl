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
#  define __has_builtin(x) 0 // Compatibility with non-clang compilers
#endif



typedef unsigned long long ticks;
inline ticks getticks()
{
#if __has_builtin(__builtin_readcyclecounter)
  return __builtin_readcyclecounter();
#elif defined __x86_64__
  ticks a, d;
  asm volatile("rdtsc" : "=a" (a), "=d" (d));
  return a | (d << 32);
#elif defined __powerpc__
  unsigned int tbl, tbu, tbu1;
  do {
    asm volatile("mftbu %0": "=r"(tbu));
    asm volatile("mftb %0": "=r"(tbl));
    asm volatile("mftbu %0": "=r"(tbu1));
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
inline double elapsed(ticks t1, ticks t0)
{
  return t1-t0;
}

double get_sys_time()
{
  timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + 1.0e-6 * tv.tv_usec;
  // timespec ts;
  // clock_gettime(CLOCK_REALTIME, &ts);
  // return ts.tv_sec + 1.0e-9 * ts.tv_nsec;
}

double measure_tick()
{
  ticks const rstart = getticks();
  double const wstart = get_sys_time();
  while (get_sys_time() - wstart < 0.1) {
    // do nothing, just wait
  }
  ticks const rend = getticks();
  double const wend = get_sys_time();
  assert(wend-wstart >= 0.09);
  return (wend - wstart) / elapsed(rend, rstart);
}



double global_result = 0.0;
template<typename realvec_t>
void save_result(realvec_t result)
{
  for (int i=0; i<realvec_t::size; ++i) {
    global_result += result[i];
  }
}



template<typename T> inline T nop(T x) { return x; }

#define DECLARE_FUNCTOR(func)                   \
  template<typename T>                          \
  struct functor_##func {                       \
    static char const* name() { return #func; } \
    T operator()(T x) { return func(x); }       \
  }

DECLARE_FUNCTOR(nop);
DECLARE_FUNCTOR(sqrt);
DECLARE_FUNCTOR(exp);
DECLARE_FUNCTOR(log);
DECLARE_FUNCTOR(sin);
DECLARE_FUNCTOR(cos);
DECLARE_FUNCTOR(atan);



template<typename realvec_t, template<typename> class func_t>
double run_bench()
{
  realvec_t x0, dx;
  for (int i=0; i<realvec_t::size; ++i) {
    x0.set_elt(i, 1.0f + float(i));
    dx.set_elt(i, 1.0e-6f);
  }
  realvec_t x, y;
  ticks t0, t1;
  double const cycles_per_tick = 1.0; // measure_tick();
  int const numiters = 10000000;
  
  func_t<realvec_t> func;
  t0 = getticks();
  x = y = x0;
  for (int n=0; n<numiters; ++n) {
    y += func(x);
    x += dx;
  }
  t1 = getticks();
  save_result(y);
  
  return cycles_per_tick * elapsed(t1,t0) * realvec_t::size / numiters;
}

template<typename realvec_t, template<typename> class func_t>
void bench_type_func()
{
  cout << "   "
       << setw(-5) << func_t<realvec_t>::name() << " "
       << setw(18) << realvec_t::name() << ": " << flush;
  double const cycles = run_bench<realvec_t, func_t>();
  cout << cycles << " cycles\n" << flush;
}

template<template<typename> class func_t>
void bench_func()
{
  cout << "\n"
       << "Benchmarking " << func_t<float>().name() << ":\n";
  
  bench_type_func<realpseudovec<float,1>, func_t>();
  // bench_type_func<realbuiltinvec<float,1>, func_t>();
  bench_type_func<realtestvec<float,1>, func_t>();
#ifdef VECMATHLIB_HAVE_VEC_FLOAT_1
  bench_type_func<realvec<float,1>, func_t>();
#endif
#ifdef VECMATHLIB_HAVE_VEC_FLOAT_2
  bench_type_func<realpseudovec<float,2>, func_t>();
  // bench_type_func<realbuiltinvec<float,2>, func_t>();
  // bench_type_func<realtestvec<float,2>, func_t>();
  bench_type_func<realvec<float,2>, func_t>();
#endif
#ifdef VECMATHLIB_HAVE_VEC_FLOAT_4
  bench_type_func<realpseudovec<float,4>, func_t>();
  // bench_type_func<realbuiltinvec<float,4>, func_t>();
  // bench_type_func<realtestvec<float,4>, func_t>();
  bench_type_func<realvec<float,4>, func_t>();
#endif
#ifdef VECMATHLIB_HAVE_VEC_FLOAT_8
  bench_type_func<realpseudovec<float,8>, func_t>();
  // bench_type_func<realbuiltinvec<float,8>, func_t>();
  // bench_type_func<realtestvec<float,8>, func_t>();
  bench_type_func<realvec<float,8>, func_t>();
#endif
  
  bench_type_func<realpseudovec<double,1>, func_t>();
  // bench_type_func<realbuiltinvec<double,1>, func_t>();
  bench_type_func<realtestvec<double,1>, func_t>();
#ifdef VECMATHLIB_HAVE_VEC_DOUBLE_1
  bench_type_func<realvec<double,1>, func_t>();
#endif
#ifdef VECMATHLIB_HAVE_VEC_DOUBLE_2
  bench_type_func<realpseudovec<double,2>, func_t>();
  // bench_type_func<realbuiltinvec<double,2>, func_t>();
  // bench_type_func<realtestvec<double,2>, func_t>();
  bench_type_func<realvec<double,2>, func_t>();
#endif
#ifdef VECMATHLIB_HAVE_VEC_DOUBLE_4
  bench_type_func<realpseudovec<double,4>, func_t>();
  // bench_type_func<realbuiltinvec<double,4>, func_t>();
  // bench_type_func<realtestvec<double,4>, func_t>();
  bench_type_func<realvec<double,4>, func_t>();
#endif
}

void bench()
{
  bench_func<functor_nop>();
  bench_func<functor_sqrt>();
  bench_func<functor_exp>();
  bench_func<functor_log>();
  bench_func<functor_sin>();
  bench_func<functor_cos>();
  bench_func<functor_atan>();
}



int main(int argc, char** argv)
{
  using namespace vecmathlib;

  cout << "Benchmarking math functions:\n";
  
  bench();
  
  // Checking global accumulator to prevent optimisation
  if (! std::isfinite(global_result)) {
    cout << "\n"
         << "WARNING: Global accumulator is not finite\n";
  }
  
  return 0;
}
