// Instantiante some functions to be able to inspect the generated
// machine code

#define VML_NODEBUG

#define restrict __restrict__

#include "vecmathlib.h"

namespace vecmathlib {

template <typename realvec_t, int n>
typename realvec_t::real_t get_elt(realvec_t x) {
  return x[n];
}
template <typename realvec_t, int n>
realvec_t set_elt(realvec_t x, typename realvec_t::real_t a) {
  return x.set_elt(n, a);
}

// template realbuiltinvec<float,1> fabs(realbuiltinvec<float,1> x);
// template realbuiltinvec<float,1> fmin(realbuiltinvec<float,1> x,
// realbuiltinvec<float,1> y);
// template intbuiltinvec<float,1> lsr(intbuiltinvec<float,1> x,
// intbuiltinvec<float,1>::int_t n);
// template intbuiltinvec<double,1> lsr(intbuiltinvec<double,1> x,
// intbuiltinvec<double,1>::int_t n);
// template intbuiltinvec<double,2> lsr(intbuiltinvec<double,2> x,
// intbuiltinvec<double,2>::int_t n);
// template intbuiltinvec<double,2> lsr(intbuiltinvec<double,2> x,
// intbuiltinvec<double,2> n);
// template realbuiltinvec<float,1> ifthen(realbuiltinvec<float,1>::boolvec_t c,
// realbuiltinvec<float,1> x, realbuiltinvec<float,1> y);
// template realbuiltinvec<double,1> ifthen(realbuiltinvec<double,1>::boolvec_t
// c, realbuiltinvec<double,1> x, realbuiltinvec<double,1> y);
// template realbuiltinvec<float,4> ifthen(realbuiltinvec<float,4>::boolvec_t c,
// realbuiltinvec<float,4> x, realbuiltinvec<float,4> y);
// template realbuiltinvec<double,2> ifthen(realbuiltinvec<double,2>::boolvec_t
// c, realbuiltinvec<double,2> x, realbuiltinvec<double,2> y);

#ifdef VECMATHLIB_HAVE_VEC_FLOAT_1
template realvec<float, 1> round(realvec<float, 1> x);
#endif

#ifdef VECMATHLIB_HAVE_VEC_FLOAT_8
template intvec<float, 8> popcount(intvec<float, 8>);
#endif

#ifdef VECMATHLIB_HAVE_VEC_DOUBLE_1
template realvec<double, 1> exp(realvec<double, 1> x);
template realvec<double, 1> log(realvec<double, 1> x);
template realvec<double, 1> sin(realvec<double, 1> x);
template realvec<double, 1> sqrt(realvec<double, 1> x);
template realvec<double, 1>::real_t
get_elt<realvec<double, 1>, 0>(realvec<double, 1> x);
template realvec<double, 1>
set_elt<realvec<double, 1>, 0>(realvec<double, 1> x,
                               realvec<double, 1>::real_t a);
#endif

#ifdef VECMATHLIB_HAVE_VEC_DOUBLE_2
template realvec<double, 2> exp(realvec<double, 2> x);
template realvec<double, 2> log(realvec<double, 2> x);
template realvec<double, 2> sin(realvec<double, 2> x);
template realvec<double, 2> sqrt(realvec<double, 2> x);
template realvec<double, 2>::real_t
get_elt<realvec<double, 2>, 0>(realvec<double, 2>);
template realvec<double, 2>::real_t
get_elt<realvec<double, 2>, 1>(realvec<double, 2>);
template realvec<double, 2>
set_elt<realvec<double, 2>, 0>(realvec<double, 2> x,
                               realvec<double, 2>::real_t a);
template realvec<double, 2>
set_elt<realvec<double, 2>, 1>(realvec<double, 2> x,
                               realvec<double, 2>::real_t a);
#endif

#ifdef VECMATHLIB_HAVE_VEC_DOUBLE_4
template realvec<double, 4> exp(realvec<double, 4> x);
template realvec<double, 4> log(realvec<double, 4> x);
template realvec<double, 4> sin(realvec<double, 4> x);
template realvec<double, 4> sqrt(realvec<double, 4> x);
template realvec<double, 4>::real_t
get_elt<realvec<double, 4>, 0>(realvec<double, 4>);
template realvec<double, 4>::real_t
get_elt<realvec<double, 4>, 1>(realvec<double, 4>);
template realvec<double, 4>::real_t
get_elt<realvec<double, 4>, 2>(realvec<double, 4>);
template realvec<double, 4>::real_t
get_elt<realvec<double, 4>, 3>(realvec<double, 4>);
template realvec<double, 4>
set_elt<realvec<double, 4>, 0>(realvec<double, 4> x,
                               realvec<double, 4>::real_t a);
template realvec<double, 4>
set_elt<realvec<double, 4>, 1>(realvec<double, 4> x,
                               realvec<double, 4>::real_t a);
template realvec<double, 4>
set_elt<realvec<double, 4>, 2>(realvec<double, 4> x,
                               realvec<double, 4>::real_t a);
template realvec<double, 4>
set_elt<realvec<double, 4>, 3>(realvec<double, 4> x,
                               realvec<double, 4>::real_t a);
template intvec<double, 4> popcount(intvec<double, 4>);
#endif
}

// Various tests to detect auto-vectorization features

#include <cassert>
#include <cstdlib>
using namespace std;

using namespace vecmathlib;

#if defined VECMATHLIB_HAVE_VEC_DOUBLE_4
typedef realvec<double, 4> realV;
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_2
typedef realvec<double, 2> realV;
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_8
typedef realvec<float, 8> realV;
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_4
typedef realvec<float, 4> realV;
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_2
typedef realvec<float, 2> realV;
#else
#error "There are no vector types"
#endif

typedef realV::scalar_t real;
const int vecsize = realV::size;

// Simple, naive loop adding two arrays
extern "C" void loop_add(real *a, real *b, real *c, ptrdiff_t n) {
  for (ptrdiff_t i = 0; i < n; i += vecsize) {
    realV tmpb = realV::loadu(&b[i]);
    realV tmpc = realV::loadu(&c[i]);
    realV tmpa = tmpb + tmpc;
    storeu(tmpa, &a[i]);
  }
}

// Declare pointers as restrict
extern "C" void loop_add_restrict(real *restrict a, real *restrict b,
                                  real *restrict c, ptrdiff_t n) {
  for (ptrdiff_t i = 0; i < n; i += vecsize) {
    realV tmpb = realV::loadu(&b[i]);
    realV tmpc = realV::loadu(&c[i]);
    realV tmpa = tmpb + tmpc;
    storeu(tmpa, &a[i]);
  }
}

// Declare pointers as restrict and aligned
extern "C" void loop_add_aligned(real *restrict a, real *restrict b,
                                 real *restrict c, ptrdiff_t n) {
  for (ptrdiff_t i = 0; i < n; i += vecsize) {
    realV tmpb = realV::loada(&b[i]);
    realV tmpc = realV::loada(&c[i]);
    realV tmpa = tmpb + tmpc;
    storea(tmpa, &a[i]);
  }
}

// Reduction loop
extern "C" real loop_dot_reduce(real *restrict a, real *restrict b,
                                ptrdiff_t n) {
  realV sumV = 0.0;
  for (ptrdiff_t i = 0; i < n; i += vecsize) {
    realV tmpa = realV::loada(&a[i]);
    realV tmpb = realV::loada(&b[i]);
    sumV += tmpa * tmpb;
  }
  return sum(sumV);
}

// Loop with a simple if condition (fmax)
extern "C" void loop_if_simple(real *restrict a, real *restrict b,
                               real *restrict c, ptrdiff_t n) {
  for (ptrdiff_t i = 0; i < n; i += vecsize) {
    realV tmpb = realV::loada(&b[i]);
    realV tmpc = realV::loada(&c[i]);
    realV tmpa = ifthen(tmpb > tmpc, tmpb, tmpc);
    storea(tmpa, &a[i]);
  }
}

// Loop with a complex if condition (select)
extern "C" void loop_if(real *restrict a, real *restrict b, real *restrict c,
                        ptrdiff_t n) {
  for (ptrdiff_t i = 0; i < n; i += vecsize) {
    realV tmpb = realV::loada(&b[i]);
    realV tmpc = realV::loada(&c[i]);
    realV tmpa = ifthen(tmpb > realV(0.0), tmpb * tmpc, realV(1.0));
    storea(tmpa, &a[i]);
  }
}

// Skip ghost points
extern "C" void loop_add_masked(real *restrict a, real *restrict b,
                                real *restrict c, ptrdiff_t n) {
  for (realV::mask_t mask(1, n - 1, 0); mask; ++mask) {
    ptrdiff_t i = mask.index();
    realV tmpb = realV::loada(&b[i]);
    realV tmpc = realV::loada(&c[i]);
    realV tmpa = tmpb + tmpc;
    storea(tmpa, &a[i], mask);
  }
}
