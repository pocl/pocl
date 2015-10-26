// -*-C++-*-

#define restrict __restrict__

#include "vecmathlib.h"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <sys/time.h>

using namespace std;
using namespace vecmathlib;

////////////////////////////////////////////////////////////////////////////////
// Helpers
////////////////////////////////////////////////////////////////////////////////

#ifndef __has_builtin
#define __has_builtin(x) 0 // Compatibility with non-clang compilers
#endif

// align upwards
static size_t align_up(size_t i, size_t size) {
  return (i + size - 1) / size * size;
}

////////////////////////////////////////////////////////////////////////////////
// High-resolution timer
////////////////////////////////////////////////////////////////////////////////

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
  timeval tp;
  gettimeofday(&tp, NULL);
  return tp.tv_sec + 1.0e-6 * tp.tv_usec;
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

////////////////////////////////////////////////////////////////////////////////
// Initialize the grid
////////////////////////////////////////////////////////////////////////////////

template <typename realvec_t>
void init(typename realvec_t::real_t *restrict xptr, ptrdiff_t m, ptrdiff_t ldm,
          ptrdiff_t n) {
  for (ptrdiff_t j = 0; j < n; ++j) {
    for (ptrdiff_t i = 0; i < m; ++i) {
      const ptrdiff_t ij = ldm * j + i;
      xptr[ij] = (i + j) % 2;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Evolution loop: Simple stencil example (Gaussian smoothing)
////////////////////////////////////////////////////////////////////////////////

// Introduce a delay, so that cache access is not so important
template <typename T> static T delay(const T x) {
  return x;
  // return log(exp(x));
}

// Original version, unvectorized
template <typename realvec_t>
void smooth_scalar(typename realvec_t::real_t const *restrict xptr,
                   typename realvec_t::real_t *restrict yptr, ptrdiff_t m,
                   ptrdiff_t ldm, ptrdiff_t n) {
  typedef typename realvec_t::real_t real_t;
  for (ptrdiff_t j = 1; j < n - 1; ++j) {
    for (ptrdiff_t i = 1; i < m - 1; ++i) {
      const ptrdiff_t ij = ldm * j + i;
      const real_t x = xptr[ij];
      const real_t xil = xptr[ij - 1];
      const real_t xir = xptr[ij + 1];
      const real_t xjl = xptr[ij - ldm];
      const real_t xjr = xptr[ij + ldm];
      const real_t y =
          real_t(0.5) * x + real_t(0.125) * (xil + xir + xjl + xjr);
      yptr[ij] = delay(y);
    }
  }
}

// Assuming no particular alignment
template <typename realvec_t>
void smooth_unaligned(typename realvec_t::real_t const *restrict xptr,
                      typename realvec_t::real_t *restrict yptr, ptrdiff_t m,
                      ptrdiff_t ldm, ptrdiff_t n) {
  typedef typename realvec_t::real_t real_t;
  typedef typename realvec_t::mask_t mask_t;
  for (ptrdiff_t j = 1; j < n - 1; ++j) {
    // Desired loop bounds
    const ptrdiff_t imin = 1;
    const ptrdiff_t imax = m - 1;
    // Align actual loop iterations with vector size
    const ptrdiff_t ioff = ldm * j;
    for (mask_t mask(imin, imax, ioff); mask; ++mask) {
      const ptrdiff_t i = mask.index();
      const ptrdiff_t ij = ioff + i;
      const realvec_t x = realvec_t::loadu(xptr + ij);
      const realvec_t xil = realvec_t::loadu(xptr + ij, -1);
      const realvec_t xir = realvec_t::loadu(xptr + ij, +1);
      const realvec_t xjl = realvec_t::loadu(xptr + ij - ldm);
      const realvec_t xjr = realvec_t::loadu(xptr + ij + ldm);
      const realvec_t y = realvec_t(real_t(0.5)) * x +
                          realvec_t(real_t(0.125)) * (xil + xir + xjl + xjr);
      storeu(delay(y), yptr + ij, mask);
    }
  }
}

// Assuming that xptr and yptr are aligned, but ldm can be arbitrary
template <typename realvec_t>
void smooth_aligned(typename realvec_t::real_t const *restrict xptr,
                    typename realvec_t::real_t *restrict yptr, ptrdiff_t m,
                    ptrdiff_t ldm, ptrdiff_t n) {
  typedef typename realvec_t::real_t real_t;
  typedef typename realvec_t::mask_t mask_t;
  for (ptrdiff_t j = 1; j < n - 1; ++j) {
    // Desired loop bounds
    const ptrdiff_t imin = 1;
    const ptrdiff_t imax = m - 1;
    // Align actual loop iterations with vector size
    const ptrdiff_t ioff = ldm * j;
    for (mask_t mask(imin, imax, ioff); mask; ++mask) {
      const ptrdiff_t i = mask.index();
      const ptrdiff_t ij = ioff + i;
      const realvec_t x = realvec_t::loada(xptr + ij);
      const realvec_t xil = realvec_t::loadu(xptr + ij, -1);
      const realvec_t xir = realvec_t::loadu(xptr + ij, +1);
      const realvec_t xjl = realvec_t::loadu(xptr + ij - ldm);
      const realvec_t xjr = realvec_t::loadu(xptr + ij + ldm);
      const realvec_t y = realvec_t(real_t(0.5)) * x +
                          realvec_t(real_t(0.125)) * (xil + xir + xjl + xjr);
      storea(delay(y), yptr + ij, mask);
    }
  }
}

// Assuming that xptr and yptr are aligned, and ldm is a multiple of
// the vector size
template <typename realvec_t>
void smooth_padded(typename realvec_t::real_t const *restrict xptr,
                   typename realvec_t::real_t *restrict yptr, ptrdiff_t m,
                   ptrdiff_t ldm, ptrdiff_t n) {
  typedef typename realvec_t::real_t real_t;
  typedef typename realvec_t::mask_t mask_t;
  assert(ldm % realvec_t::size == 0);
  for (ptrdiff_t j = 1; j < n - 1; ++j) {
    // Desired loop bounds
    const ptrdiff_t imin = 1;
    const ptrdiff_t imax = m - 1;
    // Align actual loop iterations with vector size
    const ptrdiff_t ioff = ldm * j;
    for (mask_t mask(imin, imax, ioff); mask; ++mask) {
      const ptrdiff_t i = mask.index();
      const ptrdiff_t ij = ioff + i;
      const realvec_t x = realvec_t::loada(xptr + ij);
      const realvec_t xil = realvec_t::loadu(xptr + ij, -1);
      const realvec_t xir = realvec_t::loadu(xptr + ij, +1);
      const realvec_t xjl = realvec_t::loada(xptr + ij - ldm);
      const realvec_t xjr = realvec_t::loada(xptr + ij + ldm);
      const realvec_t y = realvec_t(real_t(0.5)) * x +
                          realvec_t(real_t(0.125)) * (xil + xir + xjl + xjr);
      storea(delay(y), yptr + ij, mask);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Main routine
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
  // Number of iterations
  const int niters = 100;

  // Grid size
  const ptrdiff_t m = 100;
  const ptrdiff_t n = 100;

// Choose a vector size
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_4
  typedef realvec<double, 4> realvec_t;
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_2
  typedef realvec<double, 2> realvec_t;
#else
  typedef realpseudovec<double, 1> realvec_t;
#endif

  // Ensure the grid size is aligned
  const ptrdiff_t ldm = align_up(m, realvec_t::size);
  typedef realvec_t::real_t real_t;
  vector<real_t> x0(ldm * n + realvec_t::size - 1),
      y0(ldm * n + realvec_t::size - 1);
  real_t *restrict const x =
      (real_t *)align_up(intptr_t(&x0[0]), sizeof(realvec_t));
  real_t *restrict const y =
      (real_t *)align_up(intptr_t(&y0[0]), sizeof(realvec_t));
  for (ptrdiff_t i = 0; i < ldm * n; ++i)
    y[i] = 0.0;

  // Initialize
  init<realvec_t>(&x[0], m, ldm, n);

  // Timers
  ticks t0, t1;
  double const cycles_per_tick = 1.0; // measure_tick();
  double cycles;

  // Run the different evolution loop versions
  t0 = getticks();
  for (int iter = 0; iter < niters; ++iter) {
    smooth_scalar<realvec_t>(&x[0], &y[0], m, ldm, n);
  }
  t1 = getticks();
  cycles =
      cycles_per_tick * elapsed(t1, t0) / (1.0 * (n - 1) * (m - 1) * niters);
  cout << "smooth_scalar:    " << cycles << " cycles/point\n";

  t0 = getticks();
  for (int iter = 0; iter < niters; ++iter) {
    smooth_unaligned<realvec_t>(&x[0], &y[0], m, ldm, n);
  }
  t1 = getticks();
  cycles =
      cycles_per_tick * elapsed(t1, t0) / (1.0 * (n - 1) * (m - 1) * niters);
  cout << "smooth_unaligned: " << cycles << " cycles/point\n";

  t0 = getticks();
  for (int iter = 0; iter < niters; ++iter) {
    smooth_aligned<realvec_t>(&x[0], &y[0], m, ldm, n);
  }
  t1 = getticks();
  cycles =
      cycles_per_tick * elapsed(t1, t0) / (1.0 * (n - 1) * (m - 1) * niters);
  cout << "smooth_aligned:   " << cycles << " cycles/point\n";

  t0 = getticks();
  for (int iter = 0; iter < niters; ++iter) {
    smooth_padded<realvec_t>(&x[0], &y[0], m, ldm, n);
  }
  t1 = getticks();
  cycles =
      cycles_per_tick * elapsed(t1, t0) / (1.0 * (n - 1) * (m - 1) * niters);
  cout << "smooth_padded:    " << cycles << " cycles/point\n";

  return 0;
}
