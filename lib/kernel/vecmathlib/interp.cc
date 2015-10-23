#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
using namespace std;

#include "vecmathlib.h"
using namespace vecmathlib;

typedef float64_vec realvec_t;
typedef realvec_t::real_t real_t;
typedef realvec_t::intvec_t intvec_t;
typedef intvec_t::int_t int_t;

realvec_t interp(const real_t *array, ptrdiff_t size, real_t xmin, real_t xmax,
                 realvec_t x) {
  assert(size >= 2);
  // spacing
  real_t dx = (xmax - xmin) / (size - 1);
  real_t idx = 1.0 / dx;
  // determine location in array
  realvec_t scaled = (x - realvec_t(xmin)) * realvec_t(idx);
  realvec_t cell = floor(scaled);
  intvec_t n = convert_int(cell);
  // gather values from array
  realvec_t x0, x1;
  for (ptrdiff_t i = 0; i < realvec_t::size; ++i) {
    // ensure location is not out of bounds
    ptrdiff_t j = max(ptrdiff_t(0), min(size - 2, ptrdiff_t(n[i])));
    x0.set_elt(i, array[j]);
    x1.set_elt(i, array[j + 1]);
  }
  // determine interpolation weights
  realvec_t offset = scaled - cell;
  realvec_t w0 = realvec_t(1.0) - offset;
  realvec_t w1 = offset;
  // interpolate
  realvec_t y = w0 * x0 + w1 * x1;
  return y;
}

int main(int argc, char **argv) {
  ptrdiff_t size = 1001;
  vector<real_t> array(size);
  for (ptrdiff_t i = 0; i < size; ++i)
    array[i] = real_t(i) / 1000.0;

  real_t xmin = 0.0;
  real_t xmax = 0.5;
  realvec_t x = 0.333;
  cout << "x=" << x << "\n";
  realvec_t y = interp(&array[0], size, xmin, xmax, x);
  cout << "y=" << y << "\n";

  return 0;
}
