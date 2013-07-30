// -*-C++-*-

#include "vecmathlib.h"

#include <iostream>

using namespace std;
using namespace vecmathlib;



int main(int argc, char** argv)
{
  // Choose an "interesting" vector type
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_2
  typedef realvec<double,2> realvec_t;
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_4
  typedef realvec<float,4> realvec_t;
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_1
  typedef realvec<float,1> realvec_t;
#else
  typedef realpseudovec<float,1> realvec_t;
#endif

  typedef realvec_t::boolvec_t boolvec_t;
  typedef realvec_t::intvec_t intvec_t;
  
  realvec_t x = 1.0;
  realvec_t y = x + realvec_t(1.0);
  y = sqrt(y);
  realvec_t z = log(y);
  boolvec_t b = x < y;
  intvec_t i = convert_int(y);
  
  cout << "x=" << x << "\n";
  cout << "y=" << y << "\n";
  cout << "z=" << z << "\n";
  cout << "b=" << b << "\n";
  cout << "i=" << i << "\n";
  
  return 0;
}
