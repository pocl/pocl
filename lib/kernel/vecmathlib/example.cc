// -*-C++-*-

#include "vecmathlib.h"

#include <iostream>

using namespace std;
using namespace vecmathlib;



int main(int argc, char** argv)
{
  // Declare a double precision vector with an architecture-dependent
  // number of elements
  double_vec x;
  // Set each element separately. This is inefficient and should be
  // avoided if possible, but we want to demonstrate it here anyway.
  for (int i=0; i<double_vec::size; ++i) x.set_elt(i, double(i));
  double_vec y = x + double_vec(1.0);
  y = sqrt(y);
  double_vec z = log(y);
  
  // Boolean vectors are closely related to either double or float
  // vectors, thus we need to make a distinction
  bool_double_vec b = x < y;
  // Integer vectors are closely related to either double or float,
  // thus we need to make a distinction -- there is "long_vec"
  // corresponding to "double_vec", and there is "int_vec"
  // correpsonding to "float_vec".
  long_vec i = convert_int(y);
  
  cout << "x=" << x << "\n";
  cout << "y=" << y << "\n";
  cout << "z=" << z << "\n";
  cout << "b=" << b << "\n";
  cout << "i=" << i << "\n";
  
  return 0;
}
