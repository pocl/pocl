// -*-C++-*-

#include "vecmathlib.h"

#include <iostream>

using namespace std;
using namespace vecmathlib;



int main(int argc, char** argv)
{
  // Declare a float precision vector with an architecture-dependent
  // number of elements
  float_vec x;
  // Set each element separately. This is inefficient and should be
  // avoided if possible, but we want to demonstrate it here anyway.
  for (int i=0; i<float_vec::size; ++i) x.set_elt(i, float(i));
  float_vec y = x + float_vec(1.0);
  y = sqrt(y);
  float_vec z = log(y);
  
  // Boolean vectors are closely related to either float or float
  // vectors, thus we need to make a distinction
  bool_float_vec b = x < y;
  // Integer vectors are closely related to either float or float,
  // thus we need to make a distinction -- there is "int_vec"
  // corresponding to "float_vec", and there is "int_vec"
  // correpsonding to "float_vec".
  int_vec i = convert_int(y);
  
  cout << "x=" << x << "\n";
  cout << "y=" << y << "\n";
  cout << "z=" << z << "\n";
  cout << "b=" << b << "\n";
  cout << "i=" << i << "\n";
  
  return 0;
}
