// TESTING: C++ kernels

#include <iostream>

template<typename T>
T add(T x, T y) { return x+y; }

extern "C"
__kernel void test_cxx()
{
  std::cout << "Hello, World from an OpenCL C++ kernel!" << std::endl;
  std::cout << "2+3=" << add(2, 3) << std::endl;
}
