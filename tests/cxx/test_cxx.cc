// TESTING: C++ kernels

// #define __kernel

// unsigned _test_cxx_NUM_ARGS     = 0;
// int _test_cxx_ARG_IS_POINTER[]  = {};
// int _test_cxx_ARG_IS_LOCAL[]    = {};
// int _test_cxx_ARG_IS_IMAGE[]    = {};
// int _test_cxx_ARG_IS_SAMPLER[]  = {};
// int _test_cxx_REQD_WG_SIZE[3]   = {0, 0, 0};
// unsigned _test_cxx_NUM_LOCALS   = 0;
// unsigned _test_cxx_LOCAL_SIZE[] = {};



#include <iostream>

template<typename T>
T add(T x, T y) { return x+y; }

extern "C"
__kernel void test_cxx()
{
  std::cout << "Hello, World from an OpenCL C++ kernel!" << std::endl;
  std::cout << "2+3=" << add(2, 3) << std::endl;
}
