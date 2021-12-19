// Trying to build a faulty program twice results in NULL deref
// See https://github.com/pocl/pocl/issues/577
// should print "BUILD ERROR" twice then "OK" once.

#include "pocl_opencl.h"

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>
#include <iostream>

const char *SOURCE = R"RAW(

  __kernel void foo(__global int *input) {
    !@#$%^&*();
  }

)RAW";

int main(int argc, char *argv[]) {
  cl_int err;
  unsigned error_count = 0;
  cl::Program program(SOURCE, false, &err);

  for (unsigned i = 0; i < 2; i++) {
    try {
      program.compile();
    } catch (cl::BuildError &e) {
      std::cout << "BUILD ERROR\n";
      error_count++;
    }
  }

  cl::Platform::getDefault().unloadCompiler();

  if (error_count == 2) {
    std::cout << "OK\n";
    return 0;
  } else {
    std::cout << "FAIL\n";
    return 1;
  }
}
