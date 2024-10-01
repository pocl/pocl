// Trying to build a faulty program twice results in NULL deref
// See https://github.com/pocl/pocl/issues/577
// should print "BUILD ERROR" twice then "OK" once.

#include "pocl_opencl.h"

#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>
#include <iostream>

const char *SOURCE = R"RAW(

  __kernel void foo(__global int *input) {
    !@#$%^&*();
  }

)RAW";

int main(int argc, char *argv[]) {
  cl::Platform platform = cl::Platform::getDefault();
  cl::Device device = cl::Device::getDefault();
  unsigned error_count = 0;

  try {
    cl::Program program(SOURCE, false);
    for (unsigned i = 0; i < 2; i++) {
      try {
        program.compile();
      } catch (cl::BuildError &e) {
        std::cout << "BUILD ERROR\n";
        error_count++;
      }
    }
  } catch (cl::Error &err) {
    std::cout << "FAIL with OpenCL error = " << err.err() << std::endl;
    return EXIT_FAILURE;
  }

  platform.unloadCompiler();

  if (error_count == 2) {
    std::cout << "OK\n";
    return EXIT_SUCCESS;
  } else {
    std::cout << "FAIL\n";
    return EXIT_FAILURE;
  }
}
