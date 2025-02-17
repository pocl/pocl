// https://github.com/pocl/pocl/issues/1608
// triggers on some LLVM versions
// "LLVM ERROR: Instruction Combining did not reach a fixpoint after 1 iterations"

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>
#include <iostream>

const char *SOURCE = R"RAW(
float fn1(float *b, int c, int d) {
  float e;
  if (c * d)
    e = b[0];
  return e;
}
void fn2(int g, int h) {
  float B[8], C[8], *p = 0;
  int i = get_local_id(0);
  for (int k = 0; k < g; ++k) {
    int l = get_local_id(0) * get_local_id(1) * i * i * i;
    p[l / 8 + l % 8] = B[0];
  }
  float f = fn1(0, i / h, i % h);
  for (int m = 0; m < 8; ++m)
    for (int n = 0; n < 2; ++n)
      C[m] = mad(B[m], f, C[m]);
}
__kernel void krnl(int g) {
  fn2(g, 1);
}
)RAW";

int main(int argc, char *argv[]) {
  cl_int err = CL_INVALID_VALUE;

  try {
    cl::Program program(SOURCE);
    try {
      program.build();
    } catch (cl::BuildError &e) {
      std::cout << "FAIL with BUILD ERROR = " << e.err() << " " << e.what() << std::endl;
      for (auto &bl : e.getBuildLog())
        std::cout << std::get<1>(bl);
      return EXIT_FAILURE;
    }
    // This triggers compilation of dynamic WG binaries.
    cl::Program::Binaries binaries{};
    err = program.getInfo<>(CL_PROGRAM_BINARIES, &binaries);
  } catch (cl::Error &e) {
    std::cout << "FAIL with OpenCL error = " << e.err() << " " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  if (err == CL_SUCCESS) {
    printf("OK\n");
    return EXIT_SUCCESS;
  } else {
    printf("FAIL\n");
    return EXIT_FAILURE;
  }
}
