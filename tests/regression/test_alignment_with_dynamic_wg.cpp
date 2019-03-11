#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>
#include <cassert>
#include <iostream>

using namespace std;

const char *SOURCE = R"RAW(

__kernel void test(){
  /* Variable declarations */
  int lid = get_local_id(0);
  int i, j;
  double trialValue[3];
  double result[1][3];
  __local double localResult[2][1][3];
  for (i = 0; i < 1; ++i)
    for (j = 0; j < 3; ++j) {
      result[i][j] = 0.0;
    }
    trialValue[0] = 0.0;
    trialValue[1] = 0.0;
    trialValue[2] = 0.0;
    for (i = 0; i < 1; ++i)
      for (j = 0; j < 3; ++j) {
        result[i][j] +=  trialValue[j];
      }
    localResult[lid][0][0] += localResult[lid][0][0];
}

)RAW";

int main(int argc, char *argv[]) {
  cl::Device device = cl::Device::getDefault();
  cl::CommandQueue queue = cl::CommandQueue::getDefault();
  cl::Program program(SOURCE);
  program.build("-cl-std=CL1.2");

  // This triggers compilation of dynamic WG binaries.
  cl::Program::Binaries binaries{};
  int err = program.getInfo<>(CL_PROGRAM_BINARIES, &binaries);
  assert(err == CL_SUCCESS);

  auto kernel = cl::KernelFunctor<>(program, "test");
  kernel(cl::EnqueueArgs(queue, cl::NDRange(2), cl::NDRange(2)));

  queue.finish();
}
