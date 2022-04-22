#include "pocl_opencl.h"

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>
#include <cassert>
#include <iostream>

using namespace std;

const char *SOURCE = R"RAW(

__kernel void test(global uint *output, global const uint* trialValue){
  /* Variable declarations */
  int lid = get_local_id(0);

  int i, j, k;

  uint result[Y][Z];

  __local uint localResult[X][Y][Z];

  for (i = 0; i < Y; ++i)
    for (j = 0; j < Z; ++j) {
      result[i][j] = 0;
      localResult[lid][i][j] = 0;
    }

  for (i = 0; i < Y; ++i)
    for (j = 0; j < Z; ++j) {
      result[i][j] += trialValue[j] * 4;
      localResult[lid][i][j] += result[i][j];
    }

  barrier (CLK_LOCAL_MEM_FENCE);

  uint sum = 0;

  for (k = 0; k < X; ++k)
    for (i = 0; i < Y; ++i)
      for (j = 0; j < Z; ++j) {
        sum += localResult[k][i][j];
      }

  output[lid] = sum;
}

)RAW";

bool test_invocation(unsigned x, unsigned y, unsigned z,
                     const std::string &arg_x, const std::string &arg_y,
                     const std::string &arg_z, cl::CommandQueue &queue) {

  unsigned expected_sum = x * y * z * 4;

  unsigned local_size = x;
  assert(local_size > 0);
  assert(local_size <= 256);

  cl::Program program(SOURCE);
  std::string options = "-cl-std=CL1.2";
  options += " -DX=" + arg_x + " -DY=" + arg_y + " -DZ=" + arg_z;
  program.build(options.c_str());

  cl_uint *in1 = new cl_uint[z];
  cl_uint *out = new cl_uint[x];
  for (size_t i = 0; i < x; ++i) {
    out[i] = 0;
  }
  for (size_t i = 0; i < z; ++i) {
    in1[i] = 1;
  }

  cl::Buffer inbuf((cl_mem_flags)(CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR),
                   (z * sizeof(cl_uint)), in1);
  cl::Buffer outbuf((cl_mem_flags)(CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR),
                    (x * sizeof(cl_uint)), out);

  // This triggers compilation of dynamic WG binaries.
  cl::Program::Binaries binaries{};
  int err = program.getInfo<>(CL_PROGRAM_BINARIES, &binaries);
  assert(err == CL_SUCCESS);

  auto kernel = cl::KernelFunctor<cl::Buffer, cl::Buffer>(program, "test");

  kernel(
      cl::EnqueueArgs(queue, cl::NDRange(local_size), cl::NDRange(local_size)),
      outbuf, inbuf);

  queue.enqueueReadBuffer(outbuf, 1, 0, (x * sizeof(cl_uint)), out);

  queue.finish();

  bool correct = true;
  for (size_t i = 0; i < x; ++i) {
    if (out[i] != expected_sum)
      correct = false;
  }

  std::cout << (correct ? "OK\n" : "FAIL\n");

  delete[] in1;
  delete[] out;

  return correct;
}

int main(int argc, char *argv[]) {
  cl::Device device = cl::Device::getDefault();
  cl::CommandQueue queue = cl::CommandQueue::getDefault();

  if (argc < 4) {
    std::cout << "USAGE: $0 X Y Z\n";
    return EXIT_FAILURE;
  }

  std::string arg_x(argv[1]);
  std::string arg_y(argv[2]);
  std::string arg_z(argv[3]);

  unsigned x = std::stoi(argv[1]);
  unsigned y = std::stoi(argv[2]);
  unsigned z = std::stoi(argv[3]);

  if (!test_invocation(x, y, z, arg_x, arg_y, arg_z, queue))
    return EXIT_FAILURE;

  if (!test_invocation(y, z, x, arg_y, arg_z, arg_x, queue))
    return EXIT_FAILURE;

  if (!test_invocation(z, x, y, arg_z, arg_x, arg_y, queue))
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
