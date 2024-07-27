/*
  Github Issue #1525
*/

#include "pocl_opencl.h"

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110
#include <CL/opencl.hpp>
#include <cassert>
#include <iostream>
#include <random>

using namespace std;

const char *SOURCE = R"RAW(

__kernel void test(__global long *output, __global long *input)
{
    size_t i = get_global_id(0) * 16;

    long a[16];

    if (i + 0 < 16385) a[0] = input[i + 0];
    if (i + 1 < 16385) a[1] = input[i + 1];
    if (i + 2 < 16385) a[2] = input[i + 2];
    if (i + 3 < 16385) a[3] = input[i + 3];
    if (i + 4 < 16385) a[4] = input[i + 4];
    if (i + 5 < 16385) a[5] = input[i + 5];
    if (i + 6 < 16385) a[6] = input[i + 6];
    if (i + 7 < 16385) a[7] = input[i + 7];
    if (i + 8 < 16385) a[8] = input[i + 8];
    if (i + 9 < 16385) a[9] = input[i + 9];
    if (i + 10 < 16385) a[10] = input[i + 10];
    if (i + 11 < 16385) a[11] = input[i + 11];
    if (i + 12 < 16385) a[12] = input[i + 12];
    if (i + 13 < 16385) a[13] = input[i + 13];
    if (i + 14 < 16385) a[14] = input[i + 14];
    if (i + 15 < 16385) a[15] = input[i + 15];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (i + 0 < 16385) output[i + 0] = a[0];
    if (i + 1 < 16385) output[i + 1] = a[1];
    if (i + 2 < 16385) output[i + 2] = a[2];
    if (i + 3 < 16385) output[i + 3] = a[3];
    if (i + 4 < 16385) output[i + 4] = a[4];
    if (i + 5 < 16385) output[i + 5] = a[5];
    if (i + 6 < 16385) output[i + 6] = a[6];
    if (i + 7 < 16385) output[i + 7] = a[7];
    if (i + 8 < 16385) output[i + 8] = a[8];
    if (i + 9 < 16385) output[i + 9] = a[9];
    if (i + 10 < 16385) output[i + 10] = a[10];
    if (i + 11 < 16385) output[i + 11] = a[11];
    if (i + 12 < 16385) output[i + 12] = a[12];
    if (i + 13 < 16385) output[i + 13] = a[13];
    if (i + 14 < 16385) output[i + 14] = a[14];
    if (i + 15 < 16385) output[i + 15] = a[15];
}

)RAW";

using TestKernel = cl::KernelFunctor<cl::Buffer, cl::Buffer>;

int main(int argc, char *argv[]) {
  std::random_device RandomDevice;

  cl::Device device = cl::Device::getDefault();
  cl::CommandQueue Queue = cl::CommandQueue::getDefault();
  cl::Program Program(SOURCE);
  Program.build();

  auto Kernel = TestKernel(Program, "test");

  unsigned SIZE = 16 * 1024 + 1;
  cl::Buffer InBuffer(CL_MEM_READ_ONLY, SIZE*sizeof(long));
  cl::Buffer OutBuffer(CL_MEM_WRITE_ONLY, SIZE*sizeof(long));
  try {
    Kernel(cl::EnqueueArgs(Queue, cl::NDRange(2048), cl::NDRange(2048)), OutBuffer, InBuffer);
    Queue.finish();
  } catch (cl::Error &err) {
    std::cout << "FAIL with OpenCL error = " << err.err() << std::endl;
    return EXIT_FAILURE;
  }
}
