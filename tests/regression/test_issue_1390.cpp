/*
  Github Issue #1390
*/

#include "pocl_opencl.h"

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#ifdef CL_HPP_TARGET_OPENCL_VERSION
#undef CL_HPP_TARGET_OPENCL_VERSION
#endif
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>
#include <cassert>
#include <iostream>
#include <random>

constexpr unsigned ELEMS = 4096;

using namespace std;

const char *SOURCE = R"RAW(

__kernel void findRange(__global const float* restrict data, uint length, __global float* restrict range) {
    __local float minBuffer[256];
    __local float maxBuffer[256];
    float minimum = MAXFLOAT;
    float maximum = -MAXFLOAT;

    // Each thread calculates the range of a subset of values.

    for (uint index = get_local_id(0); index < length; index += get_local_size(0)) {
        float value = data[index];
        minimum = min(minimum, value);
        maximum = max(maximum, value);
    }

    // Now reduce them.

    minBuffer[get_local_id(0)] = minimum;
    maxBuffer[get_local_id(0)] = maximum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint step = 1; step < get_local_size(0); step *= 2) {
        if (get_local_id(0)+step < get_local_size(0) && get_local_id(0)%(2*step) == 0) {
            minBuffer[get_local_id(0)] = min(minBuffer[get_local_id(0)], minBuffer[get_local_id(0)+step]);
            maxBuffer[get_local_id(0)] = max(maxBuffer[get_local_id(0)], maxBuffer[get_local_id(0)+step]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (get_local_id(0) == 0) {
        range[0] = minBuffer[0];
        range[1] = maxBuffer[0];
    }
}

)RAW";

using FindRangeKernel = cl::KernelFunctor<cl::Buffer, unsigned, cl::Buffer>;

int main(int argc, char *argv[]) {
  std::random_device RandomDevice;
  std::mt19937 Mersenne{RandomDevice()};
  std::uniform_real_distribution<float> UniDist{-1000.0f, +2200.0f};

  cl::Device device = cl::Device::getDefault();
  cl::CommandQueue Queue = cl::CommandQueue::getDefault();
  cl::Program Program(SOURCE);
  Program.build("-cl-std=CL1.2");

  auto Kernel = FindRangeKernel(Program, "findRange");

  float *Input = new float[ELEMS];
  float Min = 10e20, Max = -10e20;
  float Output[2] = {0.0f, 0.0f};
  for (unsigned i = 0; i < ELEMS; ++i) {
    Input[i] = UniDist(Mersenne);
    Min = std::min(Min, Input[i]);
    Max = std::max(Max, Input[i]);
  }
  std::cout << "Min: " << Min << "  Max: " << Max << "\n";

  cl::Buffer InBuffer(CL_MEM_READ_ONLY, ELEMS*sizeof(float));
  cl::Buffer OutBuffer(CL_MEM_WRITE_ONLY, 8*sizeof(float));
  Queue.enqueueWriteBuffer(InBuffer, CL_FALSE, 0, ELEMS*sizeof(float), Input);

  // force single WG with 256 size
  Kernel(cl::EnqueueArgs(Queue, cl::NDRange(256), cl::NDRange(256)),
         InBuffer, ELEMS, OutBuffer);

  Queue.enqueueReadBuffer(OutBuffer, CL_TRUE, 0, 2*sizeof(int), Output);
  Queue.finish();

  bool Verify = (Min == Output[0]) && (Max == Output[1]);

  if (Verify) {
    printf("OK\n");
    return EXIT_SUCCESS;
  } else {
    printf("FAIL\n");
    return EXIT_FAILURE;
  }
}
