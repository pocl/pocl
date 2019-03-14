#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>
#include <cassert>
#include <iostream>

using namespace std;

const char *SOURCE = R"RAW(

__kernel void test(__global float* buf){
  size_t gid[2];

  gid[0] = get_global_id(0);
  gid[1] = get_global_id(1);

  size_t i, j, k;

  float tempResult[3][3][2];

  for (i = 0; i < 3; ++i) {
        tempResult[i][0][0] = 1.0f;
        tempResult[i][0][1] = 1.0f;
        tempResult[i][1][0] = 1.0f;
        tempResult[i][1][1] = 1.0f;
        tempResult[i][2][0] = 1.0f;
        tempResult[i][2][1] = 1.0f;
  }

  if (gid[0] == 0 && gid[1] == 0)
    buf[0] = tempResult[1][1][1];
}

)RAW";

#define ARRAY_SIZE 16

int main(int argc, char *argv[]) {
  cl::Device device = cl::Device::getDefault();
  cl::CommandQueue queue = cl::CommandQueue::getDefault();
  cl::Program program(SOURCE);
  program.build("-cl-std=CL1.2");

  float out[ARRAY_SIZE] = {0.0f};

  cl::Buffer outbuf((cl_mem_flags)(CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR),
                    (ARRAY_SIZE * sizeof(float)), out);

  auto kernel = cl::KernelFunctor<cl::Buffer>(program, "test");

  kernel(cl::EnqueueArgs(queue, cl::NDRange(1, 4), cl::NDRange(1, 4)), outbuf);

  queue.enqueueReadBuffer(outbuf, 1, 0, (ARRAY_SIZE * sizeof(float)), out);

  queue.finish();

  if (out[0] == 1.0f)
    printf("OK\n");
  else
    printf("FAIL\n");

  return (out[0] == 1.0f) ? 0 : 1;
}
