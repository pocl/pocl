/*
  Issue #701
*/

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>
#include <cassert>
#include <iostream>

using namespace std;

const char *SOURCE = R"RAW(

__kernel void evaluate(global const float *in, global float *out)
{
  /* Variable declarations */

  size_t elementIndex = get_global_id(1);

  size_t i, j;

  float testValue[3];
  float trialValue[1];

  float shapeIntegral[3][1][1];

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 1; ++j)
      shapeIntegral[i][j][j] = 0.0f;

  trialValue[0] = 1.5f;

  for (i = 0; i < 3; ++i)
    for (j = 0; j < 1; ++j)
      {
        shapeIntegral[i][j][j] += in[i] * trialValue[j];
      }

  if (elementIndex == 0)
  {
     out[0] = shapeIntegral[0][0][0];
     out[1] = shapeIntegral[1][0][0];
     out[2] = shapeIntegral[2][0][0];
  }

}

)RAW";

#define ARRAY_SIZE 4

int main(int argc, char *argv[]) {
  cl::Device device = cl::Device::getDefault();
  cl::CommandQueue queue = cl::CommandQueue::getDefault();
  cl::Program program(SOURCE);
  program.build("-cl-std=CL1.2");

  float in1[ARRAY_SIZE];
  float out[ARRAY_SIZE];

  cl::Buffer inbuf((cl_mem_flags)(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR),
                   (ARRAY_SIZE * sizeof(float)), in1);
  cl::Buffer outbuf((cl_mem_flags)(CL_MEM_WRITE_ONLY),
                    (ARRAY_SIZE * sizeof(float)), NULL);

  // This triggers compilation of dynamic WG binaries.
  cl::Program::Binaries binaries{};
  int err = program.getInfo<>(CL_PROGRAM_BINARIES, &binaries);
  assert(err == CL_SUCCESS);

  auto kernel = cl::KernelFunctor<cl::Buffer, cl::Buffer>(program, "evaluate");

  kernel(cl::EnqueueArgs(queue, cl::NDRange(1, 2), cl::NDRange(1, 1)), inbuf,
         outbuf);

  queue.enqueueReadBuffer(outbuf, 1, 0, (ARRAY_SIZE * sizeof(float)), out);

  queue.finish();

  printf("Value: %le \n", out[0]);
  printf("Value: %le \n", out[1]);
  printf("Value: %le \n", out[2]);
}
