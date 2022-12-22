// 0.14rc1 introduced a regression where private kernel local variable (array)
// was detected as an automatic local address space variable.
// See https://github.com/pocl/pocl/issues/445

#include "pocl_opencl.h"

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#include <CL/opencl.hpp>
#include <iostream>

using namespace std;

const char *SOURCE = R"CLC(
#if __OPENCL_C_VERSION__ < 120
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1)))
private_local_array(__global int *__restrict__ out)
{
  int tmp[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };

//  for (int j = 0; j <= 2; ++j)
#pragma no unroll
    for (int i = 0; i < 9; ++i)
      out[i] = tmp[i];
}
)CLC";

int main(int, char **)
{
  bool success = true;
  try {
    int N = 9;

    cl::CommandQueue queue((cl_command_queue_properties)0);
    cl::Program program(SOURCE, true);

    auto kernel = cl::KernelFunctor<cl::Buffer>
      (program, "private_local_array");

    cl::Buffer buffer(CL_MEM_WRITE_ONLY, N*sizeof(cl_int));
    kernel(cl::EnqueueArgs(queue, cl::NDRange(1), cl::NDRange(1)), buffer);

    queue.finish();

    cl_int *output = (cl_int*)queue.enqueueMapBuffer(
      buffer, CL_TRUE, CL_MAP_READ, 0, N*sizeof(int));
    for (int i = 0; i < N; i++) {
      if ((int)output[i] != i + 1) {
        std::cout << "FAIL: " << output[i] << " should be " << i + 1
		  << std::endl;
        success = false;
      }
    }
    queue.enqueueUnmapMemObject(buffer, output);
    queue.finish();
    cl::Platform::getDefault().unloadCompiler();
  }
  catch (cl::Error& err) {
    std::cout << "FAIL with OpenCL error = " << err.err() << std::endl;
    return EXIT_FAILURE;
  }

  if (success) {
    std::cout << "OK" << std::endl;
    return EXIT_SUCCESS;
  }
  return EXIT_FAILURE;
}
