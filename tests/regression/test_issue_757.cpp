// See https://github.com/pocl/pocl/issues/757
// Caused an out of bounds read.

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include "pocl_opencl.h"

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#include <CL/opencl.hpp>

const char *SOURCE = R"RAW(
#define lid(N) ((int) get_local_id(N))
#define gid(N) ((int) get_group_id(N))

__kernel void __attribute__ ((reqd_work_group_size(128, 1, 1))) grudge_assign_0(
  int const grdg_n,
  __global double *__restrict__ expr_8,
  int const expr_8_offset,
  __global double const *__restrict__ grdg_sub_discr_dx0_dr0,
  int const grdg_sub_discr_dx0_dr0_offset)
{
  if (-1 + -128 * gid(0) + -1 * lid(0) + grdg_n >= 0)
    expr_8[expr_8_offset + 128 * gid(0) + lid(0)] = grdg_sub_discr_dx0_dr0[grdg_sub_discr_dx0_dr0_offset + 128 * gid(0) + lid(0)];
}
)RAW";

int main(int argc, char *argv[]) {
  int n = 8;

  cl::Device device = cl::Device::getDefault();
  cl::CommandQueue queue = cl::CommandQueue::getDefault();
  cl::Program program(SOURCE, true);

  if (poclu_supports_extension(device.get(), "cl_khr_fp64") == 0) {
    std::cout << "this test requires cl_khr_fp64, test SKIPPED\n";
    return 77;
  }

  // Create buffers on the device.
  cl::Buffer buffer_A(CL_MEM_READ_WRITE, sizeof(double) * n);
  cl::Buffer buffer_B(CL_MEM_READ_WRITE, sizeof(double) * n);

  std::vector<double> A(n);
  std::vector<double> B(n);
  std::fill(B.begin(), B.end(), 1);

  // Write arrays to the device.
  queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(double) * n, A.data());
  queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(double) * n, B.data());

  // Run the kernel.
  cl::Kernel knl(program, "grudge_assign_0");
  int sz = 1;
  knl.setArg(0, sz);
  knl.setArg(1, buffer_A);
  knl.setArg(2, 0);
  knl.setArg(3, buffer_B);
  knl.setArg(4, 0);
  queue.enqueueNDRangeKernel(
    knl,
    cl::NullRange,
    cl::NDRange(((sz+127)/128)*128),
    cl::NDRange(128));
  queue.finish();

  // Read result A from the device.
  queue.enqueueReadBuffer(buffer_A, CL_TRUE, 0, sizeof(double) * n, A.data());

  for (int i = 0; i < n; ++i) {
    if (i < sz) {
      assert(A[i] == 1);
    } else {
      assert(A[i] == 0);
    }
  }
  
  std::cout << "OK" << std::endl;
  return EXIT_SUCCESS;
}
