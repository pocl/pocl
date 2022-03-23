// Copyright (c) 2021 PoCL developers
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/* See https://github.com/pocl/pocl/issues/893.
 *
 * This test will test that PoCL will not crash on LLVM 11 onward. The reason
 * was a jump threading optimizations which will duplicate basic blocks and
 * cause PoCL to form illegal parallel regions which does not follow
 * single-entry single-exit rule. This test doesn't care about the kernel result
 * or the arguments given to it. Point is to see that the kernel compiles
 * without any problems.
 *
 * More specifically the issue was the jump threading will cause kernel for loop
 * latch block to be duplicated when OpenCL is compiled to IR. Then later when
 * standard -O3 is ran as part of the PoCL passes the loop backedge is
 * duplicated, causing the loop duplication. When PoCL tries to form parallel
 * regions from explicit barrier to loop latch barrier the parallel region is
 * not form correctly. It contains incoming edges from outside of the parallel
 * region which will cause assert to fail. Those incoming edges are coming from
 * duplicated latch blocks.
 */

#include "pocl_opencl.h"

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>
#include <iostream>

const char *SOURCE = R"RAW(
#define lid(N) ((int) get_local_id(N))
#define gid(N) ((int) get_group_id(N))

__kernel void loopy_kernel(__global float *__restrict__ out, int const im_h, int const im_w)
{
  float acc_f_x_f_y_icolor;

  if (-1 + -1 * /* im_y_inner */ lid(1) + -16 * /* im_y_outer */ gid(1) + im_h >= 0 && -1 + -1 * /* im_x_inner */ lid(0) + -16 * /* im_x_outer */ gid(0) + im_w >= 0)
    acc_f_x_f_y_icolor = 0.0f;
  for (int icolor = 0; icolor <= 2; ++icolor)
  {
    barrier(CLK_LOCAL_MEM_FENCE) /* for img_fetch (insn_f_x_f_y_icolor_update depends on img_fetch_rule) */;
    if (-1 + -1 * /* im_y_inner */ lid(1) + -16 * /* im_y_outer */ gid(1) + im_h >= 0 && -1 + -1 * /* im_x_inner */ lid(0) + -16 * /* im_x_outer */ gid(0) + im_w >= 0)
      for (int f_x = -3; f_x <= 3; ++f_x)
        acc_f_x_f_y_icolor = acc_f_x_f_y_icolor + 10 * 3;
  }
  if (-1 + -1 * /* im_y_inner */ lid(1) + -16 * /* im_y_outer */ gid(1) + im_h >= 0 && -1 + -1 * /* im_x_inner */ lid(0) + -16 * /* im_x_outer */ gid(0) + im_w >= 0)
    out[im_h * im_w * /* ifeat */ gid(2) + im_h * (16 * /* im_x_outer */ gid(0) + /* im_x_inner */ lid(0)) + 16 * /* im_y_outer */ gid(1) + /* im_y_inner */ lid(1)] = acc_f_x_f_y_icolor;
}
)RAW";

int main() {
  int n = 8;
  cl::Device device = cl::Device::getDefault();
  cl::CommandQueue queue = cl::CommandQueue::getDefault();
  cl::Program program(SOURCE, true);
  cl::Buffer buffer(CL_MEM_WRITE_ONLY, sizeof(float) * 256);
  cl::Kernel kernel(program, "loopy_kernel");
  kernel.setArg(0, buffer);
  kernel.setArg(1, n);
  kernel.setArg(2, n);
  queue.enqueueNDRangeKernel(
    kernel,
    cl::NullRange,
    cl::NDRange(n),
    cl::NDRange(n));
  queue.finish();

  std::cout << "OK" << std::endl;
  return EXIT_SUCCESS;
}
