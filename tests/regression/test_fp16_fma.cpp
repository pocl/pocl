/* Regression test: half-precision fma must be correctly rounded (OpenCL C
   requires 0 ulp). On x86, Clang before 22 lowers @llvm.fma.f16 through a
   float fma, rounding the result twice, to float and then to half, which is
   off by half an ulp for about one random input in 24,000 (OpenCL-CTS
   math_fma, fp16). The inputs below are cases where that happens; the
   expected results are the correctly rounded ones. Checked for the scalar and
   the half4 overloads. Skips where the device lacks cl_khr_fp16.

   Copyright (c) 2026 PoCL developers. MIT license, see COPYING. */

#include "pocl_opencl.h"
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const char *SOURCE = R"RAW(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void k1(__global const ushort *a, __global const ushort *b,
                 __global const ushort *c, __global ushort *out) {
  size_t i = get_global_id(0);
  out[i] = as_ushort(fma(as_half(a[i]), as_half(b[i]), as_half(c[i])));
}
__kernel void k4(__global const ushort4 *a, __global const ushort4 *b,
                 __global const ushort4 *c, __global ushort4 *out) {
  size_t i = get_global_id(0);
  out[i] = as_ushort4(fma(as_half4(a[i]), as_half4(b[i]), as_half4(c[i])));
}
)RAW";

/* a, b, c, the correctly rounded fma; through float it would be one ulp off */
static const uint16_t CASES[][4] = {
  {0x72d9, 0x3697, 0x7aa7, 0x7b5b}, {0x05e7, 0xc56c, 0xbff0, 0xbff1},
  {0x532a, 0xacce, 0x867e, 0xc44d}, {0xd4c0, 0xdd58, 0x0be0, 0x7659},
  {0x8f7d, 0xf046, 0xf20c, 0xf20b}, {0x0601, 0xa1ff, 0x0c21, 0x0c1d},
  {0x3329, 0x1119, 0x1f3a, 0x1f5f}, {0xe020, 0x4ad0, 0x87d6, 0xef07},
  {0xbb4a, 0x6e00, 0x05de, 0xed77},
  /* exact cases, the same either way: 1*1+1, 2*3-6, -0*1+0 */
  {0x3c00, 0x3c00, 0x3c00, 0x4000}, {0x4000, 0x4200, 0xc600, 0x0000},
  {0x8000, 0x3c00, 0x0000, 0x0000}};

int main() {
  cl::Device device = cl::Device::getDefault();
  if (device.getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_fp16")
      == std::string::npos) {
    std::printf("SKIP: no cl_khr_fp16\nOK\n");
    return EXIT_SUCCESS;
  }
  cl::Context context(device);
  cl::CommandQueue queue(context, device);
  cl::Program program(context, SOURCE);
  program.build("-cl-std=CL1.2");

  const size_t n = sizeof(CASES) / sizeof(CASES[0]); /* 12, three half4 */
  std::vector<uint16_t> a(n), b(n), c(n), out1(n), out4(n);
  for (size_t i = 0; i < n; ++i) {
    a[i] = CASES[i][0];
    b[i] = CASES[i][1];
    c[i] = CASES[i][2];
  }
  const size_t bytes = n * sizeof(uint16_t);
  cl::Buffer ba(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, a.data());
  cl::Buffer bb(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, b.data());
  cl::Buffer bc(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, c.data());
  cl::Buffer bo1(context, CL_MEM_WRITE_ONLY, bytes);
  cl::Buffer bo4(context, CL_MEM_WRITE_ONLY, bytes);

  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> k1(program, "k1");
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> k4(program, "k4");
  k1(cl::EnqueueArgs(queue, cl::NDRange(n)), ba, bb, bc, bo1);
  k4(cl::EnqueueArgs(queue, cl::NDRange(n / 4)), ba, bb, bc, bo4);
  queue.enqueueReadBuffer(bo1, CL_TRUE, 0, bytes, out1.data());
  queue.enqueueReadBuffer(bo4, CL_TRUE, 0, bytes, out4.data());

  int bad = 0;
  for (size_t i = 0; i < n; ++i) {
    for (int v = 0; v < 2; ++v) {
      uint16_t got = v ? out4[i] : out1[i];
      if (got != CASES[i][3]) {
        std::printf("%s fma(0x%04x, 0x%04x, 0x%04x) = 0x%04x, want 0x%04x\n",
                    v ? "half4" : "half", a[i], b[i], c[i], got, CASES[i][3]);
        ++bad;
      }
    }
  }
  if (bad) {
    std::printf("FAIL: %d result(s) not correctly rounded\n", bad);
    return EXIT_FAILURE;
  }
  std::printf("OK\n");
  return EXIT_SUCCESS;
}
