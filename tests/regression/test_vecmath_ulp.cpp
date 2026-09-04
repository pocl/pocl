/* Regression test: float math builtins that may be routed to a vector
   math library (libmvec, SLEEF) must stay within the OpenCL C ULP bounds.

   Uses scalar-typed kernels so the work-item loop vectorizes and the
   vector library is reached (explicit vector types are scalarized and
   never reach it). Inputs concentrate where the libraries are weakest:
   log near 1, plus the known worst input of glibc libmvec's logf.

   Reference is double precision; ULP is measured as the CTS does, in
   units of the ULP of the reference rounded to float.

   Copyright (c) 2026 PoCL developers. MIT license, see COPYING. */

#include "pocl_opencl.h"
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <cmath>
#include <cstdio>
#include <vector>

static const char *SOURCE = R"RAW(
__kernel void k_log(__global const float *in, __global float *out) {
  size_t i = get_global_id(0); out[i] = log(in[i]); }
__kernel void k_exp(__global const float *in, __global float *out) {
  size_t i = get_global_id(0); out[i] = exp(in[i]); }
__kernel void k_pow(__global const float *in, __global float *out) {
  size_t i = get_global_id(0); out[i] = pow(in[i], 2.5f); }
)RAW";

static float ulpError(float got, double ref) {
  float refF = (float)ref;
  if (got == refF) return 0.0f;
  int exp;
  std::frexp((float)std::fabs(ref), &exp);
  if (exp < -125) exp = -125;
  double ulp = std::ldexp(1.0, exp - 24);
  return (float)(std::fabs((double)got - ref) / ulp);
}

int main() {
  cl::Platform platform = cl::Platform::getDefault();
  cl::Device device = cl::Device::getDefault();
  cl::Context context(device);
  cl::CommandQueue queue(context, device);
  cl::Program program(context, SOURCE);
  program.build();

  const size_t N = 4096;
  std::vector<float> in(N), out(N);
  /* dense around 1 for log, moderate range for exp and pow */
  for (size_t i = 0; i < N; ++i)
    in[i] = 0.5f + 1.0f * (float)i / (float)N;
  in[0] = 0x1.5566p-1f;             /* glibc libmvec logf worst case seen */
  in[1] = 0.8863354921340942f;

  struct Case { const char *name; double (*ref)(double); double bound; } cases[] = {
    {"k_log", [](double x) { return std::log(x); }, 3.0},
    {"k_exp", [](double x) { return std::exp(x); }, 3.0},
    {"k_pow", [](double x) { return std::pow(x, 2.5); }, 16.0},
  };
  int failures = 0;
  cl::Buffer inBuf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), in.data());
  cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY, N * sizeof(float));
  for (const Case &c : cases) {
    cl::Kernel kernel(program, c.name);
    kernel.setArg(0, inBuf);
    kernel.setArg(1, outBuf);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N), cl::NullRange);
    queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, N * sizeof(float), out.data());
    float worst = 0.0f; size_t worstIdx = 0;
    for (size_t i = 0; i < N; ++i) {
      float e = ulpError(out[i], c.ref((double)in[i]));
      if (e > worst) { worst = e; worstIdx = i; }
    }
    printf("%s: max %.3f ULP (bound %.0f) at %a\n", c.name, worst, c.bound, in[worstIdx]);
    if (worst > c.bound) ++failures;
  }
  if (failures) { printf("FAIL\n"); return 1; }
  printf("OK\n");
  return 0;
}
