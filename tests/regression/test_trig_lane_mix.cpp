/* Regression test: float sin/cos/tan with explicit vector types must give
   per-lane results. A lane holding a tiny value must not be affected by
   another lane holding a large one (>= 2^23), which selects the
   large-argument reduction path. Compare each lane of a mixed vector with
   the scalar result for the same input.

   Copyright (c) 2026 PoCL developers. MIT license, see COPYING. */

#include "pocl_opencl.h"
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <vector>

static const char *SOURCE = R"RAW(
__kernel void scalar_k(__global const float *in, __global float *s, __global float *c, __global float *t) {
  size_t i = get_global_id(0); float x = in[i]; s[i] = sin(x); c[i] = cos(x); t[i] = tan(x); }
__kernel void vec4_k(__global const float4 *in, __global float4 *s, __global float4 *c, __global float4 *t) {
  size_t i = get_global_id(0); float4 x = in[i]; s[i] = sin(x); c[i] = cos(x); t[i] = tan(x); }
__kernel void vec8_k(__global const float8 *in, __global float8 *s, __global float8 *c, __global float8 *t) {
  size_t i = get_global_id(0); float8 x = in[i]; s[i] = sin(x); c[i] = cos(x); t[i] = tan(x); }
)RAW";

int main() {
  cl::Platform platform = cl::Platform::getDefault();
  cl::Device device = cl::Device::getDefault();
  cl::Context context(device);
  cl::CommandQueue queue(context, device);
  cl::Program program(context, SOURCE);
  program.build();

  /* 8 lanes: tiny values mixed with large ones */
  const float lanes[8] = {FLT_MIN, 1e8f, 1e-10f, INFINITY, 1e-5f, 1e20f, 0.1f, 3e7f};
  const size_t N = 8;
  std::vector<float> in(lanes, lanes + N), s(N), c(N), t(N), vs(N), vc(N), vt(N);
  cl::Buffer inBuf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), in.data());
  cl::Buffer sBuf(context, CL_MEM_WRITE_ONLY, N * sizeof(float));
  cl::Buffer cBuf(context, CL_MEM_WRITE_ONLY, N * sizeof(float));
  cl::Buffer tBuf(context, CL_MEM_WRITE_ONLY, N * sizeof(float));

  auto run = [&](const char *name, size_t global, std::vector<float> &S, std::vector<float> &C, std::vector<float> &T) {
    cl::Kernel k(program, name);
    k.setArg(0, inBuf); k.setArg(1, sBuf); k.setArg(2, cBuf); k.setArg(3, tBuf);
    queue.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(global), cl::NullRange);
    queue.enqueueReadBuffer(sBuf, CL_TRUE, 0, N * sizeof(float), S.data());
    queue.enqueueReadBuffer(cBuf, CL_TRUE, 0, N * sizeof(float), C.data());
    queue.enqueueReadBuffer(tBuf, CL_TRUE, 0, N * sizeof(float), T.data());
  };
  run("scalar_k", N, s, c, t);
  int failures = 0;
  const char *vecKernels[] = {"vec4_k", "vec8_k"};
  const size_t vecGlobal[] = {N / 4, N / 8};
  for (int v = 0; v < 2; ++v) {
    run(vecKernels[v], vecGlobal[v], vs, vc, vt);
    for (size_t i = 0; i < N; ++i) {
      auto same = [](float a, float b) { return a == b || (std::isnan(a) && std::isnan(b)); };
      if (!same(vs[i], s[i]) || !same(vc[i], c[i]) || !same(vt[i], t[i])) {
        printf("%s lane %zu (x=%a): sin %a vs scalar %a, cos %a vs %a, tan %a vs %a\n",
               vecKernels[v], i, in[i], vs[i], s[i], vc[i], c[i], vt[i], t[i]);
        ++failures;
      }
    }
  }
  if (failures) { printf("FAIL: %d lane mismatches\n", failures); return 1; }
  printf("OK\n");
  return 0;
}
