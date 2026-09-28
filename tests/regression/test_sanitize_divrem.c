// Copyright (c) 2026 PoCL developers
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

/*
  Integer division and remainder must not trap in OpenCL: dividing by zero or
  INT_MIN by -1 gives an unspecified value. On x86, the SanitizeUBofDivRem pass
  (or the SIGFPE handler, where enabled) takes care of this. The pass folds a
  value divided by itself to a constant; it used to fold x / x to 0 instead
  of 1. Clang folds x / x itself for OpenCL C, but SPIR-V input is not folded,
  so the kernel is given as SPIR-V, generated from test_sanitize_divrem.ll with

      llvm-as test_sanitize_divrem.ll
      llvm-spirv --spirv-max-version=1.0 test_sanitize_divrem.bc

  Per work-item i, it computes x / x and x % x (signed and unsigned, scalar and
  <4 x i32>) and n / d and n % d for runtime n and d, which include d == 0 and
  INT_MIN / -1.
*/

#include "pocl_opencl.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_ERROR(err)                                                       \
  if (err != CL_SUCCESS) {                                                     \
    printf("OpenCL Error %d at %s:%d\n", err, __FILE__, __LINE__);             \
    return 1;                                                                  \
  }

#define N 4

int main(int argc, char **argv) {
  cl_uint platform_index = argc > 1 ? (cl_uint)atoi(argv[1]) : 0;
  cl_int err;

  cl_uint num_platforms;
  CHECK_ERROR(clGetPlatformIDs(0, NULL, &num_platforms));
  if (platform_index >= num_platforms) {
    printf("Platform index %u out of range\n", platform_index);
    return 1;
  }
  cl_platform_id *platforms = malloc(sizeof(cl_platform_id) * num_platforms);
  CHECK_ERROR(clGetPlatformIDs(num_platforms, platforms, NULL));
  cl_platform_id platform = platforms[platform_index];
  free(platforms);

  cl_device_id device;
  CHECK_ERROR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL));

  if (!poclu_device_supports_il(device, "SPIR-V_1.0")) {
    printf("SKIP: The test requires support for SPIR-V 1.0\n");
    return 77;
  }

  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR(err);
  cl_queue_properties props[] = {0};
  cl_command_queue queue =
      clCreateCommandQueueWithProperties(context, device, props, &err);
  CHECK_ERROR(err);

  FILE *f = fopen(SRCDIR "/test_sanitize_divrem.spv", "rb");
  if (!f) {
    printf("Failed to open test_sanitize_divrem.spv\n");
    return 1;
  }
  fseek(f, 0, SEEK_END);
  size_t size = ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char *binary = malloc(size);
  if (fread(binary, 1, size, f) != size) {
    printf("Failed to read SPIR-V\n");
    free(binary);
    fclose(f);
    return 1;
  }
  fclose(f);

  cl_program program = clCreateProgramWithIL(context, binary, size, &err);
  CHECK_ERROR(err);
  CHECK_ERROR(clBuildProgram(program, 1, &device, NULL, NULL, NULL));
  cl_kernel kernel = clCreateKernel(program, "divrem", &err);
  CHECK_ERROR(err);

  cl_int x[N] = {1, 7, -3, INT_MAX};
  cl_int v[N * 4] = {1,  -1,      2,  INT_MIN, 3, -5,      9,       100,
                     -8, INT_MAX, 13, 42,      6, INT_MIN, INT_MAX, -1};
  cl_int num[N] = {7, INT_MIN, 7, -9};
  cl_int den[N] = {0, -1, 2, 4};
  cl_int out[N * 8];
  cl_int vout[N * 16];

  cl_mem bufs[6];
  bufs[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(x), x, &err);
  CHECK_ERROR(err);
  bufs[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(v), v, &err);
  CHECK_ERROR(err);
  bufs[2] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(num), num, &err);
  CHECK_ERROR(err);
  bufs[3] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(den), den, &err);
  CHECK_ERROR(err);
  bufs[4] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(out), NULL, &err);
  CHECK_ERROR(err);
  bufs[5] =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(vout), NULL, &err);
  CHECK_ERROR(err);
  for (cl_uint i = 0; i < 6; i++)
    CHECK_ERROR(clSetKernelArg(kernel, i, sizeof(cl_mem), &bufs[i]));

  size_t global_size = N;
  CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL,
                                     0, NULL, NULL));
  CHECK_ERROR(clEnqueueReadBuffer(queue, bufs[4], CL_TRUE, 0, sizeof(out), out,
                                  0, NULL, NULL));
  CHECK_ERROR(clEnqueueReadBuffer(queue, bufs[5], CL_TRUE, 0, sizeof(vout),
                                  vout, 0, NULL, NULL));

  int errors = 0;
#define EXPECT(what, got, expected)                                            \
  if ((got) != (expected)) {                                                   \
    printf("%s: got %d, expected %d\n", what, (int)(got), (int)(expected));    \
    errors++;                                                                  \
  }

  for (int i = 0; i < N; i++) {
    // x / x and x % x (all x are nonzero)
    EXPECT("x / x (signed)", out[i * 8 + 0], 1);
    EXPECT("x / x (unsigned)", out[i * 8 + 1], 1);
    EXPECT("x % x (signed)", out[i * 8 + 2], 0);
    EXPECT("x % x (unsigned)", out[i * 8 + 3], 0);
    for (int j = 0; j < 4; j++) {
      EXPECT("v / v (signed)", vout[i * 16 + 0 + j], 1);
      EXPECT("v / v (unsigned)", vout[i * 16 + 4 + j], 1);
      EXPECT("v % v (signed)", vout[i * 16 + 8 + j], 0);
      EXPECT("v % v (unsigned)", vout[i * 16 + 12 + j], 0);
    }
  }

  // n / d and n % d: the results for d == 0 and signed INT_MIN / -1 are
  // unspecified, so only check that computing them did not trap, and check
  // the defined ones.
  EXPECT("INT_MIN / -1 (unsigned)", out[1 * 8 + 5], 0);
  EXPECT("INT_MIN % -1 (unsigned)", (cl_uint)out[1 * 8 + 7], (cl_uint)INT_MIN);
  EXPECT("7 / 2 (signed)", out[2 * 8 + 4], 3);
  EXPECT("7 / 2 (unsigned)", out[2 * 8 + 5], 3);
  EXPECT("7 % 2 (signed)", out[2 * 8 + 6], 1);
  EXPECT("7 % 2 (unsigned)", out[2 * 8 + 7], 1);
  EXPECT("-9 / 4 (signed)", out[3 * 8 + 4], -2);
  EXPECT("-9 / 4 (unsigned)", (cl_uint)out[3 * 8 + 5], (cl_uint)-9 / 4);
  EXPECT("-9 % 4 (signed)", out[3 * 8 + 6], -1);
  EXPECT("-9 % 4 (unsigned)", (cl_uint)out[3 * 8 + 7], (cl_uint)-9 % 4);

  for (int i = 0; i < 6; i++)
    clReleaseMemObject(bufs[i]);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  free(binary);

  if (errors) {
    printf("FAIL: %d errors\n", errors);
    return 1;
  }
  printf("OK\n");
  return 0;
}
