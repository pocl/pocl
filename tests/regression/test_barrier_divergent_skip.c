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
  Reduced from a bounds-checked KernelAbstractions.jl radix-sort kernel. One
  arm of a branch after a barrier skips a variable-trip loop, then reconverges
  with the other arm at the next barrier:

      barrier();
      br i1 %skip, label %join, label %loop
      ...
      join:                  ; reconvergence point
      barrier();

  Treating the successors as disjoint region entries produced a malformed
  region ending at the loop latch. Building the work-group function then
  aborted in ParallelRegion::chainAfter.

  The crash happened when building the work-group function; completing the
  build and a launch with correct results is the test. The launch takes the
  skip path in all work-items so the barrier behavior is defined.
*/

#include "pocl_opencl.h"

#include <stdio.h>
#include <stdlib.h>
#ifndef _WIN32
#include <unistd.h>
#endif

#define CHECK_ERROR(err)                                                       \
  if (err != CL_SUCCESS) {                                                     \
    printf("OpenCL Error %d at %s:%d\n", err, __FILE__, __LINE__);             \
    return 1;                                                                  \
  }

int main(int argc, char **argv) {
  cl_uint platform_index = argc > 1 ? (cl_uint)atoi(argv[1]) : 0;
  cl_int err;

#ifndef _WIN32
  // Fail quickly instead of tripping the much larger ctest timeout in case
  // a regressed build hangs.
  alarm(300);
#endif

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

  FILE *f = fopen(SRCDIR "/test_barrier_divergent_skip.spv", "rb");
  if (!f) {
    printf("Failed to open test_barrier_divergent_skip.spv\n");
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
  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);
    char *log = malloc(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log,
                          NULL);
    printf("Build error:\n%s\n", log);
    return 1;
  }

  // Building the work-group function is what used to crash; force it even
  // when the driver would otherwise defer compilation to launch time.
  size_t binary_size;
  CHECK_ERROR(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                               sizeof(binary_size), &binary_size, NULL));

  cl_kernel kernel = clCreateKernel(program, "vartrip", &err);
  CHECK_ERROR(err);

  // All work-items enter the barrier and take the loop-skipping arm, so the
  // barrier behavior is defined and every work-item stores 42.
  const size_t local_size = 64, global_size = 128;
  const cl_long opaque = 0;
  const cl_uchar enter = 1, skip = 1, exitl = 1;
  cl_int result[128];
  cl_mem out =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(result), NULL, &err);
  CHECK_ERROR(err);
  CHECK_ERROR(clSetKernelArg(kernel, 0, sizeof(out), &out));
  CHECK_ERROR(clSetKernelArg(kernel, 1, sizeof(opaque), &opaque));
  CHECK_ERROR(clSetKernelArg(kernel, 2, sizeof(enter), &enter));
  CHECK_ERROR(clSetKernelArg(kernel, 3, sizeof(skip), &skip));
  CHECK_ERROR(clSetKernelArg(kernel, 4, sizeof(exitl), &exitl));
  CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                                     &local_size, 0, NULL, NULL));
  CHECK_ERROR(clEnqueueReadBuffer(queue, out, CL_TRUE, 0, sizeof(result),
                                  result, 0, NULL, NULL));

  for (size_t i = 0; i < global_size; ++i) {
    if (result[i] != 42) {
      printf("FAIL at %zu: got %d, expected 42\n", i, result[i]);
      return 1;
    }
  }

  clReleaseMemObject(out);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  free(binary);

  printf("OK\n");
  return 0;
}
