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
  Reduced from a Julia (OpenCL.jl) kernel with a printf-then-abort exception
  path (https://github.com/pocl/pocl/issues/1958). After a barrier, a
  work-item-varying bounds check conditionally calls a noreturn helper that
  printfs a diagnostic before ending in OpUnreachable:

      barrier();
      if (in_bounds(idx)) {         // work-item-varying!
        if (out_of_bounds(gid)) {
          printf("...");            // expands to a string-length loop
          __builtin_unreachable();
        }
      }
      // return

  For the work-item loop handlers, ConvertUnreachablesToReturns removes the
  blocks leading to the unreachable. Detaching only the unreachable-terminated
  blocks severed the exit edge of the printf expansion's string-length loop,
  leaving an infinite loop that never reaches the parallel region exit, which
  WorkitemLoops cannot handle (it asserted in createLoopAround). Blocks that
  cannot reach the kernel exit must be deleted wholesale.

  The crash happened when building the work-group function, so completing the
  build and a (non-aborting) launch is the test.
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

  FILE *f = fopen(SRCDIR "/test_issue_1958.spv", "rb");
  if (!f) {
    printf("Failed to open test_issue_1958.spv\n");
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

  char kernel_name[1024];
  CHECK_ERROR(clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES,
                               sizeof(kernel_name), kernel_name, NULL));
  cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
  CHECK_ERROR(err);

  // The kernel arguments are pointers to structs whose first word is a
  // bounds value the work-item indices are checked against; a large bound
  // steers all work-items away from the printf-then-abort path. The loaded
  // values are only compared, never dereferenced.
  cl_ulong bound[4] = {(cl_ulong)1 << 40, 0, 0, 0};
  cl_mem buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(bound), bound, &err);
  CHECK_ERROR(err);
  cl_uint num_args;
  CHECK_ERROR(clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(num_args),
                              &num_args, NULL));
  for (cl_uint i = 0; i < num_args; i++)
    CHECK_ERROR(clSetKernelArg(kernel, i, sizeof(buf), &buf));

  size_t global_size = 64, local_size = 16;
  CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                                     &local_size, 0, NULL, NULL));
  CHECK_ERROR(clFinish(queue));

  clReleaseMemObject(buf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  free(binary);

  printf("OK\n");
  return 0;
}
