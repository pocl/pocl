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
  Reduced (via llvm-reduce) from a Julia KernelAbstractions kernel:
  AcceleratedKernels.jl's Blelloch scan, whose second launch hung. The
  work-group function is, per work-item:

      if (inclusive) {            // uniform kernel argument
        barrier();
        if (lid >= 1) {           // work-item-varying!
          local_memory[lid] = 0;
          if (group_id >= 0)
            local_memory[0] = 0;
        }
      }
      // join

  The parallel region entered at the barrier reaches the join through three
  edges. BarrierTailReplication used to clone the join tail once per edge,
  which turned them into three separate region exits selected by the
  work-item-varying branch. WorkitemLoops keeps only the exit path the first
  work-item takes and marks the rest unreachable in the replicated iterations,
  so the optimizer folded the work-item loop's exit test into an
  assume(lid < 1): an unconditional self-loop, hanging the kernel. Edges from
  one region to the same join must share a single tail replica.

  Launched as global=local=64; completing at all is the test.
*/

#include "pocl_opencl.h"

#include <stdio.h>
#include <stdlib.h>
#ifndef _WIN32
#include <unistd.h>
#endif

// Mirrors the Julia argument layout the SPIR-V kernel expects.
typedef struct { cl_uint random_seed; } KernelState;
typedef struct { cl_long ndrange; cl_long numblocks; } Ctx;
typedef struct { cl_uint *ptr; size_t maxsize; size_t dim1; size_t len; } DeviceArray;

#define CHECK_ERROR(err)                                                       \
  if (err != CL_SUCCESS) {                                                     \
    printf("OpenCL Error %d at %s:%d\n", err, __FILE__, __LINE__);             \
    return 1;                                                                  \
  }

static const char *KERNEL_NAME =
    "_Z21gpu_accumulate_block_16CompilerMetadataI11DynamicSize12DynamicCheckv"
    "16CartesianIndicesILi1E5TupleI5OneToI5Int64EEE7NDRangeILi1ES0_10StaticSize"
    "I5_64__ES8_vEE1_13CLDeviceArrayI6UInt32Li1ELi1EESG_SG_4Boolvv";

int main(int argc, char **argv) {
  cl_uint platform_index = argc > 1 ? (cl_uint)atoi(argv[1]) : 0;
  cl_int err;

#ifndef _WIN32
  // A regressed build hangs in clFinish; fail quickly instead of tripping the
  // much larger ctest timeout.
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

  DeviceArray V = {NULL, sizeof(cl_uint), 1, 1};
  V.ptr = clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(cl_uint), 0);
  if (V.ptr == NULL) {
    printf("SVM allocation failed\n");
    return 1;
  }
  V.ptr[0] = 42;

  FILE *f = fopen(SRCDIR "/test_multi_exit_barrier_region.spv", "rb");
  if (!f) {
    printf("Failed to open test_multi_exit_barrier_region.spv\n");
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
  cl_kernel kernel = clCreateKernel(program, KERNEL_NAME, &err);
  CHECK_ERROR(err);

  KernelState state = {0};
  Ctx ctx = {64, 1};
  cl_uint init = 0, neutral = 0;
  cl_uchar inclusive = 1;
  void *svm_ptrs[] = {V.ptr};
  CHECK_ERROR(clSetKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS,
                                  sizeof(void *), svm_ptrs));
  CHECK_ERROR(clSetKernelArg(kernel, 0, sizeof(KernelState), &state));
  CHECK_ERROR(clSetKernelArg(kernel, 1, sizeof(Ctx), &ctx));
  CHECK_ERROR(clSetKernelArg(kernel, 2, sizeof(DeviceArray), &V));
  CHECK_ERROR(clSetKernelArg(kernel, 3, sizeof(init), &init));
  CHECK_ERROR(clSetKernelArg(kernel, 4, sizeof(neutral), &neutral));
  CHECK_ERROR(clSetKernelArg(kernel, 5, sizeof(inclusive), &inclusive));

  size_t global_size = 64, local_size = 64;
  CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                                     &local_size, 0, NULL, NULL));
  CHECK_ERROR(clFinish(queue));

  clSVMFree(context, V.ptr);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  free(binary);

  printf("OK\n");
  return 0;
}
