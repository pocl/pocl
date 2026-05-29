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
  Reduced from a Julia KernelAbstractions kernel (get_global_id reduced via
  llvm-reduce). The kernel is, per work-item:

      if (get_local_id(0) > 7) return;   // work-item-varying, dead for localsize 8
      barrier();
      A[get_global_id(0)] = get_global_id(0) + 1;

  The varying early-return before the barrier makes WorkitemLoops peel work-item
  (0,0,0). The peeled copy used to read a stale global id (the loop-updated value),
  so get_global_id(0) returned the work-group size instead of 0: element 0 was
  left untouched. Launched as global=local=8, the correct result is {1,..,8}.
*/

#include "pocl_opencl.h"

#include <stdio.h>
#include <stdlib.h>

// Mirrors the Julia argument layout the SPIR-V kernel expects.
typedef struct { cl_long ndrange; cl_long numblocks; } Ctx;
typedef struct { cl_long *ptr; size_t maxsize; size_t dim1; size_t len; } DeviceArray;

#define CHECK_ERROR(err)                                                       \
  if (err != CL_SUCCESS) {                                                     \
    printf("OpenCL Error %d at %s:%d\n", err, __FILE__, __LINE__);             \
    return 1;                                                                  \
  }

static const char *KERNEL_NAME =
    "_Z7gpu_mwe16CompilerMetadataI11DynamicSize12DynamicCheckv16CartesianIndices"
    "ILi1E5TupleI5OneToI5Int64EEE7NDRangeILi1ES0_10StaticSizeI4_8__ES8_vEE13CL"
    "DeviceArrayIS5_Li1ELi1EE";

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

  // The reduced kernel's barrier lowers to a sub_group_barrier, so the program
  // only links on devices whose builtin library has the subgroup builtins.
  if (!poclu_supports_extension(device, "cl_khr_subgroups")) {
    printf("SKIP: The test requires cl_khr_subgroups\n");
    return 77;
  }

  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR(err);
  cl_queue_properties props[] = {0};
  cl_command_queue queue =
      clCreateCommandQueueWithProperties(context, device, props, &err);
  CHECK_ERROR(err);

  const size_t N = 8;
  DeviceArray A = {NULL, sizeof(cl_long) * N, N, N};
  A.ptr = clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(cl_long) * N, 0);
  if (A.ptr == NULL) {
    printf("SVM allocation failed\n");
    return 1;
  }
  for (size_t i = 0; i < N; i++)
    A.ptr[i] = -1;

  FILE *f = fopen(SRCDIR "/test_peeled_wi_global_id.spv", "rb");
  if (!f) {
    printf("Failed to open test_peeled_wi_global_id.spv\n");
    return 1;
  }
  fseek(f, 0, SEEK_END);
  size_t size = ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char *binary = malloc(size);
  fread(binary, 1, size, f);
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

  Ctx ctx = {(cl_long)N, 1};
  void *svm_ptrs[] = {A.ptr};
  CHECK_ERROR(clSetKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS,
                                  sizeof(void *), svm_ptrs));
  CHECK_ERROR(clSetKernelArg(kernel, 0, sizeof(Ctx), &ctx));
  CHECK_ERROR(clSetKernelArg(kernel, 1, sizeof(DeviceArray), &A));

  size_t global_size = N, local_size = N;
  CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                                     &local_size, 0, NULL, NULL));
  CHECK_ERROR(clFinish(queue));

  int failed = 0;
  for (size_t i = 0; i < N; i++)
    if (A.ptr[i] != (cl_long)(i + 1)) {
      printf("A[%zu] = %lld, expected %zu\n", i, (long long)A.ptr[i], i + 1);
      failed = 1;
    }

  clSVMFree(context, A.ptr);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  free(binary);

  if (failed)
    return 1;
  printf("OK\n");
  return 0;
}
