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
  SPIR-V input using SPV_INTEL_subgroups, which the CPU device accepts along
  with cl_intel_subgroups. Unlike the khr shuffles, OpSubgroupShuffleINTEL and
  OpSubgroupShuffleXorINTEL also take vectors, so a uint4 is exchanged with a
  single shuffle (one work-group barrier on the CPU) rather than four.

  test_spirv_intel_subgroups.spv is built from this OpenCL C source:

    __attribute__((intel_reqd_sub_group_size(8)))
    kernel void shuffle(global uint4 *out)
    {
      uint gid = get_global_id(0);
      uint4 v = (uint4)(gid, gid + 100, gid + 200, gid + 300);
      out[2 * gid] = intel_sub_group_shuffle(v, (get_sub_group_local_id() + 3) % 8);
      out[2 * gid + 1] = intel_sub_group_shuffle_xor(v, 5);
    }

  with LLVM 22 and the SPIRV-LLVM-Translator:

    clang -cc1 -triple spir64-unknown-unknown -cl-std=CL2.0 \
      -cl-ext=+cl_intel_subgroups,+cl_intel_required_subgroup_size \
      -finclude-default-header -emit-llvm-bc -O0 -disable-llvm-passes \
      test_spirv_intel_subgroups.cl -o test_spirv_intel_subgroups.bc
    llvm-spirv --spirv-ext=+SPV_INTEL_subgroups \
      test_spirv_intel_subgroups.bc -o test_spirv_intel_subgroups.spv

  Launched with a work-group of 16, i.e. two sub-groups of 8.
*/

#include "pocl_opencl.h"

#include <stdio.h>
#include <stdlib.h>

#define CHECK_ERROR(err)                                                       \
  if (err != CL_SUCCESS) {                                                     \
    printf("OpenCL Error %d at %s:%d\n", err, __FILE__, __LINE__);             \
    return 1;                                                                  \
  }

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

  // SPIR-V 1.1 for the SubgroupSize execution mode.
  if (!poclu_device_supports_il(device, "SPIR-V_1.1")) {
    printf("SKIP: The test requires support for SPIR-V 1.1\n");
    return 77;
  }
  if (!poclu_supports_extension(device, "cl_intel_subgroups") ||
      !poclu_supports_extension(device, "cl_intel_required_subgroup_size")) {
    printf("SKIP: The test requires cl_intel_subgroups and "
           "cl_intel_required_subgroup_size\n");
    return 77;
  }
  size_t sg_sizes[32], sg_sizes_size;
  CHECK_ERROR(clGetDeviceInfo(device, CL_DEVICE_SUB_GROUP_SIZES_INTEL,
                              sizeof(sg_sizes), sg_sizes, &sg_sizes_size));
  int has_sg8 = 0;
  for (size_t i = 0; i < sg_sizes_size / sizeof(size_t); i++)
    has_sg8 |= sg_sizes[i] == 8;
  if (!has_sg8) {
    printf("SKIP: The test requires a sub-group size of 8\n");
    return 77;
  }

  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR(err);
  cl_queue_properties props[] = {0};
  cl_command_queue queue =
      clCreateCommandQueueWithProperties(context, device, props, &err);
  CHECK_ERROR(err);

  FILE *f = fopen(SRCDIR "/test_spirv_intel_subgroups.spv", "rb");
  if (!f) {
    printf("Failed to open test_spirv_intel_subgroups.spv\n");
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
  cl_kernel kernel = clCreateKernel(program, "shuffle", &err);
  CHECK_ERROR(err);

  enum { N = 32, LOCAL = 16, SG = 8 };
  cl_mem out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                              sizeof(cl_uint4) * 2 * N, NULL, &err);
  CHECK_ERROR(err);
  CHECK_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &out));

  size_t global_size = N, local_size = LOCAL;
  CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                                     &local_size, 0, NULL, NULL));
  cl_uint4 res[2 * N];
  CHECK_ERROR(clEnqueueReadBuffer(queue, out, CL_TRUE, 0, sizeof(res), res, 0,
                                  NULL, NULL));

  int failed = 0;
  for (cl_uint i = 0; i < N; i++) {
    cl_uint first = i / SG * SG;
    cl_uint src[2] = {first + (i % SG + 3) % SG, first + ((i % SG) ^ 5)};
    for (int op = 0; op < 2; op++)
      for (int c = 0; c < 4; c++) {
        cl_uint expected = src[op] + 100 * c;
        if (res[2 * i + op].s[c] != expected) {
          printf("%s: out[%u].s%d = %u, expected %u\n",
                 op ? "shuffle_xor" : "shuffle", i, c, res[2 * i + op].s[c],
                 expected);
          failed = 1;
        }
      }
  }

  clReleaseMemObject(out);
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
