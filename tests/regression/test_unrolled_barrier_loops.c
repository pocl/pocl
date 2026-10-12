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
  Sub-group reductions and scans written as loops of shuffles over the
  sub-group size. With a sub-group size that is known at compile time, from
  intel_reqd_sub_group_size or from the local size X of a work-group function
  specialized for it, these loops are fully unrolled before the barrier
  passes. A loop bounded by a kernel argument stays a b-loop. The same
  happens to loops with explicit barriers, which reuse local memory in every
  iteration: a tree reduction over the local size, and a rotation by a
  constant. All of them must give the same results.
*/

#include "pocl_opencl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SG_SIZE 8
#define ROWS 2
#define WG_SIZE (SG_SIZE * ROWS)
#define NUM_GROUPS 4
#define N (WG_SIZE * NUM_GROUPS)

static const char KernelSource[] =
    "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n"
    "__attribute__((intel_reqd_sub_group_size(8)))\n"
    "__kernel void reduce_required(__global int *out, __global const int *in) "
    "{\n"
    "  size_t i = get_global_linear_id();\n"
    "  int v = in[i];\n"
    "  for (uint m = 1; m < get_max_sub_group_size(); m <<= 1)\n"
    "    v += sub_group_shuffle_xor(v, m);\n"
    "  out[i] = v;\n"
    "}\n"
    /* without a required size, the sub-group size is the local size X */
    "__kernel void scan_local_size(__global int *out, __global const int *in) "
    "{\n"
    "  size_t i = get_global_linear_id();\n"
    "  uint lane = get_sub_group_local_id();\n"
    "  int v = in[i];\n"
    "  for (uint d = 1; d < get_max_sub_group_size(); d <<= 1) {\n"
    "    int other = sub_group_shuffle(v, lane >= d ? lane - d : lane);\n"
    "    if (lane >= d)\n"
    "      v += other;\n"
    "  }\n"
    "  out[i] = v;\n"
    "}\n"
    "__kernel void reduce_runtime(__global int *out, __global const int *in,\n"
    "                             uint width) {\n"
    "  size_t i = get_global_linear_id();\n"
    "  int v = in[i];\n"
    "  for (uint m = 1; m < width; m <<= 1)\n"
    "    v += sub_group_shuffle_xor(v, m);\n"
    "  out[i] = v;\n"
    "}\n"
    "size_t group_size(void) {\n"
    "  return get_local_size(0) * get_local_size(1);\n"
    "}\n"
    "__kernel void tree_local_size(__global int *out, __global const int *in) "
    "{\n"
    "  __local int s[16];\n"
    "  size_t lid = get_local_linear_id();\n"
    "  s[lid] = in[get_global_linear_id()];\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  for (size_t k = group_size() / 2; k > 0; k >>= 1) {\n"
    "    if (lid < k)\n"
    "      s[lid] += s[lid + k];\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  }\n"
    "  out[get_global_linear_id()] = s[0];\n"
    "}\n"
    /* rotates the values of the work-group by 4 work-items */
    "__kernel void rotate_local(__global int *out, __global const int *in) {\n"
    "  __local int scratch[16];\n"
    "  size_t lid = get_local_linear_id();\n"
    "  int v = in[get_global_linear_id()];\n"
    "  for (uint k = 0; k < 4; ++k) {\n"
    "    scratch[lid] = v;\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    v = scratch[(lid + 1) % 16];\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  }\n"
    "  out[get_global_linear_id()] = v;\n"
    "}\n";

enum Check { REDUCE, SCAN, GROUP_SUM, ROTATE };

static int run(cl_context context, cl_command_queue queue, cl_program program,
               const char *name, const cl_int *input, enum Check check,
               size_t rows) {
  const size_t global[] = {SG_SIZE, ROWS * NUM_GROUPS};
  const size_t local[] = {SG_SIZE, rows};
  const size_t wg_size = SG_SIZE * rows;
  const cl_uint width = SG_SIZE;
  cl_int output[N];
  cl_mem in_buf = NULL, out_buf = NULL;
  cl_kernel kernel = NULL;
  int result = 1;
  cl_int err;

  kernel = clCreateKernel(program, name, &err);
  if (err != CL_SUCCESS)
    goto cleanup;
  in_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          sizeof(output), (void *)input, &err);
  if (err != CL_SUCCESS)
    goto cleanup;
  out_buf =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(output), NULL, &err);
  if (err != CL_SUCCESS)
    goto cleanup;
  err = clSetKernelArg(kernel, 0, sizeof(out_buf), &out_buf);
  err |= clSetKernelArg(kernel, 1, sizeof(in_buf), &in_buf);
  if (strcmp(name, "reduce_runtime") == 0)
    err |= clSetKernelArg(kernel, 2, sizeof(width), &width);
  if (err != CL_SUCCESS)
    goto cleanup;
  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL,
                               NULL);
  if (err != CL_SUCCESS)
    goto cleanup;
  err = clEnqueueReadBuffer(queue, out_buf, CL_TRUE, 0, sizeof(output), output,
                            0, NULL, NULL);
  if (err != CL_SUCCESS)
    goto cleanup;

  if (check == GROUP_SUM || check == ROTATE) {
    for (size_t i = 0; i < N; ++i) {
      size_t group = i / wg_size * wg_size;
      cl_int expected = 0;
      if (check == ROTATE)
        expected = input[group + (i - group + 4) % wg_size];
      else
        for (size_t j = 0; j < wg_size; ++j)
          expected += input[group + j];
      if (output[i] != expected) {
        fprintf(stderr, "%s: output[%zu] = %d, expected %d\n", name, i,
                output[i], expected);
        goto cleanup;
      }
    }
    result = 0;
    goto cleanup;
  }

  for (size_t sg = 0; sg < N / SG_SIZE; ++sg) {
    cl_int acc = 0, total = 0;
    for (size_t lane = 0; lane < SG_SIZE; ++lane)
      total += input[sg * SG_SIZE + lane];
    for (size_t lane = 0; lane < SG_SIZE; ++lane) {
      size_t i = sg * SG_SIZE + lane;
      acc += input[i];
      cl_int expected = check == SCAN ? acc : total;
      if (output[i] != expected) {
        fprintf(stderr, "%s: output[%zu] = %d, expected %d\n", name, i,
                output[i], expected);
        goto cleanup;
      }
    }
  }
  result = 0;

cleanup:
  if (err != CL_SUCCESS)
    fprintf(stderr, "%s: OpenCL error %d\n", name, err);
  if (out_buf != NULL)
    clReleaseMemObject(out_buf);
  if (in_buf != NULL)
    clReleaseMemObject(in_buf);
  if (kernel != NULL)
    clReleaseKernel(kernel);
  return result;
}

int main(void) {
  cl_platform_id platform;
  cl_device_id device;
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_int input[N];
  int result = 1;
  cl_int err;

  err = clGetPlatformIDs(1, &platform, NULL);
  if (err != CL_SUCCESS)
    goto error;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
  if (err != CL_SUCCESS)
    goto error;
  if (!poclu_supports_extension(device, "cl_khr_subgroups") ||
      !poclu_supports_extension(device, "cl_intel_required_subgroup_size")) {
    puts("SKIP: The test requires cl_khr_subgroups and "
         "cl_intel_required_subgroup_size");
    return 77;
  }
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  if (err != CL_SUCCESS)
    goto error;
  queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
  if (err != CL_SUCCESS)
    goto error;

  const char *sources[] = {KernelSource};
  program = clCreateProgramWithSource(context, 1, sources, NULL, &err);
  if (err != CL_SUCCESS)
    goto error;
  err = clBuildProgram(program, 1, &device, "-cl-std=CL3.0", NULL, NULL);
  if (err != CL_SUCCESS) {
    char log[4096];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log),
                          log, NULL);
    fprintf(stderr, "Build error:\n%s\n", log);
    goto error;
  }

  for (int i = 0; i < N; ++i)
    input[i] = (i * 7) % 13 - 6;

  result = run(context, queue, program, "reduce_required", input, REDUCE, ROWS);
  result |= run(context, queue, program, "scan_local_size", input, SCAN, ROWS);
  result |= run(context, queue, program, "reduce_runtime", input, REDUCE, ROWS);
  /* the same kernel with two local sizes, which must not share a fold */
  result |=
      run(context, queue, program, "tree_local_size", input, GROUP_SUM, ROWS);
  result |=
      run(context, queue, program, "tree_local_size", input, GROUP_SUM, 1);
  result |= run(context, queue, program, "rotate_local", input, ROTATE, ROWS);
  if (result == 0)
    puts("OK");

error:
  if (err != CL_SUCCESS)
    fprintf(stderr, "OpenCL error %d\n", err);
  if (program != NULL)
    clReleaseProgram(program);
  if (queue != NULL)
    clReleaseCommandQueue(queue);
  if (context != NULL)
    clReleaseContext(context);
  return result;
}
