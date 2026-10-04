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
  The value passed to the sub-group collectives is computed in a divergent
  branch that contains a (never taken) early return, as emitted for
  bounds-checked array accesses by e.g. Julia's OpenCL.jl. The return makes
  the barriers in the inlined collectives conditional, so WorkitemLoops peels
  the first work-item of their parallel region. The peeled copy must use the
  same __pocl_work_group_alloca storage as the rest of the work-items;
  otherwise work-item 0 combines its own value with uninitialized data and
  broadcasts that result to the whole sub-group.
*/

#include "pocl_opencl.h"

#include <stdio.h>
#include <stdlib.h>

#define WORK_SIZE 8
#define ENTER_N 5

#define CHECK_ERROR(err)                                                       \
  do {                                                                         \
    if ((err) != CL_SUCCESS) {                                                 \
      fprintf(stderr, "OpenCL error %d at %s:%d\n", (err), __FILE__,           \
              __LINE__);                                                       \
      goto error;                                                              \
    }                                                                          \
  } while (0)

static const char KernelSource[] =
    "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n"
    "__kernel void test_kernel(__global int *out,\n"
    "                          __global const int *input,\n"
    "                          uint enter_n, uint logical_len) {\n"
    "  uint i = get_sub_group_local_id();\n"
    "  uint n = get_sub_group_size();\n"
    "  int x = 0;\n"
    "  if (i < enter_n) {\n"
    "    if (i >= logical_len) return;\n"
    "    x = input[i];\n"
    "  }\n"
    "  out[0 * n + i] = sub_group_reduce_add(x);\n"
    "  out[1 * n + i] = sub_group_scan_inclusive_add(x);\n"
    "  out[2 * n + i] = sub_group_scan_exclusive_add(x);\n"
    "  out[3 * n + i] = sub_group_broadcast(x, 1);\n"
    "  out[4 * n + i] = sub_group_any(x > 4);\n"
    "}\n";

int main(void) {
  const cl_int input[WORK_SIZE] = {1, 2, 3, 4, 5, 6, 7, 8};
  cl_int output[5 * WORK_SIZE];
  cl_int expected[5 * WORK_SIZE];
  const cl_uint enter_n = ENTER_N;
  const cl_uint logical_len = WORK_SIZE;
  const size_t work_size = WORK_SIZE;
  cl_int err;
  cl_uint num_platforms;
  cl_platform_id *platforms = NULL;
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem output_buffer = NULL;
  cl_mem input_buffer = NULL;
  int result = 1;

  /* Only the first ENTER_N work-items contribute their input. */
  cl_int sum = 0;
  for (size_t i = 0; i < WORK_SIZE; ++i) {
    cl_int x = i < ENTER_N ? input[i] : 0;
    expected[1 * WORK_SIZE + i] = sum + x;
    expected[2 * WORK_SIZE + i] = sum;
    sum += x;
  }
  for (size_t i = 0; i < WORK_SIZE; ++i) {
    expected[0 * WORK_SIZE + i] = sum;
    expected[3 * WORK_SIZE + i] = input[1];
    expected[4 * WORK_SIZE + i] = 1;
  }
  for (size_t i = 0; i < 5 * WORK_SIZE; ++i)
    output[i] = -1;

  err = clGetPlatformIDs(0, NULL, &num_platforms);
  CHECK_ERROR(err);
  if (num_platforms == 0) {
    puts("SKIP: no OpenCL platforms");
    return 77;
  }
  platforms = malloc(num_platforms * sizeof(*platforms));
  if (platforms == NULL)
    goto error;
  err = clGetPlatformIDs(num_platforms, platforms, NULL);
  CHECK_ERROR(err);

  cl_device_id device;
  err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 1, &device, NULL);
  CHECK_ERROR(err);
  if (!poclu_supports_extension(device, "cl_khr_subgroups")) {
    puts("SKIP: The test requires cl_khr_subgroups");
    free(platforms);
    return 77;
  }
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR(err);
  const cl_queue_properties properties[] = {0};
  queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
  CHECK_ERROR(err);

  const char *sources[] = {KernelSource};
  program = clCreateProgramWithSource(context, 1, sources, NULL, &err);
  CHECK_ERROR(err);
  err = clBuildProgram(program, 1, &device, "-cl-std=CL3.0", NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t log_size = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);
    char *log = malloc(log_size);
    if (log != NULL) {
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size,
                            log, NULL);
      fprintf(stderr, "Build error:\n%s\n", log);
      free(log);
    }
    goto error;
  }
  kernel = clCreateKernel(program, "test_kernel", &err);
  CHECK_ERROR(err);

  /* The kernel assumes that the work-group forms a single sub-group. */
  size_t sg_size = 0;
  CHECK_ERROR(clGetKernelSubGroupInfo(
      kernel, device, CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE,
      sizeof(work_size), &work_size, sizeof(sg_size), &sg_size, NULL));
  if (sg_size != WORK_SIZE) {
    printf("SKIP: sub-group size %zu differs from the work-group size\n",
           sg_size);
    result = 77;
    goto error;
  }

  output_buffer =
      clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(output), output, &err);
  CHECK_ERROR(err);
  input_buffer =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(input), (void *)input, &err);
  CHECK_ERROR(err);

  CHECK_ERROR(clSetKernelArg(kernel, 0, sizeof(output_buffer), &output_buffer));
  CHECK_ERROR(clSetKernelArg(kernel, 1, sizeof(input_buffer), &input_buffer));
  CHECK_ERROR(clSetKernelArg(kernel, 2, sizeof(enter_n), &enter_n));
  CHECK_ERROR(clSetKernelArg(kernel, 3, sizeof(logical_len), &logical_len));
  CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_size,
                                     &work_size, 0, NULL, NULL));
  CHECK_ERROR(clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
                                  sizeof(output), output, 0, NULL, NULL));

  static const char *const names[] = {"reduce_add", "scan_inclusive_add",
                                      "scan_exclusive_add", "broadcast",
                                      "any"};
  int failed = 0;
  for (size_t op = 0; op < 5; ++op) {
    for (size_t i = 0; i < WORK_SIZE; ++i) {
      size_t idx = op * WORK_SIZE + i;
      if (output[idx] != expected[idx]) {
        fprintf(stderr, "sub_group_%s: output[%zu] = %d, expected %d\n",
                names[op], i, output[idx], expected[idx]);
        failed = 1;
      }
    }
  }
  if (failed)
    goto error;
  puts("OK");
  result = 0;

error:
  free(platforms);
  if (input_buffer != NULL)
    clReleaseMemObject(input_buffer);
  if (output_buffer != NULL)
    clReleaseMemObject(output_buffer);
  if (kernel != NULL)
    clReleaseKernel(kernel);
  if (program != NULL)
    clReleaseProgram(program);
  if (queue != NULL)
    clReleaseCommandQueue(queue);
  if (context != NULL)
    clReleaseContext(context);
  return result;
}
