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
  Flattening sub_group_shuffle exposes a source barrier after the return check.
  Both before and after the fix, ImplicitConditionalBarriers inserts the two
  barriers marked below. The block after the check dominates the source barrier,
  so the source barrier's all-or-none execution guarantee makes both insertions
  legal. VariableUniformityAnalysis cannot prove this because the bounds check
  depends on the work-item id. At launch, logical_len equals the work-group size,
  so every work-item takes the normal path.
*/

#include "pocl_opencl.h"

#include <stdio.h>
#include <stdlib.h>

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
    "                          uint logical_len) {\n"
    "  uint i = get_sub_group_local_id();\n"
    "  /* implicit_barrier(); */\n"
    "  if (i >= logical_len) return;\n"
    "  /* implicit_barrier(); */\n"
    "  uint j = get_sub_group_size() - i - 1;\n"
    "  out[i] = sub_group_shuffle(input[i], j);\n"
    "}\n";

int main(void) {
  const cl_int input[] = {0, 1, 2, 3, 4, 5, 6, 7};
  cl_int output[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  const cl_uint logical_len = 8;
  const size_t work_size = 8;
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
  CHECK_ERROR(clSetKernelArg(kernel, 2, sizeof(logical_len), &logical_len));
  CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_size,
                                     &work_size, 0, NULL, NULL));
  CHECK_ERROR(clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
                                  sizeof(output), output, 0, NULL, NULL));

  for (size_t i = 0; i < work_size; ++i) {
    if (output[i] != input[work_size - i - 1]) {
      fprintf(stderr, "output[%zu] = %d, expected %d\n", i, output[i],
              input[work_size - i - 1]);
      goto error;
    }
  }
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
