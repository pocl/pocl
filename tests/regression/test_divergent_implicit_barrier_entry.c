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

#include "pocl_opencl.h"

#include <stdio.h>
#include <stdlib.h>

#define CHECK_ERROR(err)                                                       \
  do {                                                                         \
    if ((err) != CL_SUCCESS) {                                                 \
      fprintf(stderr, "OpenCL error %d at %s:%d\\n", (err), __FILE__,          \
              __LINE__);                                                       \
      goto error;                                                              \
    }                                                                          \
  } while (0)

static const char KernelSource[] =
    "__kernel void test_kernel(__global int *out,\n"
    "                          __global const int *input,\n"
    "                          uint active_n, uint logical_len) {\n"
    "  __local int temp;\n"
    "  size_t i = get_local_id(0);\n"
    "  if (i == 0) temp = 0;\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (i < active_n) {\n"
    "    if (i >= logical_len) return;\n"
    "    if (input[i] < 0) temp = 1;\n"
    "  }\n"
    "  barrier(CLK_LOCAL_MEM_FENCE);\n"
    "  if (i == 0) out[0] = temp;\n"
    "}\n";

int main(void) {
  const cl_int input[] = {0, 0, -1, 0, 0, 0, 0, 0};
  const cl_uint active_n = 2;
  const cl_uint logical_len = 2;
  cl_int output = -1;
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
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR(err);
  const cl_queue_properties properties[] = {0};
  queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
  CHECK_ERROR(err);

  const char *sources[] = {KernelSource};
  program = clCreateProgramWithSource(context, 1, sources, NULL, &err);
  CHECK_ERROR(err);
  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t log_size = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);
    char *log = malloc(log_size);
    if (log != NULL) {
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size,
                            log, NULL);
      fprintf(stderr, "Build error:\\n%s\\n", log);
      free(log);
    }
    goto error;
  }
  kernel = clCreateKernel(program, "test_kernel", &err);
  CHECK_ERROR(err);
  output_buffer =
      clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(output), &output, &err);
  CHECK_ERROR(err);
  input_buffer =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(input), (void *)input, &err);
  CHECK_ERROR(err);

  CHECK_ERROR(clSetKernelArg(kernel, 0, sizeof(output_buffer), &output_buffer));
  CHECK_ERROR(clSetKernelArg(kernel, 1, sizeof(input_buffer), &input_buffer));
  CHECK_ERROR(clSetKernelArg(kernel, 2, sizeof(active_n), &active_n));
  CHECK_ERROR(clSetKernelArg(kernel, 3, sizeof(logical_len), &logical_len));
  CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_size,
                                     &work_size, 0, NULL, NULL));
  CHECK_ERROR(clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
                                  sizeof(output), &output, 0, NULL, NULL));

  if (output != 0) {
    fprintf(stderr, "output = %d, expected 0\\n", output);
    goto error;
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
