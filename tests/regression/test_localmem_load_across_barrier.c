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

/* WorkitemLoops must preserve values loaded before a barrier. */

#define WG_SIZE 64

#define CHECK_ERROR(err)                                                       \
  do {                                                                         \
    if ((err) != CL_SUCCESS) {                                                 \
      fprintf(stderr, "OpenCL error %d at %s:%d\n", (err), __FILE__,           \
              __LINE__);                                                       \
      goto error;                                                              \
    }                                                                          \
  } while (0)

static const char KernelSource[]
    = "__kernel void test_local(__global int *out) {\n"
      "  __local int scratch[64];\n"
      "  int lid = get_local_id(0);\n"
      "  scratch[lid] = lid;\n"
      "  barrier(CLK_LOCAL_MEM_FENCE);\n"
      "  int t = scratch[(lid + 1) % 64];\n"
      "  barrier(CLK_LOCAL_MEM_FENCE);\n"
      "  scratch[lid] = t;\n"
      "  barrier(CLK_LOCAL_MEM_FENCE);\n"
      "  out[lid] = scratch[lid];\n"
      "}\n"
      "__kernel void test_global(__global const int *in,\n"
      "                          __global int *out) {\n"
      "  int lid = get_local_id(0);\n"
      "  int t = in[(lid + 1) % 64];\n"
      "  barrier(CLK_GLOBAL_MEM_FENCE);\n"
      "  out[lid] = t;\n"
      "}\n";

int main(void) {
  cl_int output[WG_SIZE];
  const size_t work_size = WG_SIZE;
  cl_int err;
  cl_uint num_platforms;
  cl_platform_id *platforms = NULL;
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel local_kernel = NULL;
  cl_kernel global_kernel = NULL;
  cl_mem output_buffer = NULL;
  int result = 1;

  for (int i = 0; i < WG_SIZE; ++i)
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
      fprintf(stderr, "Build error:\n%s\n", log);
      free(log);
    }
    goto error;
  }
  local_kernel = clCreateKernel(program, "test_local", &err);
  CHECK_ERROR(err);
  global_kernel = clCreateKernel(program, "test_global", &err);
  CHECK_ERROR(err);
  output_buffer
      = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                       sizeof(output), output, &err);
  CHECK_ERROR(err);

  CHECK_ERROR(clSetKernelArg(local_kernel, 0, sizeof(output_buffer),
                             &output_buffer));
  CHECK_ERROR(clEnqueueNDRangeKernel(queue, local_kernel, 1, NULL, &work_size,
                                     &work_size, 0, NULL, NULL));
  CHECK_ERROR(clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
                                  sizeof(output), output, 0, NULL, NULL));

  for (int i = 0; i < WG_SIZE; ++i) {
    const cl_int expected = (i + 1) % WG_SIZE;
    if (output[i] != expected) {
      fprintf(stderr, "out[%d] = %d, expected %d\n", i, (int)output[i],
              (int)expected);
      goto error;
    }
  }

  for (int i = 0; i < WG_SIZE; ++i)
    output[i] = i;
  CHECK_ERROR(clEnqueueWriteBuffer(queue, output_buffer, CL_TRUE, 0,
                                   sizeof(output), output, 0, NULL, NULL));
  CHECK_ERROR(clSetKernelArg(global_kernel, 0, sizeof(output_buffer),
                             &output_buffer));
  CHECK_ERROR(clSetKernelArg(global_kernel, 1, sizeof(output_buffer),
                             &output_buffer));
  CHECK_ERROR(clEnqueueNDRangeKernel(queue, global_kernel, 1, NULL, &work_size,
                                     &work_size, 0, NULL, NULL));
  CHECK_ERROR(clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
                                  sizeof(output), output, 0, NULL, NULL));

  for (int i = 0; i < WG_SIZE; ++i) {
    const cl_int expected = (i + 1) % WG_SIZE;
    if (output[i] != expected) {
      fprintf(stderr, "aliased out[%d] = %d, expected %d\n", i,
              (int)output[i], (int)expected);
      goto error;
    }
  }
  puts("OK");
  result = 0;

error:
  free(platforms);
  if (output_buffer != NULL)
    clReleaseMemObject(output_buffer);
  if (global_kernel != NULL)
    clReleaseKernel(global_kernel);
  if (local_kernel != NULL)
    clReleaseKernel(local_kernel);
  if (program != NULL)
    clReleaseProgram(program);
  if (queue != NULL)
    clReleaseCommandQueue(queue);
  if (context != NULL)
    clReleaseContext(context);
  return result;
}
