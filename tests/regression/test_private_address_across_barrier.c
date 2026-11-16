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
  A private variable whose only uses after a barrier go through a copy of its
  address, which the CBS analysis of the variable's uses must follow or treat
  as an escape: a pointer phi carried across the barrier in a context array,
  an address turned into an integer, an address stored to local memory, and
  one read back by a call. Each work-item must still get its own copy of the
  variable.
*/

#include "pocl_opencl.h"

#include <stdio.h>
#include <stdlib.h>

#define WG_SIZE 16

#define CHECK_ERROR(err)                                                       \
  do {                                                                         \
    if ((err) != CL_SUCCESS) {                                                 \
      fprintf(stderr, "OpenCL error %d at %s:%d\n", (err), __FILE__,           \
              __LINE__);                                                       \
      goto error;                                                              \
    }                                                                          \
  } while (0)

static const char KernelSource[]
    = "__kernel void phi(__global int *out, __global const int *in,\n"
      "                  int n) {\n"
      "  int lid = get_local_id(0);\n"
      "  int tmp;\n"
      "  int *p = 0;\n"
      "  for (int it = 0; it < n; it++) {\n"
      "    if (it == 0) {\n"
      "      tmp = in[lid];\n"
      "      p = &tmp;\n"
      "    } else {\n"
      "      p = 0;\n"
      "    }\n"
      "    barrier(CLK_LOCAL_MEM_FENCE);\n"
      "    if (p)\n"
      "      out[lid] = *p;\n"
      "    barrier(CLK_LOCAL_MEM_FENCE);\n"
      "  }\n"
      "}\n"
      "__kernel void ptrtoint(__global int *out, __global const int *in,\n"
      "                       int n) {\n"
      "  int lid = get_local_id(0);\n"
      "  volatile int tmp = in[lid];\n"
      "  barrier(CLK_LOCAL_MEM_FENCE);\n"
      "  ulong addr = (ulong)&tmp + (ulong)(n - 2);\n"
      "  out[lid] = *(__private volatile int *)addr;\n"
      "}\n"
      "__kernel void local_slot(__global int *out, __global const int *in,\n"
      "                         int n) {\n"
      "  __private volatile int *__local slots[16];\n"
      "  int lid = get_local_id(0);\n"
      "  volatile int tmp = in[lid];\n"
      "  slots[lid] = &tmp;\n"
      "  barrier(CLK_LOCAL_MEM_FENCE);\n"
      "  out[lid] = *slots[lid];\n"
      "}\n"
      "__attribute__((noinline)) __private volatile int *\n"
      "read_slot(__private volatile int *__private *slot) {\n"
      "  return *slot;\n"
      "}\n"
      "__kernel void call_slot(__global int *out, __global const int *in,\n"
      "                        int n) {\n"
      "  int lid = get_local_id(0);\n"
      "  volatile int tmp = in[lid];\n"
      "  __private volatile int *slot = &tmp;\n"
      "  __private volatile int *p = read_slot(&slot);\n"
      "  barrier(CLK_LOCAL_MEM_FENCE);\n"
      "  out[lid] = *p;\n"
      "}\n";

static const char *const KernelNames[]
    = { "phi", "ptrtoint", "local_slot", "call_slot" };

int main(void) {
  cl_int input[WG_SIZE], output[WG_SIZE];
  const size_t work_size = WG_SIZE;
  /* the kernels take n = 2: two iterations for phi, a zero offset otherwise */
  const cl_int n = 2;
  cl_int err;
  cl_uint num_platforms;
  cl_platform_id *platforms = NULL;
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem in_buffer = NULL;
  cl_mem out_buffer = NULL;
  int mismatch = 0;
  int result = 1;

  for (int i = 0; i < WG_SIZE; ++i)
    input[i] = 100 + i;

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
  in_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             sizeof(input), input, &err);
  CHECK_ERROR(err);
  out_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(output),
                              NULL, &err);
  CHECK_ERROR(err);

  for (size_t k = 0; k < sizeof(KernelNames) / sizeof(KernelNames[0]); ++k) {
    kernel = clCreateKernel(program, KernelNames[k], &err);
    CHECK_ERROR(err);
    CHECK_ERROR(clSetKernelArg(kernel, 0, sizeof(out_buffer), &out_buffer));
    CHECK_ERROR(clSetKernelArg(kernel, 1, sizeof(in_buffer), &in_buffer));
    CHECK_ERROR(clSetKernelArg(kernel, 2, sizeof(n), &n));
    const cl_int poison = -1;
    CHECK_ERROR(clEnqueueFillBuffer(queue, out_buffer, &poison, sizeof(poison),
                                    0, sizeof(output), 0, NULL, NULL));
    CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_size,
                                       &work_size, 0, NULL, NULL));
    CHECK_ERROR(clEnqueueReadBuffer(queue, out_buffer, CL_TRUE, 0,
                                    sizeof(output), output, 0, NULL, NULL));
    CHECK_ERROR(clReleaseKernel(kernel));
    kernel = NULL;

    for (int i = 0; i < WG_SIZE; ++i) {
      if (output[i] != input[i]) {
        fprintf(stderr, "%s: out[%d] = %d, expected %d\n", KernelNames[k], i,
                (int)output[i], (int)input[i]);
        mismatch = 1;
        break;
      }
    }
  }
  if (mismatch)
    goto error;
  puts("OK");
  result = 0;

error:
  free(platforms);
  if (out_buffer != NULL)
    clReleaseMemObject(out_buffer);
  if (in_buffer != NULL)
    clReleaseMemObject(in_buffer);
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
