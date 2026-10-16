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
  Kernels in which a guarded section between barriers has an early exit that
  leaves the kernel without reaching the remaining barriers, such as the
  bounds-check failure path of a Julia kernel. Padding work-items skip the
  guarded section:

      barrier();
      br i1 %pad, label %next, label %guarded
      guarded:
        br i1 %oob, label %exit, label %body
      body:
        ...
        br label %next
      next:
      barrier();

  The successors of the first barrier reconverge at %next, but the early exit
  makes the region between the barriers multi-exit, so IsolateRegions does not
  give it a single entry. Region formation then picked an arbitrary barrier
  successor as the entry of the work-item loop: either the padding work-items
  jumped into the loop past its per-work-item control and executed the guarded
  code (first kernel), or a bogus region was formed whose exit had two
  successors and WorkitemLoops aborted with an assertion (second kernel).

  The early exit is never taken at launch, so all work-items reach every
  barrier.
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
    "__kernel void skip_padding(__global int *out, int n, int len) {\n"
    "  int i = get_global_id(0);\n"
    "  bool pad = i >= n;\n"
    "  if (!pad) {\n"
    "    if (i >= len) goto error;\n"
    "    out[i] += 1;\n"
    "  }\n"
    "  barrier(CLK_GLOBAL_MEM_FENCE);\n"
    "  if (!pad) {\n"
    "    if (i >= len) goto error;\n"
    "    out[i] += 2;\n"
    "  }\n"
    "  barrier(CLK_GLOBAL_MEM_FENCE);\n"
    "  return;\n"
    "error:\n"
    "  out[0] = -1;\n"
    "}\n"
    "\n"
    "__kernel void skip_padding_to_exit(__global int *out, int n, int len) {\n"
    "  int i = get_global_id(0) + 1;\n"
    "  bool pad = i > n;\n"
    "  int idx = pad ? 0 : i;\n"
    "  if (pad) goto sync1;\n"
    "  if (i > len) goto error;\n"
    "  out[idx - 1] += 1;\n"
    "sync1:\n"
    "  barrier(CLK_GLOBAL_MEM_FENCE);\n"
    "  if (pad) goto sync2;\n"
    "  if (i > len) goto error;\n"
    "  out[idx - 1] += 2;\n"
    "sync2:\n"
    "  barrier(CLK_GLOBAL_MEM_FENCE);\n"
    "  goto sync3;\n"
    "error:\n"
    "  out[0] = -1;\n"
    "  goto done;\n"
    "sync3:\n"
    "  barrier(CLK_GLOBAL_MEM_FENCE);\n"
    "done:\n"
    "  return;\n"
    "}\n";

static const char *KernelNames[] = {"skip_padding", "skip_padding_to_exit"};

/* Two work-groups of 8, of which the last 3 work-items are padding. */
#define GLOBAL_SIZE 16
#define LOCAL_SIZE 8
#define LOGICAL_SIZE 13

int main(void) {
  const size_t global_size = GLOBAL_SIZE, local_size = LOCAL_SIZE;
  const cl_int n = LOGICAL_SIZE, len = GLOBAL_SIZE;
  cl_int output[GLOBAL_SIZE];
  cl_int err;
  cl_uint num_platforms;
  cl_platform_id *platforms = NULL;
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem output_buffer = NULL;
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
      fprintf(stderr, "Build error:\n%s\n", log);
      free(log);
    }
    goto error;
  }
  output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(output),
                                 NULL, &err);
  CHECK_ERROR(err);

  for (size_t k = 0; k < sizeof(KernelNames) / sizeof(*KernelNames); ++k) {
    kernel = clCreateKernel(program, KernelNames[k], &err);
    CHECK_ERROR(err);

    const cl_int zero = 0;
    CHECK_ERROR(clEnqueueFillBuffer(queue, output_buffer, &zero, sizeof(zero),
                                    0, sizeof(output), 0, NULL, NULL));
    CHECK_ERROR(
        clSetKernelArg(kernel, 0, sizeof(output_buffer), &output_buffer));
    CHECK_ERROR(clSetKernelArg(kernel, 1, sizeof(n), &n));
    CHECK_ERROR(clSetKernelArg(kernel, 2, sizeof(len), &len));
    CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                                       &local_size, 0, NULL, NULL));
    CHECK_ERROR(clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
                                    sizeof(output), output, 0, NULL, NULL));

    for (size_t i = 0; i < GLOBAL_SIZE; ++i) {
      cl_int expected = i < LOGICAL_SIZE ? 3 : 0;
      if (output[i] != expected) {
        fprintf(stderr, "%s: output[%zu] = %d, expected %d\n", KernelNames[k],
                i, output[i], expected);
        goto error;
      }
    }
    clReleaseKernel(kernel);
    kernel = NULL;
  }
  puts("OK");
  result = 0;

error:
  free(platforms);
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
