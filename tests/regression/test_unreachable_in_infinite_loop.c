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
  An intentional infinite loop with observable side effects that also contains
  a path to an unreachable instruction (here: printf followed by
  __builtin_unreachable(), which clang does not fold into an assumption).

  For the work-item loop handlers, ConvertUnreachablesToReturns deletes the
  region that can only lead to the unreachable. The loop can reach the
  unreachable and cannot reach the kernel's return, so an overly eager
  deletion removed the whole loop, including its store, and the kernel
  returned immediately. Only the unreachable path may be removed; the loop
  itself must remain, so a launch that never takes the unreachable path must
  never complete.

  The check mirrors test_infinite_loop.cpp: give the kernel some time, verify
  it is still running, then replace the process to get rid of the spinning
  kernel thread.
*/

#include "pocl_opencl.h"

#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#define sleep_ms(MS) Sleep(MS)
#else
#include <unistd.h>
#define sleep_ms(MS) usleep((MS) * 1000)
#endif

#define CHECK_ERROR(err)                                                       \
  if (err != CL_SUCCESS) {                                                     \
    printf("OpenCL Error %d at %s:%d\n", err, __FILE__, __LINE__);             \
    return 1;                                                                  \
  }

static const char KernelSource[] =
    "kernel void test_kernel(global volatile int *out, int early_exit,\n"
    "                        global const int *abort_flags) {\n"
    "  if (early_exit)\n"
    "    return;\n"
    "  while (1) {\n"
    "    if (abort_flags[get_global_id(0)]) {\n"
    "      printf(\"aborting\\n\");\n"
    "      __builtin_unreachable();\n"
    "    }\n"
    "    out[0] = 1;\n"
    "  }\n"
    "}\n";

int main(int argc, char **argv) {
  cl_int err;

  cl_platform_id platform;
  CHECK_ERROR(clGetPlatformIDs(1, &platform, NULL));
  cl_device_id device;
  CHECK_ERROR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL));

  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR(err);
  cl_queue_properties props[] = {0};
  cl_command_queue queue =
      clCreateCommandQueueWithProperties(context, device, props, &err);
  CHECK_ERROR(err);

  const char *sources[] = {KernelSource};
  cl_program program =
      clCreateProgramWithSource(context, 1, sources, NULL, &err);
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
  cl_kernel kernel = clCreateKernel(program, "test_kernel", &err);
  CHECK_ERROR(err);

  const size_t global_size = 4;
  cl_int out = 0;
  cl_int abort_flags[4] = {0, 0, 0, 0};
  cl_mem out_buf =
      clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(out), &out, &err);
  CHECK_ERROR(err);
  cl_mem flags_buf =
      clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(abort_flags), abort_flags, &err);
  CHECK_ERROR(err);
  CHECK_ERROR(clSetKernelArg(kernel, 0, sizeof(out_buf), &out_buf));
  CHECK_ERROR(clSetKernelArg(kernel, 2, sizeof(flags_buf), &flags_buf));

  // The early exit must still work.
  cl_int early_exit = 1;
  CHECK_ERROR(clSetKernelArg(kernel, 1, sizeof(early_exit), &early_exit));
  CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                                     &global_size, 0, NULL, NULL));
  CHECK_ERROR(clFinish(queue));

  // Without the early exit, the kernel must spin forever.
  early_exit = 0;
  CHECK_ERROR(clSetKernelArg(kernel, 1, sizeof(early_exit), &early_exit));
  cl_event event;
  CHECK_ERROR(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                                     &global_size, 0, NULL, &event));
  CHECK_ERROR(clFlush(queue));

  sleep_ms(1500);

  cl_int status;
  CHECK_ERROR(clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                             sizeof(status), &status, NULL));
  if (status == CL_COMPLETE) {
    printf("FAIL: the infinite loop was compiled away\n");
    return 1;
  }

  printf("OK\n");
  fflush(stdout);
  // Force exit of the process regardless of the running kernel thread by
  // replacing the process with a dummy process.
#ifdef _WIN32
  ExitProcess(EXIT_SUCCESS);
#else
  execlp("true", "true", NULL);
#endif
  return 0;
}
