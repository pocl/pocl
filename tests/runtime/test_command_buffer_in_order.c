/* Commands recorded with an in-order queue stay ordered when the command
   buffer is enqueued on an out-of-order queue.

   Copyright (c) 2026 PoCL developers

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "command_buffer_common.h"
#include "poclu.h"

#define STR(x) #x

int main(int _argc, char **_argv) {
#if defined(cl_khr_command_buffer) && cl_khr_command_buffer == 1
  struct cmdbuf_ext ext;

  cl_platform_id platform;
  CHECK_CL_ERROR(clGetPlatformIDs(1, &platform, NULL));
  cl_device_id device;
  CHECK_CL_ERROR(
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL));

  int skip = cmdbuf_load_ext(platform, &ext);
  if (skip != 0)
    return skip;

  cl_command_queue_properties supported = 0;
  CHECK_CL_ERROR(clGetDeviceInfo(
      device, CL_DEVICE_COMMAND_BUFFER_SUPPORTED_QUEUE_PROPERTIES_KHR,
      sizeof(supported), &supported, NULL));
  if (!(supported & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)) {
    printf("Out-of-order command buffer queues not supported, skipping\n");
    return 77;
  }

  cl_int error;
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
  CHECK_CL_ERROR(error);

  /* Slow enough that an unordered fill recorded after it finishes first. */
  const char *code =
      STR(kernel void slow_write_one(global int *out, int iters) {
        size_t i = get_global_id(0);
        uint acc = (uint)i;
        for (int k = 0; k < iters; ++k)
          acc = acc * 1103515245u + 12345u;
        out[i] = (acc == 42u) ? 3 : 1;
      });
  const size_t length = strlen(code);

  cl_program program =
      clCreateProgramWithSource(context, 1, &code, &length, &error);
  CHECK_CL_ERROR(error);
  CHECK_CL_ERROR(clBuildProgram(program, 1, &device, NULL, NULL, NULL));
  cl_kernel kernel = clCreateKernel(program, "slow_write_one", &error);
  CHECK_CL_ERROR(error);

  const size_t elements = 64;
  const cl_int iters = 1 << 20;
  cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 elements * sizeof(cl_int), NULL, &error);
  CHECK_CL_ERROR(error);
  CHECK_CL_ERROR(clSetKernelArg(kernel, 0, sizeof(buffer), &buffer));
  CHECK_CL_ERROR(clSetKernelArg(kernel, 1, sizeof(iters), &iters));

  cl_command_queue in_order_queue =
      clCreateCommandQueue(context, device, 0, &error);
  CHECK_CL_ERROR(error);
  cl_command_queue out_of_order_queue = clCreateCommandQueue(
      context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &error);
  CHECK_CL_ERROR(error);

  cl_command_buffer_khr command_buffer =
      ext.clCreateCommandBufferKHR(1, &in_order_queue, NULL, &error);
  CHECK_CL_ERROR(error);

  /* No sync points: only the recording queue orders the two commands. */
  CHECK_CL_ERROR(ext.clCommandNDRangeKernelKHR(command_buffer, NULL, NULL,
                                               kernel, 1, NULL, &elements, NULL,
                                               0, NULL, NULL, NULL));
  cl_int two = 2;
  CHECK_CL_ERROR(ext.clCommandFillBufferKHR(
      command_buffer, NULL, NULL, buffer, &two, sizeof(two), 0,
      elements * sizeof(cl_int), 0, NULL, NULL, NULL));
  CHECK_CL_ERROR(ext.clFinalizeCommandBufferKHR(command_buffer));

  CHECK_CL_ERROR(ext.clEnqueueCommandBufferKHR(1, &out_of_order_queue,
                                               command_buffer, 0, NULL, NULL));
  CHECK_CL_ERROR(clFinish(out_of_order_queue));

  cl_int result[64];
  CHECK_CL_ERROR(clEnqueueReadBuffer(in_order_queue, buffer, CL_TRUE, 0,
                                     sizeof(result), result, 0, NULL, NULL));
  for (size_t i = 0; i < elements; ++i)
    TEST_ASSERT(result[i] == 2);

  CHECK_CL_ERROR(ext.clReleaseCommandBufferKHR(command_buffer));
  CHECK_CL_ERROR(clReleaseCommandQueue(out_of_order_queue));
  CHECK_CL_ERROR(clReleaseCommandQueue(in_order_queue));
  CHECK_CL_ERROR(clReleaseMemObject(buffer));
  CHECK_CL_ERROR(clReleaseKernel(kernel));
  CHECK_CL_ERROR(clReleaseProgram(program));
  CHECK_CL_ERROR(clReleaseContext(context));

  CHECK_CL_ERROR(clUnloadPlatformCompiler(platform));

  printf("OK\n");
  return EXIT_SUCCESS;
#else
  return 77;
#endif
}
