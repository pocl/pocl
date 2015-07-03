/* Test that recycling wait lists doesn't lead to lockups

   Copyright (C) 2015 Giuseppe Bilotta <giuseppe.bilotta@gmail.com>

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <CL/cl.h>

#include "pocl_tests.h"
#include "poclu.h"

#define MAX_PLATFORMS 32
#define MAX_DEVICES   32

void timeout(int signum)
{
  exit(EXIT_FAILURE);
}

int
main(void)
{
  cl_int err;
  cl_platform_id platforms[MAX_PLATFORMS];
  cl_uint nplatforms;
  cl_device_id devices[MAX_DEVICES];
  cl_uint ndevices;
  cl_uint i, j;

  /* set up a signal handler for ALRM that will kill
   * the program with EXIT_FAILURE on timeout
   */
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = timeout;
  sigaction(SIGALRM, &sa, NULL);

  err = clGetPlatformIDs(MAX_PLATFORMS, platforms, &nplatforms);
  CHECK_OPENCL_ERROR_IN("clGetPlatformIDs");

  for (i = 0; i < nplatforms; i++)
  {
    err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, MAX_DEVICES,
      devices, &ndevices);
    CHECK_OPENCL_ERROR_IN("clGetDeviceIDs");

    for (j = 0; j < ndevices; j++)
    {
      cl_context context = clCreateContext(NULL, 1, &devices[j], NULL, NULL, &err);
      CHECK_OPENCL_ERROR_IN("clCreateContext");
      cl_command_queue queue = clCreateCommandQueue(context, devices[j], 0, &err);
      CHECK_OPENCL_ERROR_IN("clCreateCommandQueue");

      cl_ulong alloc;
#define MAXALLOC (128*1024U*1024U)

      if (clGetDeviceInfo(devices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE,
          sizeof(alloc), &alloc, NULL) != CL_SUCCESS)
      CHECK_OPENCL_ERROR_IN("get max alloc");

      while (alloc > MAXALLOC)
        alloc /= 2;

      const size_t buf_size = alloc;

      cl_int *host_buf1 = malloc(buf_size);
      if (host_buf1 == NULL)
        return EXIT_FAILURE;
      cl_int *host_buf2 = malloc(buf_size);
      if (host_buf2 == NULL)
        return EXIT_FAILURE;

      memset(host_buf1, 1, buf_size);
      memset(host_buf2, 2, buf_size);

      cl_mem buf1 = clCreateBuffer(context, CL_MEM_READ_WRITE, buf_size, NULL, &err);
      CHECK_OPENCL_ERROR_IN("create buf1");
      cl_mem buf2 = clCreateBuffer(context, CL_MEM_READ_WRITE, buf_size, NULL, &err);
      CHECK_OPENCL_ERROR_IN("create buf2");

      cl_event buf1_event, bufcp_event, buf2_event;

      /* we test if recycling the wait list leads to neverending loops */
      cl_event wait_list[1];

      /* Note that this must be CL_TRUE because to trigger the bug the next
       * command must have a completed event in the waiting lists */
      err = clEnqueueWriteBuffer(queue, buf1, CL_TRUE, 0, buf_size, host_buf1,
	0, NULL, &buf1_event);
      CHECK_OPENCL_ERROR_IN("write buf1");

      *wait_list = buf1_event;

      err = clEnqueueCopyBuffer(queue, buf1, buf2, 0, 0, buf_size,
	1, wait_list, &bufcp_event);
      CHECK_OPENCL_ERROR_IN("copy buffers");

      *wait_list = bufcp_event;

      err = clEnqueueReadBuffer(queue, buf2, CL_FALSE, 0, buf_size, host_buf2,
	1, wait_list, &buf2_event);
      CHECK_OPENCL_ERROR_IN("read buf");

      /* timeout after 30 seconds: if we're not done by then, timeout() will be
       * invoked and terminate the program with an EXIT_FAILURE */
      alarm(30);

      err = clFinish(queue);
      CHECK_OPENCL_ERROR_IN("clFinish");

      if (memcmp(host_buf2, host_buf1, buf_size) != 0)
        return EXIT_FAILURE;

      free(host_buf2);
      free(host_buf1);
      clReleaseEvent(buf2_event);
      clReleaseEvent(bufcp_event);
      clReleaseEvent(buf1_event);
      clReleaseMemObject(buf2);
      clReleaseMemObject(buf1);
      clReleaseCommandQueue(queue);
    }
  }
  return EXIT_SUCCESS;
}
