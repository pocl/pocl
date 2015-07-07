/* Test clEnqueue{Write,Copy,Read}Buffer

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
#include <CL/cl.h>

#define MAX_PLATFORMS 32
#define MAX_DEVICES   32

int
main(void)
{
  cl_int err;
  cl_platform_id platforms[MAX_PLATFORMS];
  cl_uint nplatforms;
  cl_device_id devices[MAX_DEVICES];
  cl_uint ndevices;
  cl_uint i, j;

  err = clGetPlatformIDs(MAX_PLATFORMS, platforms, &nplatforms);
  if (err != CL_SUCCESS)
    return EXIT_FAILURE;

  for (i = 0; i < nplatforms; i++)
  {
    err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, MAX_DEVICES,
      devices, &ndevices);
    if (err != CL_SUCCESS)
      return EXIT_FAILURE;

    /* Only test the devices we actually have room for */
    if (ndevices > MAX_DEVICES)
      ndevices = MAX_DEVICES;

    for (j = 0; j < ndevices; j++)
    {
      cl_context context = clCreateContext(NULL, 1, &devices[j], NULL, NULL, &err);
      if (err != CL_SUCCESS)
        return EXIT_FAILURE;
      cl_command_queue queue = clCreateCommandQueue(context, devices[j], 0, &err);
      if (err != CL_SUCCESS)
        return EXIT_FAILURE;

      cl_ulong alloc;
#define MAXALLOC (1024U*1024U)

      if (clGetDeviceInfo(devices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE,
          sizeof(alloc), &alloc, NULL) != CL_SUCCESS)
        return EXIT_FAILURE;

      while (alloc > MAXALLOC)
        alloc /= 2;

      const size_t buf_size = (alloc/sizeof(cl_int))*sizeof(cl_int);

      cl_int *host_buf1 = malloc(buf_size);
      if (host_buf1 == NULL)
        return EXIT_FAILURE;
      cl_int *host_buf2 = malloc(buf_size);
      if (host_buf2 == NULL)
        return EXIT_FAILURE;

      memset(host_buf1, 1, buf_size);
      memset(host_buf2, 2, buf_size);

      cl_mem buf1 = clCreateBuffer(context, CL_MEM_READ_WRITE, buf_size, NULL, &err);
      if (err != CL_SUCCESS)
        return EXIT_FAILURE;
      cl_mem buf2 = clCreateBuffer(context, CL_MEM_READ_WRITE, buf_size, NULL, &err);
      if (err != CL_SUCCESS)
        return EXIT_FAILURE;

      if (clEnqueueWriteBuffer(queue, buf1, CL_TRUE, 0, buf_size, host_buf1,
	  0, NULL, NULL) != CL_SUCCESS)
        return EXIT_FAILURE;

      if (clEnqueueCopyBuffer(queue, buf1, buf2, 0, 0, buf_size,
	  0, NULL, NULL) != CL_SUCCESS)
        return EXIT_FAILURE;

      if (clEnqueueReadBuffer(queue, buf2, CL_TRUE, 0, buf_size, host_buf2,
	  0, NULL, NULL) != CL_SUCCESS)
        return EXIT_FAILURE;

      if (clFinish(queue) != CL_SUCCESS)
        return EXIT_FAILURE;

      if (memcmp(host_buf2, host_buf1, buf_size) != 0)
        return EXIT_FAILURE;

      free(host_buf2);
      free(host_buf1);
      clReleaseMemObject(buf2);
      clReleaseMemObject(buf1);
      clReleaseCommandQueue(queue);
    }
  }
  return EXIT_SUCCESS;
}
