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

#include "pocl_opencl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
  cl_context context;
  cl_command_queue queue;
  cl_mem buf1, buf2;

  CHECK_CL_ERROR(clGetPlatformIDs(MAX_PLATFORMS, platforms, &nplatforms));

  for (i = 0; i < nplatforms; i++)
  {
    CHECK_CL_ERROR(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, MAX_DEVICES,
      devices, &ndevices));

    /* Only test the devices we actually have room for */
    if (ndevices > MAX_DEVICES)
      ndevices = MAX_DEVICES;

    for (j = 0; j < ndevices; j++)
    {
      context = clCreateContext (NULL, 1, &devices[j], NULL, NULL, &err);
      CHECK_OPENCL_ERROR_IN("clCreateContext");
      queue = clCreateCommandQueue (context, devices[j], 0, &err);
      CHECK_OPENCL_ERROR_IN("clCreateCommandQueue");

      cl_ulong alloc;
#define MAXALLOC (1024U*1024U)

      CHECK_CL_ERROR(clGetDeviceInfo(devices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE,
          sizeof(alloc), &alloc, NULL));

      while (alloc > MAXALLOC)
        alloc /= 2;

      /* ensure we are allocating an even number of elements */
      const size_t nels = (alloc/sizeof(cl_int)/2)*2;
      const size_t buf_size = nels*sizeof(cl_int);

      cl_int *host_buf1 = malloc(buf_size);
      TEST_ASSERT(host_buf1);
      cl_int *host_buf2 = malloc(buf_size);
      TEST_ASSERT(host_buf2);

      memset (host_buf1, 1, buf_size);
      memset (host_buf2, 2, buf_size);

      buf1 = clCreateBuffer (context, CL_MEM_READ_WRITE, buf_size, NULL, &err);
      CHECK_OPENCL_ERROR_IN("clCreateBuffer");
      buf2 = clCreateBuffer (context, CL_MEM_READ_WRITE, buf_size, NULL, &err);
      CHECK_OPENCL_ERROR_IN("clCreateBuffer");

      CHECK_CL_ERROR(clEnqueueWriteBuffer(queue, buf1, CL_TRUE, 0, buf_size, host_buf1, 0, NULL, NULL));

      CHECK_CL_ERROR(clEnqueueCopyBuffer(queue, buf1, buf2, 0, 0, buf_size, 0, NULL, NULL));

      CHECK_CL_ERROR(clEnqueueReadBuffer(queue, buf2, CL_TRUE, 0, buf_size, host_buf2, 0, NULL, NULL));

      CHECK_CL_ERROR(clFinish(queue));

      TEST_ASSERT(memcmp(host_buf2, host_buf1, buf_size) == 0);

      memset (host_buf1, 3, buf_size);
      memset (host_buf2, 4, buf_size);

      { /* pretend the buffers are linearized buffers with 2 rows and nels/2 columns, and
           do a rectangular copy of the two rows */
        const size_t origin[] = {0, 0, 0};
        const size_t region[] = {sizeof(cl_int)*nels/2, 2, 1};
        cl_event evts[3] = {NULL, NULL, NULL};
        memset(host_buf2, 2, buf_size);
        CHECK_CL_ERROR(clEnqueueWriteBufferRect(queue, buf2, CL_TRUE, origin, origin, region,
            0, 0, 0, 0, /* natural pitches */
            host_buf2,
            0, NULL, evts));
        CHECK_CL_ERROR(clEnqueueCopyBufferRect(queue, buf2, buf1, origin, origin, region,
            0, 0, 0, 0, /* natural pitches */
            1, evts, evts + 1));
        CHECK_CL_ERROR(clEnqueueReadBufferRect(queue, buf1, CL_TRUE, origin, origin, region,
            0, 0, 0, 0, /* natural pitches */
            host_buf1,
            2, evts, evts + 2));
        CHECK_CL_ERROR(clFinish(queue));

        CHECK_CL_ERROR (clReleaseEvent (evts[2]));
        CHECK_CL_ERROR (clReleaseEvent (evts[1]));
        CHECK_CL_ERROR (clReleaseEvent (evts[0]));

        TEST_ASSERT(memcmp(host_buf2, host_buf1, buf_size) == 0);
      }

      free(host_buf2);
      free(host_buf1);
      CHECK_CL_ERROR (clReleaseMemObject (buf2));
      CHECK_CL_ERROR (clReleaseMemObject (buf1));
      CHECK_CL_ERROR (clReleaseCommandQueue (queue));
      CHECK_CL_ERROR (clReleaseContext (context));
    }
  }

  CHECK_CL_ERROR (clUnloadCompiler ());

  printf ("OK\n");
  return EXIT_SUCCESS;
}
