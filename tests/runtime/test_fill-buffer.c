/* Test clEnqueueFillBuffer

   Copyright (C) 202o Isuru Fernando <isuruf@gmail.com>

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

#include "pocl_opencl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PLATFORMS 32
#define MAX_DEVICES 32

int
main (void)
{
  cl_int err;
  cl_platform_id platforms[MAX_PLATFORMS];
  cl_uint nplatforms;
  cl_device_id devices[MAX_DEVICES];
  cl_uint ndevices;
  cl_uint i, j;
  cl_context context;
  cl_command_queue queue;
  cl_mem buf1;
  cl_int max_pattern_size = 4;
  cl_int pattern = (1 << 8) + (2 << 16) + (3 << 24);

  CHECK_CL_ERROR (clGetPlatformIDs (MAX_PLATFORMS, platforms, &nplatforms));

  for (i = 0; i < nplatforms; i++)
    {
      CHECK_CL_ERROR (clGetDeviceIDs (platforms[i], CL_DEVICE_TYPE_ALL,
                                      MAX_DEVICES, devices, &ndevices));

      /* Only test the devices we actually have room for */
      if (ndevices > MAX_DEVICES)
        ndevices = MAX_DEVICES;

      for (j = 0; j < ndevices; j++)
        {
          context = clCreateContext (NULL, 1, &devices[j], NULL, NULL, &err);
          CHECK_OPENCL_ERROR_IN ("clCreateContext");

          cl_ulong alloc;
#define MAXALLOC (1024U * 1024U)

          CHECK_CL_ERROR (clGetDeviceInfo (devices[j],
                                           CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                                           sizeof (alloc), &alloc, NULL));

          while (alloc > MAXALLOC)
            alloc /= 2;

          /* ensure we are allocating a even number of elements */
          const size_t nels
              = (alloc / sizeof (cl_char) / (1 << max_pattern_size))
                * (1 << max_pattern_size);
          const size_t buf_size = nels * sizeof (cl_char);

          cl_char *host_buf1 = malloc (buf_size);
          TEST_ASSERT (host_buf1);

          buf1 = clCreateBuffer (context, CL_MEM_READ_WRITE, buf_size, NULL,
                                 &err);
          CHECK_OPENCL_ERROR_IN ("clCreateBuffer");

          queue = clCreateCommandQueue (context, devices[j], 0, &err);
          CHECK_OPENCL_ERROR_IN ("clCreateCommandQueue");

          for (int pattern_size = 1; pattern_size <= max_pattern_size;
               pattern_size *= 2)
            {

              memset (host_buf1, 1, buf_size);

              CHECK_CL_ERROR (clEnqueueWriteBuffer (queue, buf1, CL_TRUE, 0,
                                                    buf_size, host_buf1, 0,
                                                    NULL, NULL));

              CHECK_CL_ERROR (clEnqueueFillBuffer (
                  queue, buf1, (void *)&pattern, pattern_size,
                  pattern_size * 2, buf_size - pattern_size * 3, 0, NULL,
                  NULL));

              CHECK_CL_ERROR (clEnqueueReadBuffer (queue, buf1, CL_TRUE, 0,
                                                   buf_size, host_buf1, 0,
                                                   NULL, NULL));

              CHECK_CL_ERROR (clFinish (queue));
              for (int i = 0; i < pattern_size * 2; i++)
                {
                  if (host_buf1[i] != 1)
                    {
                      printf ("Expected value at %d: 1, actual value: %d\n", i,
                              host_buf1[i]);
                      return EXIT_FAILURE;
                    }
                }
              for (int i = buf_size - pattern_size; i < buf_size; i++)
                {
                  if (host_buf1[i] != 1)
                    {
                      printf ("Expected value at %d: 1, actual value: %d\n", i,
                              host_buf1[i]);
                      return EXIT_FAILURE;
                    }
                }
              for (int i = pattern_size * 2; i < buf_size - pattern_size;
                   i += pattern_size)
                {
                  for (int j = 0; j < pattern_size; j++)
                    {
                      cl_char expected_value = *((char *)&(pattern) + j);
                      if (host_buf1[i + j] != expected_value)
                        {
                          printf (
                              "Expected value at %d: %d, actual value: %d\n",
                              i + j, expected_value, host_buf1[i + j]);
                          return EXIT_FAILURE;
                        }
                    }
                }
            }

          memset (host_buf1, 3, buf_size);

          free (host_buf1);
          CHECK_CL_ERROR (clReleaseMemObject (buf1));
          CHECK_CL_ERROR (clReleaseCommandQueue (queue));
          CHECK_CL_ERROR (clReleaseContext (context));
        }
    }

  CHECK_CL_ERROR (clUnloadCompiler ());

  return EXIT_SUCCESS;
}
