/* Test "cl_pocl_content_size" PoCL extension

   Copyright (C) 2021-2023 Tampere University

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

#include "poclu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* must be sourced from PoCL */
#include "include/CL/cl_ext_pocl.h"

#define SRC_CHAR 1
#define DST_CHAR 2

#define CHECK_DEVICE(i)                                                       \
  do                                                                          \
    {                                                                         \
      cl_device_type devtype;                                                 \
      err = clGetDeviceInfo (devices[i], CL_DEVICE_TYPE, sizeof (devtype),    \
                             &devtype, NULL);                                 \
      CHECK_OPENCL_ERROR_IN ("clGetDeviceInfo");                              \
      if (devtype != CL_DEVICE_TYPE_CPU)                                      \
        {                                                                     \
          printf ("Device %d is not CPU -> skipping test\n", i);              \
          return 77;                                                          \
        }                                                                     \
    }                                                                         \
  while (0)

int
main (void)
{
  cl_int err;
  cl_platform_id platform;
  cl_uint num_devices;
  cl_device_id *devices;
  cl_context context;
  cl_command_queue *queues;
  cl_mem buf_content_src, buf_content_dst, buf_size;
  cl_int max_pattern_size = 4;
  char host_buf_src[1024];
  char host_buf_dst[1024];
  uint64_t content_size;

  poclu_get_multiple_devices (&platform, &context, CL_FALSE, &num_devices,
                              &devices, &queues, CL_FALSE);
  if (num_devices < 2)
    {
      printf ("Not enough devices (2 required), skipping");
      return 77;
    }

  CHECK_DEVICE (0);
  CHECK_DEVICE (1);

  void *setContentSizeBuffer_ptr = clGetExtensionFunctionAddressForPlatform (
      platform, "clSetContentSizeBufferPoCL");
  TEST_ASSERT ((setContentSizeBuffer_ptr != NULL));
  clSetContentSizeBufferPoCL_fn setContentSizeBuffer
      = (clSetContentSizeBufferPoCL_fn)setContentSizeBuffer_ptr;

  buf_content_src
      = clCreateBuffer (context, CL_MEM_READ_WRITE, 1024, NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateBuffer");
  buf_content_dst
      = clCreateBuffer (context, CL_MEM_READ_WRITE, 1024, NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateBuffer");
  buf_size = clCreateBuffer (context, CL_MEM_READ_WRITE, sizeof (content_size),
                             NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateBuffer");

  memset (host_buf_src, SRC_CHAR, 1024);
  memset (host_buf_dst, DST_CHAR, 1024);
  content_size = 128;
  CHECK_CL_ERROR (clEnqueueWriteBuffer (queues[0], buf_content_src, CL_TRUE, 0,
                                        1024, host_buf_src, 0, NULL, NULL));
  CHECK_CL_ERROR (clEnqueueWriteBuffer (queues[1], buf_content_dst, CL_TRUE, 0,
                                        1024, host_buf_dst, 0, NULL, NULL));
  CHECK_CL_ERROR (clEnqueueWriteBuffer (queues[0], buf_size, CL_TRUE, 0, 8,
                                        &content_size, 0, NULL, NULL));

  setContentSizeBuffer (buf_content_src, buf_size);

  // check copying behind the "content size" doesn't copy anything
  CHECK_CL_ERROR (clEnqueueCopyBuffer (queues[0], buf_content_src,
                                       buf_content_dst, 200, 200, 100, 0, NULL,
                                       NULL));
  CHECK_CL_ERROR (clFinish (queues[0]));

  CHECK_CL_ERROR (clEnqueueReadBuffer (queues[1], buf_content_dst, CL_TRUE, 0,
                                       1024, host_buf_dst, 0, NULL, NULL));
  size_t count = 0;
  for (size_t i = 0; i < 1024; ++i)
    if (host_buf_dst[i] == DST_CHAR)
      ++count;

  TEST_ASSERT ((count == 1024)
               && "copying outside content size boundary failed");

  // check copying the "content size" partially only copies up to content size
  CHECK_CL_ERROR (clEnqueueCopyBuffer (queues[1], buf_content_src,
                                       buf_content_dst, 100, 100, 100, 0, NULL,
                                       NULL));
  CHECK_CL_ERROR (clFinish (queues[1]));

  CHECK_CL_ERROR (clEnqueueReadBuffer (queues[0], buf_content_dst, CL_TRUE, 0,
                                       1024, host_buf_dst, 0, NULL, NULL));
  count = 0;
  for (size_t i = 0; i < 100; ++i)
    if (host_buf_dst[i] == DST_CHAR)
      ++count;
  for (size_t i = 100; i < 128; ++i)
    if (host_buf_dst[i] == SRC_CHAR)
      ++count;
  for (size_t i = 128; i < 1024; ++i)
    if (host_buf_dst[i] == DST_CHAR)
      ++count;

  TEST_ASSERT ((count == 1024) && "copying partially content size failed");

  CHECK_CL_ERROR (clReleaseMemObject (buf_content_dst));
  CHECK_CL_ERROR (clReleaseMemObject (buf_content_src));
  CHECK_CL_ERROR (clReleaseMemObject (buf_size));
  for (int i = 0; i < num_devices; ++i)
    CHECK_CL_ERROR (clReleaseCommandQueue (queues[i]));
  CHECK_CL_ERROR (clReleaseContext (context));
  CHECK_CL_ERROR (clUnloadCompiler ());
  free (queues);
  free (devices);

  printf ("OK\n");
  return EXIT_SUCCESS;
}
