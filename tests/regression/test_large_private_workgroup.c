/* Regression test for kernels whose private data doesn't fit the stack of a
   default-sized worker thread.

   The work-group function keeps a dynamically indexed private array in a
   context array replicated over all work-items, on the worker thread's stack.
   With 2 MiB of private data per work-group, workers created with the
   platform's default stack size (512 KiB on macOS) crashed instead of running
   the kernel.

   Copyright (c) 2026 Tim Besard

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

#define CHECK_ERROR(err)                                                      \
  if (err != CL_SUCCESS)                                                      \
    {                                                                         \
      printf ("OpenCL error %d at %s:%d\n", err, __FILE__, __LINE__);         \
      return EXIT_FAILURE;                                                    \
    }

#define PRIVATE_BYTES_PER_WORK_GROUP (2 * 1024 * 1024)

/* The data-dependent index keeps buf in memory. buf[i] = gid + i, so
   (gid + i) % N visits every index once, and
   out[gid] = sum_i buf[i] = N * gid + N * (N - 1) / 2. */
static const char *kernel_source
  = "kernel void big_private (global long *out)\n"
    "{\n"
    "  size_t gid = get_global_id (0);\n"
    "  long buf[N];\n"
    "  for (int i = 0; i < N; i++)\n"
    "    buf[i] = (long)gid + i;\n"
    "  long acc = 0;\n"
    "  for (int i = 0; i < N; i++)\n"
    "    acc += buf[buf[i] % N];\n"
    "  out[gid] = acc;\n"
    "}\n";

int
main (void)
{
  cl_context context;
  cl_device_id device;
  cl_command_queue queue;
  cl_platform_id platform;

  cl_int err = poclu_get_any_device2 (&context, &device, &queue, &platform);
  CHECK_ERROR (err);

  cl_device_type type;
  err = clGetDeviceInfo (device, CL_DEVICE_TYPE, sizeof (type), &type, NULL);
  CHECK_ERROR (err);
  if (!(type & CL_DEVICE_TYPE_CPU))
    {
      printf ("SKIP: the test targets CPU devices\n");
      return 77;
    }

  size_t max_wg_size;
  err = clGetDeviceInfo (device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                         sizeof (max_wg_size), &max_wg_size, NULL);
  CHECK_ERROR (err);

  /* Size the array so the largest work-group needs the target footprint. */
  long n = PRIVATE_BYTES_PER_WORK_GROUP / (sizeof (cl_long) * max_wg_size);
  if (n < 16)
    n = 16;
  char build_options[64];
  snprintf (build_options, sizeof (build_options), "-DN=%ld", n);

  cl_program program
    = clCreateProgramWithSource (context, 1, &kernel_source, NULL, &err);
  CHECK_ERROR (err);
  err = clBuildProgram (program, 1, &device, build_options, NULL, NULL);
  CHECK_ERROR (err);
  cl_kernel kernel = clCreateKernel (program, "big_private", &err);
  CHECK_ERROR (err);

  /* Builds with HOST_CPU_ENABLE_STACK_SIZE_CHECK may lower the limit to what
     the stack holds; launch at whatever the kernel supports. */
  size_t local_size;
  err = clGetKernelWorkGroupInfo (kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                  sizeof (local_size), &local_size, NULL);
  CHECK_ERROR (err);
  size_t global_size = local_size;

  cl_mem out = clCreateBuffer (context, CL_MEM_WRITE_ONLY,
                               global_size * sizeof (cl_long), NULL, &err);
  CHECK_ERROR (err);
  err = clSetKernelArg (kernel, 0, sizeof (cl_mem), &out);
  CHECK_ERROR (err);
  err = clEnqueueNDRangeKernel (queue, kernel, 1, NULL, &global_size,
                                &local_size, 0, NULL, NULL);
  CHECK_ERROR (err);

  cl_long *result = malloc (global_size * sizeof (cl_long));
  err = clEnqueueReadBuffer (queue, out, CL_TRUE, 0,
                             global_size * sizeof (cl_long), result, 0, NULL,
                             NULL);
  CHECK_ERROR (err);

  int failures = 0;
  for (size_t gid = 0; gid < global_size; gid++)
    {
      cl_long expected = n * (cl_long)gid + n * (n - 1) / 2;
      if (result[gid] != expected && failures++ < 10)
        printf ("out[%zu] = %ld, expected %ld\n", gid, (long)result[gid],
                (long)expected);
    }

  free (result);
  CHECK_ERROR (clReleaseMemObject (out));
  CHECK_ERROR (clReleaseKernel (kernel));
  CHECK_ERROR (clReleaseProgram (program));
  CHECK_ERROR (clReleaseCommandQueue (queue));
  CHECK_ERROR (clReleaseContext (context));
  CHECK_ERROR (clUnloadPlatformCompiler (platform));

  if (failures)
    {
      printf ("FAIL: %d mismatches\n", failures);
      return EXIT_FAILURE;
    }
  printf ("OK\n");
  return EXIT_SUCCESS;
}
