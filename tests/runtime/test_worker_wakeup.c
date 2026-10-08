/* Test that the CPU driver doesn't lose worker threads while launching many
   small kernels: afterwards, a kernel with one work-group per compute unit
   must still get all of them running at the same time.

   A worker that went to sleep right after another one took the work could
   miss its wake-up signal and never be woken again, slowly shrinking the
   pool to a few threads.

   Copyright (c) 2026 Tim Besard

   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

#include <stdio.h>
#include <stdlib.h>

#include "poclu.h"

#define ROUNDS 20
#define LAUNCHES_PER_ROUND 2000

/* Every work-group checks in and then waits, for a bounded number of
   iterations, until all of them have. The work-groups can only all check
   in if as many workers run them concurrently. One that gives up releases
   the others, so that a failure doesn't make all of them wait in turn. */
static const char *SRC
  = "kernel void\n"
    "rendezvous (volatile global int *arrived, global int *ok, int n)\n"
    "{\n"
    "  atomic_inc (arrived);\n"
    "  int seen = 0;\n"
    "  for (int i = 0; i < (1 << 26) && seen < n; ++i)\n"
    "    seen = atomic_add (arrived, 0);\n"
    "  if (seen < n)\n"
    "    {\n"
    "      atomic_xchg (ok, 0);\n"
    "      atomic_add (arrived, n);\n"
    "    }\n"
    "}\n"
    "kernel void\n"
    "tiny (global int *x)\n"
    "{\n"
    "  x[get_global_id (0)] += 1;\n"
    "}\n";

int
main (int argc, char **argv)
{
  cl_context ctx;
  cl_device_id dev;
  cl_command_queue queue;
  cl_platform_id platform;
  cl_int err;

  CHECK_CL_ERROR (poclu_get_any_device2 (&ctx, &dev, &queue, &platform));

  cl_uint cus = 0;
  CHECK_CL_ERROR (clGetDeviceInfo (dev, CL_DEVICE_MAX_COMPUTE_UNITS,
                                   sizeof (cus), &cus, NULL));
  if (cus < 2)
    {
      printf ("SKIP: the device has a single compute unit\n");
      return 77;
    }

  cl_program program = clCreateProgramWithSource (ctx, 1, &SRC, NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateProgramWithSource");
  CHECK_CL_ERROR (clBuildProgram (program, 1, &dev, NULL, NULL, NULL));
  cl_kernel rendezvous = clCreateKernel (program, "rendezvous", &err);
  CHECK_OPENCL_ERROR_IN ("clCreateKernel");
  cl_kernel tiny = clCreateKernel (program, "tiny", &err);
  CHECK_OPENCL_ERROR_IN ("clCreateKernel");

  cl_mem arrived
    = clCreateBuffer (ctx, CL_MEM_READ_WRITE, sizeof (cl_int), NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateBuffer");
  cl_mem ok
    = clCreateBuffer (ctx, CL_MEM_READ_WRITE, sizeof (cl_int), NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateBuffer");
  cl_mem x = clCreateBuffer (ctx, CL_MEM_READ_WRITE, 4096 * sizeof (cl_int),
                             NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateBuffer");

  cl_int n = cus;
  CHECK_CL_ERROR (clSetKernelArg (rendezvous, 0, sizeof (cl_mem), &arrived));
  CHECK_CL_ERROR (clSetKernelArg (rendezvous, 1, sizeof (cl_mem), &ok));
  CHECK_CL_ERROR (clSetKernelArg (rendezvous, 2, sizeof (cl_int), &n));
  CHECK_CL_ERROR (clSetKernelArg (tiny, 0, sizeof (cl_mem), &x));

  size_t tiny_global = 1024, tiny_local = 64;
  size_t rdv_global = cus, rdv_local = 1;
  cl_int zero = 0, one = 1, result = 0;
  for (int round = 0; round < ROUNDS; ++round)
    {
      /* a mix of back-to-back and synchronized launches of small kernels,
         to have workers go to sleep while others are being woken up */
      for (int i = 0; i < LAUNCHES_PER_ROUND; ++i)
        {
          CHECK_CL_ERROR (clEnqueueNDRangeKernel (
            queue, tiny, 1, NULL, &tiny_global, &tiny_local, 0, NULL, NULL));
          if (i % 3 == 0)
            CHECK_CL_ERROR (clFinish (queue));
        }

      CHECK_CL_ERROR (clEnqueueWriteBuffer (
        queue, arrived, CL_FALSE, 0, sizeof (cl_int), &zero, 0, NULL, NULL));
      CHECK_CL_ERROR (clEnqueueWriteBuffer (
        queue, ok, CL_FALSE, 0, sizeof (cl_int), &one, 0, NULL, NULL));
      CHECK_CL_ERROR (clFinish (queue));
      CHECK_CL_ERROR (clEnqueueNDRangeKernel (
        queue, rendezvous, 1, NULL, &rdv_global, &rdv_local, 0, NULL, NULL));
      CHECK_CL_ERROR (clEnqueueReadBuffer (
        queue, ok, CL_TRUE, 0, sizeof (cl_int), &result, 0, NULL, NULL));
      if (!result)
        {
          printf ("FAIL: not all %u compute units ran concurrently after %d "
                  "launches\n",
                  cus, (round + 1) * LAUNCHES_PER_ROUND);
          return EXIT_FAILURE;
        }
    }

  CHECK_CL_ERROR (clReleaseMemObject (x));
  CHECK_CL_ERROR (clReleaseMemObject (ok));
  CHECK_CL_ERROR (clReleaseMemObject (arrived));
  CHECK_CL_ERROR (clReleaseKernel (tiny));
  CHECK_CL_ERROR (clReleaseKernel (rendezvous));
  CHECK_CL_ERROR (clReleaseProgram (program));
  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseContext (ctx));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (platform));

  printf ("OK\n");
  return EXIT_SUCCESS;
}
