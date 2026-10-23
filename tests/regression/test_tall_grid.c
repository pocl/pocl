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
  A grid that is only wide in its third dimension, or that is small but far
  away from the origin, must not get the work-group function specialized for
  small grids: that one tells the optimizer that the group counts and global
  offsets are below the specialization limit (65535 on CPUs), so the kernels
  below would see get_num_groups(2) and get_global_offset(0) <= 65535
  regardless of the launch.
*/

#include "pocl_opencl.h"

#include <stdio.h>
#include <stdlib.h>

#define CHECK_ERROR(err)                                                      \
  if (err != CL_SUCCESS)                                                      \
    {                                                                         \
      printf ("OpenCL Error %d at %s:%d\n", err, __FILE__, __LINE__);         \
      return EXIT_FAILURE;                                                    \
    }

static const char *source
  = "kernel void tall (global uint *out) {\n"
    "  if (get_global_id (2) == get_global_size (2) - 1)\n"
    "    out[0] = get_num_groups (2) > 65535 ? get_num_groups (2) : 0;\n"
    "}\n"
    "kernel void offset (global uint *out) {\n"
    "  out[0] = get_global_offset (0) > 65535 ? get_global_offset (0) : 0;\n"
    "}\n";

static int
run (cl_context context,
     cl_command_queue queue,
     cl_program program,
     const char *name,
     const size_t *offset,
     const size_t *global,
     cl_uint expected)
{
  cl_int err;
  cl_kernel kernel = clCreateKernel (program, name, &err);
  CHECK_ERROR (err);

  cl_uint result = 0;
  cl_mem out
    = clCreateBuffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                      sizeof (result), &result, &err);
  CHECK_ERROR (err);
  CHECK_ERROR (clSetKernelArg (kernel, 0, sizeof (cl_mem), &out));

  const size_t local[3] = { 1, 1, 1 };
  CHECK_ERROR (clEnqueueNDRangeKernel (queue, kernel, 3, offset, global, local,
                                       0, NULL, NULL));
  CHECK_ERROR (clEnqueueReadBuffer (queue, out, CL_TRUE, 0, sizeof (result),
                                    &result, 0, NULL, NULL));

  CHECK_ERROR (clReleaseMemObject (out));
  CHECK_ERROR (clReleaseKernel (kernel));

  if (result != expected)
    {
      printf ("FAIL: kernel %s returned %u, expected %u\n", name, result,
              expected);
      return EXIT_FAILURE;
    }
  return EXIT_SUCCESS;
}

int
main (void)
{
  cl_int err;
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;

  CHECK_ERROR (poclu_get_any_device2 (&context, &device, &queue, &platform));

  cl_program program
    = clCreateProgramWithSource (context, 1, &source, NULL, &err);
  CHECK_ERROR (err);
  CHECK_ERROR (clBuildProgram (program, 1, &device, NULL, NULL, NULL));
  const size_t tall_global[3] = { 1, 1, 70000 };
  int failed = run (context, queue, program, "tall", NULL, tall_global, 70000);

  const size_t offset[3] = { 100000, 0, 0 };
  const size_t small_global[3] = { 1, 1, 1 };
  failed
    |= run (context, queue, program, "offset", offset, small_global, 100000);

  CHECK_ERROR (clReleaseProgram (program));
  CHECK_ERROR (clReleaseCommandQueue (queue));
  CHECK_ERROR (clReleaseContext (context));
  CHECK_ERROR (clUnloadPlatformCompiler (platform));

  if (failed)
    return EXIT_FAILURE;
  printf ("OK\n");
  return EXIT_SUCCESS;
}
