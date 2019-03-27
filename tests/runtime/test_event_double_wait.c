/* Test that waiting for an event twice doesn't lead to double-frees

   Copyright (C) 2020 Giuseppe Bilotta <giuseppe.bilotta@gmail.com>

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

int main(int argc, char **argv)
{
  cl_int err;
  const char *krn_src = "kernel void test () { }";
  cl_program program;
  cl_context ctx;
  cl_command_queue queue;
  cl_device_id did;
  cl_kernel kernel;
  cl_event kern_evt;
  const size_t gws[] = { 1 };

  CHECK_CL_ERROR (poclu_get_any_device (&ctx, &did, &queue));
  TEST_ASSERT (ctx);
  TEST_ASSERT (did);
  TEST_ASSERT (queue);

  program = clCreateProgramWithSource (ctx, 1, &krn_src, NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateProgramWithSource");

  CHECK_CL_ERROR (clBuildProgram (program, 1, &did, "", NULL, NULL));

  kernel = clCreateKernel (program, "test", &err);
  CHECK_OPENCL_ERROR_IN ("clCreateKernel");

  CHECK_CL_ERROR (clEnqueueNDRangeKernel (queue, kernel, 1, NULL, gws, NULL, 0, NULL, &kern_evt));

  CHECK_CL_ERROR (clWaitForEvents (1, &kern_evt));
  CHECK_CL_ERROR (clWaitForEvents (1, &kern_evt));
  CHECK_CL_ERROR (clFinish (queue));

  CHECK_CL_ERROR (clReleaseEvent (kern_evt));
  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseKernel (kernel));
  CHECK_CL_ERROR (clReleaseProgram (program));
  CHECK_CL_ERROR (clReleaseContext (ctx));
  CHECK_CL_ERROR (clUnloadCompiler ());

  return EXIT_SUCCESS;

}


