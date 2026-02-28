/* Regression test for kernel exit modes (__pocl_exit, __pocl_trap, plain
   unreachable).  Verifies the event execution status for each mode.

   Copyright (c) 2026 Tim Besard

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

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_ERROR(err)                                                      \
  if (err != CL_SUCCESS)                                                      \
    {                                                                         \
      printf ("OpenCL Error %d at %s:%d\n", err, __FILE__, __LINE__);         \
      return EXIT_FAILURE;                                                    \
    }

static int
test_kernel (cl_context context, cl_device_id device, cl_program program,
             const char *kernel_name, int expect_failure)
{
  cl_int err;

  cl_kernel kernel = clCreateKernel (program, kernel_name, &err);
  CHECK_ERROR (err);

  cl_command_queue queue
    = clCreateCommandQueueWithProperties (context, device, NULL, &err);
  CHECK_ERROR (err);

  /* n=0 triggers the bad path in every kernel */
  cl_long n = 0;
  err = clSetKernelArg (kernel, 0, sizeof (cl_long), &n);
  CHECK_ERROR (err);

  cl_event event;
  size_t global_size = 1;
  err = clEnqueueNDRangeKernel (queue, kernel, 1, NULL, &global_size, NULL, 0,
                                NULL, &event);
  CHECK_ERROR (err);

  clFinish (queue);

  cl_int status;
  err = clGetEventInfo (event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                        sizeof (cl_int), &status, NULL);
  CHECK_ERROR (err);

  int failed;
  if (expect_failure)
    {
      failed = status >= 0;
      if (failed)
        printf ("FAIL: %s: expected negative status, got %d\n", kernel_name,
                status);
      else
        printf ("OK: %s: status = %d (expected negative)\n", kernel_name,
                status);
    }
  else
    {
      failed = status != CL_COMPLETE;
      if (failed)
        printf ("FAIL: %s: expected CL_COMPLETE (%d), got %d\n", kernel_name,
                (int)CL_COMPLETE, status);
      else
        printf ("OK: %s: status = CL_COMPLETE\n", kernel_name);
    }

  clReleaseEvent (event);
  clReleaseCommandQueue (queue);
  clReleaseKernel (kernel);

  return failed;
}

int
main (int argc, char **argv)
{
  int platform_index = 0;
  if (argc > 1)
    platform_index = atoi (argv[1]);

  const char *input_spirv = "test_kernel_exit_modes.spv";
  if (argc > 2)
    input_spirv = argv[2];

  cl_int err;
  cl_uint num_platforms;
  err = clGetPlatformIDs (0, NULL, &num_platforms);
  CHECK_ERROR (err);

  cl_platform_id *platforms = malloc (sizeof (cl_platform_id) * num_platforms);
  err = clGetPlatformIDs (num_platforms, platforms, NULL);
  CHECK_ERROR (err);

  if ((cl_uint)platform_index >= num_platforms)
    {
      printf ("Platform index %d out of range (max %u)\n", platform_index,
              num_platforms - 1);
      free (platforms);
      return EXIT_FAILURE;
    }

  cl_platform_id platform = platforms[platform_index];
  free (platforms);

  cl_device_id device;
  err = clGetDeviceIDs (platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
  CHECK_ERROR (err);

  if (!poclu_device_supports_il (device, "SPIR-V_1.0"))
    {
      printf ("SKIP: device does not support SPIR-V\n");
      return 77;
    }

  /* Load the pre-compiled SPIR-V module */
  FILE *f = fopen (input_spirv, "rb");
  if (!f)
    {
      printf ("Failed to open %s\n", input_spirv);
      return EXIT_FAILURE;
    }
  fseek (f, 0, SEEK_END);
  size_t size = ftell (f);
  fseek (f, 0, SEEK_SET);
  unsigned char *binary = malloc (size);
  fread (binary, 1, size, f);
  fclose (f);

  cl_context context = clCreateContext (NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR (err);

  cl_program program = clCreateProgramWithIL (context, binary, size, &err);
  free (binary);
  CHECK_ERROR (err);

  err = clBuildProgram (program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      size_t log_size;
      clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                             &log_size);
      char *log = malloc (log_size);
      clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG, log_size,
                             log, NULL);
      printf ("Build error:\n%s\n", log);
      free (log);
      clReleaseProgram (program);
      clReleaseContext (context);
      return EXIT_FAILURE;
    }

  int failures = 0;

  /* Unreachable after willreturn call (function-attrs infers willreturn,
     UTR deletes block as pure dead code) -> CL_COMPLETE */
  failures += test_kernel (context, device, program, "test_unreachable", 0);

  /* __pocl_trap through noinline wrapper -> CL_FAILED */
  failures += test_kernel (context, device, program, "test_trap", 1);

  /* __pocl_exit through noinline wrapper -> CL_SUCCESS */
  failures += test_kernel (context, device, program, "test_exit", 0);

  /* Unreachable after side-effecting call (calls printf, function-attrs
     cannot infer willreturn, UTR preserves block) -> CL_FAILED */
  failures += test_kernel (context, device, program,
                           "test_unreachable_sideeffect", 1);

  clReleaseProgram (program);
  clReleaseContext (context);

  if (failures)
    {
      printf ("%d test(s) FAILED\n", failures);
      return EXIT_FAILURE;
    }

  printf ("OK\n");
  return EXIT_SUCCESS;
}
