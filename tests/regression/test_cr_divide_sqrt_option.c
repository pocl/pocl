/* Regression test for issue #2404: clBuildProgram and clCompileProgram must
   fail with -cl-fp32-correctly-rounded-divide-sqrt for a device whose
   CL_DEVICE_SINGLE_FP_CONFIG lacks CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT, and
   succeed for a device that has it.

   Copyright (c) 2026 pocl developers

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

#define OPTION "-cl-fp32-correctly-rounded-divide-sqrt"

static const char *source
  = "kernel void k (global float *a) { a[0] = a[1] / a[2] + sqrt (a[3]); }";

/* Builds, or only compiles, the kernel for the device with OPTION and
   returns 1 if the result does not match the device's
   CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT bit. */
static int
check_option (cl_device_id device, int has_cr, int compile_only)
{
  cl_int err;
  cl_context context = clCreateContext (NULL, 1, &device, NULL, NULL, &err);
  CHECK_CL_ERROR (err);
  cl_program program
    = clCreateProgramWithSource (context, 1, &source, NULL, &err);
  CHECK_CL_ERROR (err);

  cl_int expected = CL_SUCCESS;
  if (compile_only)
    {
      err = clCompileProgram (program, 1, &device, OPTION, 0, NULL, NULL, NULL,
                              NULL);
      if (!has_cr)
        expected = CL_COMPILE_PROGRAM_FAILURE;
    }
  else
    {
      err = clBuildProgram (program, 1, &device, OPTION, NULL, NULL);
      if (!has_cr)
        expected = CL_BUILD_PROGRAM_FAILURE;
    }

  size_t log_size = 0;
  CHECK_CL_ERROR (clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG,
                                         0, NULL, &log_size));
  char *log = (char *)calloc (log_size + 1, 1);
  TEST_ASSERT (log != NULL);
  CHECK_CL_ERROR (clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG,
                                         log_size, log, NULL));

  /* Without the bit, the build must fail because of the option, not
     because the driver cannot compile the kernel. */
  int mismatch
    = err != expected
      || (!has_cr
          && strstr (log, "CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT") == NULL);
  if (mismatch)
    {
      char name[256] = "";
      clGetDeviceInfo (device, CL_DEVICE_NAME, sizeof (name) - 1, name, NULL);
      printf ("FAIL: %s with " OPTION " on %s, which %s the bit, returned "
              "%d, expected %d; build log: \"%s\"\n",
              compile_only ? "clCompileProgram" : "clBuildProgram", name,
              has_cr ? "has" : "lacks", err, expected, log);
    }

  free (log);
  CHECK_CL_ERROR (clReleaseProgram (program));
  CHECK_CL_ERROR (clReleaseContext (context));
  return mismatch;
}

int
main (void)
{
  cl_platform_id platform;
  cl_uint num_devices = 0;
  int failures = 0;

  CHECK_CL_ERROR (clGetPlatformIDs (1, &platform, NULL));
  CHECK_CL_ERROR (
    clGetDeviceIDs (platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices));
  cl_device_id *devices
    = (cl_device_id *)malloc (num_devices * sizeof (cl_device_id));
  TEST_ASSERT (devices != NULL);
  CHECK_CL_ERROR (
    clGetDeviceIDs (platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL));

  for (cl_uint i = 0; i < num_devices; ++i)
    {
      cl_bool compiler = CL_FALSE;
      cl_device_fp_config fp_config = 0;
      CHECK_CL_ERROR (clGetDeviceInfo (devices[i],
                                       CL_DEVICE_COMPILER_AVAILABLE,
                                       sizeof (compiler), &compiler, NULL));
      if (!compiler)
        continue;
      CHECK_CL_ERROR (clGetDeviceInfo (devices[i], CL_DEVICE_SINGLE_FP_CONFIG,
                                       sizeof (fp_config), &fp_config, NULL));
      int has_cr = (fp_config & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT) != 0;

      failures += check_option (devices[i], has_cr, 0);
      failures += check_option (devices[i], has_cr, 1);
    }

  free (devices);
  if (failures)
    return EXIT_FAILURE;
  printf ("OK\n");
  return EXIT_SUCCESS;
}
