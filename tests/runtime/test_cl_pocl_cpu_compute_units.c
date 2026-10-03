/* Test "cl_pocl_cpu_compute_units" PoCL extension

   Copyright (C) 2026 Tim Besard

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

#include "include/CL/cl_ext_pocl.h"

/* Tests cl_pocl_cpu_compute_units: the number of compute units set with
 * clSetCPUMaxComputeUnitsPOCL takes precedence over the environment, and can
 * only be set before the devices are initialized. */

int
main (void)
{
  cl_int err;
  cl_platform_id platform = NULL;
  char extensions[4096];

  /* the API should win over the environment */
#ifdef _WIN32
  _putenv_s ("POCL_CPU_MAX_CU_COUNT", "5");
#else
  setenv ("POCL_CPU_MAX_CU_COUNT", "5", 1);
#endif

  /* querying the platform doesn't initialize the devices */
  CHECK_CL_ERROR (clGetPlatformIDs (1, &platform, NULL));
  CHECK_CL_ERROR (clGetPlatformInfo (platform, CL_PLATFORM_EXTENSIONS,
                                     sizeof (extensions), extensions, NULL));
  if (strstr (extensions, "cl_pocl_cpu_compute_units") == NULL)
    {
      printf ("cl_pocl_cpu_compute_units not supported, skipping\n");
      return 77;
    }

  clSetCPUMaxComputeUnitsPOCL_fn setCPUMaxComputeUnits
      = (clSetCPUMaxComputeUnitsPOCL_fn)
          clGetExtensionFunctionAddressForPlatform (
              platform, "clSetCPUMaxComputeUnitsPOCL");
  TEST_ASSERT (setCPUMaxComputeUnits != NULL);

  TEST_ASSERT (setCPUMaxComputeUnits (NULL, 2) == CL_INVALID_PLATFORM);
  TEST_ASSERT (setCPUMaxComputeUnits (platform, 0) == CL_INVALID_VALUE);

  /* the last call before initialization wins */
  CHECK_CL_ERROR (setCPUMaxComputeUnits (platform, 3));
  CHECK_CL_ERROR (setCPUMaxComputeUnits (platform, 2));

  cl_device_id device;
  err = clGetDeviceIDs (platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
  if (err == CL_DEVICE_NOT_FOUND)
    {
      printf ("No CPU devices, skipping\n");
      return 77;
    }
  CHECK_OPENCL_ERROR_IN ("clGetDeviceIDs");

  cl_uint compute_units;
  CHECK_CL_ERROR (clGetDeviceInfo (device, CL_DEVICE_MAX_COMPUTE_UNITS,
                                   sizeof (compute_units), &compute_units,
                                   NULL));
  TEST_ASSERT (compute_units == 2);

  /* once the devices are initialized, it's too late */
  TEST_ASSERT (setCPUMaxComputeUnits (platform, 4) == CL_INVALID_OPERATION);

  printf ("OK\n");
  return EXIT_SUCCESS;
}
