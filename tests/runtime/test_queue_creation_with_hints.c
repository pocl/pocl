/* Tests command queue hints (cl_khr_priority_hints, cl_khr_throttle_hints)

   Copyright (c) 2024 Jan Solanti / Tampere University

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poclu.h"

#include <CL/cl_ext.h>

int
main ()
{
  cl_int err;
  cl_platform_id platforms[1];
  cl_uint nplatforms;
  cl_device_id devices[1];

  err = clGetPlatformIDs (1, platforms, &nplatforms);
  CHECK_OPENCL_ERROR_IN ("clGetPlatformIDs");
  if (!nplatforms)
    return EXIT_FAILURE;

  size_t extensions_len;
  err = clGetPlatformInfo (platforms[0], CL_PLATFORM_EXTENSIONS_WITH_VERSION,
                           0, NULL, &extensions_len);
  CHECK_OPENCL_ERROR_IN ("clGetPlatformInfo (size_ret)");

  cl_name_version *extensions = malloc (extensions_len);
  err = clGetPlatformInfo (platforms[0], CL_PLATFORM_EXTENSIONS_WITH_VERSION,
                           extensions_len, extensions, NULL);
  CHECK_OPENCL_ERROR_IN ("clGetPlatformInfo (value)");

  int cl_khr_priority_hints_found = 0;
  int cl_khr_throttle_hints_found = 0;

  for (size_t i = 0; i < extensions_len / sizeof (cl_name_version); ++i)
    {
      if (strcmp ("cl_khr_priority_hints", extensions[i].name) == 0
          && extensions[i].version >= CL_MAKE_VERSION (1, 0, 0))
        cl_khr_priority_hints_found = 1;
      if (strcmp ("cl_khr_throttle_hints", extensions[i].name) == 0
          && extensions[i].version >= CL_MAKE_VERSION (1, 0, 0))
        cl_khr_throttle_hints_found = 1;
    }

  free (extensions);
  extensions = NULL;

  if (!cl_khr_priority_hints_found || !cl_khr_throttle_hints_found)
    {
      printf ("Queue hint extensions not supported\n");
      return 77;
    }

  err = clGetDeviceIDs (platforms[0], CL_DEVICE_TYPE_ALL, 1, devices, NULL);
  CHECK_OPENCL_ERROR_IN ("clGetDeviceIDs");

  cl_context context = clCreateContext (NULL, 1, devices, NULL, NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateContext");

  err = clGetContextInfo (context, CL_CONTEXT_DEVICES, sizeof (cl_device_id),
                          devices, NULL);
  CHECK_OPENCL_ERROR_IN ("clGetContextInfo");

  cl_queue_properties valid_hints[][5] = {
    { CL_QUEUE_PRIORITY_KHR, CL_QUEUE_PRIORITY_HIGH_KHR, 0, 0, 0 },
    { CL_QUEUE_PRIORITY_KHR, CL_QUEUE_PRIORITY_MED_KHR, 0, 0, 0 },
    { CL_QUEUE_PRIORITY_KHR, CL_QUEUE_PRIORITY_LOW_KHR, 0, 0, 0 },
    { CL_QUEUE_THROTTLE_KHR, CL_QUEUE_THROTTLE_HIGH_KHR, 0, 0, 0 },
    { CL_QUEUE_THROTTLE_KHR, CL_QUEUE_THROTTLE_MED_KHR, 0, 0, 0 },
    { CL_QUEUE_THROTTLE_KHR, CL_QUEUE_THROTTLE_LOW_KHR, 0, 0, 0 },
    { CL_QUEUE_THROTTLE_KHR, CL_QUEUE_THROTTLE_HIGH_KHR, CL_QUEUE_PRIORITY_KHR,
      CL_QUEUE_PRIORITY_HIGH_KHR, 0 },
    { CL_QUEUE_THROTTLE_KHR, CL_QUEUE_THROTTLE_MED_KHR, CL_QUEUE_PRIORITY_KHR,
      CL_QUEUE_PRIORITY_HIGH_KHR, 0 },
    { CL_QUEUE_THROTTLE_KHR, CL_QUEUE_THROTTLE_LOW_KHR, CL_QUEUE_PRIORITY_KHR,
      CL_QUEUE_PRIORITY_HIGH_KHR, 0 },
    { CL_QUEUE_THROTTLE_KHR, CL_QUEUE_THROTTLE_HIGH_KHR, CL_QUEUE_PRIORITY_KHR,
      CL_QUEUE_PRIORITY_MED_KHR, 0 },
    { CL_QUEUE_THROTTLE_KHR, CL_QUEUE_THROTTLE_MED_KHR, CL_QUEUE_PRIORITY_KHR,
      CL_QUEUE_PRIORITY_MED_KHR, 0 },
    { CL_QUEUE_THROTTLE_KHR, CL_QUEUE_THROTTLE_LOW_KHR, CL_QUEUE_PRIORITY_KHR,
      CL_QUEUE_PRIORITY_MED_KHR, 0 },
    { CL_QUEUE_THROTTLE_KHR, CL_QUEUE_THROTTLE_HIGH_KHR, CL_QUEUE_PRIORITY_KHR,
      CL_QUEUE_PRIORITY_LOW_KHR, 0 },
    { CL_QUEUE_THROTTLE_KHR, CL_QUEUE_THROTTLE_MED_KHR, CL_QUEUE_PRIORITY_KHR,
      CL_QUEUE_PRIORITY_LOW_KHR, 0 },
    { CL_QUEUE_THROTTLE_KHR, CL_QUEUE_THROTTLE_LOW_KHR, CL_QUEUE_PRIORITY_KHR,
      CL_QUEUE_PRIORITY_LOW_KHR, 0 },
  };
  for (size_t i = 0;
       i < (sizeof (valid_hints) / sizeof (cl_queue_properties[5])); ++i)
    {
      fprintf (stderr, "%d\n", (int)i);
      cl_command_queue queueA = clCreateCommandQueueWithProperties (
          context, devices[0], valid_hints[i], &err);
      CHECK_OPENCL_ERROR_IN ("clCreateCommandQueue");
      TEST_ASSERT (queueA);
      CHECK_CL_ERROR (clReleaseCommandQueue (queueA));
    }

  cl_queue_properties bad_hints[][5] = {
    { CL_QUEUE_PRIORITY_KHR, 0, 0, 0, 0 },
    { CL_QUEUE_PRIORITY_KHR, CL_QUEUE_PRIORITY_LOW_KHR + 1, 0, 0, 0 },
    { CL_QUEUE_THROTTLE_KHR, 0, 0, 0, 0 },
    { CL_QUEUE_THROTTLE_KHR, CL_QUEUE_THROTTLE_LOW_KHR + 1, 0, 0, 0 },
    { CL_QUEUE_THROTTLE_KHR, CL_QUEUE_THROTTLE_HIGH_KHR, CL_QUEUE_PRIORITY_KHR,
      0, 0 },
    { CL_QUEUE_THROTTLE_KHR, 0, CL_QUEUE_PRIORITY_KHR,
      CL_QUEUE_PRIORITY_HIGH_KHR, 0 },
    { CL_QUEUE_PROPERTIES,
      CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE,
      CL_QUEUE_PRIORITY_KHR, CL_QUEUE_PRIORITY_HIGH_KHR, 0 },
    { CL_QUEUE_PROPERTIES,
      CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_ON_DEVICE,
      CL_QUEUE_THROTTLE_KHR, CL_QUEUE_THROTTLE_HIGH_KHR, 0 },
  };
  for (size_t i = 0;
       i < (sizeof (bad_hints) / sizeof (cl_queue_properties[5])); ++i)
    {
      cl_command_queue queueA = clCreateCommandQueueWithProperties (
          context, devices[0], bad_hints[i], &err);
      TEST_ASSERT (!queueA && err != CL_SUCCESS);
      if (queueA)
        CHECK_CL_ERROR (clReleaseCommandQueue (queueA));
    }

  CHECK_CL_ERROR (clReleaseContext (context));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (platforms[0]));

  printf ("OK\n");
  return EXIT_SUCCESS;
}
