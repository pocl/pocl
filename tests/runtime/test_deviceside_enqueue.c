/* Tests Device-Side Enqueue

   Copyright (c) 2021 Väinö Liukko

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
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

/* Tests for verifying the return values of API functions related to
   Device-side Enqueue.  Since currently PoCL doesn't have support for
   Device-side Enqueue on any of the device drivers. The return values should
   match the values defined in the OpenCL specification.  See,
   https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_device_side_enqueue
*/

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

  err = clGetPlatformIDs (MAX_PLATFORMS, platforms, &nplatforms);
  CHECK_OPENCL_ERROR_IN ("clGetPlatformIDs");
  if (!nplatforms)
    return EXIT_FAILURE;

#ifdef CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES
  for (i = 0; i < nplatforms; i++)
    {
      err = clGetDeviceIDs (platforms[i], CL_DEVICE_TYPE_ALL, MAX_DEVICES,
                            devices, &ndevices);
      CHECK_OPENCL_ERROR_IN ("clGetDeviceIDs");

      for (j = 0; j < ndevices; j++)
        {
          cl_uint device_queue_support;
          err = clGetDeviceInfo (devices[j], CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES,
                                 sizeof (cl_uint), &device_queue_support,
                                 NULL);
          CHECK_OPENCL_ERROR_IN ("clGetDeviceInfo");

          if (device_queue_support == 0)
            {
              cl_command_queue_properties device_queue_props;
              err = clGetDeviceInfo (devices[j],
                                     CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES,
                                     sizeof (cl_command_queue_properties),
                                     &device_queue_props, NULL);
              CHECK_OPENCL_ERROR_IN ("clGetDeviceInfo");

              TEST_ASSERT (device_queue_props == 0);

              cl_uint device_queue_pref_size;
              err = clGetDeviceInfo (
                  devices[j], CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE,
                  sizeof (cl_uint), &device_queue_pref_size, NULL);
              CHECK_OPENCL_ERROR_IN ("clGetDeviceInfo");

              TEST_ASSERT (device_queue_pref_size == 0);

              cl_uint device_queue_max_size;
              err = clGetDeviceInfo (
                  devices[j], CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE,
                  sizeof (cl_uint), &device_queue_max_size, NULL);
              CHECK_OPENCL_ERROR_IN ("clGetDeviceInfo");

              TEST_ASSERT (device_queue_max_size == 0);

              cl_uint device_queue_max_queues;
              err = clGetDeviceInfo (
                  devices[j], CL_DEVICE_MAX_ON_DEVICE_QUEUES, sizeof (cl_uint),
                  &device_queue_max_queues, NULL);
              CHECK_OPENCL_ERROR_IN ("clGetDeviceInfo");

              TEST_ASSERT (device_queue_max_queues == 0);

              cl_uint device_queue_max_events;
              err = clGetDeviceInfo (
                  devices[j], CL_DEVICE_MAX_ON_DEVICE_EVENTS, sizeof (cl_uint),
                  &device_queue_max_events, NULL);
              CHECK_OPENCL_ERROR_IN ("clGetDeviceInfo");

              TEST_ASSERT (device_queue_max_events == 0);

              context
                  = clCreateContext (NULL, 1, &devices[j], NULL, NULL, &err);
              queue = clCreateCommandQueue (context, devices[j], 0, &err);

              cl_uint queue_size;
              err = clGetCommandQueueInfo (
                  queue, CL_QUEUE_SIZE, sizeof (cl_uint), &queue_size, NULL);
              TEST_ASSERT (err == CL_INVALID_COMMAND_QUEUE);

              cl_command_queue queue_default;
              err = clGetCommandQueueInfo (queue, CL_QUEUE_DEVICE_DEFAULT,
                                           sizeof (cl_command_queue),
                                           &queue_default, NULL);
              CHECK_OPENCL_ERROR_IN ("clGetCommandQueueInfo");

              TEST_ASSERT (queue_default == NULL);

              err = clSetDefaultDeviceCommandQueue (context, devices[j],
                                                    queue);

              TEST_ASSERT (err == CL_INVALID_OPERATION);
              CHECK_CL_ERROR (clReleaseCommandQueue (queue));
              CHECK_CL_ERROR (clReleaseContext (context));
            }
        }

      CHECK_CL_ERROR (clUnloadPlatformCompiler (platforms[i]));
    }
#endif


  printf ("OK\n");
  return EXIT_SUCCESS;
}
