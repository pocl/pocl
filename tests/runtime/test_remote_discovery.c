/* test_remote_discovery.c - Loops till a device is returned and then prints
   the server's IP to which the device belongs.

   Copyright (c) 2023-2024 Yashvardhan Agarwal / Tampere University

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

#include "CL/cl_ext_pocl.h"
#include "poclu.h"

#define MAX_PLATFORMS 32

int
main (void)
{
  cl_int err;
  cl_platform_id platforms[MAX_PLATFORMS];
  cl_uint nplatforms;
  cl_device_id *devices;
  cl_uint ndevices = 0;
  cl_uint i, j;

  err = clGetPlatformIDs (MAX_PLATFORMS, platforms, &nplatforms);
  CHECK_OPENCL_ERROR_IN ("clGetPlatformIDs");
  if (!nplatforms)
    return EXIT_FAILURE;

  for (i = 0; i < nplatforms; i++)
    {
      while (!ndevices)
        {
          err = clGetDeviceIDs (platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL,
                                &ndevices);
          if (err != CL_SUCCESS && err != CL_DEVICE_NOT_FOUND)
            return EXIT_FAILURE;
        }

      devices = (cl_device_id *)malloc (sizeof (cl_device_id) * ndevices);
      err = clGetDeviceIDs (platforms[i], CL_DEVICE_TYPE_ALL, ndevices,
                            devices, NULL);

      for (j = 0; j < ndevices; j++)
        {

          size_t ip_size;
          err = clGetDeviceInfo (devices[j], CL_DEVICE_REMOTE_SERVER_IP_POCL,
                                 0, NULL, &ip_size);
          CHECK_OPENCL_ERROR_IN ("clGetDeviceInfo");
          char *ip = (char *)malloc (ip_size);
          err = clGetDeviceInfo (devices[j], CL_DEVICE_REMOTE_SERVER_IP_POCL,
                                 ip_size, ip, NULL);
          CHECK_OPENCL_ERROR_IN ("clGetDeviceInfo");

          printf ("\nServer IP: %s\n", ip);
          free (ip);
        }
      free (devices);
      devices = NULL;
    }

  return EXIT_SUCCESS;
}
