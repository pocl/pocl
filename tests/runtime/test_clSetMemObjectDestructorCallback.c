#include <stdio.h>
#include <stdlib.h>

#include "poclu.h"

#define MAX_DEVICES   1

#define FAKE_PTR 0xDEADBEEF

void callback(cl_mem memobj, void *user_data)
{
  if (user_data == (void*)FAKE_PTR)
    printf("OK\n");
  else
    printf("FAIL\n");
}

int
main(void)
{
  cl_int err;
  cl_platform_id platform[1];
  cl_uint nplatforms;
  cl_device_id devices[MAX_DEVICES];
  cl_uint ndevices;

  err = clGetPlatformIDs (1, platform, &nplatforms);
  CHECK_OPENCL_ERROR_IN("clGetPlatformIDs");

  err = clGetDeviceIDs (platform[0], CL_DEVICE_TYPE_ALL, MAX_DEVICES,
                        devices, &ndevices);
  CHECK_OPENCL_ERROR_IN("clGetDeviceIDs");

  TEST_ASSERT(ndevices >= 1);

  cl_context context = clCreateContext (NULL, 1, devices, NULL, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateContext");

  cl_mem mem = clCreateBuffer (context, 0, 1024, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer");

  err = clSetMemObjectDestructorCallback (mem, callback, (void*)FAKE_PTR);
  CHECK_OPENCL_ERROR_IN("clSetMemObjectDestructorCallback");

  CHECK_CL_ERROR (clReleaseMemObject (mem));
  CHECK_CL_ERROR (clReleaseContext (context));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (platform[0]));

  return EXIT_SUCCESS;
}
