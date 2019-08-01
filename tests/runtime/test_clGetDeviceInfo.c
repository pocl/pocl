#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "poclu.h"

#define MAX_PLATFORMS 32
#define MAX_DEVICES   32

int
main(void)
{
  cl_int err;
  cl_platform_id platforms[MAX_PLATFORMS];
  cl_uint nplatforms;
  cl_device_id devices[MAX_DEVICES];
  cl_uint ndevices;
  cl_uint i, j;

  err = clGetPlatformIDs(MAX_PLATFORMS, platforms, &nplatforms);	
  CHECK_OPENCL_ERROR_IN("clGetPlatformIDs");
  if (!nplatforms)
    return EXIT_FAILURE;

  for (i = 0; i < nplatforms; i++)
  {
    err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, MAX_DEVICES,
                         devices, &ndevices);
    CHECK_OPENCL_ERROR_IN("clGetDeviceIDs");

    for (j = 0; j < ndevices; j++)
    {
      cl_long global_memsize, max_mem_alloc_size, min_max_mem_alloc_size;

      err = clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE,
                            sizeof(global_memsize), &global_memsize, NULL);
      CHECK_OPENCL_ERROR_IN("clGetDeviceInfo");

      err = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                            sizeof(max_mem_alloc_size), &max_mem_alloc_size,
                            NULL);
      CHECK_OPENCL_ERROR_IN("clGetDeviceInfo");

      TEST_ASSERT(global_memsize > 0);

      min_max_mem_alloc_size = 128*1024*1024;
      if (min_max_mem_alloc_size < global_memsize/4)
        min_max_mem_alloc_size = global_memsize/4;

      TEST_ASSERT(max_mem_alloc_size >= min_max_mem_alloc_size);
    }
  }

  CHECK_CL_ERROR (clUnloadCompiler ());

  printf ("OK\n");
  return EXIT_SUCCESS;
}
