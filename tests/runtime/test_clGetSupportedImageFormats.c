#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "poclu.h"

#define MAX_DEVICES   2

int
main(void)
{
  cl_int err;
  cl_platform_id platform[1];
  cl_uint nplatforms;
  cl_device_id devices[MAX_DEVICES];
  cl_uint ndevices;
  cl_image_format *img_formats;
  cl_uint num_entries;
  
  err = clGetPlatformIDs (1, platform, &nplatforms);	
  CHECK_OPENCL_ERROR_IN("clGetPlatformIDs");

  err = clGetDeviceIDs (platform[0], CL_DEVICE_TYPE_ALL, MAX_DEVICES,
                        devices, &ndevices);
  CHECK_OPENCL_ERROR_IN("clGetDeviceIDs");

  TEST_ASSERT(ndevices >= 2);

  cl_context context = clCreateContext (NULL, ndevices, devices, NULL, NULL, 
                                        &err);
  CHECK_OPENCL_ERROR_IN("clCreateContext");

  err = clGetSupportedImageFormats (context, 0, CL_MEM_OBJECT_IMAGE2D, 0,
                              NULL, &num_entries);
  CHECK_OPENCL_ERROR_IN("clGetSupportedImageFormats");
  
  img_formats = (cl_image_format*)malloc (sizeof(cl_image_format)*num_entries);

  err = clGetSupportedImageFormats (context, 0, CL_MEM_OBJECT_IMAGE2D,
                                      num_entries, img_formats, NULL);
  CHECK_OPENCL_ERROR_IN("clGetSupportedImageFormats");

  TEST_ASSERT(num_entries != 0);

  CHECK_CL_ERROR (clReleaseContext (context));

  free (img_formats);

  return EXIT_SUCCESS;
}
