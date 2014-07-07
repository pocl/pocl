#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "poclu.h"
#include "pocl_tests.h"

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
  if (err != CL_SUCCESS)
    return EXIT_FAILURE;

  err = clGetDeviceIDs (platform[0], CL_DEVICE_TYPE_ALL, MAX_DEVICES,
                        devices, &ndevices);
  if (err != CL_SUCCESS)
	return EXIT_FAILURE;

  TEST_ASSERT(ndevices >= 2);

  cl_context context = clCreateContext (NULL, ndevices, devices, NULL, NULL, 
                                        &err);

  if (err != CL_SUCCESS)
    return EXIT_FAILURE;
  
  clGetSupportedImageFormats (context, 0, CL_MEM_OBJECT_IMAGE2D, 0, 
                              NULL, &num_entries);
  
  img_formats = (cl_image_format*)malloc (sizeof(cl_image_format)*num_entries);

  clGetSupportedImageFormats (context, 0, CL_MEM_OBJECT_IMAGE2D, 
                                      num_entries, img_formats, NULL);

  if (err != CL_SUCCESS || num_entries == 0) 
    return EXIT_FAILURE;
  

  return EXIT_SUCCESS;
}
