#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "poclu.h"

#define MAX_DEVICES   2

int
main(void)
{
  cl_int err;
  cl_image_format *img_formats;
  cl_uint num_entries;

  cl_context ctx = NULL;
  cl_device_id dev = NULL;
  cl_command_queue cq = NULL;
  
  err = poclu_get_any_device(&ctx, &dev, &cq);
  CHECK_OPENCL_ERROR_IN("poclu_get_any_device");

  TEST_ASSERT(dev != NULL);

  cl_bool img_support = 0;
  err = clGetDeviceInfo(dev, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &img_support, 0);

  if (img_support != CL_TRUE)
    return EXIT_SUCCESS;

  err = clGetSupportedImageFormats (ctx, 0, CL_MEM_OBJECT_IMAGE2D, 0,
                                    NULL, &num_entries);
  CHECK_OPENCL_ERROR_IN("clGetSupportedImageFormats");

  if (num_entries == 0)
    return EXIT_SUCCESS;

  img_formats = (cl_image_format*)malloc (sizeof(cl_image_format)*num_entries);

  err = clGetSupportedImageFormats (ctx, 0, CL_MEM_OBJECT_IMAGE2D,
                                    num_entries, img_formats, NULL);
  CHECK_OPENCL_ERROR_IN("clGetSupportedImageFormats");

  TEST_ASSERT(num_entries != 0);

  CHECK_CL_ERROR (clReleaseCommandQueue (cq));
  CHECK_CL_ERROR (clReleaseContext (ctx));

  free (img_formats);

  CHECK_CL_ERROR (clUnloadCompiler ());

  return EXIT_SUCCESS;
}
