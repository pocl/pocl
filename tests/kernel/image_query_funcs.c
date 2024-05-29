#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poclu.h"

#ifdef _MSC_VER
#  include "vccompat.hpp"
#endif

#define W 2
#define H 4
#define D 8
#define PIXELS (W * H * D)

int main(int argc, char **argv)
{
  /* test name */
  char name[] = "test_image_query_funcs";
  size_t global_work_size[1] = { 1 }, local_work_size[1]= { 1 };
  size_t srcdir_length, name_length, filename_size;
  char *filename = NULL;
  char *source = NULL;
  cl_platform_id pid = NULL;
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_int err;

  /* image parameters */
  cl_uchar4 *imageData;
  cl_image_format image_format;
  cl_image_desc image2_desc, image3_desc;

  printf("Running test %s...\n", name);

  memset(&image2_desc, 0, sizeof(cl_image_desc));
  image2_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  image2_desc.image_width = W;
  image2_desc.image_height = H;

  memset(&image3_desc, 0, sizeof(cl_image_desc));
  image3_desc.image_type = CL_MEM_OBJECT_IMAGE3D;
  image3_desc.image_width = W;
  image3_desc.image_height = H;
  image3_desc.image_depth = D;

  image_format.image_channel_order = CL_RGBA;
  image_format.image_channel_data_type = CL_UNSIGNED_INT8;
  imageData = (cl_uchar4 *)malloc (PIXELS * sizeof (cl_uchar4));

  TEST_ASSERT (imageData != NULL && "out of host memory\n");
  memset (imageData, 1, PIXELS * sizeof (cl_uchar4));

  /* determine file name of kernel source to load */
  srcdir_length = strlen(SRCDIR);
  name_length = strlen(name);
  filename_size = srcdir_length + name_length + 16;
  filename = (char *)malloc(filename_size + 1);
  TEST_ASSERT (filename != NULL && "out of host memory\n");

  snprintf(filename, filename_size, "%s/%s.cl", SRCDIR, name);

  /* read source code */
  source = poclu_read_file (filename);
  TEST_ASSERT (source != NULL && "Kernel .cl not found.");

  /* setup an OpenCL context and command queue using default device */
  context = poclu_create_any_context2 (&pid);
  TEST_ASSERT (context != NULL && "clCreateContextFromType call failed\n");

  cl_sampler external_sampler = clCreateSampler (
      context, CL_FALSE, CL_ADDRESS_NONE, CL_FILTER_NEAREST, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateSampler");

  size_t device_id_size = 0;
  err = clGetContextInfo (context, CL_CONTEXT_DEVICES, 0, NULL,
                          &device_id_size);
  CHECK_OPENCL_ERROR_IN ("clGetContextInfo");
  cl_device_id *devices = malloc (device_id_size);
  TEST_ASSERT (devices != NULL && "out of host memory\n");
  err = clGetContextInfo (context, CL_CONTEXT_DEVICES, device_id_size, devices,
                          NULL);
  CHECK_OPENCL_ERROR_IN ("clGetContextInfo");

  queue = clCreateCommandQueue (context, devices[0], 0, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateCommandQueue");

  /* Create image */
  cl_mem image2
      = clCreateImage (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       &image_format, &image2_desc, imageData, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateImage image2");

  cl_mem image3
      = clCreateImage (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       &image_format, &image3_desc, imageData, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateImage image3");

  unsigned color[4] = { 2, 9, 11, 7 };
  size_t orig[3] = { 0, 0, 0 };
  size_t reg[3] = { 2, 4, 1 };
  err = clEnqueueFillImage (queue, image2, color, orig, reg, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN ("clCreateImage image3");

  /* create and build program */
  program = clCreateProgramWithSource (context, 1, (const char **)&source,
                                       NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateProgramWithSource");

  err = clBuildProgram (program, 0, NULL, NULL, NULL, NULL);
  CHECK_OPENCL_ERROR_IN ("clBuildProgram");

  /* execute the kernel with give name */
  kernel = clCreateKernel (program, name, NULL);
  CHECK_OPENCL_ERROR_IN ("clCreateKernel");

  err = clSetKernelArg (kernel, 0, sizeof (cl_mem), &image2);
  CHECK_OPENCL_ERROR_IN ("clSetKernelArg 0");

  err = clSetKernelArg (kernel, 1, sizeof (cl_mem), &image3);
  CHECK_OPENCL_ERROR_IN ("clSetKernelArg 1");

  err = clSetKernelArg (kernel, 2, sizeof (cl_sampler), &external_sampler);
  CHECK_OPENCL_ERROR_IN ("clSetKernelArg 2");

  err = clEnqueueNDRangeKernel (queue, kernel, 1, NULL, global_work_size,
                                local_work_size, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN ("clEnqueueNDRangeKernel");

  err = clFinish (queue);
  CHECK_OPENCL_ERROR_IN ("clFinish");

  CHECK_CL_ERROR (clReleaseMemObject (image2));
  CHECK_CL_ERROR (clReleaseMemObject (image3));
  CHECK_CL_ERROR (clReleaseKernel (kernel));
  CHECK_CL_ERROR (clReleaseProgram (program));
  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseSampler (external_sampler));
  CHECK_CL_ERROR (clReleaseContext (context));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (pid));

  free (source);
  free (filename);
  free (imageData);
  free (devices);

  printf("OK\n");
  return EXIT_SUCCESS;
}
