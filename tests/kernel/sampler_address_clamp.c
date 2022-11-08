#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poclu.h"

#ifdef _MSC_VER
#  include "vccompat.hpp"
#endif

int main(int argc, char **argv)
{
  /* test name */
  char name[] = "test_sampler_address_clamp";
  size_t global_work_size[1] = { 1 }, local_work_size[1]= { 1 };
  size_t srcdir_length, name_length, filename_size;
  char *filename = NULL;
  char *source = NULL;
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem image = NULL;
  cl_int result;
  int retval = -1;

  /* image parameters */
  cl_uchar4 *imageData;
  cl_image_format image_format;
  cl_image_desc image_desc;

  printf("Running test %s...\n", name);
  memset(&image_desc, 0, sizeof(cl_image_desc));
  image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  image_desc.image_width = 4;
  image_desc.image_height = 4;
  image_format.image_channel_order = CL_RGBA;
  image_format.image_channel_data_type = CL_UNSIGNED_INT8;
  imageData = (cl_uchar4*)malloc (4 * 4 * sizeof(cl_uchar4));
  
  if (imageData == NULL)
    {
      puts("out of host memory\n");
      goto error;
    }
  memset (imageData, 1, 4*4*sizeof(cl_uchar4));

  /* determine file name of kernel source to load */
  srcdir_length = strlen(SRCDIR);
  name_length = strlen(name);
  filename_size = srcdir_length + name_length + 16;
  filename = (char *)malloc(filename_size + 1);
  if (!filename) 
    {
      puts("out of memory");
      goto error;
    }
  
  snprintf(filename, filename_size, "%s/%s.cl", SRCDIR, name);
  
  /* read source code */
  source = poclu_read_file (filename);
  TEST_ASSERT (source != NULL && "Kernel .cl not found.");

  /* setup an OpenCL context and command queue using default device */
  context = poclu_create_any_context();
  if (!context) 
    {
      puts("clCreateContextFromType call failed\n");
      goto error;
    }

  size_t device_id_size = 0;
  result = clGetContextInfo (context, CL_CONTEXT_DEVICES, 0, NULL,
                             &device_id_size);
  if (result != CL_SUCCESS)
    {
      puts ("clGetContextInfo call failed while fetching size\n");
      goto error;
    }
  cl_device_id *devices = malloc (device_id_size);
  TEST_ASSERT (devices != NULL && "out of host memory\n");
  result = clGetContextInfo (context, CL_CONTEXT_DEVICES, device_id_size,
                             devices, NULL);
  if (result != CL_SUCCESS)
    {
      puts ("clGetContextInfo call failed\n");
      goto error;
    }

  queue = clCreateCommandQueue(context, devices[0], 0, NULL); 
  if (!queue) 
    {
      puts("clCreateCommandQueue call failed\n");
      goto error;
    }

  /* Create image */

  image = clCreateImage (context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                         &image_format, &image_desc, imageData, &result);
  if (result != CL_SUCCESS)
    {
      puts("image creation failed\n");
      goto error;
    }


  /* create and build program */
  program = clCreateProgramWithSource (context, 1, (const char **)&source,
                                       NULL, NULL);
  if (!program) 
    {
      puts("clCreateProgramWithSource call failed\n");
      goto error;
    }

  result = clBuildProgram(program, 0, NULL, NULL, NULL, NULL); 
  if (result != CL_SUCCESS) 
    {
      puts("clBuildProgram call failed\n");
      goto error;
    }

  /* execute the kernel with give name */
  kernel = clCreateKernel(program, name, NULL); 
  if (!kernel) 
    {
      puts("clCreateKernel call failed\n");
      goto error;
    }

   result = clSetKernelArg( kernel, 0, sizeof(cl_mem), &image);
   if (result)
     {
       puts("clSetKernelArg failed\n");
       goto error;
     }

  result = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size, 
                                  local_work_size, 0, NULL, NULL); 
  if (result != CL_SUCCESS) 
    {
      puts("clEnqueueNDRangeKernel call failed\n");
      goto error;
    }

  result = clFinish(queue);
  if (result == CL_SUCCESS)
    retval = 0;

error:

  if (image)
    {
      clReleaseMemObject (image);
    }

  if (kernel) 
    {
      clReleaseKernel(kernel);
    }
  if (program) 
    {
      clReleaseProgram(program);
    }
  if (queue) 
    {
      clReleaseCommandQueue(queue);
    }
  if (context) 
    {
      clReleaseContext (context);
      clUnloadCompiler ();
    }
  if (source) 
    {
      free(source);
    }
  if (filename)
    {
      free(filename);
    }
  if (imageData)
    {
      free(imageData);
    }
  free (devices);

  CHECK_CL_ERROR (clUnloadCompiler ());

  if (retval) 
    {
      printf("FAIL\n");
      return EXIT_FAILURE;
    }
 
  printf("OK\n");
  return EXIT_SUCCESS;
}
