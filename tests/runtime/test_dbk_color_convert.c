/* test_dbk_color_convert.c - test program for color convert defined builtin
   kernels

   Copyright (c) 2024 PoCL developers

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

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "image_test_utils.h"
#include "poclu.h"

/**
 * Program arguments (and defaults tram.rgb):
 * 1. width (640)
 * 2. height (480)
 * 3. source image path (input.nv12)
 * 4. reference image path (tram.rgb)
 * 5. (optional) output location of image
 */
int
main (int argc, char const *argv[])
{

  TEST_ASSERT (argc >= 5);

  errno = 0;
  char *end_ptr;
  int width = (int)strtol (argv[1], &end_ptr, 10);
  printf ("errno: %d, %s, %d\n", errno, strerror (errno), *end_ptr);
  TEST_ASSERT (errno == 0 && *end_ptr == '\0');
  int height = (int)strtol (argv[2], &end_ptr, 10);
  TEST_ASSERT (errno == 0 && *end_ptr == '\0');

  const char *file_name = argv[3];
  size_t bytes_read = 0;
  char *input_data = poclu_read_binfile (file_name, &bytes_read);
  size_t input_size = (size_t)height * width * 3 / 2;
  TEST_ASSERT (bytes_read == input_size);

  size_t rgb_size = (size_t)height * width * 3;

  cl_platform_id platform = NULL;
  cl_context context = NULL;
  cl_device_id *devices = NULL;
  cl_command_queue *queues = NULL;
  cl_uint num_devices = 0;
  int status;

  int err = poclu_get_multiple_devices (&platform, &context, 0, &num_devices,
                                        &devices, &queues, 0);
  CHECK_OPENCL_ERROR_IN ("poclu_get_multiple_devices");

  clCreateProgramWithDefinedBuiltInKernelsEXP_fn createProgramWithDBKs;
  createProgramWithDBKs = (clCreateProgramWithDefinedBuiltInKernelsEXP_fn)
    clGetExtensionFunctionAddressForPlatform (
      platform, "clCreateProgramWithDefinedBuiltInKernelsEXP");
  TEST_ASSERT (createProgramWithDBKs != NULL);
  cl_dbk_id_exp dbk_ids[] = { CL_DBK_IMG_COLOR_CONVERT_EXP };
  const char *kernel_names[] = { "exp_img_color_convert" };

  pocl_image_attr_t input_attrs
    = { width, height, POCL_COLOR_SPACE_BT709, POCL_CHANNEL_RANGE_FULL,
        POCL_DF_IMAGE_NV12 };
  pocl_image_attr_t output_attrs = input_attrs;
  output_attrs.format = POCL_DF_IMAGE_RGB;
  cl_dbk_attributes_img_color_convert_exp convert_attrs
    = { input_attrs, output_attrs };
  const void *attributes[] = { &convert_attrs };
  cl_int device_support[] = { 0 };
  cl_program program = createProgramWithDBKs (
    context, num_devices, devices, 1, dbk_ids, kernel_names, attributes,
    device_support, &status);
  TEST_ASSERT (status == CL_SUCCESS);

  clBuildProgram (program, num_devices, devices, NULL, NULL, NULL);
  TEST_ASSERT (status == CL_SUCCESS);

  /* setup kernel */

  cl_mem input_buf
    = clCreateBuffer (context, CL_MEM_READ_ONLY, input_size, NULL, &status);
  cl_mem output_buf
    = clCreateBuffer (context, CL_MEM_READ_WRITE, rgb_size, NULL, &status);

  cl_kernel convert_kernel
    = clCreateKernel (program, kernel_names[0], &status);
  TEST_ASSERT (status == CL_SUCCESS);

  status = clSetKernelArg (convert_kernel, 0, sizeof (cl_mem), &input_buf);
  status |= clSetKernelArg (convert_kernel, 1, sizeof (cl_mem), &output_buf);
  TEST_ASSERT (status == CL_SUCCESS);

  /* run kernel */
  cl_event write_event;
  clEnqueueWriteBuffer (queues[0], input_buf, CL_TRUE, 0, input_size,
                        input_data, 0, NULL, &write_event);

  size_t global_work_size[] = { 1 };
  cl_event wait_events[] = { write_event };
  size_t wait_event_size = sizeof (wait_events) / sizeof (wait_events[0]);
  cl_event enqueue_event;

  clEnqueueNDRangeKernel (queues[0], convert_kernel, 1, NULL, global_work_size,
                          NULL, wait_event_size, wait_events, &enqueue_event);

  void *output_array = malloc (rgb_size);
  clEnqueueReadBuffer (queues[0], output_buf, CL_TRUE, 0, rgb_size,
                       output_array, 1, &enqueue_event, NULL);

  if (argc > 5)
    poclu_write_binfile (argv[5], output_array, rgb_size);

  cl_event all_events[] = { enqueue_event, write_event };
  size_t all_events_size = sizeof (all_events) / sizeof (all_events[0]);
  clWaitForEvents (all_events_size, all_events);
  for (size_t i = 0; i < all_events_size; i++)
    {
      clReleaseEvent (all_events[i]);
    }

  void *reference_data = malloc (rgb_size);
  reference_data = poclu_read_binfile (argv[4], &bytes_read);
  TEST_ASSERT (rgb_size == bytes_read);
  double psnr = calculate_PSNR (width, height, 3, (uint8_t *)output_array,
                                reference_data);
  TEST_ASSERT (psnr > 25);

  free (input_data);
  free (output_array);
  free (reference_data);
  clReleaseMemObject (input_buf);
  clReleaseMemObject (output_buf);
  clReleaseKernel (convert_kernel);
  clReleaseProgram (program);
  clReleaseContext (context);
  for (cl_uint i = 0; i < num_devices; i++)
    {
      clReleaseDevice (devices[i]);
      clReleaseCommandQueue (queues[i]);
    }
}