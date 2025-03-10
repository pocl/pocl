/* test_dbk_jpeg.c - test program for JPEG DBKs.

   Copyright (c) 2024 Robin Bijl / Tampere University

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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "image_test_utils.h"
#include "poclu.h"

/* enable this and provide an output file name to the program
   to save compression results */
/* #define WRITE_COMPRESS_OUTPUT */

/**
 * Program arguments order:
 * 1. width
 * 2. height
 * 3. source image path
 * 4. (optional) output location of image
 *
 * Arguments to use with test data:
 * 640 480 <abs path to tram.rgb>
 */

#define BUILTIN_KERNELS_STR_LEN 32768

int
main (int argc, char const *argv[])
{

  TEST_ASSERT (argc >= 4);

  errno = 0;
  char *end_ptr;
  int width = (int)strtol (argv[1], &end_ptr, 10);
  TEST_ASSERT (errno == 0 && *end_ptr == '\0');
  int height = (int)strtol (argv[2], &end_ptr, 10);
  TEST_ASSERT (errno == 0 && *end_ptr == '\0');

  const char *file_name = argv[3];
  size_t bytes_read = 0;
  char *input_data = poclu_read_binfile (file_name, &bytes_read);
  size_t input_size = height * width * 3;
  TEST_ASSERT (bytes_read == input_size);

  cl_platform_id platform = NULL;
  cl_context context = NULL;
  cl_device_id *devices = NULL;
  cl_command_queue *queues = NULL;
  cl_uint num_devices = 0;
  int status;

  int err = poclu_get_multiple_devices (&platform, &context, 0, &num_devices,
                                        &devices, &queues, 0);
  CHECK_OPENCL_ERROR_IN ("poclu_get_multiple_devices");

  char builtin_list[BUILTIN_KERNELS_STR_LEN];

  for (cl_uint i = 0; i < num_devices; ++i) {
    size_t size_ret = 0;
    int err = clGetDeviceInfo(devices[i], CL_DEVICE_BUILT_IN_KERNELS, 0, 0,
                             &size_ret);
    CHECK_OPENCL_ERROR_IN ("clGetDeviceInfo");
    TEST_ASSERT(size_ret < BUILTIN_KERNELS_STR_LEN);
    builtin_list[0] = 0;
    err = clGetDeviceInfo(devices[i], CL_DEVICE_BUILT_IN_KERNELS,
                          size_ret, builtin_list, 0 );
    CHECK_OPENCL_ERROR_IN ("clGetDeviceInfo");
    if ((strstr(builtin_list, "jpeg_encode") == NULL)
        || (strstr(builtin_list, "jpeg_decode") == NULL))
        {
           printf("one of the devices does not support jpeg_encode "
                  "or jpeg_decode DBK, skipping test\n");
           return 77;
        }
  }

  clCreateProgramWithDefinedBuiltInKernelsEXP_fn createProgramWithDBKs;
  createProgramWithDBKs = (clCreateProgramWithDefinedBuiltInKernelsEXP_fn)
    clGetExtensionFunctionAddressForPlatform (
      platform, "clCreateProgramWithDefinedBuiltInKernelsEXP");
  TEST_ASSERT (createProgramWithDBKs != NULL);
  cl_dbk_id_exp dbk_ids[] = { CL_DBK_JPEG_ENCODE_EXP, CL_DBK_JPEG_DECODE_EXP };
  const char *kernel_names[] = { "jpeg_encode_exp", "jpeg_decode_exp" };
  cl_dbk_attributes_jpeg_encode_exp encode_attributes = { width, height, 80 };
  const void *attributes[] = { &encode_attributes, NULL };
  cl_int device_support[] = { 0, 0 };
  cl_program program = createProgramWithDBKs (
    context, num_devices, devices, 2, dbk_ids, kernel_names, attributes,
    device_support, &status);
  TEST_ASSERT (status == CL_SUCCESS);

  clBuildProgram (program, num_devices, devices, NULL, NULL, NULL);
  TEST_ASSERT (status == CL_SUCCESS);

  /* setup kernel */

  cl_mem input_buf
    = clCreateBuffer (context, CL_MEM_READ_ONLY, input_size, NULL, &status);
  cl_mem output_buf
    = clCreateBuffer (context, CL_MEM_READ_WRITE, input_size, NULL, &status);
  cl_mem output_size_buf = clCreateBuffer (context, CL_MEM_READ_WRITE,
                                           sizeof (size_t), NULL, &status);

  cl_kernel encode_kernel = clCreateKernel (program, kernel_names[0], &status);
  TEST_ASSERT (status == CL_SUCCESS);

  status = clSetKernelArg (encode_kernel, 0, sizeof (cl_mem), &input_buf);
  status |= clSetKernelArg (encode_kernel, 1, sizeof (cl_mem), &output_buf);
  status
    |= clSetKernelArg (encode_kernel, 2, sizeof (cl_mem), &output_size_buf);
  TEST_ASSERT (status == CL_SUCCESS);

  cl_mem decode_out_buf
    = clCreateBuffer (context, CL_MEM_WRITE_ONLY, input_size, NULL, &status);

  cl_kernel decode_kernel = clCreateKernel (program, kernel_names[1], &status);
  TEST_ASSERT (status == CL_SUCCESS);

  status = clSetKernelArg (decode_kernel, 0, sizeof (cl_mem), &output_buf);
  status
    |= clSetKernelArg (decode_kernel, 1, sizeof (cl_mem), &output_size_buf);
  status
    |= clSetKernelArg (decode_kernel, 2, sizeof (cl_mem), &decode_out_buf);
  TEST_ASSERT (status == CL_SUCCESS);

  /* run kernel */
  cl_event write_event, clear_event;
  clEnqueueWriteBuffer (queues[0], input_buf, CL_TRUE, 0, input_size,
                        input_data, 0, NULL, &write_event);

  /* set buffer content to zero to check later for changes */
  size_t jpeg_size_value = 0;
  clEnqueueWriteBuffer (queues[0], output_size_buf, CL_TRUE, 0,
                        sizeof (size_t), &jpeg_size_value, 0, NULL,
                        &clear_event);
  size_t global_work_size[] = { 1 };
  cl_event wait_events[] = { write_event, clear_event };
  size_t wait_event_size = sizeof (wait_events) / sizeof (wait_events[0]);
  cl_event enqueue_event, decode_event;

  clEnqueueNDRangeKernel (queues[0], encode_kernel, 1, NULL, global_work_size,
                          NULL, wait_event_size, wait_events, &enqueue_event);

  clEnqueueNDRangeKernel (queues[0], decode_kernel, 1, NULL, global_work_size,
                          NULL, 1, &enqueue_event, &decode_event);

  cl_event size_read_event;
  clEnqueueReadBuffer (queues[0], output_size_buf, CL_TRUE, 0, sizeof (size_t),
                       &jpeg_size_value, 1, &decode_event, &size_read_event);

#ifdef WRITE_COMPRESS_OUTPUT
  TEST_ASSERT (argc > 4);
  void *output_array = malloc (input_size);
  clEnqueueReadBuffer (queues[0], output_buf, CL_TRUE, 0, input_size,
                       output_array, 1, &size_read_event, NULL);
  poclu_write_binfile (argv[4], output_array, input_size);
  free (output_array);
#endif

  void *decode_array = malloc (input_size);
  clEnqueueReadBuffer (queues[0], decode_out_buf, CL_TRUE, 0, input_size,
                       decode_array, 1, &decode_event, NULL);

  cl_event all_events[] = { enqueue_event, size_read_event, clear_event,
                            write_event, decode_event };
  size_t all_events_size = sizeof (all_events) / sizeof (all_events[0]);
  clWaitForEvents (all_events_size, all_events);
  for (size_t i = 0; i < all_events_size; i++)
    clReleaseEvent (all_events[i]);

  TEST_ASSERT (jpeg_size_value > 0);

  double psnr
    = calculate_PSNR (width, height, 3, (uint8_t *)input_data, decode_array);
  TEST_ASSERT (psnr > 25);

  free (input_data);
  free (decode_array);
  clReleaseMemObject (input_buf);
  clReleaseMemObject (output_buf);
  clReleaseMemObject (output_size_buf);
  clReleaseMemObject (decode_out_buf);
  clReleaseKernel (encode_kernel);
  clReleaseKernel (decode_kernel);
  clReleaseProgram (program);
  clReleaseContext (context);
  for (cl_uint i = 0; i < num_devices; i++)
    {
      clReleaseDevice (devices[i]);
      clReleaseCommandQueue (queues[i]);
    }
}
