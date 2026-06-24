/* Test float-to-normalized-integer rounding behavior for images.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <CL/opencl.h>
#include "poclu.h"

static int
test_format(cl_context context, cl_command_queue queue,
            cl_channel_type data_type, size_t type_size,
            cl_float color_val, int expected_val)
{
  cl_int err;
  cl_image_format format = {
    .image_channel_order = CL_R,
    .image_channel_data_type = data_type
  };

  cl_image_desc desc = {
    .image_type = CL_MEM_OBJECT_IMAGE2D,
    .image_width = 1,
    .image_height = 1,
  };

  cl_mem img = clCreateImage (context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
  if (err == CL_INVALID_OPERATION || err == CL_INVALID_IMAGE_FORMAT_DESCRIPTOR || err == CL_IMAGE_FORMAT_NOT_SUPPORTED)
    {
      printf ("Format 0x%X not supported (error %d), skipping test. SKIP\n", data_type, err);
      return 77;
    }
  CHECK_OPENCL_ERROR_IN ("clCreateImage");

  cl_float color[4] = { color_val, 0.0f, 0.0f, 0.0f };
  size_t origin[3] = { 0, 0, 0 };
  size_t region[3] = { 1, 1, 1 };

  err = clEnqueueFillImage (queue, img, color, origin, region, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN ("clEnqueueFillImage");

  err = clFinish (queue);
  CHECK_OPENCL_ERROR_IN ("clFinish");

  // Read back
  long long readback = 0;
  err = clEnqueueReadImage (queue, img, CL_TRUE, origin, region, 0, 0, &readback, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN ("clEnqueueReadImage");

  long long masked_val = 0;
  if (type_size == 1)
    {
      if (data_type == CL_SNORM_INT8)
        masked_val = *(char *)&readback;
      else
        masked_val = *(unsigned char *)&readback;
    }
  else if (type_size == 2)
    {
      if (data_type == CL_SNORM_INT16)
        masked_val = *(short *)&readback;
      else
        masked_val = *(unsigned short *)&readback;
    }

  printf ("Format 0x%X (input: %f) readback value: %lld (expected %d)\n", data_type, color_val, masked_val, expected_val);
  TEST_ASSERT (masked_val == expected_val);

  CHECK_CL_ERROR (clReleaseMemObject (img));
  return EXIT_SUCCESS;
}

int
main(void)
{
  cl_int err;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;
  cl_platform_id platform = NULL;

  CHECK_CL_ERROR (
    poclu_get_any_device2 (&context, &device, &queue, &platform));
  TEST_ASSERT (context);
  TEST_ASSERT (device);
  TEST_ASSERT (queue);

  cl_bool SupportsImgs = CL_FALSE;
  err = clGetDeviceInfo (device, CL_DEVICE_IMAGE_SUPPORT, sizeof (cl_bool),
                         &SupportsImgs, NULL);
  CHECK_OPENCL_ERROR_IN ("clGetDeviceInfo CL_DEVICE_IMAGE_SUPPORT\n");
  if (SupportsImgs == CL_FALSE)
    {
      puts ("Selected device doesn't support images, skipping test. SKIP");
      return 77;
    }

#define RUN_TEST_FORMAT(data_type, type_size, color_val, expected_val) \
  do { \
    int res = test_format (context, queue, data_type, type_size, color_val, expected_val); \
    if (res == 77) return 77; \
    TEST_ASSERT (res == EXIT_SUCCESS); \
  } while (0)

  // Test all normalized image formats supported by OpenCL (signed/unsigned 8 and 16-bit integers)
  // For each format, we test:
  // 1. Case A: rounds UP to nearest even value (halfway from odd value)
  // 2. Case B: rounds DOWN to nearest even value (halfway from even value)

  // CL_SNORM_INT8 (max 127)
  RUN_TEST_FORMAT (CL_SNORM_INT8, 1, 63.5f / 127.0f, 64);
  RUN_TEST_FORMAT (CL_SNORM_INT8, 1, 62.5f / 127.0f, 62);

  // CL_UNORM_INT8 (max 255)
  RUN_TEST_FORMAT (CL_UNORM_INT8, 1, 127.5f / 255.0f, 128);
  RUN_TEST_FORMAT (CL_UNORM_INT8, 1, 126.5f / 255.0f, 126);

  // CL_SNORM_INT16 (max 32767)
  RUN_TEST_FORMAT (CL_SNORM_INT16, 2, 16383.5f / 32767.0f, 16384);
  RUN_TEST_FORMAT (CL_SNORM_INT16, 2, 16382.5f / 32767.0f, 16382);

  // CL_UNORM_INT16 (max 65535)
  RUN_TEST_FORMAT (CL_UNORM_INT16, 2, 32767.5f / 65535.0f, 32768);
  RUN_TEST_FORMAT (CL_UNORM_INT16, 2, 32766.5f / 65535.0f, 32766);

#undef RUN_TEST_FORMAT

  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseContext (context));

  printf ("OK\n");
  return EXIT_SUCCESS;
}
