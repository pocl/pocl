/* Test that the associated event to a failing command isn't unduly freed

   Copyright (C) 2015 Giuseppe Bilotta <giuseppe.bilotta@gmail.com>

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

#include "pocl_opencl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
  cl_int err;
  cl_context ctx;
  cl_command_queue queue;
  cl_device_id did;
  cl_bool has_img = CL_FALSE;

  const size_t buf_size = sizeof(cl_int);

  cl_mem buf, img;



  poclu_get_any_device(&ctx, &did, &queue);
  TEST_ASSERT(ctx);
  TEST_ASSERT(did);
  TEST_ASSERT(queue);

  CHECK_CL_ERROR (clGetDeviceInfo (did, CL_DEVICE_IMAGE_SUPPORT, sizeof(has_img), &has_img, NULL));

  buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, buf_size, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer");

  if (has_img) {
    cl_image_format img_fmt = {
      .image_channel_order = CL_RGBA,
      .image_channel_data_type = CL_UNSIGNED_INT32 };
    cl_image_desc img_dsc = {
      .image_type = CL_MEM_OBJECT_IMAGE2D,
      .image_width = 1,
      .image_height = 1,
      .image_depth = 1,
      .image_array_size = 1,
      .image_row_pitch = 0,
      .image_slice_pitch = 0,
      .num_mip_levels = 0,
      .num_samples = 0,
      .buffer = NULL,
    };

    img = clCreateImage(ctx, CL_MEM_READ_WRITE,
      &img_fmt, &img_dsc, NULL, &err);
    CHECK_OPENCL_ERROR_IN("clCreateImage");
  }

  /* An invalid waiting list (e.g. a null event in it) should make
   * associated commands fail without segfaults and without touching any associated
   * event. Test that this is indeed the case.
   */

  cl_int *host_ptr = NULL;
  cl_event no_event = NULL;

  /* We will test both NULL and invalid initial value for map_event */
  cl_event initial_values[] = { NULL, (cl_event)1 };
  cl_uint i = 0;

  /***
   * Test buffer mapping/unmapping
   */

  /* Test without associated event */
  host_ptr = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_READ, 0, buf_size,
    1, &no_event, NULL, &err);
  TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);
  TEST_ASSERT(host_ptr == NULL);

  /* Test with map_event = NULL */
  cl_event map_event = NULL;
  host_ptr = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_READ, 0, buf_size,
    1, &no_event, &map_event, &err);
  TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);
  TEST_ASSERT(host_ptr == NULL);
  TEST_ASSERT(map_event == NULL); /* should not have been touched */

  /* Now do an actual mapping to test the unmapping */
  host_ptr = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_READ, 0, buf_size,
    0, NULL, NULL, &err);
  CHECK_OPENCL_ERROR_IN("map buffer");

  err = clEnqueueUnmapMemObject(queue, buf, host_ptr, 1, &no_event, NULL);
  TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);
  err = clEnqueueUnmapMemObject(queue, buf, host_ptr, 1, &no_event, &map_event);
  TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);
  TEST_ASSERT(map_event == NULL); /* should not have been touched */

  /* Actually unmap */
  err = clEnqueueUnmapMemObject(queue, buf, host_ptr, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("unmap buffer");
  host_ptr = NULL;

  /* Test with map_event != NULL but invalid */
  map_event = (cl_event)1;
  host_ptr = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_READ, 0, buf_size,
    1, &no_event, &map_event, &err);
  TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);
  TEST_ASSERT(host_ptr == NULL);
  TEST_ASSERT(map_event == (cl_event)1); /* should not have been touched */

  host_ptr = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_READ, 0, buf_size,
    0, NULL, NULL, &err);
  CHECK_OPENCL_ERROR_IN("map buffer");

  err = clEnqueueUnmapMemObject(queue, buf, host_ptr, 1, &no_event, &map_event);
  TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);
  TEST_ASSERT(map_event == (cl_event)1); /* should not have been touched */

  err = clEnqueueUnmapMemObject(queue, buf, host_ptr, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("unmap buffer");
  host_ptr = NULL;

  /***
   * Test image commands
   */

  if (has_img) {
    cl_int color = 0;
    const size_t origin[] = {0, 0, 0};
    const size_t region[] = {1, 1, 1};

    /* First, no associated event */
    err = clEnqueueFillImage(queue, img, &color, origin, region,
      1, &no_event, NULL);
    TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);
    err = clEnqueueWriteImage(queue, img, CL_TRUE, origin, region,
      0, 0, &color,
      1, &no_event, NULL);
    TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);
    err = clEnqueueWriteImage(queue, img, CL_TRUE, origin, region,
      0, 0, &color,
      1, &no_event, NULL);
    TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);
    err = clEnqueueReadImage(queue, img, CL_TRUE, origin, region,
      0, 0, &color,
      1, &no_event, NULL);
    TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);

    size_t img_pitch = 0;
    host_ptr = clEnqueueMapImage(queue, img, CL_TRUE, CL_MAP_READ,
      origin, region, &img_pitch, NULL,
      1, &no_event, NULL, &err);
    TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);
    TEST_ASSERT(host_ptr == NULL);

    /* No need to test Unmap, it's the same API call as for buffers */

    /* Next, associated event with initial NULL value and with initial
     * invalid value */

    for (i = 0; i < 2; ++i) {
      cl_event initial_value = initial_values[i];
      map_event = initial_value;

      err = clEnqueueFillImage(queue, img, &color, origin, region,
        1, &no_event, &map_event);
      TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);
      TEST_ASSERT(map_event == initial_value);
      err = clEnqueueWriteImage(queue, img, CL_TRUE, origin, region,
        0, 0, &color,
        1, &no_event, &map_event);
      TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);
      TEST_ASSERT(map_event == initial_value);
      err = clEnqueueWriteImage(queue, img, CL_TRUE, origin, region,
        0, 0, &color,
        1, &no_event, &map_event);
      TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);
      TEST_ASSERT(map_event == initial_value);
      err = clEnqueueReadImage(queue, img, CL_TRUE, origin, region,
        0, 0, &color,
        1, &no_event, &map_event);
      TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);
      TEST_ASSERT(map_event == initial_value);
      host_ptr = clEnqueueMapImage(queue, img, CL_TRUE, CL_MAP_READ,
        origin, region, &img_pitch, NULL,
        1, &no_event, &map_event, &err);
      TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);
      TEST_ASSERT(host_ptr == NULL);
      TEST_ASSERT(map_event == initial_value);

    }
  }

  err = clEnqueueMarkerWithWaitList(queue, 1, &no_event, NULL);
  TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);
  for (i = 0; i < 2; ++i) {
    cl_event initial_value = initial_values[i];
    map_event = initial_value;
    err = clEnqueueMarkerWithWaitList(queue, 1, &no_event, &map_event);
    TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);
    TEST_ASSERT(map_event == initial_value);
  }

  clFinish (queue);
  if (has_img)
    clReleaseMemObject (img);
  clReleaseMemObject (buf);

  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseContext (ctx));

  CHECK_CL_ERROR (clUnloadCompiler ());

  printf ("OK\n");
  return EXIT_SUCCESS;
}


