/* Test basic cl_khr_command_buffer image-related functions

   Copyright (c) 2022 Jan Solanti / Tampere University

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

#include <stdio.h>
#include <string.h>

#include "poclu.h"

#define STR(x) #x

int
main (int _argc, char **_argv)
{
#if defined(cl_khr_command_buffer) && cl_khr_command_buffer == 1
  struct
  {
    clCreateCommandBufferKHR_fn clCreateCommandBufferKHR;
    clCommandCopyBufferToImageKHR_fn clCommandCopyBufferToImageKHR;
    clCommandCopyImageToBufferKHR_fn clCommandCopyImageToBufferKHR;
    clCommandCopyImageKHR_fn clCommandCopyImageKHR;
    clCommandFillImageKHR_fn clCommandFillImageKHR;
    clFinalizeCommandBufferKHR_fn clFinalizeCommandBufferKHR;
    clEnqueueCommandBufferKHR_fn clEnqueueCommandBufferKHR;
    clReleaseCommandBufferKHR_fn clReleaseCommandBufferKHR;
    clGetCommandBufferInfoKHR_fn clGetCommandBufferInfoKHR;
  } ext;

  cl_platform_id platform;
  CHECK_CL_ERROR (clGetPlatformIDs (1, &platform, NULL));
  cl_device_id device;
  CHECK_CL_ERROR (
      clGetDeviceIDs (platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL));

  ext.clCreateCommandBufferKHR = clGetExtensionFunctionAddressForPlatform (
      platform, "clCreateCommandBufferKHR");
  ext.clCommandCopyBufferToImageKHR
      = clGetExtensionFunctionAddressForPlatform (
          platform, "clCommandCopyBufferToImageKHR");
  ext.clCommandCopyImageToBufferKHR
      = clGetExtensionFunctionAddressForPlatform (
          platform, "clCommandCopyImageToBufferKHR");
  ext.clCommandCopyImageKHR = clGetExtensionFunctionAddressForPlatform (
      platform, "clCommandCopyImageKHR");
  ext.clCommandFillImageKHR = clGetExtensionFunctionAddressForPlatform (
      platform, "clCommandFillImageKHR");
  ext.clFinalizeCommandBufferKHR = clGetExtensionFunctionAddressForPlatform (
      platform, "clFinalizeCommandBufferKHR");
  ext.clEnqueueCommandBufferKHR = clGetExtensionFunctionAddressForPlatform (
      platform, "clEnqueueCommandBufferKHR");
  ext.clReleaseCommandBufferKHR = clGetExtensionFunctionAddressForPlatform (
      platform, "clReleaseCommandBufferKHR");
  ext.clGetCommandBufferInfoKHR = clGetExtensionFunctionAddressForPlatform (
      platform, "clGetCommandBufferInfoKHR");

  cl_int error;
  cl_context context = clCreateContext (NULL, 1, &device, NULL, NULL, &error);
  CHECK_CL_ERROR (error);

  size_t img_width = 8;
  size_t img_height = 8;
  size_t img_depth = 8;
  cl_image_format img_format;
  img_format.image_channel_order = CL_RGBA;
  img_format.image_channel_data_type = CL_UNSIGNED_INT8;
  size_t img_channels = 4;
  size_t img_bpp = sizeof (uint8_t) * img_channels;
  size_t img_npixels = img_width * img_height * img_depth;
  size_t buf_size = img_bpp * img_npixels;

  const unsigned fill_pixel[4] = { 127, 64, 1, 255 };
  const unsigned copy_pixel[4] = { 0xF0, 0x0F, 0x66, 0x55 };
  size_t img2img_origin[3]
      = { img_width / 2 - 1, img_height / 2 - 1, img_depth / 2 - 1 };
  size_t img2img_region[3] = { 2, 2, 2 };

  cl_image_desc img_desc;
  img_desc.image_type = CL_MEM_OBJECT_IMAGE3D;
  img_desc.image_width = img_width;
  img_desc.image_height = img_height;
  img_desc.image_depth = img_depth;
  img_desc.image_array_size = 0;
  img_desc.image_row_pitch = 0;
  img_desc.image_slice_pitch = 0;
  img_desc.num_mip_levels = 0;
  img_desc.num_samples = 0;
  img_desc.mem_object = NULL;

  cl_mem buffer
      = clCreateBuffer (context, CL_MEM_READ_ONLY, buf_size, NULL, &error);
  CHECK_CL_ERROR (error);
  cl_mem img1 = clCreateImage (context, CL_MEM_READ_ONLY, &img_format,
                               &img_desc, NULL, &error);
  CHECK_CL_ERROR (error);
  cl_mem img2 = clCreateImage (context, CL_MEM_READ_ONLY, &img_format,
                               &img_desc, NULL, &error);
  CHECK_CL_ERROR (error);

  cl_command_queue command_queue = clCreateCommandQueue (
      context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &error);
  CHECK_CL_ERROR (error);

  /**** Command buffer creation ****/
  cl_command_buffer_khr command_buffer
      = ext.clCreateCommandBufferKHR (1, &command_queue, NULL, &error);
  CHECK_CL_ERROR (error);

  {
    cl_sync_point_khr fill_syncpt;
    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { img_width, img_height, img_depth };
    CHECK_CL_ERROR (ext.clCommandFillImageKHR (command_buffer, NULL, img1,
                                               fill_pixel, origin, region, 0,
                                               NULL, &fill_syncpt, NULL));
    cl_sync_point_khr buf2img_syncpt;
    CHECK_CL_ERROR (ext.clCommandCopyBufferToImageKHR (
        command_buffer, NULL, buffer, img2, 0, origin, region, 0, NULL,
        &buf2img_syncpt, NULL));
    cl_sync_point_khr img2img_syncpt;
    cl_sync_point_khr img2img_deps[2] = { fill_syncpt, buf2img_syncpt };
    CHECK_CL_ERROR (ext.clCommandCopyImageKHR (
        command_buffer, NULL, img2, img1, img2img_origin, img2img_origin,
        img2img_region, 2, img2img_deps, &img2img_syncpt, NULL));

    CHECK_CL_ERROR (ext.clCommandCopyImageToBufferKHR (
        command_buffer, NULL, img1, buffer, origin, region, 0, 1,
        &img2img_syncpt, NULL, NULL));
  }

  CHECK_CL_ERROR (ext.clFinalizeCommandBufferKHR (command_buffer));

  cl_command_buffer_state_khr cmdbuf_state;
  CHECK_CL_ERROR (ext.clGetCommandBufferInfoKHR (
      command_buffer, CL_COMMAND_BUFFER_STATE_KHR,
      sizeof (cl_command_buffer_state_khr), &cmdbuf_state, NULL));
  TEST_ASSERT (cmdbuf_state == CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR);
  /*** Command buffer is ready ***/

  /*** Main test ***/
  {
    uint8_t img_src[buf_size];
    for (size_t i = 0; i < img_npixels; ++i)
      {
        for (size_t j = 0; j < img_channels; ++j)
          img_src[4 * i + j] = copy_pixel[j];
      }
    cl_event write_src_event;
    cl_event command_buf_event;
    CHECK_CL_ERROR (clEnqueueWriteBuffer (command_queue, buffer, CL_FALSE, 0,
                                          buf_size, img_src, 0, NULL,
                                          &write_src_event));

    CHECK_CL_ERROR (ext.clEnqueueCommandBufferKHR (
        0, NULL, command_buffer, 1, &write_src_event, &command_buf_event));

    cl_int err;
    uint8_t *buf_map
        = clEnqueueMapBuffer (command_queue, buffer, CL_TRUE, CL_MAP_READ, 0,
                              buf_size, 1, &command_buf_event, NULL, &err);
    CHECK_OPENCL_ERROR_IN ("clEnqueueMapBuffer");

    /*
     * The command buffer first fills img1 with fill_pixel. Then it copies
     * buffer into img2, a small subregion of img2 into img1, and finally
     * the contents of the changed img1 back into buffer.
     *
     * Verify that the end result has fill_pixel everywhere except in the
     * region copied from img2, which should contain copy_pixel.
     * */
    for (size_t i = 0; i < img_npixels; ++i)
      {
        size_t z = i / (img_width * img_height);
        size_t y = (i / img_width) % img_height;
        size_t x = i % img_width;

        if (x >= img2img_origin[0] && x < img2img_origin[0] + img2img_region[0]
            && y >= img2img_origin[1]
            && y < img2img_origin[1] + img2img_region[1]
            && z >= img2img_origin[2]
            && z < img2img_origin[2] + img2img_region[2])
          {
            for (size_t j = 0; j < img_channels; ++j)
              TEST_ASSERT (buf_map[4 * i + j] == copy_pixel[j]);
          }
        else
          {
            for (size_t j = 0; j < img_channels; ++j)
              TEST_ASSERT (buf_map[4 * i + j] == fill_pixel[j]);
          }
      }
    CHECK_CL_ERROR (clEnqueueUnmapMemObject (command_queue, buffer, buf_map, 0,
                                             NULL, NULL));

    CHECK_CL_ERROR (clReleaseEvent (write_src_event));
    CHECK_CL_ERROR (clReleaseEvent (command_buf_event));
  }

  CHECK_CL_ERROR (ext.clReleaseCommandBufferKHR (command_buffer));
  CHECK_CL_ERROR (clReleaseCommandQueue (command_queue));

  CHECK_CL_ERROR (clReleaseMemObject (img1));
  CHECK_CL_ERROR (clReleaseMemObject (img2));
  CHECK_CL_ERROR (clReleaseMemObject (buffer));

  CHECK_CL_ERROR (clReleaseContext (context));

  CHECK_CL_ERROR (clUnloadCompiler ());

  printf ("OK\n");
  return EXIT_SUCCESS;
#else
  return 77;
#endif
}
