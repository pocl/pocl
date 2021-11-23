/* Test clEnqueueCopy{BufferToImage,ImageToBuffer}

   Copyright (C) 2016 Giuseppe Bilotta <giuseppe.bilotta@gmail.com>

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

#include "poclu.h"

#define MAX_PLATFORMS 32
#define MAX_DEVICES   32

/* we will manipulate images with 4 16-bit channels, which are manadated
 * as necessarily supported by the standard */
static const cl_image_format img_format = {
  .image_channel_order = CL_RGBA,
  .image_channel_data_type = CL_UNSIGNED_INT16
};

static const size_t pixel_size = 4*sizeof(cl_ushort);
#define CHANNEL_MAX 0xFFFFU

int
main(void)
{
  cl_int err;
  cl_platform_id platforms[MAX_PLATFORMS];
  cl_uint nplatforms;
  cl_device_id devices[MAX_DEVICES];
  cl_uint ndevices;
  cl_uint i, j;
  size_t el, row, col;
  cl_context context;
  cl_command_queue queue;
  cl_mem buf, img;

  CHECK_CL_ERROR(clGetPlatformIDs(MAX_PLATFORMS, platforms, &nplatforms));

  for (i = 0; i < nplatforms; i++)
  {
    CHECK_CL_ERROR(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, MAX_DEVICES,
      devices, &ndevices));

    /* Only test the devices we actually have room for */
    if (ndevices > MAX_DEVICES)
      ndevices = MAX_DEVICES;

    for (j = 0; j < ndevices; j++)
    {
      /* skip devices that do not support images */
      cl_bool has_img;
      CHECK_CL_ERROR(clGetDeviceInfo(devices[j], CL_DEVICE_IMAGE_SUPPORT, sizeof(has_img), &has_img, NULL));
      if (!has_img)
        continue;

      context = clCreateContext (NULL, 1, &devices[j], NULL, NULL, &err);
      CHECK_OPENCL_ERROR_IN("clCreateContext");
      queue = clCreateCommandQueue (context, devices[j], 0, &err);
      CHECK_OPENCL_ERROR_IN("clCreateCommandQueue");

      cl_ulong alloc;
      size_t max_height;
      size_t max_width;
#define MAXALLOC (1024U*1024U)

      CHECK_CL_ERROR(clGetDeviceInfo(devices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE,
          sizeof(alloc), &alloc, NULL));
      CHECK_CL_ERROR(clGetDeviceInfo(devices[j], CL_DEVICE_IMAGE2D_MAX_WIDTH,
          sizeof(max_width), &max_width, NULL));
      CHECK_CL_ERROR(clGetDeviceInfo(devices[j], CL_DEVICE_IMAGE2D_MAX_HEIGHT,
          sizeof(max_height), &max_height, NULL));


      while (alloc > MAXALLOC)
        alloc /= 2;

      // fit at least one max_width inside the alloc (shrink max_width for this)
      while (max_width*pixel_size > alloc)
        max_width /= 2;

      // round number of elements to next multiple of max_width elements
      const size_t nels = (alloc/pixel_size/max_width)*max_width;
      const size_t buf_size = nels*pixel_size;

      cl_image_desc img_desc;
      memset(&img_desc, 0, sizeof(img_desc));
      img_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
      img_desc.image_width = max_width;
      img_desc.image_height = nels/max_width;
      img_desc.image_depth = 1;

      cl_ushort null_pixel[4] = {0, 0, 0, 0};
      cl_ushort *host_buf = malloc(buf_size);
      TEST_ASSERT(host_buf);

      for (el = 0; el < nels; el+=4) {
        host_buf[el] = el & CHANNEL_MAX;
        host_buf[el+1] = (CHANNEL_MAX - el) & CHANNEL_MAX;
        host_buf[el+2] = (CHANNEL_MAX/((el & 1) + 1) - el) & CHANNEL_MAX;
        host_buf[el+3] = (CHANNEL_MAX - el/((el & 1) + 1)) & CHANNEL_MAX;
      }

      buf = clCreateBuffer (context, CL_MEM_READ_WRITE, buf_size, NULL, &err);
      CHECK_OPENCL_ERROR_IN("clCreateBuffer");
      img = clCreateImage (context, CL_MEM_READ_WRITE, &img_format, &img_desc,
                           NULL, &err);
      CHECK_OPENCL_ERROR_IN("clCreateImage");

      CHECK_CL_ERROR(clEnqueueWriteBuffer(queue, buf, CL_TRUE, 0, buf_size, host_buf, 0, NULL, NULL));

      const size_t offset = 0;
      const size_t origin[] = {0, 0, 0};
      const size_t region[] = {img_desc.image_width, img_desc.image_height, 1};

      CHECK_CL_ERROR(clEnqueueCopyBufferToImage(queue, buf, img, offset, origin, region, 0, NULL, NULL));

      size_t row_pitch, slice_pitch;
      cl_ushort *img_map = clEnqueueMapImage(queue, img, CL_TRUE, CL_MAP_READ, origin, region,
        &row_pitch, &slice_pitch, 0, NULL, NULL, &err);
      CHECK_OPENCL_ERROR_IN("clEnqueueMapImage");

      CHECK_CL_ERROR(clFinish(queue));

      for (row = 0; row < img_desc.image_height; ++row) {
        for (col = 0; col < img_desc.image_width; ++col) {
          cl_ushort *img_pixel = (cl_ushort*)((char*)img_map + row*row_pitch) + col*4;
          cl_ushort *buf_pixel = host_buf + (row*img_desc.image_width + col)*4;

          if (memcmp(img_pixel, buf_pixel, pixel_size) != 0)
            printf("%zu %zu %zu : %x %x %x %x | %x %x %x %x\n",
              row, col, (size_t)(buf_pixel - host_buf),
              buf_pixel[0],
              buf_pixel[1],
              buf_pixel[2],
              buf_pixel[3],
              img_pixel[0],
              img_pixel[1],
              img_pixel[2],
              img_pixel[3]);

          TEST_ASSERT(memcmp(img_pixel, buf_pixel, pixel_size) == 0);

        }
      }

      CHECK_CL_ERROR(clEnqueueUnmapMemObject(queue, img, img_map, 0, NULL, NULL));

      /* Clear the buffer, and ensure it has been cleared */
      CHECK_CL_ERROR(clEnqueueFillBuffer(queue, buf, null_pixel, sizeof(null_pixel), 0, buf_size, 0, NULL, NULL));
      cl_ushort *buf_map = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_READ, 0, buf_size, 0, NULL, NULL, &err);
      CHECK_OPENCL_ERROR_IN("clEnqueueMapBuffer");

      CHECK_CL_ERROR(clFinish(queue));

      for (el = 0; el < nels; ++el) {
#if 0 // debug
        if (buf_map[el] != 0) {
          printf("%zu/%zu => %u\n", el, nels, buf_map[el]);
        }
#endif
        TEST_ASSERT(buf_map[el] == 0);
      }

      CHECK_CL_ERROR(clEnqueueUnmapMemObject(queue, buf, buf_map, 0, NULL, NULL));

      /* Copy data from image to buffer, and check that it's again equal to the original buffer */
      CHECK_CL_ERROR(clEnqueueCopyImageToBuffer(queue, img, buf, origin, region, offset, 0, NULL, NULL));
      buf_map = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_READ, 0, buf_size, 0, NULL, NULL, &err);
      CHECK_CL_ERROR(clFinish(queue));

      TEST_ASSERT(memcmp(buf_map, host_buf, buf_size) == 0);

      CHECK_CL_ERROR (
          clEnqueueUnmapMemObject (queue, buf, buf_map, 0, NULL, NULL));
      CHECK_CL_ERROR (clFinish (queue));

      free(host_buf);
      CHECK_CL_ERROR (clReleaseMemObject (img));
      CHECK_CL_ERROR (clReleaseMemObject (buf));
      CHECK_CL_ERROR (clReleaseCommandQueue (queue));
      CHECK_CL_ERROR (clReleaseContext (context));
    }
  }

  CHECK_CL_ERROR (clUnloadCompiler ());

  printf ("OK\n");
  return EXIT_SUCCESS;
}
