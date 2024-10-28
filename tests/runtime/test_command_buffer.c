/* Test basic cl_khr_command_buffer functionality minus image-related functions

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
#include <stdlib.h>
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
    clCommandCopyBufferKHR_fn clCommandCopyBufferKHR;
    clCommandCopyBufferRectKHR_fn clCommandCopyBufferRectKHR;
    clCommandFillBufferKHR_fn clCommandFillBufferKHR;
    clCommandNDRangeKernelKHR_fn clCommandNDRangeKernelKHR;
    clCommandBarrierWithWaitListKHR_fn clCommandBarrierWithWaitListKHR;
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
  ext.clCommandCopyBufferKHR = clGetExtensionFunctionAddressForPlatform (
      platform, "clCommandCopyBufferKHR");
  ext.clCommandCopyBufferRectKHR = clGetExtensionFunctionAddressForPlatform (
      platform, "clCommandCopyBufferRectKHR");
  ext.clCommandFillBufferKHR = clGetExtensionFunctionAddressForPlatform (
      platform, "clCommandFillBufferKHR");
  ext.clCommandNDRangeKernelKHR = clGetExtensionFunctionAddressForPlatform (
      platform, "clCommandNDRangeKernelKHR");
  ext.clCommandBarrierWithWaitListKHR
      = clGetExtensionFunctionAddressForPlatform (
          platform, "clCommandBarrierWithWaitListKHR");
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

  const char *code = STR (kernel void vector_addition (
      global const int *tile1, global const int *tile2, global int *res) {
    size_t index = get_global_id (0);
    res[index] = tile1[index] + tile2[index];
  });
  const size_t length = strlen (code);

  cl_program program
      = clCreateProgramWithSource (context, 1, &code, &length, &error);
  CHECK_CL_ERROR (error);
  CHECK_CL_ERROR (clBuildProgram (program, 1, &device, NULL, NULL, NULL));
  cl_kernel kernel = clCreateKernel (program, "vector_addition", &error);
  CHECK_CL_ERROR (error);

  size_t frame_count = 60;
  size_t frame_elements = 1024;
  size_t frame_size = frame_elements * sizeof (cl_int);

  size_t tile_count = 16;
  size_t tile_elements = frame_elements / tile_count;
  size_t tile_size = tile_elements * sizeof (cl_int);

  cl_mem buffer_tile1
      = clCreateBuffer (context, CL_MEM_READ_ONLY, tile_size, NULL, &error);
  CHECK_CL_ERROR (error);
  cl_mem buffer_tile2
      = clCreateBuffer (context, CL_MEM_READ_ONLY, tile_size, NULL, &error);
  CHECK_CL_ERROR (error);
  cl_mem buffer_res
      = clCreateBuffer (context, CL_MEM_WRITE_ONLY, tile_size, NULL, &error);
  CHECK_CL_ERROR (error);

  CHECK_CL_ERROR (
      clSetKernelArg (kernel, 0, sizeof (buffer_tile1), &buffer_tile1));
  CHECK_CL_ERROR (
      clSetKernelArg (kernel, 1, sizeof (buffer_tile2), &buffer_tile2));
  CHECK_CL_ERROR (
      clSetKernelArg (kernel, 2, sizeof (buffer_res), &buffer_res));

  cl_command_queue_properties props = 0;
  CHECK_CL_ERROR (clGetDeviceInfo (device, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES,
                                   sizeof (props), &props, NULL));
  if (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
    props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
  cl_command_queue command_queue
    = clCreateCommandQueue (context, device, props, &error);
  CHECK_CL_ERROR (error);

  cl_command_buffer_khr command_buffer
      = ext.clCreateCommandBufferKHR (1, &command_queue, NULL, &error);
  CHECK_CL_ERROR (error);

  cl_mem buffer_src1
      = clCreateBuffer (context, CL_MEM_READ_ONLY, frame_size, NULL, &error);
  CHECK_CL_ERROR (error);
  cl_mem buffer_src2
      = clCreateBuffer (context, CL_MEM_READ_ONLY, frame_size, NULL, &error);
  CHECK_CL_ERROR (error);
  cl_mem buffer_dst
      = clCreateBuffer (context, CL_MEM_READ_WRITE, frame_size, NULL, &error);
  CHECK_CL_ERROR (error);

  cl_sync_point_khr tile_sync_point = 0;
  for (size_t tile_index = 0; tile_index < tile_count; tile_index++)
    {
      cl_sync_point_khr copy_sync_points[2];
      CHECK_CL_ERROR (ext.clCommandCopyBufferKHR (
          command_buffer, NULL, NULL, buffer_src1, buffer_tile1,
          tile_index * tile_size, 0, tile_size, tile_sync_point ? 1 : 0,
          tile_sync_point ? &tile_sync_point : NULL, &copy_sync_points[0],
          NULL));
      /* For the sake of testing, pretend we are working with a vertical stack
       * of 8x8 tiles */
      size_t src_origin[3] = { 0, tile_index * 8, 0 };
      size_t dst_origin[3] = { 0, 0, 0 };
      size_t tile_region[3] = { 8 * sizeof (cl_int), 8, 1 };
      CHECK_CL_ERROR (ext.clCommandCopyBufferRectKHR (
          command_buffer, NULL, NULL, buffer_src2, buffer_tile2, src_origin,
          dst_origin, tile_region, tile_region[0], 0, tile_region[0], 0,
          tile_sync_point ? 1 : 0, tile_sync_point ? &tile_sync_point : NULL,
          &copy_sync_points[1], NULL));

      cl_sync_point_khr nd_sync_point;
      CHECK_CL_ERROR (ext.clCommandNDRangeKernelKHR (
          command_buffer, NULL, NULL, kernel, 1, NULL, &tile_elements, NULL, 2,
          copy_sync_points, &nd_sync_point, NULL));

      cl_sync_point_khr res_copy_sync_point;
      CHECK_CL_ERROR (ext.clCommandCopyBufferKHR (
          command_buffer, NULL, NULL, buffer_res, buffer_dst, 0,
          tile_index * tile_size, tile_size, 1, &nd_sync_point,
          &res_copy_sync_point, NULL));

      char zero = 0;
      cl_sync_point_khr fill_sync_points[2];
      CHECK_CL_ERROR (ext.clCommandFillBufferKHR (
          command_buffer, NULL, NULL, buffer_tile1, &zero, sizeof (zero), 0,
          tile_size, 1, &nd_sync_point, &fill_sync_points[0], NULL));
      CHECK_CL_ERROR (ext.clCommandFillBufferKHR (
          command_buffer, NULL, NULL, buffer_tile2, &zero, sizeof (zero), 0,
          tile_size, 1, &nd_sync_point, &fill_sync_points[1], NULL));

      cl_sync_point_khr barrier_deps[4]
          = { nd_sync_point, res_copy_sync_point, fill_sync_points[0],
              fill_sync_points[1] };
      CHECK_CL_ERROR (ext.clCommandBarrierWithWaitListKHR (
          command_buffer, NULL, NULL, 4, barrier_deps, &tile_sync_point, NULL));
    }

  CHECK_CL_ERROR (ext.clFinalizeCommandBufferKHR (command_buffer));

  cl_command_buffer_state_khr cmdbuf_state;
  CHECK_CL_ERROR (ext.clGetCommandBufferInfoKHR (
      command_buffer, CL_COMMAND_BUFFER_STATE_KHR,
      sizeof (cl_command_buffer_state_khr), &cmdbuf_state, NULL));
  TEST_ASSERT (cmdbuf_state == CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR);

  cl_int *src1 = malloc (frame_elements * sizeof (cl_int));
  cl_int *src2 = malloc (frame_elements * sizeof (cl_int));
  for (size_t frame_index = 0; frame_index < frame_count; frame_index++)
    {
      for (size_t i = 0; i < frame_elements; ++i)
        {
          src1[i] = i + frame_index;
          src2[i] = i + frame_index + 1;
        }
      cl_event write_src_events[2];
      cl_event command_buf_event;
      CHECK_CL_ERROR (clEnqueueWriteBuffer (command_queue, buffer_src1,
                                            CL_FALSE, 0, frame_size, src1, 0,
                                            NULL, &write_src_events[0]));
      CHECK_CL_ERROR (clEnqueueWriteBuffer (command_queue, buffer_src2,
                                            CL_FALSE, 0, frame_size, src2, 0,
                                            NULL, &write_src_events[1]));

      CHECK_CL_ERROR (ext.clEnqueueCommandBufferKHR (
          0, NULL, command_buffer, 2, write_src_events, &command_buf_event));

      cl_int err;
      cl_int *buf_map = clEnqueueMapBuffer (
          command_queue, buffer_dst, CL_TRUE, CL_MAP_READ, 0,
          sizeof (cl_int) * frame_elements, 1, &command_buf_event, NULL, &err);
      CHECK_OPENCL_ERROR_IN ("clEnqueueMapBuffer");

      for (size_t i = 0; i < frame_elements; ++i)
        {
          TEST_ASSERT (buf_map[i] == (2 * (i + frame_index) + 1));
        }
      CHECK_CL_ERROR (clEnqueueUnmapMemObject (command_queue, buffer_dst,
                                               buf_map, 0, NULL, NULL));
      CHECK_CL_ERROR (clFinish (command_queue));

      CHECK_CL_ERROR (clReleaseEvent (write_src_events[0]));
      CHECK_CL_ERROR (clReleaseEvent (write_src_events[1]));
      CHECK_CL_ERROR (clReleaseEvent (command_buf_event));
    }

  CHECK_CL_ERROR (ext.clReleaseCommandBufferKHR (command_buffer));
  CHECK_CL_ERROR (clReleaseCommandQueue (command_queue));

  CHECK_CL_ERROR (clReleaseMemObject (buffer_src1));
  CHECK_CL_ERROR (clReleaseMemObject (buffer_src2));
  CHECK_CL_ERROR (clReleaseMemObject (buffer_dst));

  CHECK_CL_ERROR (clReleaseMemObject (buffer_tile1));
  CHECK_CL_ERROR (clReleaseMemObject (buffer_tile2));
  CHECK_CL_ERROR (clReleaseMemObject (buffer_res));

  CHECK_CL_ERROR (clReleaseKernel (kernel));
  CHECK_CL_ERROR (clReleaseProgram (program));
  CHECK_CL_ERROR (clReleaseContext (context));

  CHECK_CL_ERROR (clUnloadPlatformCompiler (platform));

  free (src1);
  free (src2);
  printf ("OK\n");
  return EXIT_SUCCESS;
#else
  return 77;
#endif
}
