/* Test basic cl_khr_command_buffer_multi_device functionality

   Copyright (c) 2022-2024 Jan Solanti / Tampere University

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
#if defined(cl_khr_command_buffer_multi_device)                               \
  && cl_khr_command_buffer_multi_device == 1
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
    clRemapCommandBufferKHR_fn clRemapCommandBufferKHR;
  } ext;

  cl_platform_id platform = NULL;
  cl_context context = NULL;
  cl_device_id *devices = NULL;
  cl_command_queue *queues = NULL;
  cl_uint i = 0, num_devices = 0;

  int err = poclu_get_multiple_devices (&platform, &context, 0, &num_devices,
                                        &devices, &queues, 0);
  CHECK_OPENCL_ERROR_IN ("poclu_get_multiple_devices");

  /* Remapping capability & automatic remapping capabilities must be present */
  cl_platform_command_buffer_capabilities_khr platform_caps;
  CHECK_CL_ERROR (
    clGetPlatformInfo (platform, CL_PLATFORM_COMMAND_BUFFER_CAPABILITIES_KHR,
                       sizeof (platform_caps), &platform_caps, NULL));
  if (!(platform_caps & CL_COMMAND_BUFFER_PLATFORM_REMAP_QUEUES_KHR))
    {
      printf ("ERROR: Command buffer remapping not supported\n");
      return 77;
    }
  if (!(platform_caps & CL_COMMAND_BUFFER_PLATFORM_AUTOMATIC_REMAP_KHR))
    {
      printf ("ERROR: Command buffer remapping not supported\n");
      return 77;
    }
  if (num_devices == 1)
    {
      printf ("NOTE: Only 1 device available, using two queues on the same "
              "device\n");
      cl_command_queue *old_queue = queues;
      queues = malloc (sizeof (cl_command_queue) * 2);
      queues[0] = old_queue[0];
      free (old_queue);

      queues[1] = clCreateCommandQueue (context, devices[0], 0, &err);
      CHECK_OPENCL_ERROR_IN ("clCreateCommandQueue");

      cl_device_id *old_device = devices;
      devices = malloc (sizeof (cl_device_id) * 2);
      /* Duplicate the device since it is accessed later */
      devices[0] = old_device[0];
      devices[1] = old_device[0];
      free (old_device);

      num_devices = 2;
    }

  for (unsigned j = 0; j < num_devices; ++j)
    {
      cl_device_command_buffer_capabilities_khr device_caps;
      CHECK_CL_ERROR (
        clGetDeviceInfo (devices[j], CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR,
                         sizeof (device_caps), &device_caps, NULL));
      if (!(device_caps & CL_COMMAND_BUFFER_CAPABILITY_MULTIPLE_QUEUE_KHR))
        {
          printf ("ERROR: capability to record command buffers with multiple "
                  "queues is required\n");
          return 77;
        }
      if (!(platform_caps & CL_COMMAND_BUFFER_PLATFORM_UNIVERSAL_SYNC_KHR))
        {
          cl_uint num_sync_devices;
          CHECK_CL_ERROR (clGetDeviceInfo (
            devices[j], CL_DEVICE_COMMAND_BUFFER_NUM_SYNC_DEVICES_KHR,
            sizeof (cl_uint), &num_sync_devices, NULL));
          cl_device_id *sync_devices = (cl_device_id *)alloca (
            sizeof (cl_device_id) * num_sync_devices);
          CHECK_CL_ERROR (clGetDeviceInfo (
            devices[j], CL_DEVICE_COMMAND_BUFFER_SYNC_DEVICES_KHR,
            sizeof (cl_device_id) * num_sync_devices, sync_devices, NULL));

          for (unsigned i = 0; i < num_devices; ++i)
            {
              int can_sync = 1;
              if (i != j)
                {
                  int found = 0;
                  for (unsigned k = 0; k < num_sync_devices; ++k)
                    {
                      found |= (sync_devices[k] == devices[i]);
                    }
                  can_sync &= found;
                }
              if (!can_sync)
                {
                  printf ("ERROR: Global sync not supported and found devices "
                          "can't sync with each other\n");
                  return 77;
                }
            }
        }
    }

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
  ext.clRemapCommandBufferKHR = clGetExtensionFunctionAddressForPlatform (
    platform, "clRemapCommandBufferKHR");

  cl_int error;

  const char *code = STR (kernel void vector_addition (
    global const int *tile1, global const int *tile2, global int *res) {
    size_t index = get_global_id (0);
    res[index] = tile1[index] + tile2[index];
  });
  const size_t length = strlen (code);

  cl_program program
    = clCreateProgramWithSource (context, 1, &code, &length, &error);
  CHECK_CL_ERROR (error);
  CHECK_CL_ERROR (
    clBuildProgram (program, num_devices, devices, NULL, NULL, NULL));
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

  /* Adapted from test_command_buffer in order to cover a wide range of
   * commands in the remap operation */
  cl_command_buffer_properties_khr props[]
    = { CL_COMMAND_BUFFER_FLAGS_KHR, CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR,
        0 };
  cl_command_buffer_khr command_buffer
    = ext.clCreateCommandBufferKHR (num_devices, queues, props, &error);
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
        command_buffer, queues[tile_index % num_devices], buffer_src1,
        buffer_tile1, tile_index * tile_size, 0, tile_size,
        tile_sync_point ? 1 : 0, tile_sync_point ? &tile_sync_point : NULL,
        &copy_sync_points[0], NULL));
      /* For the sake of testing, pretend we are working with a vertical stack
       * of 8x8 tiles */
      size_t src_origin[3] = { 0, tile_index * 8, 0 };
      size_t dst_origin[3] = { 0, 0, 0 };
      size_t tile_region[3] = { 8 * sizeof (cl_int), 8, 1 };
      CHECK_CL_ERROR (ext.clCommandCopyBufferRectKHR (
        command_buffer, queues[tile_index % num_devices], buffer_src2,
        buffer_tile2, src_origin, dst_origin, tile_region, tile_region[0], 0,
        tile_region[0], 0, tile_sync_point ? 1 : 0,
        tile_sync_point ? &tile_sync_point : NULL, &copy_sync_points[1],
        NULL));

      cl_sync_point_khr nd_sync_point;
      CHECK_CL_ERROR (ext.clCommandNDRangeKernelKHR (
        command_buffer, queues[tile_index % num_devices], NULL, kernel, 1,
        NULL, &tile_elements, NULL, 2, copy_sync_points, &nd_sync_point,
        NULL));

      cl_sync_point_khr res_copy_sync_point;
      CHECK_CL_ERROR (ext.clCommandCopyBufferKHR (
        command_buffer, queues[tile_index % num_devices], buffer_res,
        buffer_dst, 0, tile_index * tile_size, tile_size, 1, &nd_sync_point,
        &res_copy_sync_point, NULL));

      char zero = 0;
      cl_sync_point_khr fill_sync_points[2];
      CHECK_CL_ERROR (ext.clCommandFillBufferKHR (
        command_buffer, queues[tile_index % num_devices], buffer_tile1, &zero,
        sizeof (zero), 0, tile_size, 1, &nd_sync_point, &fill_sync_points[0],
        NULL));
      CHECK_CL_ERROR (ext.clCommandFillBufferKHR (
        command_buffer, queues[tile_index % num_devices], buffer_tile2, &zero,
        sizeof (zero), 0, tile_size, 1, &nd_sync_point, &fill_sync_points[1],
        NULL));

      cl_sync_point_khr barrier_deps[4]
        = { nd_sync_point, res_copy_sync_point, fill_sync_points[0],
            fill_sync_points[1] };
      CHECK_CL_ERROR (ext.clCommandBarrierWithWaitListKHR (
        command_buffer, queues[tile_index % num_devices], 4, barrier_deps,
        &tile_sync_point, NULL));
    }

  CHECK_CL_ERROR (ext.clFinalizeCommandBufferKHR (command_buffer));

  cl_command_buffer_state_khr cmdbuf_state;
  CHECK_CL_ERROR (ext.clGetCommandBufferInfoKHR (
    command_buffer, CL_COMMAND_BUFFER_STATE_KHR,
    sizeof (cl_command_buffer_state_khr), &cmdbuf_state, NULL));
  TEST_ASSERT (cmdbuf_state == CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR);

  /* Enqueue N-queue buffer normally */
  cl_int src1[frame_elements];
  cl_int src2[frame_elements];
  for (size_t frame_index = 0; frame_index < frame_count; frame_index++)
    {
      for (size_t i = 0; i < frame_elements; ++i)
        {
          src1[i] = i + frame_index;
          src2[i] = i + frame_index + 1;
        }
      cl_event write_src_events[2];
      cl_event command_buf_event;
      CHECK_CL_ERROR (clEnqueueWriteBuffer (queues[0], buffer_src1, CL_FALSE,
                                            0, frame_size, src1, 0, NULL,
                                            &write_src_events[0]));
      CHECK_CL_ERROR (clEnqueueWriteBuffer (queues[0], buffer_src2, CL_FALSE,
                                            0, frame_size, src2, 0, NULL,
                                            &write_src_events[1]));

      CHECK_CL_ERROR (
        ext.clEnqueueCommandBufferKHR (num_devices, queues, command_buffer, 2,
                                       write_src_events, &command_buf_event));

      cl_int err;
      cl_int *buf_map = clEnqueueMapBuffer (
        queues[0], buffer_dst, CL_TRUE, CL_MAP_READ, 0,
        sizeof (cl_int) * frame_elements, 1, &command_buf_event, NULL, &err);
      CHECK_OPENCL_ERROR_IN ("clEnqueueMapBuffer");

      for (size_t i = 0; i < frame_elements; ++i)
        {
          TEST_ASSERT (buf_map[i] == (2 * (i + frame_index) + 1));
        }
      CHECK_CL_ERROR (clEnqueueUnmapMemObject (queues[0], buffer_dst, buf_map,
                                               0, NULL, NULL));
      CHECK_CL_ERROR (clFinish (queues[0]));

      CHECK_CL_ERROR (clReleaseEvent (write_src_events[0]));
      CHECK_CL_ERROR (clReleaseEvent (write_src_events[1]));
      CHECK_CL_ERROR (clReleaseEvent (command_buf_event));
    }

  /* Remap from N to 1 queues & run */
  cl_command_buffer_khr remapped_cmdbuf = ext.clRemapCommandBufferKHR (
    command_buffer, CL_TRUE, 1, queues, 0, NULL, NULL, &error);
  CHECK_CL_ERROR (error);
  CHECK_CL_ERROR (ext.clGetCommandBufferInfoKHR (
    remapped_cmdbuf, CL_COMMAND_BUFFER_STATE_KHR,
    sizeof (cl_command_buffer_state_khr), &cmdbuf_state, NULL));
  TEST_ASSERT (cmdbuf_state == CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR);
  for (size_t frame_index = 0; frame_index < frame_count; frame_index++)
    {
      for (size_t i = 0; i < frame_elements; ++i)
        {
          src1[i] = i + frame_index;
          src2[i] = i + frame_index + 1;
        }
      cl_event write_src_events[2];
      cl_event command_buf_event;
      CHECK_CL_ERROR (clEnqueueWriteBuffer (queues[0], buffer_src1, CL_FALSE,
                                            0, frame_size, src1, 0, NULL,
                                            &write_src_events[0]));
      CHECK_CL_ERROR (clEnqueueWriteBuffer (queues[0], buffer_src2, CL_FALSE,
                                            0, frame_size, src2, 0, NULL,
                                            &write_src_events[1]));

      CHECK_CL_ERROR (ext.clEnqueueCommandBufferKHR (
        1, queues, remapped_cmdbuf, 2, write_src_events, &command_buf_event));

      cl_int err;
      cl_int *buf_map = clEnqueueMapBuffer (
        queues[0], buffer_dst, CL_TRUE, CL_MAP_READ, 0,
        sizeof (cl_int) * frame_elements, 1, &command_buf_event, NULL, &err);
      CHECK_OPENCL_ERROR_IN ("clEnqueueMapBuffer");

      for (size_t i = 0; i < frame_elements; ++i)
        {
          TEST_ASSERT (buf_map[i] == (2 * (i + frame_index) + 1));
        }
      CHECK_CL_ERROR (clEnqueueUnmapMemObject (queues[0], buffer_dst, buf_map,
                                               0, NULL, NULL));
      CHECK_CL_ERROR (clFinish (queues[0]));

      CHECK_CL_ERROR (clReleaseEvent (write_src_events[0]));
      CHECK_CL_ERROR (clReleaseEvent (write_src_events[1]));
      CHECK_CL_ERROR (clReleaseEvent (command_buf_event));
    }

  for (unsigned i = 0; i < num_devices; ++i)
    {
      CHECK_CL_ERROR (clReleaseCommandQueue (queues[i]));
    }
  free (queues);

  CHECK_CL_ERROR (clReleaseMemObject (buffer_src1));
  CHECK_CL_ERROR (clReleaseMemObject (buffer_src2));
  CHECK_CL_ERROR (clReleaseMemObject (buffer_dst));

  CHECK_CL_ERROR (clReleaseMemObject (buffer_tile1));
  CHECK_CL_ERROR (clReleaseMemObject (buffer_tile2));
  CHECK_CL_ERROR (clReleaseMemObject (buffer_res));

  CHECK_CL_ERROR (ext.clReleaseCommandBufferKHR (command_buffer));
  CHECK_CL_ERROR (ext.clReleaseCommandBufferKHR (remapped_cmdbuf));

  CHECK_CL_ERROR (clReleaseKernel (kernel));
  CHECK_CL_ERROR (clReleaseProgram (program));
  CHECK_CL_ERROR (clReleaseContext (context));

  free (devices);

  printf ("OK\n");
  return EXIT_SUCCESS;
#else
  return 77;
#endif
}
