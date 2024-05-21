/* OpenCL runtime library: clEnqueueWriteImage()

   Copyright (c) 2011-2024 PoCL developers

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

#include "pocl_cl.h"
#include "pocl_image_util.h"
#include "pocl_shared.h"
#include "pocl_util.h"

cl_int
pocl_validate_write_image (cl_command_queue command_queue,
                           cl_mem image,
                           const size_t *origin, /* [3] */
                           const size_t *region, /* [3] */
                           size_t row_pitch,
                           size_t slice_pitch,
                           const void *ptr)
{
  POCL_RETURN_ERROR_COND ((ptr == NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_ON((command_queue->context != image->context),
    CL_INVALID_CONTEXT, "image and command_queue are not from the same context\n");

  POCL_RETURN_ERROR_ON ((!image->is_image), CL_INVALID_MEM_OBJECT,
                        "image argument is not an image\n");
  POCL_RETURN_ERROR_ON ((image->is_gl_texture), CL_INVALID_MEM_OBJECT,
                        "image is a GL texture\n");
  POCL_RETURN_ON_UNSUPPORTED_IMAGE (image, command_queue->device);

  POCL_RETURN_ERROR_ON (
      (image->flags & (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS)),
      CL_INVALID_OPERATION,
      "image buffer has been created with CL_MEM_HOST_READ_ONLY "
      "or CL_MEM_HOST_NO_ACCESS\n");

  if (image->buffer)
    POCL_RETURN_ERROR_ON (
        (image->buffer->flags
         & (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS)),
        CL_INVALID_OPERATION,
        "image buffer has been created with CL_MEM_HOST_READ_ONLY "
        "or CL_MEM_HOST_NO_ACCESS\n");

  int errcode = pocl_check_image_origin_region (image, origin, region);
  if (errcode != CL_SUCCESS)
    return errcode;

  return CL_SUCCESS;
}

cl_int
pocl_write_image_common (cl_command_buffer_khr command_buffer,
                         cl_command_queue command_queue,
                         cl_mem image,
                         const size_t *origin, /* [3] */
                         const size_t *region, /* [3] */
                         size_t row_pitch,
                         size_t slice_pitch,
                         const void *ptr,
                         cl_uint num_items_in_wait_list,
                         const cl_event *event_wait_list,
                         cl_event *event,
                         const cl_sync_point_khr *sync_point_wait_list,
                         cl_sync_point_khr *sync_point,
                         _cl_command_node **cmd)
{
  POCL_VALIDATE_WAIT_LIST_PARAMS;

  unsigned i;
  cl_device_id device;
  POCL_CHECK_DEV_IN_CMDQ;

  cl_int errcode = pocl_validate_write_image (
      command_queue, image, origin, region, row_pitch, slice_pitch, ptr);
  if (errcode != CL_SUCCESS)
    return errcode;

  char rdonly = 0;

  if (command_buffer == NULL)
    {
      errcode = pocl_check_event_wait_list (
          command_queue, num_items_in_wait_list, event_wait_list);
      if (errcode != CL_SUCCESS)
        return errcode;
      errcode = pocl_create_command (
        cmd, command_queue, CL_COMMAND_WRITE_IMAGE, event,
        num_items_in_wait_list, event_wait_list, image, rdonly);
    }
  else
    {
      errcode = pocl_create_recorded_command (
        cmd, command_buffer, command_queue, CL_COMMAND_WRITE_IMAGE,
        num_items_in_wait_list, sync_point_wait_list, image, rdonly);
    }
  if (errcode != CL_SUCCESS)
    return errcode;

  _cl_command_node *c = *cmd;

  c->command.write_image.dst_mem_id
      = &image->device_ptrs[device->global_mem_id];
  c->command.write_image.dst = image;
  c->command.write_image.src_host_ptr = ptr;
  c->command.write_image.src_mem_id = NULL;
  c->command.write_image.origin[0] = origin[0];
  c->command.write_image.origin[1] = origin[1];
  c->command.write_image.origin[2] = origin[2];
  c->command.write_image.region[0] = region[0];
  c->command.write_image.region[1] = region[1];
  c->command.write_image.region[2] = region[2];
  c->command.write_image.src_row_pitch = row_pitch;
  c->command.write_image.src_slice_pitch = slice_pitch;
  c->command.write_image.src_offset = 0;

  return CL_SUCCESS;
}

extern CL_API_ENTRY cl_int CL_API_CALL
POname (clEnqueueWriteImage) (cl_command_queue command_queue,
                              cl_mem image,
                              cl_bool blocking_write,
                              const size_t *origin, /*[3]*/
                              const size_t *region, /*[3]*/
                              size_t row_pitch,
                              size_t slice_pitch,
                              const void *ptr,
                              cl_uint num_events_in_wait_list,
                              const cl_event *event_wait_list,
                              cl_event *event) CL_API_SUFFIX__VERSION_1_0
{
  cl_int errcode;
  _cl_command_node *cmd;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_COND ((*(command_queue->device->available) == CL_FALSE),
                          CL_DEVICE_NOT_AVAILABLE);

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (image)),
                          CL_INVALID_MEM_OBJECT);

  POCL_RETURN_ERROR_COND ((ptr == NULL), CL_INVALID_VALUE);

  if (IS_IMAGE1D_BUFFER (image))
    {
      IMAGE1D_ORIG_REG_TO_BYTES (image, origin, region);
      return POname (clEnqueueWriteBuffer) (
          command_queue, image, blocking_write, i1d_origin[0], i1d_region[0],
          ptr, num_events_in_wait_list, event_wait_list, event);
    }

  errcode = pocl_write_image_common (
      NULL, command_queue, image, origin, region, row_pitch, slice_pitch, ptr,
      num_events_in_wait_list, event_wait_list, event, NULL, NULL, &cmd);
  if (errcode != CL_SUCCESS)
    return errcode;

  pocl_command_enqueue (command_queue, cmd);

  if (blocking_write)
    errcode = POname(clFinish) (command_queue);

  return errcode;
}
POsym(clEnqueueWriteImage)
