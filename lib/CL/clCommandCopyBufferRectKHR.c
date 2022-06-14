/* OpenCL runtime library: clCommandCopyBufferRectKHR()

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

#include "pocl_cl.h"
#include "pocl_shared.h"
#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_CALL
POname (clCommandCopyBufferRectKHR) (
    cl_command_buffer_khr command_buffer, cl_command_queue command_queue,
    cl_mem src_buffer, cl_mem dst_buffer, const size_t *src_origin,
    const size_t *dst_origin, const size_t *region, size_t src_row_pitch,
    size_t src_slice_pitch, size_t dst_row_pitch, size_t dst_slice_pitch,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr *sync_point_wait_list,
    cl_sync_point_khr *sync_point,
    cl_mutable_command_khr *mutable_handle) CL_API_SUFFIX__VERSION_1_2
{
  cl_int errcode;
  _cl_recorded_command *cmd = NULL;

  CMDBUF_VALIDATE_COMMON_HANDLES;

  errcode = pocl_record_rect_copy (
      command_buffer->queues[0], CL_COMMAND_COPY_BUFFER_RECT, src_buffer,
      CL_FALSE, dst_buffer, CL_FALSE, src_origin, dst_origin, region,
      src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch,
      num_sync_points_in_wait_list, sync_point_wait_list, &cmd,
      command_buffer);

  size_t src_offset = 0;
  POCL_CONVERT_SUBBUFFER_OFFSET (src_buffer, src_offset);
  POCL_RETURN_ERROR_ON (
      (src_buffer->size > command_queue->device->max_mem_alloc_size),
      CL_OUT_OF_RESOURCES, "src is larger than device's MAX_MEM_ALLOC_SIZE\n");

  size_t dst_offset = 0;
  POCL_CONVERT_SUBBUFFER_OFFSET (dst_buffer, dst_offset);
  POCL_RETURN_ERROR_ON (
      (dst_buffer->size > command_queue->device->max_mem_alloc_size),
      CL_OUT_OF_RESOURCES, "src is larger than device's MAX_MEM_ALLOC_SIZE\n");

  POCL_FILL_COMMAND_COPY_BUFFER_RECT;

  errcode = pocl_command_record (command_buffer, cmd, sync_point);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  return CL_SUCCESS;

ERROR:
  pocl_free_recorded_command (cmd);
  return errcode;
}
POsym (clCommandCopyBufferRectKHR)
