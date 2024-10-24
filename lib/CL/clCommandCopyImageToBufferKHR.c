/* OpenCL runtime library: clCommandCopyImageToBufferKHR()

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
#include "pocl_mem_management.h"
#include "pocl_shared.h"
#include "pocl_util.h"

extern CL_API_ENTRY cl_int CL_API_CALL
POname (clCommandCopyImageToBufferKHR) (
    cl_command_buffer_khr command_buffer, cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    cl_mem src_image, cl_mem dst_buffer, const size_t *src_origin,
    const size_t *region, size_t dst_offset,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr *sync_point_wait_list,
    cl_sync_point_khr *sync_point,
    cl_mutable_command_khr *mutable_handle) CL_API_SUFFIX__VERSION_1_2
{
  cl_int errcode;
  CMDBUF_VALIDATE_COMMON_HANDLES;
  SETUP_MUTABLE_HANDLE;

  errcode = pocl_copy_image_to_buffer_common (
    command_buffer, command_queue, src_image, dst_buffer, src_origin, region,
    dst_offset, num_sync_points_in_wait_list, NULL, NULL, sync_point_wait_list,
    sync_point, mutable_handle);
  if (errcode != CL_SUCCESS)
    return errcode;

  if (*mutable_handle)
    errcode
      = pocl_command_record (command_buffer, *mutable_handle, sync_point);

  if (errcode != CL_SUCCESS)
    goto ERROR;

  return CL_SUCCESS;

ERROR:
  pocl_mem_manager_free_command (*mutable_handle);
  return errcode;
}
POsym (clCommandCopyImageToBufferKHR)
