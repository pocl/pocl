/* OpenCL runtime library: clCommandSVMMemcpyKHR()

   Copyright (c) 2022-2023 Michal Babej / Intel Finland Oy

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

CL_API_ENTRY cl_int
POname (clCommandSVMMemcpyKHR) (
    cl_command_buffer_khr command_buffer, cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    void *dst_ptr, const void *src_ptr, size_t size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr *sync_point_wait_list,
    cl_sync_point_khr *sync_point,
    cl_mutable_command_khr *mutable_handle) CL_API_SUFFIX__VERSION_2_0
{
  cl_int errcode;

  CMDBUF_VALIDATE_COMMON_HANDLES;

  if (command_queue != NULL
      && strstr (command_buffer->queues[0]->device->extensions,
                 "cl_khr_command_buffer_multi_device")
           == NULL)
    return CL_INVALID_COMMAND_QUEUE;

  return pocl_svm_memcpy_common (command_buffer, command_queue,
                                 CL_COMMAND_SVM_MEMCPY, dst_ptr, src_ptr, size,
                                 num_sync_points_in_wait_list, NULL, NULL,
                                 sync_point_wait_list, sync_point, NULL);
}
POsym (clCommandSVMMemcpyKHR)
