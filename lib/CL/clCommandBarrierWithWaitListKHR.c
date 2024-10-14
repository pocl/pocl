/* OpenCL runtime library: clCommandBarrierWithWaitListKHR()

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

#include <CL/cl_ext.h>

#include "pocl_cl.h"
#include "pocl_mem_management.h"
#include "pocl_util.h"

CL_API_ENTRY cl_int
POname (clCommandBarrierWithWaitListKHR) (
    cl_command_buffer_khr command_buffer, cl_command_queue command_queue,
    const cl_command_properties_khr* properties,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr *sync_point_wait_list,
    cl_sync_point_khr *sync_point,
    cl_mutable_command_khr *mutable_handle) CL_API_SUFFIX__VERSION_1_2
{
  cl_int errcode;
  _cl_command_node *cmd = NULL;

  CMDBUF_VALIDATE_COMMON_HANDLES;

  errcode = pocl_create_recorded_command (
    &cmd, command_buffer, command_queue, CL_COMMAND_BARRIER,
    num_sync_points_in_wait_list, sync_point_wait_list, NULL);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  cmd->command.barrier.has_wait_list = num_sync_points_in_wait_list != 0;
  errcode = pocl_command_record (command_buffer, cmd, sync_point);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  return CL_SUCCESS;

ERROR:
  pocl_mem_manager_free_command (cmd);
  return errcode;
}
POsym (clCommandBarrierWithWaitListKHR)
