/* OpenCL runtime library: clCommandNDRangeKernelKHR()

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

CL_API_ENTRY cl_int
POname (clCommandNDRangeKernelKHR) (
    cl_command_buffer_khr command_buffer, cl_command_queue command_queue,
    const cl_ndrange_kernel_command_properties_khr *properties,
    cl_kernel kernel, cl_uint work_dim, const size_t *global_work_offset,
    const size_t *global_work_size, const size_t *local_work_size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr *sync_point_wait_list,
    cl_sync_point_khr *sync_point,
    cl_mutable_command_khr *mutable_handle) CL_API_SUFFIX__VERSION_1_2
{
  cl_int errcode = CL_SUCCESS;
  _cl_command_node *cmd = NULL;

  CMDBUF_VALIDATE_COMMON_HANDLES;

  errcode = pocl_ndrange_kernel_common (
      command_buffer, command_queue, properties, kernel, work_dim,
      global_work_offset, global_work_size, local_work_size,
      num_sync_points_in_wait_list, NULL, NULL, sync_point_wait_list,
      sync_point, &cmd);

  for (unsigned i = 0; i < kernel->meta->num_args; ++i)
    {
      struct pocl_argument_info *ai
          = &cmd->command.run.kernel->meta->arg_info[i];
      struct pocl_argument *a = &cmd->command.run.kernel->dyn_arguments[i];
      if (ai->type == POCL_ARG_TYPE_SAMPLER)
        POname (clRetainSampler) (cmd->command.run.arguments[i].value);
    }

  errcode = pocl_command_record (command_buffer, cmd, sync_point);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  return CL_SUCCESS;

ERROR:
  pocl_mem_manager_free_command (cmd);
  return errcode;
}
POsym (clCommandNDRangeKernelKHR)
