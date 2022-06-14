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
  size_t offset[3] = { 0, 0, 0 };
  size_t num_groups[3] = { 0, 0, 0 };
  size_t local[3] = { 0, 0, 0 };
  cl_int errcode = CL_SUCCESS;
  _cl_recorded_command *cmd = NULL;

  cl_uint memobj_count = 0;
  cl_mem memobj_list[kernel->meta->num_args];
  char readonly_flags[kernel->meta->num_args];

  cl_sync_point_khr next_syncpoint = 0;

  CMDBUF_VALIDATE_COMMON_HANDLES;

  POCL_LOCK (command_buffer->mutex);
  next_syncpoint = command_buffer->num_syncpoints;
  POCL_UNLOCK (command_buffer->mutex);

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (kernel)), CL_INVALID_KERNEL);

  POCL_RETURN_ERROR_ON (
      (command_queue->context != kernel->context), CL_INVALID_CONTEXT,
      "kernel and command_queue are not from the same context\n");

  errcode = pocl_kernel_calc_wg_size (
      command_queue, kernel, work_dim, global_work_offset, global_work_size,
      local_work_size, offset, local, num_groups);
  if (errcode != CL_SUCCESS)
    return errcode;

  errcode = pocl_kernel_collect_mem_objs (command_queue, kernel, &memobj_count,
                                          memobj_list, readonly_flags);
  if (errcode != CL_SUCCESS)
    return errcode;

  errcode = pocl_create_recorded_command (
      &cmd, command_buffer, command_queue, CL_COMMAND_NDRANGE_KERNEL,
      num_sync_points_in_wait_list, sync_point_wait_list, memobj_count,
      memobj_list, readonly_flags);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  cl_uint program_dev_i = CL_UINT_MAX;
  POCL_FILL_COMMAND_NDRANGEKERNEL;

  errcode = pocl_kernel_copy_args (kernel, &cmd->command.run);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  for (unsigned i = 0; i < kernel->meta->num_args; ++i)
    {
      struct pocl_argument_info *ai
          = &cmd->command.run.kernel->meta->arg_info[i];
      struct pocl_argument *a = &cmd->command.run.kernel->dyn_arguments[i];
      if (ai->type == POCL_ARG_TYPE_SAMPLER)
        POname (clRetainSampler) (cmd->command.run.arguments[i].value);
    }

  errcode = POname (clRetainKernel) (kernel);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  errcode = pocl_command_record (command_buffer, cmd, sync_point);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  return CL_SUCCESS;

ERROR:
  pocl_free_recorded_command (cmd);
  return errcode;
}
POsym (clCommandNDRangeKernelKHR)
