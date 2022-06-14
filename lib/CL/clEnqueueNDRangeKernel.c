/* OpenCL runtime library: clEnqueueNDRangeKernel()

   Copyright (c) 2011 Universidad Rey Juan Carlos and
                 2012-2020 Pekka Jääskeläinen

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

#include "config.h"
#include "pocl_binary.h"
#include "pocl_cache.h"
#include "pocl_cl.h"
#include "pocl_context.h"
#include "pocl_cq_profiling.h"
#include "pocl_llvm.h"
#include "pocl_local_size.h"
#include "pocl_mem_management.h"
#include "pocl_shared.h"
#include "pocl_util.h"
#include "utlist.h"

#ifndef _WIN32
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif
#include <assert.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>

//#define DEBUG_NDRANGE

CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueNDRangeKernel)(cl_command_queue command_queue,
                       cl_kernel kernel,
                       cl_uint work_dim,
                       const size_t *global_work_offset,
                       const size_t *global_work_size,
                       const size_t *local_work_size,
                       cl_uint num_events_in_wait_list,
                       const cl_event *event_wait_list,
                       cl_event *event) CL_API_SUFFIX__VERSION_1_0
{
  size_t offset[3] = { 0, 0, 0 };
  size_t num_groups[3] = { 0, 0, 0 };
  size_t local[3] = { 0, 0, 0 };
  /* cached values for max_work_item_sizes,
   * since we are going to access them repeatedly */
  size_t max_local_x, max_local_y, max_local_z;
  /* cached values for max_work_group_size,
   * since we are going to access them repeatedly */
  size_t max_group_size;

  int errcode = 0;

  /* no need for malloc, pocl_create_event will memcpy anyway.
   * num_args is the absolute max needed */
  cl_uint memobj_count = 0;
  cl_mem memobj_list[kernel->meta->num_args];
  char readonly_flag_list[kernel->meta->num_args];
  _cl_command_node *cmd;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (kernel)), CL_INVALID_KERNEL);


  POCL_RETURN_ERROR_ON((command_queue->context != kernel->context),
    CL_INVALID_CONTEXT,
    "kernel and command_queue are not from the same context\n");

  errcode = pocl_kernel_calc_wg_size (
      command_queue, kernel, work_dim, global_work_offset, global_work_size,
      local_work_size, offset, local, num_groups);
  if (errcode != CL_SUCCESS)
    return errcode;
  errcode = pocl_kernel_collect_mem_objs (command_queue, kernel, &memobj_count,
                                          memobj_list, readonly_flag_list);

  if (errcode != CL_SUCCESS)
    return errcode;

  errcode = pocl_check_event_wait_list (command_queue, num_events_in_wait_list,
                                        event_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;

  errcode
      = pocl_create_command (&cmd, command_queue, CL_COMMAND_NDRANGE_KERNEL,
                             event, num_events_in_wait_list, event_wait_list,
                             memobj_count, memobj_list, readonly_flag_list);

  if (errcode != CL_SUCCESS)
    {
      POCL_MSG_ERR ("Failed to create command: %i\n", errcode);
      return errcode;
    }

  cl_uint program_dev_i = CL_UINT_MAX;
  POCL_FILL_COMMAND_NDRANGEKERNEL;

  cmd->program_device_i = program_dev_i;
  cmd->next = NULL;

  POname (clRetainKernel) (kernel);

  errcode = pocl_kernel_copy_args (kernel, &cmd->command.run);
  if (errcode != CL_SUCCESS)
    {
      pocl_ndrange_node_cleanup (cmd);
      pocl_mem_manager_free_command (cmd);
      return errcode;
    }

  if (pocl_cq_profiling_enabled)
    {
      pocl_cq_profiling_register_event (cmd->event);
      POname(clRetainKernel) (kernel);
      cmd->event->meta_data->kernel = kernel;
    }

  pocl_command_enqueue (command_queue, cmd);
  return CL_SUCCESS;
}
POsym(clEnqueueNDRangeKernel)
