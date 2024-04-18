/* OpenCL runtime library: clEnqueueNDRangeKernel()

   Copyright (c) 2011 Universidad Rey Juan Carlos and
                 2012-2020 Pekka Jääskeläinen
                 2020-2024 PoCL Developers

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
#include "pocl_cq_profiling.h"
#include "pocl_llvm.h"
#include "pocl_mem_management.h"
#include "pocl_shared.h"
#include "pocl_util.h"

#ifndef _WIN32
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif
#include <sys/stat.h>
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
  int errcode = 0;

  _cl_command_node *cmd;

  POCL_RETURN_ERROR_COND ((*(command_queue->device->available) == CL_FALSE),
                          CL_DEVICE_NOT_AVAILABLE);
  errcode = pocl_ndrange_kernel_common (
    NULL, command_queue, NULL, kernel, kernel->dyn_arguments, work_dim,
    global_work_offset, global_work_size, local_work_size,
    num_events_in_wait_list, event_wait_list, event, NULL, NULL, &cmd);
  POCL_RETURN_ERROR_COND (errcode != CL_SUCCESS, errcode);

  if (pocl_cq_profiling_enabled)
    {
      pocl_cq_profiling_register_event (cmd->sync.event.event);
      POname(clRetainKernel) (kernel);
      cmd->sync.event.event->meta_data->kernel = kernel;
    }

  pocl_command_enqueue (command_queue, cmd);
  return CL_SUCCESS;
}
POsym(clEnqueueNDRangeKernel)
