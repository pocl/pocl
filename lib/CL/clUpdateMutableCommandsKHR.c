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

#include "pocl_util.h"
#include "pocl_shared.h"

CL_API_ENTRY cl_int
POname (clUpdateMutableCommandsKHR) (
  cl_command_buffer_khr command_buffer,
  cl_uint num_configs,
  const cl_command_buffer_update_type_khr *config_types,
  const void **configs)
{
  cl_int errcode;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_buffer)),
                          CL_INVALID_COMMAND_BUFFER_KHR);

  POCL_RETURN_ERROR_COND (
    (command_buffer->state != CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR),
    CL_INVALID_OPERATION);

  POCL_RETURN_ERROR_COND ((command_buffer->is_mutable == CL_FALSE),
                          CL_INVALID_OPERATION);

  for (cl_uint i = 0; i < num_configs; ++i)
    {
      POCL_RETURN_ERROR_COND (
        (config_types[i] != CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR),
        CL_INVALID_VALUE);
      POCL_RETURN_ERROR_COND ((configs[i] == NULL), CL_INVALID_VALUE);
    }

  for (cl_uint i = 0; i < num_configs; ++i)
    {
      const cl_mutable_dispatch_config_khr *cfg
        = (const cl_mutable_dispatch_config_khr *)configs[i];
      cl_device_id dev = NULL;
      if (cfg->command->device)
        dev = cfg->command->device;
      else if (command_buffer->queues)
        dev = command_buffer->queues[cfg->command->queue_idx]->device;
      else
        POCL_RETURN_ERROR_ON(1, CL_INVALID_COMMAND_BUFFER_KHR,
                             "Command buffer has no assigned device\n");

      cl_mutable_dispatch_fields_khr support
        = dev ? dev->cmdbuf_mutable_dispatch_capabilities : 0;
      POCL_RETURN_ERROR_COND (
        (cfg->command->type != CL_COMMAND_NDRANGE_KERNEL), CL_INVALID_VALUE);
      POCL_RETURN_ERROR_ON((support == 0), CL_INVALID_DEVICE,
                           "The device does not support any mutable fields\n");

      POCL_RETURN_ERROR_COND ((cfg->num_args > 0
         && ((support & CL_MUTABLE_DISPATCH_ARGUMENTS_KHR) == 0)),
        CL_INVALID_VALUE);
      POCL_RETURN_ERROR_COND ((cfg->num_svm_args > 0
         && ((support & CL_MUTABLE_DISPATCH_ARGUMENTS_KHR) == 0)),
        CL_INVALID_VALUE);
      POCL_RETURN_ERROR_COND ((cfg->num_exec_infos > 0
         && ((support & CL_MUTABLE_DISPATCH_EXEC_INFO_KHR) == 0)),
        CL_INVALID_VALUE);
      POCL_RETURN_ERROR_COND ((cfg->global_work_offset != NULL
         && ((support & CL_MUTABLE_DISPATCH_GLOBAL_WORK_OFFSET_KHR) == 0)),
        CL_INVALID_VALUE);
      POCL_RETURN_ERROR_COND ((cfg->global_work_size != NULL
         && ((support & CL_MUTABLE_DISPATCH_GLOBAL_WORK_SIZE_KHR) == 0)),
        CL_INVALID_VALUE);
      POCL_RETURN_ERROR_COND ((cfg->local_work_size != NULL
         && ((support & CL_MUTABLE_DISPATCH_LOCAL_WORK_SIZE_KHR) == 0)),
        CL_INVALID_VALUE);

      cl_uint program_dev_i = cfg->command->program_device_i;
      cl_kernel kernel = cfg->command->command.run.kernel;
      cl_device_id realdev = pocl_real_dev (dev);
      cl_uint work_dim = cfg->work_dim
          ? cfg->work_dim
          : cfg->command->command.run.pc.work_dim;
      unsigned copy_size = sizeof (size_t) * work_dim;

      //const size_t zero_sizes[] = {0,0,0};
      const size_t *global_work_offset = cfg->global_work_offset
          ? cfg->global_work_offset
          : NULL;
          //: cfg->command->command.run.pc.global_offset;

      size_t *LS = cfg->command->command.run.pc.local_size;
      size_t *NG = cfg->command->command.run.pc.num_groups;
      size_t orig_global_work_size[] = { LS[0] * NG[0],
                                       LS[1] * NG[1],
                                       LS[2] * NG[2]};
      const size_t *global_work_size = cfg->global_work_size
          ? cfg->global_work_size
          : NULL;
          //: orig_global_work_size;

      const size_t *local_work_size = cfg->local_work_size
          ? cfg->local_work_size
          : NULL;
          //: cfg->command->command.run.pc.local_size;

      size_t out_global_work_offset[3];
      size_t out_num_groups[3];
      size_t out_local_work_size[3];
      int errcode = pocl_kernel_calc_wg_size (
          dev, kernel, program_dev_i, work_dim,
          global_work_offset, global_work_size, local_work_size,
          out_global_work_offset, out_local_work_size, out_num_groups);
      POCL_RETURN_ERROR_ON (errcode != CL_SUCCESS, errcode,
                            "Error calculating WorkGroup size\n");

      memcpy (cfg->command->command.run.pc.local_size,
              out_local_work_size, copy_size);
      POCL_MSG_PRINT_INFO ("UPDATE MUTABLE CMD: NEW LOCAL SIZE %zu %zu %zu\n",
                     out_local_work_size[0], out_local_work_size[1],
                     out_local_work_size[2]);
      memcpy (cfg->command->command.run.pc.global_offset,
              out_global_work_offset, copy_size);
      POCL_MSG_PRINT_INFO ("UPDATE MUTABLE CMD: NEW GLOBAL OFFSET %zu %zu %zu\n",
                     out_global_work_offset[0],
                     out_global_work_offset[1],
                     out_global_work_offset[2]);
      memcpy (cfg->command->command.run.pc.num_groups,
              out_num_groups, copy_size);
      POCL_MSG_PRINT_INFO ("UPDATE MUTABLE CMD: NEW GLOBAL SIZE %zu %zu %zu\n",
                     out_num_groups[0], out_num_groups[1],
                     out_num_groups[2]);
    }

  return CL_SUCCESS;
}
POsym (clUpdateMutableCommandsKHR)
