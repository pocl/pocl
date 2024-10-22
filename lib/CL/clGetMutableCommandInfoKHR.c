/* OpenCL runtime library: clGetMutableCommandInfoKHR()

   Copyright (c) 2024 Michal Babej / Intel Finland Oy

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

CL_API_ENTRY cl_int CL_API_CALL
POname (clGetMutableCommandInfoKHR) (cl_mutable_command_khr command,
                                     cl_mutable_command_info_khr param_name,
                                     size_t param_value_size,
                                     void *param_value,
                                     size_t *param_value_size_ret)
{
  POCL_RETURN_ERROR_COND ((command == NULL), CL_INVALID_MUTABLE_COMMAND_KHR);

#define PARAM_SIZE(expr)                                                      \
  {                                                                           \
    if (param_value_size_ret != NULL)                                         \
      {                                                                       \
        *param_value_size_ret = expr;                                         \
      }                                                                       \
  }
#define PARAM_VALUE(val, size)                                                \
  {                                                                           \
    if (param_value != NULL)                                                  \
      {                                                                       \
        POCL_RETURN_ERROR_COND ((param_value_size < size), CL_INVALID_VALUE); \
        memcpy (param_value, val, size);                                      \
      }                                                                       \
  }

  _cl_command_node *node = command;
  cl_command_buffer_khr cmd_buffer = node->cmd_buffer;
  /* All queues must have the same OpenCL context */
  //  cl_context ref_ctx = cmd_buffer->queues[0]->context;
  const cl_command_queue *queues = cmd_buffer->queues;

  switch (param_name)
    {
    case CL_MUTABLE_COMMAND_COMMAND_QUEUE_KHR:
      PARAM_SIZE (sizeof (cl_command_queue));
      PARAM_VALUE (&queues[node->queue_idx], sizeof (cl_command_queue));
      break;
    case CL_MUTABLE_COMMAND_COMMAND_BUFFER_KHR:
      PARAM_SIZE (sizeof (cl_command_buffer_khr));
      PARAM_VALUE (&cmd_buffer, sizeof (cl_command_buffer_khr));
      break;
    case CL_MUTABLE_COMMAND_COMMAND_TYPE_KHR:
      PARAM_SIZE (sizeof (cl_command_type));
      PARAM_VALUE (&node->type, sizeof (cl_command_type));
      break;
    case CL_MUTABLE_COMMAND_PROPERTIES_ARRAY_KHR:
      {
        size_t num_properties = cmd_buffer->num_properties > 0
                                  ? 2 * cmd_buffer->num_properties + 1
                                  : 0;
        POCL_RETURN_GETINFO_ARRAY (cl_command_buffer_properties_khr,
                                   num_properties, cmd_buffer->properties);
        break;
      }
    case CL_MUTABLE_DISPATCH_KERNEL_KHR:
      PARAM_SIZE (sizeof (cl_kernel));
      PARAM_VALUE (&node->command.run.kernel, sizeof (cl_kernel));
      break;
    case CL_MUTABLE_DISPATCH_DIMENSIONS_KHR:
      PARAM_SIZE (sizeof (cl_uint));
      PARAM_VALUE (&node->command.run.pc.work_dim, sizeof (cl_uint));
      break;
    case CL_MUTABLE_DISPATCH_GLOBAL_WORK_OFFSET_KHR:
      PARAM_SIZE (sizeof (size_t) * node->command.run.pc.work_dim);
      PARAM_VALUE (&node->command.run.pc.global_offset,
                   sizeof (size_t) * node->command.run.pc.work_dim);
      break;
    case CL_MUTABLE_DISPATCH_GLOBAL_WORK_SIZE_KHR:
      PARAM_SIZE (sizeof (size_t) * node->command.run.pc.work_dim);
      size_t *NG = node->command.run.pc.num_groups;
      size_t *LS = node->command.run.pc.local_size;
      assert (LS[0]);
      size_t Res[] = { (NG[0] * LS[0]), (NG[1] * LS[1]), (NG[2] * LS[2]) };
      POCL_MSG_WARN ("GET MUTABLE INFO : GLOBAL SIZE: %zu %zu %zu \n", Res[0],
                     Res[1], Res[2]);
      PARAM_VALUE (&Res, sizeof (size_t) * node->command.run.pc.work_dim);
      break;
    case CL_MUTABLE_DISPATCH_LOCAL_WORK_SIZE_KHR:
      PARAM_SIZE (sizeof (size_t) * node->command.run.pc.work_dim);
      PARAM_VALUE (&node->command.run.pc.local_size,
                   sizeof (size_t) * node->command.run.pc.work_dim);
      break;
    default:
      return CL_INVALID_VALUE;
    }
#undef PARAM_SIZE
#undef PARAM_VALUE

  return CL_SUCCESS;
}
POsym (clGetMutableCommandInfoKHR)
