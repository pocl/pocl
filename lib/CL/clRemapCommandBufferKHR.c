/* OpenCL runtime library: clRemapCommandBufferKHR()

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
#include "pocl_shared.h"
#include "pocl_util.h"


#define CMD_NODE_COPY_ARRAY(item_t, count, name) \
  do { \
  new_cmd->name = (item_t *) malloc(sizeof (item_t) * count); \
  /* TODO: avoid leaking new_cmd? */ \
  POCL_GOTO_ERROR_COND ((new_cmd->name == NULL), CL_OUT_OF_HOST_MEMORY); \
  memcpy(new_cmd->name, cmd->name, sizeof (item_t) * count); \
  }while(0)

CL_API_ENTRY cl_command_buffer_khr CL_API_CALL
POname (clRemapCommandBufferKHR) (
    cl_command_buffer_khr command_buffer,
    cl_uint num_queues, const cl_command_queue *queues,
    cl_uint num_handles,
    const cl_mutable_command_khr* handles,
    cl_mutable_command_khr* handles_ret,
    cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_2
{
  int errcode = 0;
  cl_command_buffer_khr new_cmdbuf = NULL;

  if ((errcode = pocl_cmdbuf_validate_queue_list(num_queues, queues))
      != CL_SUCCESS)
  {
    *errcode_ret = errcode;
    return NULL;
  }

  cl_command_buffer_properties_khr universal_sync = pocl_cmdbuf_get_property(command_buffer, CL_COMMAND_BUFFER_UNIVERSAL_SYNC_KHR);
  POCL_GOTO_ERROR_COND((universal_sync == 0 && num_queues > 1), CL_INCOMPATIBLE_COMMAND_QUEUE_KHR);

  POname (clCreateCommandBufferKHR) (num_queues, queues, command_buffer->properties, &errcode);
  if (errcode != CL_SUCCESS)
    {
      *errcode_ret = errcode;
      return NULL;
    }

  _cl_command_node *cmd;
  LL_FOREACH(command_buffer->cmds, cmd)
  {
    _cl_command_node *new_cmd = pocl_mem_manager_new_command ();
    POCL_GOTO_ERROR_COND ((new_cmd == NULL), CL_OUT_OF_HOST_MEMORY);
    memcpy(new_cmd, cmd, sizeof(_cl_command_node));

    /* TODO: be smarter about this */
    new_cmd->queue_idx = new_cmd->queue_idx % new_cmdbuf->num_queues;
    
    CMD_NODE_COPY_ARRAY(cl_mem, new_cmd->memobj_count, memobj_list);
    CMD_NODE_COPY_ARRAY(char, new_cmd->memobj_count, readonly_flag_list);

    for (unsigned i = 0; i < new_cmd->memobj_count; ++i)
      {
        clRetainMemObject(new_cmd->memobj_list[i]);
      }

    switch (new_cmd->type)
      {
        case CL_COMMAND_NDRANGE_KERNEL:
        pocl_kernel_copy_args(new_cmd->command.run.kernel, &new_cmd->command.run);
        for (unsigned i = 0; i < new_cmd->command.run.kernel->meta->num_args; ++i)
          {
            struct pocl_argument_info *ai = &new_cmd->command.run.kernel->meta->arg_info[i];
            if (ai->type == POCL_ARG_TYPE_SAMPLER)
              POname (clRetainSampler) (new_cmd->command.run.arguments[i].value);
          }
        break;

        case CL_COMMAND_FILL_BUFFER:
        CMD_NODE_COPY_ARRAY(char, new_cmd->command.memfill.pattern_size, command.memfill.pattern);
        break;

        default:
        break;
      }

    LL_APPEND(new_cmdbuf->cmds, new_cmd);
  }

  if (command_buffer->state == CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR
        || command_buffer->state == CL_COMMAND_BUFFER_STATE_PENDING_KHR)
    {
      errcode = POname (clFinalizeCommandBufferKHR) (new_cmdbuf);
      if (errcode != CL_SUCCESS)
        goto ERROR;
    }

  if (errcode_ret != NULL)
    *errcode_ret = errcode;
  return new_cmdbuf;

ERROR:
  POname (clReleaseCommandBufferKHR) (new_cmdbuf);
  if (errcode_ret != NULL)
    *errcode_ret = errcode;
  return NULL;
}
POsym (clRemapCommandBufferKHR)
