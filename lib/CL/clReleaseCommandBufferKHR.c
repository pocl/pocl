/* OpenCL runtime library: clReleaseCommandBufferKHR()

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
#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_CALL
POname (clReleaseCommandBufferKHR) (cl_command_buffer_khr command_buffer)
    CL_API_SUFFIX__VERSION_1_2
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_buffer)),
                          CL_INVALID_COMMAND_BUFFER_KHR);

  int new_refcount;
  int errcode_ret = CL_SUCCESS;
  POCL_LOCK_OBJ (command_buffer);
  POCL_RELEASE_OBJECT_UNLOCKED (command_buffer, new_refcount);
  POCL_MSG_PRINT_REFCOUNTS ("Release Command Buffer %p  : %d\n",
                            command_buffer, new_refcount);

  if (new_refcount == 0)
    {
      POCL_UNLOCK_OBJ (command_buffer);
      VG_REFC_ZERO (command_buffer);

      /* Avoid freeing twice if there are multiple queues on the same device */
      cl_device_id *freed_devs = (cl_device_id *)alloca (
          command_buffer->num_queues * sizeof (cl_device_id *));
      POCL_RETURN_ERROR_COND ((freed_devs == NULL), CL_OUT_OF_HOST_MEMORY);

      unsigned num_freed = 0;

      for (int i = 0; i < command_buffer->num_queues; ++i)
        {
          cl_command_queue q = command_buffer->queues[i];
          int is_freed = 0;
          for (int j = 0; j < num_freed; ++j)
            {
              if (freed_devs[j] == q->device)
                is_freed = 1;
            }
          if (!is_freed)
            {
              int errcode = CL_SUCCESS;
              if (q->device->ops->free_command_buffer)
                errcode = q->device->ops->free_command_buffer (q->device,
                                                               command_buffer);
              if (errcode != CL_SUCCESS)
                errcode_ret = errcode;

              freed_devs[num_freed++] = q->device;
            }
          POname (clReleaseCommandQueue) (q);
        }

      _cl_command_node *cmd = command_buffer->cmds;
      while (cmd != NULL)
        {
          switch (cmd->type)
            {
            case CL_COMMAND_NDRANGE_KERNEL:
              for (unsigned i = 0; i < cmd->command.run.kernel->meta->num_args;
                   ++i)
                {
                  struct pocl_argument *a
                      = &cmd->command.run.kernel->dyn_arguments[i];
                  struct pocl_argument_info *ai
                      = &cmd->command.run.kernel->meta->arg_info[i];
                  if (ai->type == POCL_ARG_TYPE_SAMPLER)
                    POname (clReleaseSampler) (
                        cmd->command.run.arguments[i].value);
                  if (cmd->command.run.arguments[i].value != NULL)
                    POCL_MEM_FREE (cmd->command.run.arguments[i].value);
                }
              POname (clReleaseKernel) (cmd->command.run.kernel);
              POCL_MEM_FREE (cmd->command.run.arguments);
              break;
            case CL_COMMAND_COPY_BUFFER:
            case CL_COMMAND_COPY_BUFFER_RECT:
            case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
            case CL_COMMAND_COPY_IMAGE:
            case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
              break;
            case CL_COMMAND_FILL_BUFFER:
              POCL_MEM_FREE (cmd->command.memfill.pattern);
              break;
            case CL_COMMAND_SVM_MEMFILL:
              POCL_MEM_FREE (cmd->command.svm_fill.pattern);
              break;
            case CL_COMMAND_FILL_IMAGE:
              break;
            case CL_COMMAND_BARRIER:
              break;
            default:
              break;
            }

          for (unsigned i = 0; i < cmd->memobj_count; ++i)
            {
              POname (clReleaseMemObject) (cmd->memobj_list[i]);
            }
          _cl_command_node *next = cmd->next;
          pocl_mem_manager_free_command (cmd);
          cmd = next;
        }

      POCL_DESTROY_OBJECT (command_buffer);
      POCL_MEM_FREE (command_buffer->queues);
      POCL_MEM_FREE (command_buffer->properties);
      POCL_MEM_FREE (command_buffer);
    }
  else
    {
      VG_REFC_NONZERO (command_buffer);
      POCL_UNLOCK_OBJ (command_buffer);
    }

  return errcode_ret;
}
POsym (clReleaseCommandBufferKHR)
