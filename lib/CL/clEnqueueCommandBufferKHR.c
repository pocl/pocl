/* OpenCL runtime library: clEnqueueCommandBufferKHR()

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

static void CL_CALLBACK
buffer_finished_callback (cl_event event, cl_int event_command_status,
                          void *user_data)
{
  cl_command_buffer_khr command_buffer = (cl_command_buffer_khr)user_data;
  POCL_LOCK (command_buffer->mutex);
  command_buffer->pending -= 1;
  if (command_buffer->pending == 0)
    command_buffer->state = CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR;
  POCL_UNLOCK (command_buffer->mutex);
  POname (clReleaseCommandBufferKHR) (command_buffer);
}

static cl_command_buffer_properties_khr
get_cmdbuf_property (cl_command_buffer_khr command_buffer,
              cl_command_buffer_properties_khr name)
{
  for (unsigned i = 0; i < command_buffer->num_properties; ++i)
    {
      if (command_buffer->properties[2 * i] == name)
        return command_buffer->properties[2 * i + 1];
    }
  return 0;
}

CL_API_ENTRY cl_int
POname (clEnqueueCommandBufferKHR) (cl_uint num_queues,
                                    cl_command_queue *queues,
                                    cl_command_buffer_khr command_buffer,
                                    cl_uint num_events_in_wait_list,
                                    const cl_event *event_wait_list,
                                    cl_event *event) CL_API_SUFFIX__VERSION_1_2
{
  int errcode = CL_SUCCESS;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_buffer)),
                          CL_INVALID_COMMAND_BUFFER_KHR);

  POCL_RETURN_ERROR_COND ((command_buffer->queues == NULL),
                          CL_INVALID_COMMAND_BUFFER_KHR);

  const cl_command_queue *used_queues;

  POCL_RETURN_ERROR_COND ((num_queues != 0 && queues == NULL),
                          CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND ((num_queues == 0 && queues != NULL),
                          CL_INVALID_VALUE);

  /* All queues must have the same OpenCL context as the command buffer was
   * created on */
  cl_context ref_ctx = command_buffer->queues[0]->context;

  if (queues != NULL && num_queues != 0)
    {
      POCL_RETURN_ERROR_COND ((num_queues != command_buffer->num_queues),
                              CL_INVALID_VALUE);

      /* All queues must be valid */
      for (unsigned i = 0; i < num_queues; ++i)
        {
          POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (queues[i])),
                                  CL_INVALID_COMMAND_QUEUE);

          POCL_RETURN_ERROR_COND ((queues[i]->device == NULL),
                                  CL_INVALID_COMMAND_QUEUE);

          POCL_RETURN_ERROR_COND ((queues[i]->context == NULL),
                                  CL_INVALID_COMMAND_QUEUE);

          /* check queue compatibility with parameters at matching [i] in
           * command buffer. A compatible command-queue is defined as a
           * command-queue with identical properties targeting the same device
           * and in the same OpenCL context.
           */
          POCL_RETURN_ERROR_COND ((queues[i]->context != ref_ctx),
                                  CL_INVALID_CONTEXT);

          POCL_RETURN_ERROR_COND (
              (queues[i]->device != command_buffer->queues[i]->device),
              CL_INCOMPATIBLE_COMMAND_QUEUE_KHR);

          POCL_RETURN_ERROR_COND (
              (queues[i]->properties != command_buffer->queues[i]->properties),
              CL_INCOMPATIBLE_COMMAND_QUEUE_KHR);

          used_queues = queues;
        }
    }
  else
    {
      used_queues = command_buffer->queues;
    }

  errcode = pocl_check_event_wait_list (
      used_queues[0], num_events_in_wait_list, event_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;

  cl_command_buffer_flags_khr flags
      = (cl_command_buffer_flags_khr)get_cmdbuf_property (
          command_buffer, CL_COMMAND_BUFFER_FLAGS_KHR);
  POCL_LOCK (command_buffer->mutex);
  int is_ready
      = command_buffer->state == CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR
        || (command_buffer->state == CL_COMMAND_BUFFER_STATE_PENDING_KHR
            && flags & CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR);
  if (is_ready)
    {
      command_buffer->state = CL_COMMAND_BUFFER_STATE_PENDING_KHR;
      command_buffer->pending += 1;
    }
  POCL_UNLOCK (command_buffer->mutex);
  POCL_RETURN_ERROR_COND ((!is_ready), CL_INVALID_OPERATION);

  /* Submit to queue(s) */
  /* TODO: figure out how submitting should work with multiple queues */
  cl_command_queue q = used_queues[0];
  if (q->device->ops->run_command_buffer)
    {
      return q->device->ops->run_command_buffer (q->device, command_buffer);
    }
  else
    {
      /* Submit individual commands manually */
      _cl_command_node *cmd;

      cl_event syncpoints[command_buffer->num_syncpoints];
      cl_event *deps = (cl_event *)alloca (
          sizeof (event)
          * (command_buffer->num_syncpoints + num_events_in_wait_list));

      unsigned sync_id = 0;
      LL_FOREACH (command_buffer->cmds, cmd)
      {
        unsigned j, k;
        assert (cmd->queue_idx == 0);

        /* Add events from syncpoints to waitlist */
        for (j = 0; j < cmd->sync.syncpoint.num_sync_points_in_wait_list; ++j)
          {
            // sync point ids start at 1
            deps[j]
                = syncpoints[cmd->sync.syncpoint.sync_point_wait_list[j] - 1];
          }
        /* Add events from command buffer dependencies to waitlist */
        for (k = 0; k < num_events_in_wait_list; ++k, ++j)
          {
            deps[j] = event_wait_list[k];
          }

        _cl_command_node *node = NULL;
        errcode = pocl_create_command (
            &node, q, cmd->type, &syncpoints[sync_id], j, deps,
            cmd->memobj_count, cmd->memobj_list, cmd->readonly_flag_list);
        ++sync_id;

        if (errcode != CL_SUCCESS)
          {
            POCL_MSG_ERR ("Failed to instantiate recorded command: %i\n",
                          errcode);
            pocl_mem_manager_free_command (node);
            return errcode;
          }

        memcpy (&node->command, &cmd->command, sizeof (_cl_command_t));

        // Copy variables that are freed when the command finishes
        switch (cmd->type)
          {
          case CL_COMMAND_NDRANGE_KERNEL:
            POname (clRetainKernel) (node->command.run.kernel);
            errcode = pocl_kernel_copy_args (node->command.run.kernel,
                                             &node->command.run);
            if (errcode != CL_SUCCESS)
              {
                errcode = CL_OUT_OF_HOST_MEMORY;
                break;
              }
            break;
          case CL_COMMAND_FILL_BUFFER:
            node->command.memfill.pattern
                = pocl_aligned_malloc (cmd->command.memfill.pattern_size,
                                       cmd->command.memfill.pattern_size);
            if (node->command.memfill.pattern == NULL)
              {
                errcode = CL_OUT_OF_HOST_MEMORY;
                break;
              }
            memcpy (node->command.memfill.pattern,
                    cmd->command.memfill.pattern,
                    cmd->command.memfill.pattern_size);
            break;

          default:
            break;
          }

        if (errcode != CL_SUCCESS)
          {
            POCL_MSG_ERR ("Failed to allocate temporary command parameters\n");
            pocl_mem_manager_free_command (node);
            return errcode;
          }

        pocl_command_enqueue (q, node);
      }

      /* We need an event for the completion of the command buffer as a whole.
       * TODO: grab start timestamp before submitting any of the constituent
       * commands
       */
      _cl_command_node *node = NULL;
      errcode = pocl_create_command (&node, q, CL_COMMAND_COMMAND_BUFFER_KHR,
                                     event, command_buffer->num_syncpoints,
                                     syncpoints, 0, NULL, NULL);
      if (errcode != CL_SUCCESS)
        {
          pocl_mem_manager_free_command (node);
          return errcode;
        }
      POname (clRetainCommandBufferKHR) (command_buffer);
      pocl_command_enqueue (q, node);
      return POname (clSetEventCallback) (*event, CL_COMPLETE,
                                          buffer_finished_callback,
                                          (void *)command_buffer);
    }
}
POsym (clEnqueueCommandBufferKHR)
