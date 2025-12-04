/* OpenCL runtime library: command buffer utility functions

   Copyright (c) 2022 Jan Solanti / Tampere University
                 2025 Topi Leppänen / Tampere University

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

#include "pocl_mem_management.h"
#include "pocl_util.h"
#include "utlist.h"

#include "pocl_cmdbuf.h"

cl_int
pocl_cmdbuf_create_command (_cl_command_node **cmd,
                            cl_command_buffer_khr command_buffer,
                            cl_command_queue command_queue,
                            cl_command_type command_type,
                            cl_uint num_sync_points_in_wait_list,
                            const cl_sync_point_khr *sync_point_wait_list,
                            pocl_buffer_migration_info *buffer_usage)
{
  cl_int errcode = pocl_check_syncpoint_wait_list (
    command_buffer, num_sync_points_in_wait_list, sync_point_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;

  if (buffer_usage != NULL)
    {
      /* If the buffer is an image backed by buffer storage,
         replace with actual storage. */
      pocl_buffer_migration_info *migr_info = NULL;
      LL_FOREACH (buffer_usage, migr_info)
        if (migr_info->buffer->buffer)
          migr_info->buffer = migr_info->buffer->buffer;

      if (!pocl_preallocate_buffers (command_queue->device, buffer_usage))
        return CL_OUT_OF_RESOURCES;
    }

  *cmd = pocl_mem_manager_new_command ();
  POCL_RETURN_ERROR_COND ((*cmd == NULL), CL_OUT_OF_HOST_MEMORY);
  (*cmd)->type = command_type;
  (*cmd)->buffered = 1;

  /* pocl_cmdbuf_choose_recording_queue should have been called to ensure we
   * have a valid command queue, usually via CMDBUF_VALIDATE_COMMON_HANDLES
   * but at that time *cmd was not allocated at that time, so find the queue
   * index again here */
  for (unsigned i = 0; i < command_buffer->num_queues; ++i)
    {
      if (command_buffer->queues[i] == command_queue)
        (*cmd)->queue_idx = i;
    }

  (*cmd)->sync.syncpoint.num_sync_points_in_wait_list
    = num_sync_points_in_wait_list;
  if (num_sync_points_in_wait_list > 0)
    {
      cl_sync_point_khr *wait_list
        = malloc (sizeof (cl_sync_point_khr) * num_sync_points_in_wait_list);
      if (wait_list == NULL)
        {
          POCL_MEM_FREE (*cmd);
          return CL_OUT_OF_HOST_MEMORY;
        }
      memcpy (wait_list, sync_point_wait_list,
              sizeof (cl_sync_point_khr) * num_sync_points_in_wait_list);
      (*cmd)->sync.syncpoint.sync_point_wait_list = wait_list;
    }

  (*cmd)->migr_infos = buffer_usage;
  pocl_buffer_migration_info *migr_info = NULL;

  /* We need to retain the buffers as we expect them to be executed
     later. They are retained again for each executed instance in
     pocl_create_migration_commands() and those references are freed
     after the executed instance is freed.  This one is freed at
     command buffer free time. */
  LL_FOREACH (buffer_usage, migr_info)
    POname (clRetainMemObject) (migr_info->buffer);

  return CL_SUCCESS;
}

cl_int
pocl_cmdbuf_record_command (cl_command_buffer_khr command_buffer,
                            _cl_command_node *cmd,
                            cl_sync_point_khr *sync_point)
{
  POCL_LOCK (command_buffer->mutex);
  if (command_buffer->state != CL_COMMAND_BUFFER_STATE_RECORDING_KHR)
    {
      POCL_UNLOCK (command_buffer->mutex);
      return CL_INVALID_OPERATION;
    }
  pocl_buffer_migration_info *mi;
  LL_FOREACH (cmd->migr_infos, mi)
    {
      /* Note: mem object refcounts are NOT bumped here as deduplicating them
       * to match the migration info list would introduce unnecessary
       * complexity. The recorded commands themselves already hold counted
       * references to their buffers and they are expected to live until the
       * entire command buffer is destroyed. */
      command_buffer->migr_infos = pocl_append_unique_migration_info (
        command_buffer->migr_infos, mi->buffer, mi->read_only);
    }
  LL_APPEND (command_buffer->cmds, cmd);

  if (sync_point != NULL)
    *sync_point = command_buffer->num_syncpoints + 1;
  command_buffer->num_syncpoints++;
  cmd->cmd_buffer = command_buffer;
  POCL_UNLOCK (command_buffer->mutex);
  return CL_SUCCESS;
}

cl_int
pocl_cmdbuf_validate_queue_list (cl_uint num_queues,
                                 const cl_command_queue *queues)
{
  POCL_RETURN_ERROR_COND ((num_queues == 0), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND ((queues == NULL), CL_INVALID_VALUE);

  /* All queues must have the same OpenCL context */
  cl_context ref_ctx = queues[0]->context;

  for (unsigned i = 0; i < num_queues; ++i)
    {
      /* All queues must be valid Command queue objects */
      POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (queues[i])),
                              CL_INVALID_COMMAND_QUEUE);

      POCL_RETURN_ERROR_COND ((queues[i]->device == NULL),
                              CL_INVALID_COMMAND_QUEUE);

      POCL_RETURN_ERROR_COND ((queues[i]->context == NULL),
                              CL_INVALID_COMMAND_QUEUE);

      POCL_RETURN_ERROR_COND ((queues[i]->context != ref_ctx),
                              CL_INVALID_COMMAND_QUEUE);
    }

  return CL_SUCCESS;
}

cl_int
pocl_cmdbuf_choose_recording_queue (cl_command_buffer_khr command_buffer,
                                    cl_command_queue *command_queue)
{
  assert (command_queue != NULL);
  cl_command_queue q = *command_queue;

  POCL_RETURN_ERROR_COND ((q == NULL && command_buffer->num_queues != 1),
                          CL_INVALID_COMMAND_QUEUE);

  if (q)
    {
      POCL_RETURN_ERROR_COND (
        (command_buffer->queues[0]->context != q->context),
        CL_INVALID_CONTEXT);
      int queue_in_buffer = 0;
      for (unsigned i = 0; i < command_buffer->num_queues; ++i)
        {
          if (q == command_buffer->queues[i])
            queue_in_buffer = 1;
        }
      POCL_RETURN_ERROR_COND ((!queue_in_buffer), CL_INVALID_COMMAND_QUEUE);
    }
  else
    q = command_buffer->queues[0];

  *command_queue = q;
  return CL_SUCCESS;
}

cl_command_buffer_properties_khr
pocl_cmdbuf_get_property (cl_command_buffer_khr command_buffer,
                          cl_command_buffer_properties_khr name)
{
  for (unsigned i = 0; i < command_buffer->num_properties; ++i)
    {
      if (command_buffer->properties[2 * i] == name)
        return command_buffer->properties[2 * i + 1];
    }
  return 0;
}

/**
 * Check if command buffer is ready to be enqueued or mutated
 *
 * The command buffer must either be in executable state,
 * or it can be in a pending state, if the cmd buf was created with the
 * simultaneous-flag
 *
 */
int
pocl_cmdbuf_is_ready (cl_command_buffer_khr command_buffer)
{
  cl_command_buffer_flags_khr flags
    = (cl_command_buffer_flags_khr)pocl_cmdbuf_get_property (
      command_buffer, CL_COMMAND_BUFFER_FLAGS_KHR);
  int is_ready
    = command_buffer->state == CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR
      || (command_buffer->state == CL_COMMAND_BUFFER_STATE_PENDING_KHR
          && flags & CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR);
  return is_ready;
}
