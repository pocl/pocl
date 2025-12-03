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

#ifndef POCL_CMDBUF_H
#define POCL_CMDBUF_H

#include "pocl_cl.h"

#ifdef __cplusplus
extern "C"
{
#endif

cl_int pocl_command_record (cl_command_buffer_khr command_buffer,
                            _cl_command_node *cmd,
                            cl_sync_point_khr *sync_point);

cl_int
pocl_create_recorded_command (_cl_command_node **cmd,
                              cl_command_buffer_khr command_buffer,
                              cl_command_queue command_queue,
                              cl_command_type command_type,
                              cl_uint num_sync_points_in_wait_list,
                              const cl_sync_point_khr *sync_point_wait_list,
                              pocl_buffer_migration_info *migration_infos);

cl_int
pocl_cmdbuf_choose_recording_queue (cl_command_buffer_khr command_buffer,
                                    cl_command_queue *command_queue);

cl_int pocl_cmdbuf_validate_queue_list (cl_uint num_queues,
                                        const cl_command_queue *queues);

cl_command_buffer_properties_khr
pocl_cmdbuf_get_property (cl_command_buffer_khr command_buffer,
                          cl_command_buffer_properties_khr name);

/* Returns 1 if command buffer is ready to be enqueued or mutated */
int pocl_cmdbuf_is_ready (cl_command_buffer_khr command_buffer);

#ifdef __cplusplus
}
#endif

#define CMDBUF_VALIDATE_COMMON_HANDLES                                        \
  do                                                                          \
    {                                                                         \
      POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_buffer)),         \
                              CL_INVALID_COMMAND_BUFFER_KHR);                 \
      cl_device_id dev = command_buffer->queues[0]->device;                   \
      if (strstr (dev->extensions, "cl_khr_command_buffer_multi_device"))     \
        {                                                                     \
          POCL_RETURN_ERROR_COND (                                            \
            (command_queue == NULL && command_buffer->num_queues > 1),        \
            CL_INVALID_COMMAND_QUEUE);                                        \
          int queue_in_buffer = 0;                                            \
          for (unsigned ii = 0; ii < command_buffer->num_queues; ++ii)        \
            {                                                                 \
              queue_in_buffer                                                 \
                |= (command_queue == command_buffer->queues[ii]);             \
            }                                                                 \
          POCL_RETURN_ERROR_COND (                                            \
            (command_queue != NULL && !queue_in_buffer),                      \
            CL_INVALID_COMMAND_QUEUE);                                        \
        }                                                                     \
      else                                                                    \
        {                                                                     \
          POCL_RETURN_ERROR_ON (                                              \
            (command_queue != NULL), CL_INVALID_COMMAND_QUEUE,                \
            "device does not support cl_khr_command_buffer_multi_device");    \
        }                                                                     \
      if (dev->cmdbuf_mutable_dispatch_capabilities == 0)                     \
        {                                                                     \
          POCL_RETURN_ERROR_COND ((mutable_handle != NULL),                   \
                                  CL_INVALID_VALUE);                          \
        }                                                                     \
      errcode = pocl_cmdbuf_choose_recording_queue (command_buffer,           \
                                                    &command_queue);          \
      if (errcode != CL_SUCCESS)                                              \
        return errcode;                                                       \
    }                                                                         \
  while (0)

#define SETUP_MUTABLE_HANDLE                                                  \
  _cl_command_node *cmd_temp = NULL;                                          \
  if (mutable_handle == NULL)                                                 \
    mutable_handle = &cmd_temp;

#endif
