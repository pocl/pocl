/* OpenCL runtime library: clEnqueueReadBuffer()

   Copyright (c) 2011 Universidad Rey Juan Carlos
                 2014 Pekka Jääskeläinen / Intel Finland Oy

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
#include "pocl_shared.h"
#include "pocl_util.h"

cl_int
pocl_validate_read_buffer (cl_command_queue command_queue,
                           cl_mem buffer,
                           size_t offset,
                           size_t size,
                           void *ptr)
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (buffer)),
                          CL_INVALID_MEM_OBJECT);

  POCL_RETURN_ON_SUB_MISALIGN (buffer, command_queue);

  POCL_RETURN_ERROR_ON (
      (command_queue->context != buffer->context), CL_INVALID_CONTEXT,
      "buffer and command_queue are not from the same context\n");

  POCL_RETURN_ERROR_ON (
      (buffer->flags & (CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS)),
      CL_INVALID_OPERATION,
      "buffer has been created with CL_MEM_HOST_WRITE_ONLY "
      "or CL_MEM_HOST_NO_ACCESS\n");

  POCL_RETURN_ERROR_COND((ptr == NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND ((size == 0), CL_INVALID_VALUE);

  if (pocl_buffer_boundcheck (buffer, offset, size) != CL_SUCCESS)
    return CL_INVALID_VALUE;

  return CL_SUCCESS;
}

cl_int
pocl_read_buffer_common (cl_command_buffer_khr command_buffer,
                         cl_command_queue command_queue,
                         cl_mem buffer,
                         size_t offset,
                         size_t size,
                         void *ptr,
                         cl_uint num_items_in_wait_list,
                         const cl_event *event_wait_list,
                         cl_event *event,
                         const cl_sync_point_khr *sync_point_wait_list,
                         cl_sync_point_khr *sync_point,
                         _cl_command_node **cmd)
{
  POCL_VALIDATE_WAIT_LIST_PARAMS;

  unsigned i;
  cl_device_id device;
  POCL_CHECK_DEV_IN_CMDQ;

  cl_int errcode
      = pocl_validate_read_buffer (command_queue, buffer, offset, size, ptr);
  if (errcode != CL_SUCCESS)
    return errcode;

  POCL_RETURN_ERROR_ON (
      (buffer->size > command_queue->device->max_mem_alloc_size),
      CL_OUT_OF_RESOURCES,
      "buffer is larger than device's MAX_MEM_ALLOC_SIZE\n");

  char rdonly = 1;

  if (command_buffer == NULL)
    {
      errcode = pocl_check_event_wait_list (
          command_queue, num_items_in_wait_list, event_wait_list);
      if (errcode != CL_SUCCESS)
        return errcode;
      errcode = pocl_create_command (
        cmd, command_queue, CL_COMMAND_READ_BUFFER, event,
        num_items_in_wait_list, event_wait_list,
        pocl_append_unique_migration_info (NULL, buffer, rdonly));
    }
  else
    {
      errcode = pocl_create_recorded_command (
        cmd, command_buffer, command_queue, CL_COMMAND_READ_BUFFER,
        num_items_in_wait_list, sync_point_wait_list,
        pocl_append_unique_migration_info (NULL, buffer, rdonly));
    }
  if (errcode != CL_SUCCESS)
    return errcode;

  _cl_command_node *c = *cmd;

  c->command.read.dst_host_ptr = ptr;
  c->command.read.offset = offset;
  c->command.read.size = size;
  c->command.read.src = buffer;

  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
POname (clEnqueueReadBuffer) (cl_command_queue command_queue,
                              cl_mem buffer,
                              cl_bool blocking_read,
                              size_t offset,
                              size_t size,
                              void *ptr,
                              cl_uint num_events_in_wait_list,
                              const cl_event *event_wait_list,
                              cl_event *event) CL_API_SUFFIX__VERSION_1_0
{
  cl_int errcode;
  _cl_command_node *cmd = NULL;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_COND ((*(command_queue->device->available) == CL_FALSE),
                          CL_DEVICE_NOT_AVAILABLE);

  errcode = pocl_read_buffer_common (NULL, command_queue, buffer, offset, size,
                                     ptr, num_events_in_wait_list,
                                     event_wait_list, event, NULL, NULL, &cmd);
  if (errcode != CL_SUCCESS)
    return errcode;

  pocl_command_enqueue (command_queue, cmd);

  if (blocking_read)
    POname(clFinish) (command_queue);

  return CL_SUCCESS;
}
POsym(clEnqueueReadBuffer)
