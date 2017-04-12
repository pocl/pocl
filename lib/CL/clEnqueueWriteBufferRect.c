/* OpenCL runtime library: clEnqueueWriteBufferRect()

   Copyright (c) 2011 Universidad Rey Juan Carlos
   
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "pocl_cl.h"
#include <assert.h>
#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueWriteBufferRect)(cl_command_queue command_queue,
                         cl_mem buffer,
                         cl_bool blocking_write,
                         const size_t *buffer_origin,
                         const size_t *host_origin, 
                         const size_t *region,
                         size_t buffer_row_pitch,
                         size_t buffer_slice_pitch,
                         size_t host_row_pitch,
                         size_t host_slice_pitch,
                         const void *ptr,
                         cl_uint num_events_in_wait_list,
                         const cl_event *event_wait_list,
                         cl_event *event) CL_API_SUFFIX__VERSION_1_1
{
  cl_device_id device;
  unsigned i;
  _cl_command_node *cmd;
  cl_int errcode;

  POCL_RETURN_ERROR_COND((command_queue == NULL), CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_COND((buffer == NULL), CL_INVALID_MEM_OBJECT);

  POCL_RETURN_ERROR_ON((buffer->type != CL_MEM_OBJECT_BUFFER),
      CL_INVALID_MEM_OBJECT, "buffer is not a CL_MEM_OBJECT_BUFFER\n");

  POCL_RETURN_ERROR_ON((buffer->flags & (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS)),
    CL_INVALID_OPERATION, "buffer has been created with CL_MEM_HOST_READ_ONLY "
    "or CL_MEM_HOST_NO_ACCESS\n");

  POCL_RETURN_ERROR_ON((command_queue->context != buffer->context),
    CL_INVALID_CONTEXT, "buffer and command_queue are not from the same context\n");

  errcode = pocl_check_event_wait_list(command_queue, num_events_in_wait_list, event_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;


  POCL_RETURN_ERROR_COND((ptr == NULL), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND((buffer_origin == NULL), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND((host_origin == NULL), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND((region == NULL), CL_INVALID_VALUE);

  size_t region_bytes = region[0] * region[1] * region[2];
  POCL_RETURN_ERROR_ON((region_bytes <= 0), CL_INVALID_VALUE, "All items in region must be >0\n");

  if (pocl_buffer_boundcheck_3d(buffer->size, buffer_origin, region, &buffer_row_pitch,
      &buffer_slice_pitch, "") != CL_SUCCESS) return CL_INVALID_VALUE;

  if (pocl_buffer_boundcheck_3d(((size_t)-1), host_origin, region, &host_row_pitch,
      &host_slice_pitch, "") != CL_SUCCESS) return CL_INVALID_VALUE;




  device = POCL_REAL_DEV(command_queue->device);

  for (i = 0; i < command_queue->context->num_devices; ++i)
    {
      if (command_queue->context->devices[i] == device)
        break;
    }
  assert(i < command_queue->context->num_devices);

  POname(clRetainMemObject) (buffer);

  pocl_create_command (&cmd, command_queue, CL_COMMAND_WRITE_BUFFER_RECT,
                       event, num_events_in_wait_list, event_wait_list, 1,
                       &buffer);

  cmd->command.write_image.device_ptr =
    buffer->device_ptrs[device->dev_id].mem_ptr;
  cmd->command.write_image.host_ptr = ptr;
  memcpy (&cmd->command.write_image.origin,
          buffer_origin, sizeof (size_t) * 3);
  memcpy (&cmd->command.write_image.h_origin,
          host_origin, sizeof (size_t) * 3);
  memcpy (&cmd->command.write_image.region, region, sizeof (size_t) * 3);
  cmd->command.write_image.h_rowpitch = host_row_pitch;
  cmd->command.write_image.h_slicepitch = host_slice_pitch;
  cmd->command.write_image.b_rowpitch = buffer_row_pitch;
  cmd->command.write_image.b_slicepitch = buffer_slice_pitch;
  cmd->command.write_image.buffer = buffer;

  buffer->owning_device = command_queue->device;
  pocl_command_enqueue (command_queue, cmd);

  if (blocking_write)
    POname(clFinish)(command_queue);

  return CL_SUCCESS;
}
POsym(clEnqueueWriteBufferRect)
