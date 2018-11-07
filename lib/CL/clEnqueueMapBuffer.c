/* OpenCL runtime library: clEnqueueMapBuffer()

   Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
   
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
#include "utlist.h"
#include <assert.h>
#include "pocl_util.h"
#include "pocl_shared.h"

CL_API_ENTRY void * CL_API_CALL
POname(clEnqueueMapBuffer)(cl_command_queue command_queue,
                   cl_mem           buffer,
                   cl_bool          blocking_map,
                   cl_map_flags     map_flags,
                   size_t           offset,
                   size_t           size,
                   cl_uint          num_events_in_wait_list,
                   const cl_event * event_wait_list,
                   cl_event *       event,
                   cl_int *         errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  cl_int errcode = CL_SUCCESS;
  cl_device_id device = NULL;
  cl_int mapping_result = CL_FAILED;
  mem_mapping_t *mapping_info = NULL;
  unsigned i;
  _cl_command_node *cmd = NULL;
  /* need to release the memobject before returning? */
  int must_release = 0;

  POCL_GOTO_ERROR_COND((command_queue == NULL), CL_INVALID_COMMAND_QUEUE);

  POCL_GOTO_ERROR_COND((buffer == NULL), CL_INVALID_MEM_OBJECT);

  POCL_GOTO_ON_SUB_MISALIGN (buffer, command_queue);

  POCL_GOTO_ERROR_COND((size == 0), CL_INVALID_VALUE);

  POCL_GOTO_ERROR_ON((buffer->type != CL_MEM_OBJECT_BUFFER),
      CL_INVALID_MEM_OBJECT, "buffer is not a CL_MEM_OBJECT_BUFFER\n");

  POCL_GOTO_ERROR_ON((command_queue->context != buffer->context),
    CL_INVALID_CONTEXT, "buffer and command_queue are not from the same context\n");

  errcode = pocl_check_event_wait_list (command_queue, num_events_in_wait_list,
                                        event_wait_list);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  errcode = pocl_buffer_boundcheck(buffer, offset, size);
  if (errcode != CL_SUCCESS)
    goto ERROR;

  POCL_GOTO_ERROR_ON((buffer->flags & (CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS) &&
    map_flags & CL_MAP_READ), CL_INVALID_OPERATION, "buffer has been created with "
    "CL_MEM_HOST_WRITE_ONLY or CL_MEM_HOST_NO_ACCESS and CL_MAP_READ is set in map_flags\n");

  POCL_GOTO_ERROR_ON((buffer->flags & (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS) &&
      map_flags & (CL_MAP_WRITE | CL_MAP_WRITE_INVALIDATE_REGION)), CL_INVALID_OPERATION,
      "buffer has been created with CL_MEM_HOST_READ_ONL or CL_MEM_HOST_NO_ACCESS "
      "and CL_MAP_WRITE or CL_MAP_WRITE_INVALIDATE_REGION is set in map_flags\n");

  POCL_CHECK_DEV_IN_CMDQ;

  POCL_CONVERT_SUBBUFFER_OFFSET (buffer, offset);

  /* Ensure the parent buffer is not freed prematurely. */
  POname(clRetainMemObject) (buffer);
  must_release = 1;

  mapping_info = (mem_mapping_t*) calloc (1, sizeof (mem_mapping_t));
  POCL_GOTO_ERROR_COND ((mapping_info == NULL), CL_OUT_OF_HOST_MEMORY);

  mapping_info->host_ptr = NULL;
  mapping_info->map_flags = map_flags;
  mapping_info->offset = offset;
  mapping_info->size = size;

  // TODO COPY_HOST_PTR
  if ((buffer->flags & CL_MEM_USE_HOST_PTR)
      || (buffer->flags & CL_MEM_ALLOC_HOST_PTR))
    {
      /* In this case it should use the given host_ptr + offset as
         the mapping area in the host memory. */   
      assert (buffer->mem_host_ptr != NULL);
      mapping_info->host_ptr = (char*)buffer->mem_host_ptr + offset;
    }
  else
    {
      /* The first call to the device driver's map mem tells where
         the mapping will be stored in the host memory.
         When return value (mapping_info->host_ptr) is non-NULL, the
         buffer will be mapped there (assumed it will succeed).  */
      mapping_info->host_ptr = NULL;
      mapping_result = device->ops->map_mem (
          device->data,
          &buffer->device_ptrs[device->dev_id],
          buffer,
          mapping_info);
    }

  POCL_GOTO_ERROR_ON ((mapping_info->host_ptr == NULL), CL_MAP_FAILURE,
                      "device map failed\n");

  errcode = pocl_create_command (&cmd, command_queue, CL_COMMAND_MAP_BUFFER, 
                                 event, num_events_in_wait_list, 
                                 event_wait_list, 1, &buffer);
  
  if (errcode != CL_SUCCESS)
      goto ERROR;

  cmd->command.map.mem_id = &buffer->device_ptrs[device->dev_id];
  cmd->command.map.mapping = mapping_info;

  POCL_LOCK_OBJ (buffer);
  DL_APPEND (buffer->mappings, mapping_info);
  POCL_UNLOCK_OBJ (buffer);

  POCL_MSG_PRINT_MEMORY ("Buffer %p New Mapping: host_ptr %p offset %zu\n",
                         buffer, mapping_info->host_ptr, mapping_info->offset);

  buffer->owning_device = command_queue->device;
  pocl_command_enqueue(command_queue, cmd);

  if (blocking_map)
    {
      POname (clFinish) (command_queue);
    }

  if (errcode_ret)
    *errcode_ret = errcode;

  POCL_SUCCESS ();
  return mapping_info->host_ptr;

ERROR:
  if (must_release)
    POname(clReleaseMemObject)(buffer);
  if (mapping_result == CL_SUCCESS)
    device->ops->unmap_mem (device->data,
                            &buffer->device_ptrs[device->dev_id],
                            buffer,
                            mapping_info);
  POCL_MEM_FREE(mapping_info);
  POCL_MEM_FREE (cmd);
  if (errcode_ret)
    *errcode_ret = errcode;

  return NULL;
}
POsym(clEnqueueMapBuffer)
