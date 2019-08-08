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

  POCL_GOTO_ERROR_ON (
      (buffer->size > command_queue->device->global_memory->max_alloc),
      CL_OUT_OF_RESOURCES,
      "buffer is larger than device's MAX_MEM_ALLOC_SIZE\n");

  mapping_info = (mem_mapping_t*) calloc (1, sizeof (mem_mapping_t));
  POCL_GOTO_ERROR_COND ((mapping_info == NULL), CL_OUT_OF_HOST_MEMORY);

  POCL_LOCK_OBJ (buffer);
  /* increment refcount twice.
     one is for duration of Map command, and is implicitly decreased
     after EnqueueMap is finished; the second one is for the duration
     of mapping, and is decreased by UnMap. */
  POCL_RETAIN_OBJECT_UNLOCKED (buffer);
  POCL_RETAIN_OBJECT_UNLOCKED (buffer);
  must_release = 1;

  assert (buffer->mem_host_ptr != NULL);
  mapping_info->host_ptr = (char *)buffer->mem_host_ptr + offset;
  mapping_info->map_flags = map_flags;
  mapping_info->offset = offset;
  mapping_info->size = size;
  DL_APPEND (buffer->mappings, mapping_info);
  ++buffer->map_count;
  buffer->owning_device = command_queue->device;
  POCL_UNLOCK_OBJ (buffer);

  errcode = pocl_create_command (&cmd, command_queue, CL_COMMAND_MAP_BUFFER, 
                                 event, num_events_in_wait_list, 
                                 event_wait_list, 1, &buffer);
  
  if (errcode != CL_SUCCESS)
      goto ERROR;

  cmd->command.map.mem_id = &buffer->gmem_ptrs[device->global_mem_id];
  cmd->command.map.mapping = mapping_info;

  POCL_MSG_PRINT_MEMORY ("Buffer %p New Mapping: host_ptr %p offset %zu\n",
                         buffer, mapping_info->host_ptr, mapping_info->offset);

  pocl_command_enqueue (command_queue, cmd);

  if (blocking_map)
    {
      POname (clFinish) (command_queue);
    }

  if (errcode_ret)
    *errcode_ret = CL_SUCCESS;

  return mapping_info->host_ptr;

ERROR:
  if (must_release)
    {
      POCL_LOCK_OBJ (buffer);
      assert (mapping_info);
      DL_DELETE (buffer->mappings, mapping_info);
      --buffer->map_count;
      POCL_UNLOCK_OBJ (buffer);
      POname (clReleaseMemObject) (buffer);
    }
  POCL_MEM_FREE (mapping_info);
  POCL_MEM_FREE (cmd);
  if (errcode_ret)
    *errcode_ret = errcode;

  return NULL;
}
POsym(clEnqueueMapBuffer)
