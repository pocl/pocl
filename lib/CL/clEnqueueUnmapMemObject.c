/* OpenCL runtime library: clEnqueueUnmapMemObject()

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

CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueUnmapMemObject)(cl_command_queue command_queue,
                        cl_mem           memobj,
                        void *           mapped_ptr,
                        cl_uint          num_events_in_wait_list,
                        const cl_event * event_wait_list,
                        cl_event *       event) CL_API_SUFFIX__VERSION_1_0
{
  int errcode;
  cl_device_id device;
  unsigned i;
  mem_mapping_t *mapping = NULL;
  _cl_command_node *cmd = NULL;

  POCL_RETURN_ERROR_COND((memobj == NULL), CL_INVALID_MEM_OBJECT);

  POCL_RETURN_ERROR_COND((command_queue == NULL), CL_INVALID_COMMAND_QUEUE);

  POCL_RETURN_ERROR_ON((command_queue->context != memobj->context),
    CL_INVALID_CONTEXT, "memobj and command_queue are not from the same context\n");

  errcode = pocl_check_event_wait_list (command_queue, num_events_in_wait_list,
                                        event_wait_list);
  if (errcode != CL_SUCCESS)
    return errcode;

  HANDLE_IMAGE1D_BUFFER (memobj);

  POCL_LOCK_OBJ (memobj);
  DL_FOREACH (memobj->mappings, mapping)
    {
      POCL_MSG_PRINT_MEMORY (
          "UnMap %p search Mapping: host_ptr %p offset %zu\n", mapped_ptr,
          mapping->host_ptr, mapping->offset);

      if (mapping->host_ptr == mapped_ptr)
          break;
    }
  POCL_UNLOCK_OBJ (memobj);
  POCL_RETURN_ERROR_ON((mapping == NULL), CL_INVALID_VALUE,
      "Could not find mapping of this memobj\n");

  /* find the index of the device's ptr in the buffer */
  POCL_CHECK_DEV_IN_CMDQ;

  errcode = pocl_create_command (&cmd, command_queue, 
                                 CL_COMMAND_UNMAP_MEM_OBJECT, 
                                 event, num_events_in_wait_list, 
                                 event_wait_list, 1, &memobj);

  if (errcode != CL_SUCCESS)
    goto ERROR;

  cmd->command.unmap.data = command_queue->device->data;
  cmd->command.unmap.memobj = memobj;
  cmd->command.unmap.mapping = mapping;

  POname(clRetainMemObject) (memobj);
  memobj->owning_device = command_queue->device;
  pocl_command_enqueue(command_queue, cmd);

  return CL_SUCCESS;

ERROR:
  POCL_MEM_FREE(cmd);
  return errcode;
}
POsym(clEnqueueUnmapMemObject)
