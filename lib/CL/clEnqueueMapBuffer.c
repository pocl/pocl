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
#include "clEnqueueMapBuffer.h"

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
  cl_device_id device;
  void *host_ptr = NULL;
  mem_mapping_t *mapping_info = NULL;
  int errcode;
  _cl_command_node *cmd = NULL;

  if (buffer == NULL)
    POCL_ERROR(CL_INVALID_MEM_OBJECT);

  if (command_queue == NULL || command_queue->device == NULL ||
      command_queue->context == NULL)
    POCL_ERROR(CL_INVALID_COMMAND_QUEUE);

  if (command_queue->context != buffer->context)
    POCL_ERROR(CL_INVALID_CONTEXT);

  if (offset + size > buffer->size)
    POCL_ERROR(CL_INVALID_VALUE);

  if (buffer->flags & (CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS) &&
      map_flags & CL_MAP_READ)
    POCL_ERROR(CL_INVALID_OPERATION);

  if (buffer->flags & (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_NO_ACCESS) &&
      map_flags & (CL_MAP_WRITE | CL_MAP_WRITE_INVALIDATE_REGION))
    POCL_ERROR(CL_INVALID_OPERATION);

  device = command_queue->device;
 
  /* Ensure the parent buffer is not freed prematurely. */
  POname(clRetainMemObject) (buffer);

  mapping_info = (mem_mapping_t*) calloc (1, sizeof (mem_mapping_t));
  if (mapping_info == NULL)
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR;
    }

  if (buffer->flags & CL_MEM_USE_HOST_PTR)
    {
      /* In this case it should use the given host_ptr + offset as
         the mapping area in the host memory. */   
      assert (buffer->mem_host_ptr != NULL);
      host_ptr = buffer->mem_host_ptr + offset;
    }
  else
    {
      /* The first call to the device driver's map mem tells where
         the mapping will be stored (the last argument is NULL) in
         the host memory. When the last argument is non-NULL, the
         buffer will be mapped there (assumed it will succeed).  */
      
      host_ptr = device->ops->map_mem 
        (device->data, buffer->device_ptrs[device->dev_id].mem_ptr, offset, 
         size, 
         NULL);
    }

  if (host_ptr == NULL)
    {
      POCL_UPDATE_EVENT_COMPLETE(event, command_queue);
      errcode = CL_MAP_FAILURE;
      goto ERROR;
    }

  errcode = pocl_create_command (&cmd, command_queue, CL_COMMAND_MAP_BUFFER, 
                                 event, num_events_in_wait_list, 
                                 event_wait_list);
  
  if (errcode != CL_SUCCESS)
      goto ERROR;
  
  cmd->command.map.buffer = buffer;
  cmd->command.map.mapping = mapping_info;
    
  mapping_info->host_ptr = host_ptr;
  mapping_info->offset = offset;
  mapping_info->size = size;
  DL_APPEND (buffer->mappings, mapping_info);  
  pocl_command_enqueue(command_queue, cmd);

  if (blocking_map != CL_TRUE)
    {
      POCL_SUCCESS ();
      return mapping_info->host_ptr;
    }
  else
    errcode = POname(clFinish) (command_queue);

  if (errcode_ret)
    *errcode_ret = errcode;
  

  POCL_SUCCESS ();
  return host_ptr;

 ERROR:
  if (event)
    free(*event);
  free(cmd);
  free(event);
  free (mapping_info);
  if (errcode_ret)
    *errcode_ret = errcode;
  return NULL;
}
POsym(clEnqueueMapBuffer)

void*
pocl_map_mem_cmd(cl_device_id device, 
                 cl_mem buffer, 
                 mem_mapping_t *mapping_info) {


  
  /* The second call ensures the memory is flushed/updated to the
     host location. */
  device->ops->map_mem 
    (device->data, buffer->device_ptrs[device->dev_id].mem_ptr, 
     mapping_info->offset, mapping_info->size, mapping_info->host_ptr);
  
  buffer->map_count++;
  return mapping_info->host_ptr;

}
