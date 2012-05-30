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
#include "pocl_icd.h"
#include "utlist.h"
#include <assert.h>

CL_API_ENTRY void * CL_API_CALL
clEnqueueMapBuffer(cl_command_queue command_queue,
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
 
  mapping_info = (mem_mapping_t*) malloc (sizeof (mem_mapping_t));
  if (mapping_info == NULL)
    POCL_ERROR(CL_OUT_OF_HOST_MEMORY);

  /* Ensure the parent buffer is not freed prematurely. */
  clRetainMemObject (buffer);
  if (blocking_map != CL_TRUE)
    {
      POCL_ABORT_UNIMPLEMENTED();
    }
  else
    {
      clFinish (command_queue);
    }

  if (event != NULL)
    {
      *event = (cl_event)malloc (sizeof(struct _cl_event));
      if (*event == NULL)
        return CL_OUT_OF_HOST_MEMORY; 
      POCL_INIT_OBJECT(*event);
      (*event)->queue = command_queue;
      POCL_INIT_ICD_OBJECT(*event);
      clRetainCommandQueue (command_queue);

      POCL_PROFILE_QUEUED;
    }

  POCL_PROFILE_SUBMITTED;
  POCL_PROFILE_RUNNING;

  host_ptr = device->map_mem 
      (device->data, buffer->device_ptrs[device->dev_id], offset, size, 
       buffer->mem_host_ptr);

  POCL_PROFILE_COMPLETE;

  if (host_ptr == NULL)
      POCL_ERROR (CL_MAP_FAILURE);

  mapping_info->host_ptr = host_ptr;
  mapping_info->offset = offset;
  mapping_info->size = size;
  DL_APPEND (buffer->mappings, mapping_info);
  
  buffer->map_count++;

  POCL_SUCCESS ();
  return host_ptr;
}
