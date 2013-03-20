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

CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueUnmapMemObject)(cl_command_queue command_queue,
                        cl_mem           memobj,
                        void *           mapped_ptr,
                        cl_uint          num_events_in_wait_list,
                        const cl_event * event_wait_list,
                        cl_event *       event) CL_API_SUFFIX__VERSION_1_0
{
  cl_device_id device_id;
  unsigned i;
  mem_mapping_t *mapping = NULL;

  if (memobj == NULL)
    return CL_INVALID_MEM_OBJECT;

  if (command_queue == NULL || command_queue->device == NULL ||
      command_queue->context == NULL)
    return CL_INVALID_COMMAND_QUEUE;

  if (command_queue->context != memobj->context)
    return CL_INVALID_CONTEXT;

  DL_FOREACH (memobj->mappings, mapping)
    {
      if (mapping->host_ptr == mapped_ptr)
          break;
    }
  if (mapping == NULL)
    return CL_INVALID_VALUE;

  /* find the index of the device's ptr in the buffer */
  device_id = command_queue->device;
  for (i = 0; i < command_queue->context->num_devices; ++i)
    {
      if (command_queue->context->devices[i] == device_id)
        break;
    }

  assert(i < command_queue->context->num_devices);

  if (event != NULL)
    {
      *event = (cl_event)malloc (sizeof(struct _cl_event));
      if (*event == NULL)
        return CL_OUT_OF_HOST_MEMORY; 
      POCL_INIT_OBJECT(*event);
      (*event)->queue = command_queue;
      (*event)->command_type = CL_COMMAND_UNMAP_MEM_OBJECT;
      POname(clRetainCommandQueue) (command_queue);

      POCL_UPDATE_EVENT_QUEUED;
    }

  POCL_UPDATE_EVENT_SUBMITTED;
  POCL_UPDATE_EVENT_RUNNING;

  if (memobj->flags & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR))
    {
      /* TODO: should we ensure the device global region is updated from
         the host memory? How does the specs define it,
         can the host_ptr be assumed to point to the host and the
         device accessible memory or just point there until the
         kernel(s) get executed or similar? */
      /* Assume the region is automatically up to date. */
    } else 
    {
      /* TODO: fixme. The offset computation must be done at the device driver. */
      if (device_id->unmap_mem != NULL)        
        device_id->unmap_mem
          (device_id->data, mapping->host_ptr, memobj->device_ptrs[device_id->dev_id] + mapping->offset, 
           mapping->size);
    }

  POCL_UPDATE_EVENT_COMPLETE;

  DL_DELETE(memobj->mappings, mapping);
  memobj->map_count--;
  POname(clReleaseMemObject) (memobj);

  return CL_SUCCESS;
}
POsym(clEnqueueUnmapMemObject)
