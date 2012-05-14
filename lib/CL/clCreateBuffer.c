/* OpenCL runtime library: clCreateBuffer()

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
#include "devices.h"

CL_API_ENTRY cl_mem CL_API_CALL
clCreateBuffer(cl_context context,
               cl_mem_flags flags,
               size_t size,
               void *host_ptr,
               cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  cl_mem mem;
  cl_device_id device;
  void *device_ptr;
  unsigned i, j;

  if (context == NULL)
    POCL_ERROR(CL_INVALID_CONTEXT);

  mem = (cl_mem) malloc(sizeof(struct _cl_mem));
  if (mem == NULL)
    POCL_ERROR(CL_OUT_OF_HOST_MEMORY);

  POCL_INIT_OBJECT(mem);
  mem->parent = NULL;
  mem->map_count = 0;
  mem->mappings = NULL;
  mem->flags = flags;

  /* Store the per device buffer pointers always to a known
     location in the buffer (dev_id), even though the context
     might not contain all the devices. */
  mem->device_ptrs = (void **) malloc(pocl_num_devices * sizeof(void *));
  if (mem->device_ptrs == NULL)
    {
      free(mem);
      POCL_ERROR(CL_OUT_OF_HOST_MEMORY);
    }  

  for (i = 0; i < pocl_num_devices; ++i)
    mem->device_ptrs[i] = NULL;
  
  for (i = 0; i < context->num_devices; ++i)
    {
      if (i > 0)
        clRetainMemObject (mem);
      device = context->devices[i];
      device_ptr = device->malloc(device->data, flags, size, host_ptr);
      if (device_ptr == NULL)
        {
          for (j = 0; j < i; ++j)
            {
              device = context->devices[j];
              device->free(device->data, flags, mem->device_ptrs[device->dev_id]);
            }
          free(mem);
          POCL_ERROR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
        }
      mem->device_ptrs[device->dev_id] = device_ptr;
      /* The device allocator allocated from a device-host shared memory. */
      if (flags & CL_MEM_ALLOC_HOST_PTR ||
          flags & CL_MEM_USE_HOST_PTR)
          mem->mem_host_ptr = device_ptr;      
    }

  mem->size = size;
  mem->context = context;

  POCL_RETAIN_OBJECT(context);
  
  if (errcode_ret != NULL)
    *errcode_ret = CL_SUCCESS;
  return mem;
}
