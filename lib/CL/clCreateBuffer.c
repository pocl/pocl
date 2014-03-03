/* OpenCL runtime library: clCreateBuffer()

   Copyright (c) 2011-2013 Universidad Rey Juan Carlos
                           Pekka Jääskeläinen / Tampere University of Technology
   
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
POname(clCreateBuffer)(cl_context context,
               cl_mem_flags flags,
               size_t size,
               void *host_ptr,
               cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  cl_mem mem;
  cl_device_id device;
  void *device_ptr;
  int errcode;
  unsigned i, j;

  if (size == 0)
    {
      errcode = CL_INVALID_BUFFER_SIZE;
      goto ERROR;
    }

  if (context == NULL)
    {
      errcode = CL_INVALID_CONTEXT;
      goto ERROR;
    }
  
  mem = (cl_mem) malloc(sizeof(struct _cl_mem));
  if (mem == NULL)
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR;
    }
  
  if (size == 0)
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR_CLEAN_MEM;
    }
  
  if (flags == 0)
    flags = CL_MEM_READ_WRITE;
  
  /* validate flags */
  
  if (flags > (1<<10)-1)
    {
      errcode = CL_INVALID_VALUE;
      goto ERROR_CLEAN_MEM;
    }
  
  if (((flags & CL_MEM_READ_WRITE) &&
       (flags & CL_MEM_WRITE_ONLY ||
        flags & CL_MEM_READ_ONLY)) ||
      
      ((flags & CL_MEM_READ_ONLY) &&
       (flags & CL_MEM_WRITE_ONLY)) ||
      
      ((flags & CL_MEM_USE_HOST_PTR) &&
       (flags & CL_MEM_ALLOC_HOST_PTR ||
        flags & CL_MEM_COPY_HOST_PTR)) || 
      
      ((flags & CL_MEM_HOST_WRITE_ONLY) &&
       (flags & CL_MEM_HOST_READ_ONLY)) ||
      
      ((flags & CL_MEM_HOST_NO_ACCESS) &&
       ((flags & CL_MEM_HOST_READ_ONLY) ||
        (flags & CL_MEM_HOST_WRITE_ONLY))))
    {
      errcode = CL_INVALID_VALUE;
      goto ERROR_CLEAN_MEM;
    }
  
  if (host_ptr == NULL)
    {
      if ((flags & CL_MEM_USE_HOST_PTR) ||
          (flags & CL_MEM_COPY_HOST_PTR))
        {
          errcode = CL_INVALID_HOST_PTR;
          goto ERROR_CLEAN_MEM;
        }
    }
  else
    {
      if ((~flags & CL_MEM_USE_HOST_PTR) &&
          (~flags & CL_MEM_COPY_HOST_PTR))
        {
          errcode = CL_INVALID_HOST_PTR;
          goto ERROR_CLEAN_MEM;
        }
    }
  
  for (i = 0; i < context->num_devices; ++i)
    {
      cl_ulong max_alloc;
      
      POname(clGetDeviceInfo) (context->devices[i], 
                               CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), 
                               &max_alloc, NULL);
      
      if (size > max_alloc)
        {
          errcode = CL_INVALID_BUFFER_SIZE;
          goto ERROR_CLEAN_MEM;
        }
    }
  
  POCL_INIT_OBJECT(mem);
  mem->parent = NULL;
  mem->map_count = 0;
  mem->mappings = NULL;
  mem->type = CL_MEM_OBJECT_BUFFER;
  mem->flags = flags;
  mem->is_image = CL_FALSE;
  
  /* Store the per device buffer pointers always to a known
     location in the buffer (dev_id), even though the context
     might not contain all the devices. */
  mem->device_ptrs = 
    (pocl_mem_identifier*) malloc(pocl_num_devices * sizeof(pocl_mem_identifier));
  if (mem->device_ptrs == NULL)
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR_CLEAN_MEM;
    }  
  
  for (i = 0; i < pocl_num_devices; ++i)
    {
      mem->device_ptrs[i].mem_ptr = NULL;
    }

  mem->mem_host_ptr = host_ptr; 
  mem->size = size;
  mem->context = context;
  
  for (i = 0; i < context->num_devices; ++i)
    {
      if (i > 0)
        POname(clRetainMemObject) (mem);
      device = context->devices[i];
      assert (device->ops->alloc_mem_obj != NULL);
      if (device->ops->alloc_mem_obj (device, mem) != CL_SUCCESS)
        {
          errcode = CL_MEM_OBJECT_ALLOCATION_FAILURE;
          goto ERROR_CLEAN_MEM_AND_DEVICE;
        }
    }
  
  POCL_RETAIN_OBJECT(context);
  
  if (errcode_ret != NULL)
    *errcode_ret = CL_SUCCESS;
  return mem;
  
 ERROR_CLEAN_MEM_AND_DEVICE:
  for (j = 0; j < i; ++j)
    {
      device = context->devices[j];
      device->ops->free(device->data, flags, 
                        mem->device_ptrs[device->dev_id].mem_ptr);
    }
 ERROR_CLEAN_MEM:
  free(mem);
 ERROR:
  if(errcode_ret)
    {
      *errcode_ret = errcode;
    }
  return NULL;
}
POsym(clCreateBuffer)
