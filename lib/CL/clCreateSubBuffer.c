/* OpenCL runtime library: clCreateSubBuffer()

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
#include "devices.h"

/* NOTE: this function is untested! */
CL_API_ENTRY cl_mem CL_API_CALL
POname(clCreateSubBuffer)(cl_mem                   buffer,
                  cl_mem_flags             flags,
                  cl_buffer_create_type    buffer_create_type,
                  const void *             buffer_create_info,
                  cl_int *                 errcode_ret) CL_API_SUFFIX__VERSION_1_1 
{
  cl_device_id device;
  cl_mem mem;
  int errcode;
  int i;

  if (buffer == NULL || buffer->parent != NULL)
  {
    errcode = CL_INVALID_MEM_OBJECT;
    goto ERROR;
  }

  if (buffer_create_type != CL_BUFFER_CREATE_TYPE_REGION ||
      buffer_create_info == NULL)
  {
    errcode = CL_INVALID_VALUE;
    goto ERROR;
  }

  cl_buffer_region* info = 
    (cl_buffer_region*)buffer_create_info;

  if (info->size == 0)
  {
    errcode = CL_INVALID_BUFFER_SIZE;
    goto ERROR;
  }
  
  if (info->size + info->origin > buffer->size)
  {
    errcode = CL_INVALID_VALUE;
    goto ERROR;
  }

  mem = (cl_mem) malloc(sizeof(struct _cl_mem));
  if (mem == NULL)
  {
    errcode = CL_OUT_OF_HOST_MEMORY;
    goto ERROR;
  }

  POCL_INIT_OBJECT(mem);
  mem->mappings = NULL;
  mem->parent = buffer;

  mem->type = CL_MEM_OBJECT_BUFFER;
  mem->size = info->size;
  mem->context = buffer->context;

  if ((buffer->flags & CL_MEM_WRITE_ONLY &&
       flags & (CL_MEM_READ_WRITE | CL_MEM_READ_ONLY)) ||
      (buffer->flags & CL_MEM_READ_ONLY &&
       flags & (CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY)) ||
      (flags & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR | 
                CL_MEM_COPY_HOST_PTR)))
  {
    errcode = CL_INVALID_VALUE;
    goto ERROR_CLEAN_MEM;
  }

  if ((buffer->flags & CL_MEM_HOST_WRITE_ONLY &&
       flags & CL_MEM_HOST_READ_ONLY) ||
      (buffer->flags & CL_MEM_HOST_READ_ONLY &&
       flags & CL_MEM_HOST_WRITE_ONLY) ||
      (buffer->flags & CL_MEM_HOST_NO_ACCESS &&
       flags & (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_WRITE_ONLY)))
  {
    errcode = CL_INVALID_VALUE;
    goto ERROR_CLEAN_MEM;
  }

  if ((flags & CL_MEM_READ_WRITE) |
      (flags & CL_MEM_READ_ONLY) |
      (flags & CL_MEM_WRITE_ONLY)) 
    { 
      mem->flags =
        (flags & CL_MEM_READ_WRITE) |
        (flags & CL_MEM_READ_ONLY) |
        (flags & CL_MEM_WRITE_ONLY);
    } else 
    {
      mem->flags =
        (buffer->flags & CL_MEM_READ_WRITE) |
        (buffer->flags & CL_MEM_READ_ONLY) |
        (buffer->flags & CL_MEM_WRITE_ONLY);
    }

  mem->flags = mem->flags |
    (buffer->flags & CL_MEM_USE_HOST_PTR) |
    (buffer->flags & CL_MEM_ALLOC_HOST_PTR) |
    (buffer->flags & CL_MEM_COPY_HOST_PTR);

  if (mem->flags & CL_MEM_USE_HOST_PTR)
    {
      mem->mem_host_ptr = buffer->mem_host_ptr + info->origin;
    }

  mem->device_ptrs = 
    malloc(pocl_num_devices * sizeof(pocl_mem_identifier *));
  if (mem->device_ptrs == NULL)
    {
        errcode = CL_OUT_OF_HOST_MEMORY;
        goto ERROR_CLEAN_MEM;
    }

  for (i = 0; i < pocl_num_devices; ++i)
    mem->device_ptrs[i].mem_ptr = NULL;
  
  for (i = 0; i < mem->context->num_devices; ++i)
    {
      device = mem->context->devices[i];

      /* device_ptrs can contain a pointer to a book keeping
         structure instead of the actual buffer in memory, therefore
         call the device driver layer to produce the sub buffer
         reference */
      if (device->ops->create_sub_buffer != NULL)
        mem->device_ptrs[device->dev_id].mem_ptr = 
          device->ops->create_sub_buffer
          (device->data, buffer->device_ptrs[device->dev_id].mem_ptr, 
           info->origin, info->size);
      else
        mem->device_ptrs[device->dev_id].mem_ptr = 
          buffer->device_ptrs[device->dev_id].mem_ptr + info->origin;
    }

  POCL_RETAIN_OBJECT(mem->parent);
  POCL_RETAIN_OBJECT(mem->context);

  if (errcode_ret != NULL)
    *errcode_ret = CL_SUCCESS;
  return mem;

#if 0
ERROR_CLEAN_MEM_AND_DEVPTR:
    free(mem->device_ptrs);
#endif
ERROR_CLEAN_MEM:
    free(mem);
ERROR:
  if(errcode_ret)
  {
    *errcode_ret = errcode;
  }
  return NULL;
}
POsym(clCreateSubBuffer)
