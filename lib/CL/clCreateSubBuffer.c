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

#include "devices.h"
#include "pocl_cl.h"
#include "pocl_util.h"

/* NOTE: this function is untested! */
CL_API_ENTRY cl_mem CL_API_CALL
POname(clCreateSubBuffer)(cl_mem                   buffer,
                  cl_mem_flags             flags,
                  cl_buffer_create_type    buffer_create_type,
                  const void *             buffer_create_info,
                  cl_int *                 errcode_ret) CL_API_SUFFIX__VERSION_1_1 
{
  cl_device_id device;
  cl_mem mem = NULL;
  int errcode;
  unsigned i;

  HANDLE_IMAGE1D_BUFFER (buffer);

  POCL_GOTO_ERROR_COND((buffer == NULL), CL_INVALID_MEM_OBJECT);

  POCL_GOTO_ERROR_ON((buffer->parent != NULL), CL_INVALID_MEM_OBJECT,
    "buffer is already a sub-buffer\n");

  POCL_GOTO_ERROR_COND((buffer_create_info == NULL), CL_INVALID_VALUE);

  POCL_GOTO_ERROR_COND((buffer_create_type != CL_BUFFER_CREATE_TYPE_REGION),
    CL_INVALID_VALUE);

  cl_buffer_region* info = 
    (cl_buffer_region*)buffer_create_info;

  POCL_GOTO_ERROR_ON((info->size == 0), CL_INVALID_BUFFER_SIZE,
    "buffer_create_info->size == 0\n");
  
  POCL_GOTO_ERROR_ON((info->size + info->origin > buffer->size), CL_INVALID_VALUE,
    "buffer_create_info->size+origin > buffer size\n");

  mem = (cl_mem) malloc(sizeof(struct _cl_mem));
  if (mem == NULL)
  {
    errcode = CL_OUT_OF_HOST_MEMORY;
    goto ERROR;
  }

  POCL_INIT_OBJECT(mem);
  mem->mappings = NULL;
  mem->destructor_callbacks = NULL;
  mem->parent = buffer;
  mem->type = CL_MEM_OBJECT_BUFFER;
  mem->size = info->size;
  mem->origin = info->origin;
  mem->context = buffer->context;
  mem->latest_event = NULL;
  mem->owning_device = buffer->owning_device;
  mem->is_pipe = CL_FALSE;
  mem->mem_host_ptr = NULL;

  POCL_GOTO_ERROR_ON((buffer->flags & CL_MEM_WRITE_ONLY &&
       flags & (CL_MEM_READ_WRITE | CL_MEM_READ_ONLY)), CL_INVALID_VALUE,
       "Invalid flags: buffer is CL_MEM_WRITE_ONLY, requested sub-buffer "
       "CL_MEM_READ_WRITE or CL_MEM_READ_ONLY\n");

  POCL_GOTO_ERROR_ON((buffer->flags & CL_MEM_READ_ONLY &&
       flags & (CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY)), CL_INVALID_VALUE,
       "Invalid flags: buffer is CL_MEM_READ_ONLY, requested sub-buffer "
       "CL_MEM_READ_WRITE or CL_MEM_WRITE_ONLY\n");

  POCL_GOTO_ERROR_ON((flags & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR |
                CL_MEM_COPY_HOST_PTR)), CL_INVALID_VALUE,
                "Invalid flags: (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR | "
                "CL_MEM_COPY_HOST_PTR)\n");

  POCL_GOTO_ERROR_ON((buffer->flags & CL_MEM_HOST_WRITE_ONLY &&
       flags & CL_MEM_HOST_READ_ONLY), CL_INVALID_VALUE,
       "Invalid flags: buffer is CL_MEM_HOST_WRITE_ONLY, requested sub-buffer "
       "CL_MEM_HOST_READ_ONLY\n");

  POCL_GOTO_ERROR_ON((buffer->flags & CL_MEM_HOST_READ_ONLY &&
       flags & CL_MEM_HOST_WRITE_ONLY), CL_INVALID_VALUE,
       "Invalid flags: buffer is CL_MEM_HOST_READ_ONLY, requested sub-buffer "
       "CL_MEM_HOST_WRITE_ONLY\n");

  POCL_GOTO_ERROR_ON((buffer->flags & CL_MEM_HOST_NO_ACCESS &&
       flags & (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_WRITE_ONLY)), CL_INVALID_VALUE,
       "Invalid flags: buffer is CL_MEM_HOST_NO_ACCESS, requested sub-buffer "
       "(CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_WRITE_ONLY)\n");

  pocl_cl_mem_inherit_flags (mem, buffer, flags);

  if (mem->flags & CL_MEM_USE_HOST_PTR || mem->flags & CL_MEM_ALLOC_HOST_PTR)
    {
      mem->mem_host_ptr = (char*)buffer->mem_host_ptr + info->origin;
    }

  mem->device_ptrs = (pocl_mem_identifier*)
    malloc(buffer->context->num_devices * sizeof(pocl_mem_identifier));
  if (mem->device_ptrs == NULL)
    {
        errcode = CL_OUT_OF_HOST_MEMORY;
        goto ERROR;
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
        mem->device_ptrs[device->dev_id].mem_ptr = (char*)
          buffer->device_ptrs[device->dev_id].mem_ptr + info->origin;
    }

  POCL_RETAIN_OBJECT(mem->parent);
  POCL_RETAIN_OBJECT(mem->context);

  POCL_MSG_PRINT_INFO ("Created Subbuffer %p, parent %p\n", mem, mem->parent);

  if (errcode_ret != NULL)
    *errcode_ret = CL_SUCCESS;
  return mem;

ERROR:
  POCL_MEM_FREE(mem);
  if(errcode_ret)
  {
    *errcode_ret = errcode;
  }
  return NULL;
}
POsym(clCreateSubBuffer)
