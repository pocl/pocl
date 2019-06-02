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
#include "common.h"
#include "pocl_util.h"

static unsigned long buffer_ids = 0;

CL_API_ENTRY cl_mem CL_API_CALL
POname(clCreateBuffer)(cl_context   context,
                       cl_mem_flags flags,
                       size_t       size,
                       void         *host_ptr,
                       cl_int       *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  cl_mem mem = NULL;
  cl_device_id device;
  int errcode;
  unsigned i, j;

  POCL_GOTO_ERROR_COND((size == 0), CL_INVALID_BUFFER_SIZE);

  POCL_GOTO_ERROR_COND((context == NULL), CL_INVALID_CONTEXT);
  
  mem = (cl_mem) malloc(sizeof(struct _cl_mem));
  if (mem == NULL)
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR;
    }
  mem->device_ptrs = NULL;

  if (flags == 0)
    flags = CL_MEM_READ_WRITE;
  
  /* validate flags */
  
  POCL_GOTO_ERROR_ON((flags > (1<<10)-1), CL_INVALID_VALUE, "Flags must "
    "be < 1024 (there are only 10 flags)\n");

  POCL_GOTO_ERROR_ON(((flags & CL_MEM_READ_WRITE) &&
    (flags & CL_MEM_WRITE_ONLY || flags & CL_MEM_READ_ONLY)),
    CL_INVALID_VALUE, "Invalid flags: CL_MEM_READ_WRITE cannot be used "
    "together with CL_MEM_WRITE_ONLY or CL_MEM_READ_ONLY\n");

  POCL_GOTO_ERROR_ON(((flags & CL_MEM_READ_ONLY) &&
    (flags & CL_MEM_WRITE_ONLY)), CL_INVALID_VALUE, "Invalid flags: "
    "can't have both CL_MEM_WRITE_ONLY and CL_MEM_READ_ONLY\n");

  POCL_GOTO_ERROR_ON(((flags & CL_MEM_USE_HOST_PTR) &&
    (flags & CL_MEM_ALLOC_HOST_PTR || flags & CL_MEM_COPY_HOST_PTR)),
    CL_INVALID_VALUE, "Invalid flags: CL_MEM_USE_HOST_PTR cannot be used "
    "together with CL_MEM_ALLOC_HOST_PTR or CL_MEM_COPY_HOST_PTR\n");

  POCL_GOTO_ERROR_ON(((flags & CL_MEM_HOST_WRITE_ONLY) &&
    (flags & CL_MEM_HOST_READ_ONLY)), CL_INVALID_VALUE, "Invalid flags: "
    "can't have both CL_MEM_HOST_READ_ONLY and CL_MEM_HOST_WRITE_ONLY\n");

  POCL_GOTO_ERROR_ON(((flags & CL_MEM_HOST_NO_ACCESS) &&
    ((flags & CL_MEM_HOST_READ_ONLY) || (flags & CL_MEM_HOST_WRITE_ONLY))),
    CL_INVALID_VALUE, "Invalid flags: CL_MEM_HOST_NO_ACCESS cannot be used "
    "together with CL_MEM_HOST_READ_ONLY or CL_MEM_HOST_WRITE_ONLY\n");

  if (host_ptr == NULL)
    {
      POCL_GOTO_ERROR_ON(((flags & CL_MEM_USE_HOST_PTR) ||
        (flags & CL_MEM_COPY_HOST_PTR)), CL_INVALID_HOST_PTR,
        "host_ptr is NULL, but flags specify {COPY|USE}_HOST_PTR\n");
    }
  else
    {
      POCL_GOTO_ERROR_ON(((~flags & CL_MEM_USE_HOST_PTR) &&
        (~flags & CL_MEM_COPY_HOST_PTR)), CL_INVALID_HOST_PTR,
        "host_ptr is not NULL, but flags don't specify {COPY|USE}_HOST_PTR\n");
    }

  POCL_GOTO_ERROR_ON ((size > context->max_mem_alloc_size),
                      CL_INVALID_BUFFER_SIZE,
                      "Size (%zu) is bigger than max mem alloc size (%zu) "
                      "of all devices in context\n",
                      size, context->max_mem_alloc_size);

  POCL_INIT_OBJECT(mem);
  mem->id = ATOMIC_INC (buffer_ids);
  mem->parent = NULL;
  mem->map_count = 0;
  mem->mappings = NULL;
  mem->buffer = NULL;
  mem->destructor_callbacks = NULL;
  mem->type = CL_MEM_OBJECT_BUFFER;
  mem->flags = flags;
  mem->is_image = CL_FALSE;
  mem->owning_device = NULL;
  mem->is_pipe = 0;
  mem->pipe_packet_size = 0;
  mem->pipe_max_packets = 0;

  /* Store the per device buffer pointers always to a known
     location in the buffer (dev_id), even though the context
     might not contain all the devices. */
  mem->device_ptrs =
    (pocl_mem_identifier*) calloc(pocl_num_devices,
                                  sizeof(pocl_mem_identifier));
  POCL_GOTO_ERROR_COND((mem->device_ptrs == NULL), CL_OUT_OF_HOST_MEMORY);

  /* init dev pointer structure to ease inter device pointer sharing
     in ops->alloc_mem_obj */
  for (i = 0; i < context->num_devices; ++i)
    {
      int dev_id = context->devices[i]->dev_id;

      mem->device_ptrs[dev_id].global_mem_id =
        context->devices[i]->global_mem_id;
      mem->device_ptrs[i].available = 1;
    }

  mem->size = size;
  mem->origin = 0;
  mem->context = context;
  mem->mem_host_ptr = host_ptr;
  mem->shared_mem_allocation_owner = NULL;

  /* if there is a "special needs" device (hsa) operating in the host memory 
     let it alloc memory first for shared memory use */
  if (context->svm_allocdev)
    {
      if (context->svm_allocdev->ops->alloc_mem_obj (context->svm_allocdev, mem, host_ptr) != CL_SUCCESS)
        {
          errcode = CL_MEM_OBJECT_ALLOCATION_FAILURE;
          goto ERROR_CLEAN_MEM_AND_DEVICE;
        }
    }

  for (i = 0; i < context->num_devices; ++i)
    {
      /* this is already handled iff available */
      if (context->svm_allocdev == context->devices[i])
        continue;

      device = context->devices[i];
      assert (device->ops->alloc_mem_obj != NULL);
      if (device->ops->alloc_mem_obj (device, mem, host_ptr) != CL_SUCCESS)
        {
          errcode = CL_MEM_OBJECT_ALLOCATION_FAILURE;
          goto ERROR_CLEAN_MEM_AND_DEVICE;
        }
    }

  /* Some device driver may already have allocated host accessible memory */
  if ((flags & CL_MEM_ALLOC_HOST_PTR) && (mem->mem_host_ptr == NULL))
    {
      assert(mem->shared_mem_allocation_owner == NULL);
      mem->mem_host_ptr = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, size);
      if (mem->mem_host_ptr == NULL)
        {
          errcode = CL_OUT_OF_HOST_MEMORY;
          goto ERROR_CLEAN_MEM_AND_DEVICE;
        }
    }

  POCL_RETAIN_OBJECT(context);

  POCL_MSG_PRINT_MEMORY ("Created Buffer ID %zu / %p, MEM_HOST_PTR: %p, "
                         "DEVICE_PTR[0]: %p, SIZE %zu, FLAGS %zu \n",
                         mem->id, mem, mem->mem_host_ptr,
                         mem->device_ptrs[0].mem_ptr, size, flags);

  if (errcode_ret != NULL)
    *errcode_ret = CL_SUCCESS;
  return mem;

ERROR_CLEAN_MEM_AND_DEVICE:
  for (j = 0; j < i; ++j)
    {
      device = context->devices[j];
      device->ops->free(device, mem);
    }
ERROR:
  if (mem)
    POCL_MEM_FREE (mem->device_ptrs);
  POCL_MEM_FREE(mem);
  if(errcode_ret)
    {
      *errcode_ret = errcode;
    }
  return NULL;
}
POsym(clCreateBuffer)
