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

#include "common.h"
#include "devices.h"
#include "pocl_cl.h"
#include "pocl_shared.h"
#include "pocl_util.h"

cl_mem
pocl_create_memobject (cl_context context, cl_mem_flags flags, size_t size,
                       cl_mem_object_type type, int* device_image_support,
                       void *host_ptr, cl_int *errcode_ret)
{
  cl_mem mem = NULL;
  int errcode = CL_SUCCESS;
  unsigned i;

  POCL_GOTO_ERROR_COND((size == 0), CL_INVALID_BUFFER_SIZE);

  POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (context)), CL_INVALID_CONTEXT);

  if (flags == 0)
    flags = CL_MEM_READ_WRITE;

  /* validate flags */

  POCL_GOTO_ERROR_ON ((flags > (1 << 10) - 1), CL_INVALID_VALUE,
                      "Flags must "
                      "be < 1024 (there are only 10 flags)\n");

  POCL_GOTO_ERROR_ON (
      ((flags & CL_MEM_READ_WRITE)
       && (flags & CL_MEM_WRITE_ONLY || flags & CL_MEM_READ_ONLY)),
      CL_INVALID_VALUE,
      "Invalid flags: CL_MEM_READ_WRITE cannot be used "
      "together with CL_MEM_WRITE_ONLY or CL_MEM_READ_ONLY\n");

  POCL_GOTO_ERROR_ON (
      ((flags & CL_MEM_READ_ONLY) && (flags & CL_MEM_WRITE_ONLY)),
      CL_INVALID_VALUE,
      "Invalid flags: "
      "can't have both CL_MEM_WRITE_ONLY and CL_MEM_READ_ONLY\n");

  POCL_GOTO_ERROR_ON (
      ((flags & CL_MEM_USE_HOST_PTR)
       && (flags & CL_MEM_ALLOC_HOST_PTR || flags & CL_MEM_COPY_HOST_PTR)),
      CL_INVALID_VALUE,
      "Invalid flags: CL_MEM_USE_HOST_PTR cannot be used "
      "together with CL_MEM_ALLOC_HOST_PTR or CL_MEM_COPY_HOST_PTR\n");

  POCL_GOTO_ERROR_ON (
      ((flags & CL_MEM_HOST_WRITE_ONLY) && (flags & CL_MEM_HOST_READ_ONLY)),
      CL_INVALID_VALUE,
      "Invalid flags: "
      "can't have both CL_MEM_HOST_READ_ONLY and CL_MEM_HOST_WRITE_ONLY\n");

  POCL_GOTO_ERROR_ON (
      ((flags & CL_MEM_HOST_NO_ACCESS)
       && ((flags & CL_MEM_HOST_READ_ONLY)
           || (flags & CL_MEM_HOST_WRITE_ONLY))),
      CL_INVALID_VALUE,
      "Invalid flags: CL_MEM_HOST_NO_ACCESS cannot be used "
      "together with CL_MEM_HOST_READ_ONLY or CL_MEM_HOST_WRITE_ONLY\n");

  if (host_ptr == NULL)
    {
      POCL_GOTO_ERROR_ON (
          ((flags & CL_MEM_USE_HOST_PTR) || (flags & CL_MEM_COPY_HOST_PTR)),
          CL_INVALID_HOST_PTR,
          "host_ptr is NULL, but flags specify {COPY|USE}_HOST_PTR\n");
    }
  else
    {
      POCL_GOTO_ERROR_ON (
          ((~flags & CL_MEM_USE_HOST_PTR) && (~flags & CL_MEM_COPY_HOST_PTR)),
          CL_INVALID_HOST_PTR,
          "host_ptr is not NULL, but flags don't specify "
          "{COPY|USE}_HOST_PTR\n");
    }

  POCL_GOTO_ERROR_ON ((size > context->max_mem_alloc_size),
                      CL_INVALID_BUFFER_SIZE,
                      "Size (%zu) is bigger than max mem alloc size (%zu) "
                      "of all devices in context\n",
                      size, (size_t)context->max_mem_alloc_size);

  mem = (cl_mem)calloc (1, sizeof (struct _cl_mem));
  POCL_GOTO_ERROR_COND ((mem == NULL), CL_OUT_OF_HOST_MEMORY);

  POCL_INIT_OBJECT (mem);
  mem->type = type;
  mem->flags = flags;
  mem->device_supports_this_image = device_image_support;

  mem->device_ptrs = (pocl_mem_identifier *)calloc (
      pocl_num_devices, sizeof (pocl_mem_identifier));
  POCL_GOTO_ERROR_COND ((mem->device_ptrs == NULL), CL_OUT_OF_HOST_MEMORY);

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
  mem->context = context;
  mem->is_image = (type != CL_MEM_OBJECT_PIPE && type != CL_MEM_OBJECT_BUFFER);
  /* https://www.khronos.org/registry/OpenCL/sdk/2.0/docs/man/xhtml/dataTypes.html
   *
   * The user is responsible for ensuring that data passed into and out of
   * OpenCL buffers are natively aligned relative to the start of the buffer as
   * described above. This implies that OpenCL buffers created with
   * CL_MEM_USE_HOST_PTR need to provide an appropriately aligned host memory
   * pointer that is aligned to the data types used to access these buffers in
   * a kernel(s).
   */
  if (flags & CL_MEM_USE_HOST_PTR)
    {
      assert (host_ptr);
      mem->mem_host_ptr = host_ptr;
      if (((uintptr_t)host_ptr % context->min_buffer_alignment) != 0)
        {
          POCL_MSG_WARN ("host_ptr (%p) given to "
                         "clCreateBuffer(CL_MEM_USE_HOST_PTR, ..)\n"
                         "isn't aligned for any device in context;\n"
                         "The minimum required alignment is: %zu;\n"
                         "This can cause various problems later.\n",
                         host_ptr, context->min_buffer_alignment);
        }
    }

  /* if there is a "special needs" device (hsa) operating in the host memory
     let it alloc memory first for shared memory use */
  if (context->svm_allocdev)
    {
      errcode = context->svm_allocdev->ops->alloc_mem_obj
          (context->svm_allocdev, mem, host_ptr);
      POCL_GOTO_ERROR_ON( (errcode != CL_SUCCESS),
                           CL_MEM_OBJECT_ALLOCATION_FAILURE,
                           "SVM device failed to allocate memory");
    }

  /* allocate on every device memory */
  for (i = 0; i < context->num_devices; ++i)
    {
      cl_device_id dev = context->devices[i];

      /* some devices might not support this image format */
      if (mem->is_image && (mem->device_supports_this_image[i] == 0))
        continue;

      /* this is already handled if available */
      if (context->svm_allocdev == context->devices[i])
        continue;

      assert (dev->ops->alloc_mem_obj != NULL);
      errcode = dev->ops->alloc_mem_obj (dev, mem, host_ptr);
      POCL_GOTO_ERROR_COND ((errcode != CL_SUCCESS),
                            CL_MEM_OBJECT_ALLOCATION_FAILURE);
    }

  /* Some device driver may already have allocated host accessible memory */
  if ((flags & CL_MEM_ALLOC_HOST_PTR) && (mem->mem_host_ptr == NULL))
    {
      assert(mem->shared_mem_allocation_owner == NULL);
      mem->mem_host_ptr = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, size);
      POCL_GOTO_ERROR_COND ((mem->mem_host_ptr == NULL), CL_OUT_OF_HOST_MEMORY);
    }

  goto SUCCESS;

ERROR:
  if (mem)
  {
    if (mem->device_ptrs)
    {
      for (i = 0; i < context->num_devices; ++i)
        {
          cl_device_id dev = context->devices[i];
          pocl_mem_identifier *p = &mem->device_ptrs[dev->dev_id];
          if (p->mem_ptr)
            dev->ops->free (dev, mem);
        }
      POCL_MEM_FREE (mem->device_ptrs);
    }

    if (((flags & CL_MEM_USE_HOST_PTR) == 0) && mem->mem_host_ptr)
      POCL_MEM_FREE (mem->mem_host_ptr);

    POCL_MEM_FREE (mem);
  }

SUCCESS:
  if (errcode_ret)
    {
      *errcode_ret = errcode;
    }

  return mem;
}




CL_API_ENTRY cl_mem CL_API_CALL POname (clCreateBuffer) (
    cl_context context, cl_mem_flags flags, size_t size, void *host_ptr,
    cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  cl_mem mem = NULL;

  mem = pocl_create_memobject (context, flags, size, CL_MEM_OBJECT_BUFFER,
                               NULL, host_ptr, errcode_ret);

  if (mem == NULL)
    return NULL;

  POCL_RETAIN_OBJECT(context);

  POCL_MSG_PRINT_MEMORY ("Created Buffer ID %" PRIu64 " / %p, MEM_HOST_PTR: %p, "
                         "device_ptrs[0]: %p, SIZE %zu, FLAGS %" PRIu64 " \n",
                         mem->id, mem, mem->mem_host_ptr,
                         mem->device_ptrs[0].mem_ptr, size, flags);

  return mem;
}
POsym(clCreateBuffer)
