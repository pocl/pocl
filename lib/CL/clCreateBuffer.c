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

static unsigned long buffer_ids = 0;

cl_mem
pocl_create_memobject (cl_context context, cl_mem_flags flags, size_t size,
                       void *host_ptr, cl_int *errcode_ret)
{
  cl_mem mem = NULL;
  int errcode;

  POCL_GOTO_ERROR_COND((size == 0), CL_INVALID_BUFFER_SIZE);

  POCL_GOTO_ERROR_COND((context == NULL), CL_INVALID_CONTEXT);

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
  mem->id = ATOMIC_INC (buffer_ids);
  mem->type = CL_MEM_OBJECT_BUFFER;
  mem->flags = flags;

  mem->gmem_ptrs = (pocl_mem_identifier *)calloc (
      pocl_num_global_mem, sizeof (pocl_mem_identifier));
  POCL_GOTO_ERROR_COND ((mem->gmem_ptrs == NULL), CL_OUT_OF_HOST_MEMORY);

  mem->size = size;
  mem->origin = 0;
  mem->context = context;

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
      POCL_MSG_PRINT_MEMORY ("USE_HOST_PTR %p \n", host_ptr);
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

  return mem;

ERROR:
  if (mem)
    POCL_MEM_FREE (mem->gmem_ptrs);

  POCL_MEM_FREE (mem);
  if (errcode_ret)
    {
      *errcode_ret = errcode;
    }

  return NULL;
}

/******************************************************************************/

CL_API_ENTRY cl_mem CL_API_CALL POname (clCreateBuffer) (
    cl_context context, cl_mem_flags flags, size_t size, void *host_ptr,
    cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  cl_mem mem = NULL;
  int errcode;
  unsigned i, j;

  POCL_GOTO_ERROR_COND ((size == 0), CL_INVALID_BUFFER_SIZE);

  POCL_GOTO_ERROR_COND ((context == NULL), CL_INVALID_CONTEXT);

  mem = pocl_create_memobject (context, flags, size, host_ptr, errcode_ret);
  if (mem == NULL)
    return NULL;
  mem->is_image = CL_FALSE;

  /* allocate on every global memory */
  for (i = 0; i < context->num_devices; ++i)
    {
      cl_device_id dev = context->devices[i];
      assert (dev->global_memory->alloc_mem_obj != NULL);
      pocl_mem_identifier *p = &mem->gmem_ptrs[dev->global_mem_id];
      if (p->mem_ptr != NULL) // already allocated
        continue;

      errcode = dev->global_memory->alloc_mem_obj (dev->global_memory, mem, p,
                                                   host_ptr);
      if (errcode)
        goto ERROR_CLEAN_MEM_AND_DEVICE;
    }

  /* allocate host backing memory now, if no driver has allocated it. */
  if (mem->mem_host_ptr == NULL)
    {
      POCL_MSG_PRINT_MEMORY ("mem_host_ptr NOT preallocated, allocating\n");
      size_t max
          = (size > MAX_EXTENDED_ALIGNMENT) ? size : MAX_EXTENDED_ALIGNMENT;
      mem->mem_host_ptr = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, max);
      assert (mem->mem_host_ptr);

      if (flags & CL_MEM_COPY_HOST_PTR)
        {
          POCL_MSG_PRINT_MEMORY (
              "allocated mem_host_ptr + CL_MEM_COPY_HOST_PTR\n");
          memcpy (mem->mem_host_ptr, host_ptr, size);
        }
    }

  POCL_RETAIN_OBJECT(context);

  POCL_MSG_PRINT_MEMORY (
      "Created Buffer %p, MEM_HOST_PTR: %p, GMEM_PTRS[0]: %p, "
      "SIZE %zu, FLAGS %" PRIu64 " \n",
      mem, mem->mem_host_ptr, mem->gmem_ptrs[0].mem_ptr, size, flags);

  if (errcode_ret != NULL)
    *errcode_ret = CL_SUCCESS;
  return mem;

ERROR_CLEAN_MEM_AND_DEVICE:
  for (j = 0; j < i; ++j)
    {
      cl_device_id dev = context->devices[i];
      pocl_mem_identifier *p = &mem->gmem_ptrs[dev->global_mem_id];
      if (p->mem_ptr)
        dev->global_memory->free_mem_obj (dev->global_memory, mem, p);
    }

ERROR:
  if (mem)
    POCL_MEM_FREE (mem->gmem_ptrs);
  POCL_MEM_FREE(mem);
  if(errcode_ret)
    {
      *errcode_ret = errcode;
    }
  return NULL;
}
POsym(clCreateBuffer)
