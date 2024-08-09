/* OpenCL runtime library: clCreateSubBuffer()

   Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
                 2024 Pekka Jääskeläinen / Intel Finland Oy

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include "devices.h"
#include "pocl_cl.h"
#include "pocl_util.h"
#include "utlist.h"

CL_API_ENTRY cl_mem CL_API_CALL
POname (clCreateSubBuffer) (cl_mem parent,
                            cl_mem_flags flags,
                            cl_buffer_create_type buffer_create_type,
                            const void *buffer_create_info,
                            cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_1
{
  cl_mem mem = NULL;
  int errcode, parent_locked = 0;

  POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (parent)), CL_INVALID_MEM_OBJECT);

  POCL_GOTO_ERROR_ON ((parent->is_image || IS_IMAGE1D_BUFFER (parent)),
                      CL_INVALID_MEM_OBJECT,
                      "subbuffers on images not supported\n");

  POCL_GOTO_ERROR_ON ((parent->parent != NULL), CL_INVALID_MEM_OBJECT,
                      "parent is already a sub-buffer\n");

  POCL_GOTO_ERROR_COND((buffer_create_info == NULL), CL_INVALID_VALUE);

  POCL_GOTO_ERROR_COND((buffer_create_type != CL_BUFFER_CREATE_TYPE_REGION),
    CL_INVALID_VALUE);

  cl_buffer_region *info = (cl_buffer_region *)buffer_create_info;

  POCL_GOTO_ERROR_ON ((info->size == 0 && !parent->has_device_address),
                      CL_INVALID_BUFFER_SIZE,
                      "buffer_create_info->size == 0\n");

  POCL_GOTO_ERROR_ON ((info->size + info->origin > parent->size),
                      CL_INVALID_VALUE,
                      "buffer_create_info->size+origin > buffer size\n");

  POCL_GOTO_ERROR_ON (
    (parent->flags & CL_MEM_WRITE_ONLY
     && flags & (CL_MEM_READ_WRITE | CL_MEM_READ_ONLY)),
    CL_INVALID_VALUE,
    "Invalid flags: parent is CL_MEM_WRITE_ONLY, requested sub-parent "
    "CL_MEM_READ_WRITE or CL_MEM_READ_ONLY\n");

  POCL_GOTO_ERROR_ON (
    (parent->flags & CL_MEM_READ_ONLY
     && flags & (CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY)),
    CL_INVALID_VALUE,
    "Invalid flags: parent is CL_MEM_READ_ONLY, requested sub-parent "
    "CL_MEM_READ_WRITE or CL_MEM_WRITE_ONLY\n");

  POCL_GOTO_ERROR_ON((flags & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR |
                CL_MEM_COPY_HOST_PTR)), CL_INVALID_VALUE,
                "Invalid flags: (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR | "
                "CL_MEM_COPY_HOST_PTR)\n");

  POCL_GOTO_ERROR_ON (
    (parent->flags & CL_MEM_HOST_WRITE_ONLY && flags & CL_MEM_HOST_READ_ONLY),
    CL_INVALID_VALUE,
    "Invalid flags: parent is CL_MEM_HOST_WRITE_ONLY, requested sub-parent "
    "CL_MEM_HOST_READ_ONLY\n");

  POCL_GOTO_ERROR_ON (
    (parent->flags & CL_MEM_HOST_READ_ONLY && flags & CL_MEM_HOST_WRITE_ONLY),
    CL_INVALID_VALUE,
    "Invalid flags: parent is CL_MEM_HOST_READ_ONLY, requested sub-parent "
    "CL_MEM_HOST_WRITE_ONLY\n");

  POCL_GOTO_ERROR_ON (
    (parent->flags & CL_MEM_HOST_NO_ACCESS
     && flags & (CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_WRITE_ONLY)),
    CL_INVALID_VALUE,
    "Invalid flags: parent is CL_MEM_HOST_NO_ACCESS, requested sub-parent "
    "(CL_MEM_HOST_READ_ONLY | CL_MEM_HOST_WRITE_ONLY)\n");

  POCL_GOTO_ERROR_ON (
    (!parent->has_device_address
     && (info->origin % parent->context->min_buffer_alignment) != 0),
    CL_MISALIGNED_SUB_BUFFER_OFFSET,
    "no devices for which the origin value (%zu) is "
    "aligned to the CL_DEVICE_MEM_BASE_ADDR_ALIGN value (%zu)\n",
    info->origin, parent->context->min_buffer_alignment);

  mem = (cl_mem) calloc (1, sizeof (struct _cl_mem));
  POCL_GOTO_ERROR_COND ((mem == NULL), CL_OUT_OF_HOST_MEMORY);

  POCL_INIT_OBJECT (mem);

  mem->device_ptrs = (pocl_mem_identifier *)calloc (
    POCL_ATOMIC_LOAD (pocl_num_devices), sizeof (pocl_mem_identifier));
  POCL_GOTO_ERROR_COND ((mem->device_ptrs == NULL), CL_OUT_OF_HOST_MEMORY);

  mem->context = parent->context;
  mem->parent = parent;
  mem->latest_version = parent->latest_version;
  mem->mem_host_ptr_version = parent->mem_host_ptr_version;

  /* Initially the sub-buffer's latest contents are in the same memory where
     the parent's latest contents are due to aliasing. */
  for (unsigned i = 0; i < mem->context->num_devices; ++i)
    {
      cl_device_id d = mem->context->devices[i];
      if (parent->device_ptrs[d->global_mem_id].version == mem->latest_version)
        {
          mem->device_ptrs[d->global_mem_id].version = mem->latest_version;
          break;
        }
    }

  mem->type = CL_MEM_OBJECT_BUFFER;
  mem->size = info->size;
  mem->origin = info->origin;
  pocl_cl_mem_inherit_flags (mem, parent, flags);
  /* All other struct members are NULL (not valid). */

  /* The sub-parents should keep the parent parent alive until all of them are
   * released. */
  POCL_RETAIN_OBJECT (mem->parent);
  POCL_RETAIN_OBJECT (mem->context);

  cl_mem_list_item_t *sub_buf
    = (cl_mem_list_item_t *)calloc (1, sizeof (cl_mem_list_item_t));
  sub_buf->mem = mem;
  POCL_GOTO_ERROR_COND ((mem->device_ptrs == NULL), CL_OUT_OF_HOST_MEMORY);

  POCL_LOCK_OBJ (parent); parent_locked = 1;

  LL_APPEND (parent->sub_buffers, sub_buf);

  for (unsigned i = 0; i < parent->context->num_devices; ++i)
    {
      cl_device_id dev = parent->context->devices[i];
      if (parent->device_ptrs[dev->global_mem_id].mem_ptr == NULL)
        {
          /* Due to the lazy parent allocation we might not have allocated
             the parent parent yet. Let's do so now to get addresses to the
             sub-parents generated. */
          int err
            = dev->ops->alloc_mem_obj (dev, parent, parent->mem_host_ptr);
          POCL_GOTO_ERROR_COND ((err != CL_SUCCESS), err);
          POCL_GOTO_ERROR_COND (
            (parent->device_ptrs[dev->global_mem_id].mem_ptr == NULL),
            CL_MEM_OBJECT_ALLOCATION_FAILURE);
        }

      mem->device_ptrs[dev->global_mem_id].mem_ptr
        = parent->device_ptrs[dev->global_mem_id].mem_ptr + info->origin;

      /* Allocate/register the sub-parent with such devices which need to do
         something device-specific with them. */
      if (dev->ops->alloc_subbuffer != NULL)
        {
          int ret_val = dev->ops->alloc_subbuffer (dev, mem);
          POCL_GOTO_ERROR_COND ((ret_val != CL_SUCCESS), ret_val);
        }
    }
  if (parent->mem_host_ptr != NULL)
    mem->mem_host_ptr = parent->mem_host_ptr + info->origin;

  POCL_UNLOCK_OBJ (parent); parent_locked = 0;

  POCL_MSG_PRINT_MEMORY ("Created sub-buffer %zu (%p) with size %zu, origin "
                         "%zu and parent %zu (%p)\n",
                         mem->id, mem, info->size, info->origin,
                         mem->parent->id, mem->parent);

  if (errcode_ret != NULL)
    *errcode_ret = CL_SUCCESS;
  return mem;

ERROR:
  if (parent_locked)
    POCL_UNLOCK_OBJ (parent);

  if (mem != NULL && mem->device_ptrs)
    {
      for (unsigned i = 0; i < parent->context->num_devices; ++i)
        {
          cl_device_id dev = parent->context->devices[i];
          pocl_mem_identifier *p = &mem->device_ptrs[dev->global_mem_id];
          if (p->mem_ptr != NULL && dev->ops->free_subbuffer != NULL)
            dev->ops->free_subbuffer (dev, mem);
        }
      POCL_MEM_FREE (mem->device_ptrs);
    }

  POCL_MEM_FREE(mem);
  if(errcode_ret)
  {
    *errcode_ret = errcode;
  }
  return NULL;
}
POsym (clCreateSubBuffer)
