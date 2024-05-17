/* OpenCL runtime library: clReleaseMemObject()

   Copyright (c) 2011 Universidad Rey Juan Carlos
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
#include "utlist.h"

#ifdef ENABLE_RDMA
#include "pocl_rdma.h"
#endif

extern unsigned long buffer_c;

extern unsigned long image_c;

CL_API_ENTRY cl_int CL_API_CALL
POname(clReleaseMemObject)(cl_mem memobj) CL_API_SUFFIX__VERSION_1_0
{
  int new_refcount;
  cl_device_id dev;
  cl_mem parent = NULL;
  unsigned i;
  mem_destructor_callback_t *callback, *next_callback;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (memobj)),
                          CL_INVALID_MEM_OBJECT);

  cl_context context = memobj->context;

  POCL_LOCK_OBJ (memobj);
  POCL_RELEASE_OBJECT_UNLOCKED (memobj, new_refcount);

  if (memobj->parent != NULL)
    POCL_MSG_PRINT_REFCOUNTS (
      "Release subbuffer %" PRId64 " (%p), Refcount: %d, Parent %zu\n",
      memobj->id, memobj, new_refcount, memobj->parent->id);
  else
    POCL_MSG_PRINT_REFCOUNTS ("Release memory object %" PRId64
                              " (%p), Refcount: %d\n",
                              memobj->id, memobj, new_refcount);

  /* OpenCL 1.2 Page 118:

     After the memobj reference count becomes zero and commands queued for
     execution on a command-queue(s) that use memobj have finished, the memory
     object is deleted. If memobj is a buffer object, memobj cannot be deleted
     until all sub-buffer objects associated with memobj are deleted.
  */

  cl_int err = CL_SUCCESS;
  if (new_refcount == 0)
    {
      POCL_UNLOCK_OBJ (memobj);
      VG_REFC_ZERO (memobj);

      cl_event last = memobj->last_updater;
      if (memobj->is_image)
        {
          TP_FREE_IMAGE (context->id, memobj->id);
          POCL_ATOMIC_DEC (image_c);
        }
      else
        {
          TP_FREE_BUFFER (context->id, memobj->id);
          POCL_ATOMIC_DEC (buffer_c);
        }

      if (memobj->is_image && (memobj->type == CL_MEM_OBJECT_IMAGE1D_BUFFER))
        {
          cl_mem b = memobj->buffer;
          assert (b);
          err = POname (clReleaseMemObject) (b);
          POCL_MEM_FREE (memobj);
          return err;
        }

      parent = memobj->parent;
      if (parent == NULL)
        {
          assert (memobj->mappings == NULL);
          assert (memobj->map_count == 0);

          POCL_MSG_PRINT_REFCOUNTS ("Free Memory Object %" PRId64
                                    " (%p), Flags: %" PRIu64 "\n",
                                    memobj->id, memobj, memobj->flags);

          for (i = 0; i < context->num_devices; ++i)
            {
              dev = context->devices[i];
              if (*(dev->available) == CL_FALSE)
                continue;
              if (memobj->device_ptrs[dev->global_mem_id].mem_ptr == NULL)
                continue;

              dev->ops->free (dev, memobj);

              memobj->device_ptrs[dev->global_mem_id].mem_ptr = NULL;
            }

          /* Release the implicit sub-buffers. */
          cl_mem_list_item_t *sb = NULL, *tmp;
          LL_FOREACH_SAFE (memobj->implicit_sub_buffers, sb, tmp)
            {
              for (i = 0; i < context->num_devices; ++i)
                {
                  cl_device_id dev = context->devices[i];
                  if (dev->ops->free_subbuffer != NULL)
                    dev->ops->free_subbuffer (dev, sb->mem);
                }
              LL_DELETE (memobj->implicit_sub_buffers, sb);
              POCL_MEM_FREE (sb->mem);
              free (sb);
            }
          /* Free host mem allocated by the runtime. */
          if (memobj->mem_host_ptr != NULL)
            {
              if (memobj->flags & CL_MEM_USE_HOST_PTR)
                /* User allocated, do not free. */
                memobj->mem_host_ptr = NULL;
              else
                {
                  POCL_MEM_FREE (memobj->mem_host_ptr);
                }
            }
        }
      else
        {
          /* It's a sub-buffer. Some devices might have resources associated to
           * them. */
          for (i = 0; i < context->num_devices; ++i)
            {
              cl_device_id dev = context->devices[i];
              if (dev->ops->free_subbuffer != NULL)
                dev->ops->free_subbuffer (dev, memobj);
            }
          /* Remove the sub-buffer record from the parent buffer. */
          cl_mem_list_item_t *sub_buf;

          assert (parent->sub_buffers != NULL);

          POCL_LOCK_OBJ (parent);

          LL_SEARCH_SCALAR (parent->sub_buffers, sub_buf, mem, memobj)
            ;
          assert (sub_buf != NULL);
          assert (sub_buf->mem == memobj);

          LL_DELETE (parent->sub_buffers, sub_buf);
          free (sub_buf);

          POCL_UNLOCK_OBJ (parent);
          /* Let the parent buffer free the host pointer. */
          memobj->mem_host_ptr = NULL;
        }
      POCL_MEM_FREE (memobj->device_ptrs);

      /* Fire any registered destructor callbacks. */
      callback = memobj->destructor_callbacks;
      while (callback)
      {
        callback->pfn_notify (memobj, callback->user_data);
        next_callback = callback->next;
        free (callback);
        callback = next_callback;
      }

      if (memobj->is_image)
        POCL_MEM_FREE (memobj->device_supports_this_image);

      if (memobj->content_buffer)
        {
          POCL_LOCK_OBJ (memobj->content_buffer);
          assert (memobj->content_buffer->size_buffer == memobj);
          memobj->content_buffer->size_buffer = NULL;
          POCL_UNLOCK_OBJ (memobj->content_buffer);
          memobj->content_buffer = NULL;
        }

      if (memobj->size_buffer)
        {
          POCL_LOCK_OBJ (memobj->size_buffer);
          assert (memobj->size_buffer->content_buffer == memobj);
          memobj->size_buffer->content_buffer = NULL;
          POCL_UNLOCK_OBJ (memobj->size_buffer);
          memobj->size_buffer = NULL;
        }

      if (memobj->has_device_address)
        {
          POCL_LOCK_OBJ (context);
          pocl_raw_ptr *tmp = NULL, *item = NULL;
          DL_FOREACH_SAFE (context->raw_ptrs, item, tmp)
          {
            if (item->shadow_cl_mem == memobj)
              {
                DL_DELETE (context->raw_ptrs, item);
                free (item);
                break;
              }
          }
          POCL_UNLOCK_OBJ (context);
        }

      POCL_DESTROY_OBJECT (memobj);
      POCL_MEM_FREE(memobj);

      if (parent)
        POname(clReleaseMemObject)(parent);

      POname(clReleaseContext)(context);

      if (last)
        POname (clReleaseEvent) (last);
    }
  else
    {
      VG_REFC_NONZERO (memobj);
      POCL_UNLOCK_OBJ (memobj);
    }

  return CL_SUCCESS;
}
POsym(clReleaseMemObject)
