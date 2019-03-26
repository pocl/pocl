/* OpenCL runtime library: clReleaseMemObject()

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

#include "utlist.h"
#include "pocl_cl.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clReleaseMemObject)(cl_mem memobj) CL_API_SUFFIX__VERSION_1_0
{
  int new_refcount;
  cl_device_id dev;
  cl_mem parent = NULL;
  unsigned i;
  mem_mapping_t *mapping, *temp;
  mem_destructor_callback_t *callback, *next_callback;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (memobj)),
                          CL_INVALID_MEM_OBJECT);

  cl_context context = memobj->context;

  POCL_RELEASE_OBJECT(memobj, new_refcount);

  POCL_MSG_PRINT_REFCOUNTS ("Release mem obj %p  %d\n", memobj, new_refcount);

  /* OpenCL 1.2 Page 118:

     After the memobj reference count becomes zero and commands queued for execution on 
     a command-queue(s) that use memobj have finished, the memory object is deleted. If 
     memobj is a buffer object, memobj cannot be deleted until all sub-buffer objects associated 
     with memobj are deleted.
  */

  if (new_refcount == 0)
    {
      if (memobj->is_image)
        {
          TP_FREE_IMAGE (context->id, memobj->id)
        }
      else
        {
          TP_FREE_BUFFER (context->id, memobj->id);
        }

      if (memobj->is_image && (memobj->type == CL_MEM_OBJECT_IMAGE1D_BUFFER))
        {
          cl_mem b = memobj->buffer;
          assert (b);
          cl_int err = POname (clReleaseMemObject) (b);
          POCL_MEM_FREE (memobj);
          return err;
        }

      POCL_MSG_PRINT_REFCOUNTS ("Free mem obj %p\n", memobj);
      if (memobj->parent == NULL)
        {
          cl_device_id shared_mem_owner_dev =
            memobj->shared_mem_allocation_owner;

          for (i = 0; i < context->num_devices; ++i)
            {
              /* owner is called last */
              if (shared_mem_owner_dev == context->devices[i])
                 continue;
              dev = context->devices[i];
              if (dev->available != CL_TRUE)
                continue;
              if (memobj->device_ptrs[dev->dev_id].mem_ptr == NULL)
                continue;

              dev->ops->free (dev, memobj);

              memobj->device_ptrs[dev->dev_id].mem_ptr = NULL;
            }
          if (shared_mem_owner_dev)
            shared_mem_owner_dev->ops->free (shared_mem_owner_dev, memobj);

        }
      DL_FOREACH_SAFE(memobj->mappings, mapping, temp)
        {
          POCL_MEM_FREE(mapping);
        }
      memobj->mappings = NULL;

      parent = memobj->parent;

      /* Free host mem allocated by the runtime (not for sub buffers) */
      if (memobj->parent == NULL && (memobj->flags & CL_MEM_ALLOC_HOST_PTR)
          && memobj->mem_host_ptr != NULL)
        {
          POCL_MEM_FREE(memobj->mem_host_ptr);
        }
      POCL_MEM_FREE(memobj->device_ptrs);

      /* Fire any registered destructor callbacks */
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

      POCL_DESTROY_OBJECT (memobj);
      POCL_MEM_FREE(memobj);

      if (parent)
        POname(clReleaseMemObject)(parent);
      POname(clReleaseContext)(context);
    }
  return CL_SUCCESS;
}
POsym(clReleaseMemObject)
