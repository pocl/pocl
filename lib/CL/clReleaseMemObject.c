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

#include "devices.h"
#include "pocl_cl.h"
#include "utlist.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clReleaseMemObject)(cl_mem memobj) CL_API_SUFFIX__VERSION_1_0
{
  int new_refcount;
  cl_device_id dev;
  cl_mem parent = NULL;
  unsigned i;
  mem_destructor_callback_t *callback, *next_callback;

  POCL_RETURN_ERROR_COND((memobj == NULL), CL_INVALID_MEM_OBJECT);

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
      if (memobj->is_image && (memobj->type == CL_MEM_OBJECT_IMAGE1D_BUFFER))
        {
          cl_mem b = memobj->buffer;
          assert (b);
          cl_int err = POname (clReleaseMemObject) (b);
          POCL_MEM_FREE (memobj);
          return err;
        }

      parent = memobj->parent;
      if (parent == NULL)
        {
          assert (memobj->mappings == NULL);
          assert (memobj->map_count == 0);

          POCL_MSG_PRINT_REFCOUNTS ("Free mem obj %p FLAGS %" PRIu64 "\n",
                                    memobj, memobj->flags);

          for (i = 0; i < context->num_devices; ++i)
            {
              dev = context->devices[i];
              pocl_mem_identifier *p = &memobj->gmem_ptrs[dev->global_mem_id];
              if (p->mem_ptr == NULL) // already free or never allocated
                continue;

              dev->global_memory->free_mem_obj (dev->global_memory, memobj, p);
            }

          /* Free host mem allocated by the runtime */
          if (memobj->mem_host_ptr != NULL)
            {
              if (memobj->flags & CL_MEM_USE_HOST_PTR)
                memobj->mem_host_ptr = NULL; // user allocated, do not free
              else
                POCL_MEM_FREE (memobj->mem_host_ptr);
            }

          POCL_MEM_FREE (memobj->gmem_ptrs);
        }

      assert (memobj->gmem_ptrs == NULL);

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

      assert (memobj->mem_host_ptr == NULL);

      POCL_DESTROY_OBJECT (memobj);
      POCL_MEM_FREE(memobj);

      if (parent)
        POname(clReleaseMemObject)(parent);

      POname(clReleaseContext)(context);
    }
  return CL_SUCCESS;
}
POsym(clReleaseMemObject)
