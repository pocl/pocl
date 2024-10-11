/* OpenCL runtime library: clSVMAlloc()

   Copyright (c) 2015 Michal Babej / Tampere University of Technology
                 2023-2024 Pekka Jääskeläinen / Intel Finland Oy

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
#include "pocl_shared.h"
#include "pocl_util.h"
#include "utlist.h"

CL_API_ENTRY void* CL_API_CALL
POname(clSVMAlloc)(cl_context context,
                   cl_svm_mem_flags flags,
                   size_t size,
                   unsigned int alignment) CL_API_SUFFIX__VERSION_2_0
{
  unsigned i;
  int p;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (context)), NULL);

  POCL_RETURN_ERROR_ON((!context->svm_allocdev), NULL,
                       "None of the devices in this context is SVM-capable\n");

  POCL_RETURN_ERROR_COND((size == 0), NULL);

  POCL_RETURN_ERROR_ON ((size > context->max_mem_alloc_size), NULL,
                        "size(%zu) > CL_DEVICE_MAX_MEM_ALLOC_SIZE value "
                        "for some device in context\n",
                        size);

  /* flags does not contain CL_MEM_SVM_FINE_GRAIN_BUFFER
   * but does contain CL_MEM_SVM_ATOMICS. */
  POCL_RETURN_ERROR_COND((flags & CL_MEM_SVM_ATOMICS) &&
                         ((flags & CL_MEM_SVM_FINE_GRAIN_BUFFER) == 0), NULL);

  /* Flags  */
  p = __builtin_popcount(flags & (CL_MEM_READ_WRITE
                                           | CL_MEM_WRITE_ONLY | CL_MEM_READ_ONLY));
  POCL_RETURN_ERROR_ON((p > 1), NULL, "flags may contain only one of "
                   "CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY | CL_MEM_READ_ONLY\n");

  const cl_svm_mem_flags valid_flags = (CL_MEM_SVM_ATOMICS | CL_MEM_SVM_FINE_GRAIN_BUFFER
                                  | CL_MEM_READ_WRITE | CL_MEM_WRITE_ONLY
                                  | CL_MEM_READ_ONLY);
  POCL_RETURN_ERROR_ON ((flags & (~valid_flags)), NULL,
                        "flags argument "
                        "contains invalid bits (nonexistent flags)\n");

  /* CL_MEM_SVM_FINE_GRAIN_BUFFER or CL_MEM_SVM_ATOMICS is specified in flags
   * and these are not supported by at least one device in context. */
  if (flags & CL_MEM_SVM_FINE_GRAIN_BUFFER)
    for (i=0; i < context->num_devices; i++)
      POCL_RETURN_ERROR_ON((DEVICE_SVM_FINEGR(context->devices[i]) == 0), NULL,
                           "One of the devices in the context doesn't support "
                           "fine-grained buffers, and it's in flags\n");

  if (flags & CL_MEM_SVM_ATOMICS)
    for (i=0; i < context->num_devices; i++)
      POCL_RETURN_ERROR_ON((DEVICE_SVM_ATOM(context->devices[i]) == 0), NULL,
                           "One of the devices in the context doesn't support "
                           "SVM atomics buffers, and it's in flags\n");

  pocl_raw_ptr *item = calloc (1, sizeof (pocl_raw_ptr));
  POCL_RETURN_ERROR_ON ((item == NULL), NULL, "out of host memory\n");

  if (alignment == 0)
    alignment = context->svm_allocdev->min_data_type_align_size;

  /* alignment is not a power of two or the OpenCL implementation cannot support
   * the specified alignment for at least one device in context. */
  p = __builtin_popcount(alignment);
  POCL_RETURN_ERROR_ON((p > 1), NULL, "aligment argument must be a power of 2\n");

  for (i=0; i < context->num_devices; i++)
    POCL_RETURN_ERROR_ON((context->devices[i]->min_data_type_align_size < alignment),
                         NULL, "All devices must support the requested memory "
                         "aligment (%u) \n", alignment);

  void *ptr = context->svm_allocdev->ops->svm_alloc (context->svm_allocdev,
                                                     flags, size);
  if (ptr == NULL)
    {
      POCL_MEM_FREE (item);
      POCL_MSG_ERR ("SVM manager device failed to allocate memory.\n");
      return NULL;
    }

  POCL_LOCK_OBJ (context);
  /* Register the pointer as a SVM pointer so clCreateBuffer() detects it. */
  item->vm_ptr = ptr;
  item->size = size;
  DL_APPEND (context->raw_ptrs, item);
  POCL_UNLOCK_OBJ (context);

  /* Create a shadow cl_mem object for keeping track of the SVM
     allocation and to implement automated migrations, cl_pocl_content_size,
     etc. for CG SVM using the same code as with non-SVM cl_mems. */

  /* For remote devices using CL_MEM_DEVICE_ADDRESS actually allocates storage
     from the remote as well. */
  cl_int errcode = CL_SUCCESS;
  cl_mem clmem_shadow = POname (clCreateBuffer) (
      context,
      CL_MEM_DEVICE_ADDRESS_EXT | CL_MEM_DEVICE_PRIVATE_EXT
          | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
      size, ptr, &errcode);

  if (errcode != CL_SUCCESS)
    {
      POCL_MSG_ERR ("Failed to allocate memory a shadow cl_mem object.\n");
      return NULL;
    }

  item->shadow_cl_mem = clmem_shadow;

  POname (clRetainContext) (context);

  POCL_MSG_PRINT_MEMORY ("Allocated SVM: PTR %p, SIZE %zu, FLAGS %" PRIu64
                         " \n",
                         ptr, size, flags);

  POCL_ATOMIC_INC (svm_buffer_c);

  return ptr;
}
POsym(clSVMAlloc)

