/* OpenCL runtime library: clSVMAlloc()

   Copyright (c) 2015 Michal Babej / Tampere University of Technology

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

#include "pocl_util.h"
#include "devices.h"

CL_API_ENTRY void* CL_API_CALL
POname(clSVMAlloc)(cl_context context,
                   cl_svm_mem_flags flags,
                   size_t size,
                   unsigned int alignment) CL_API_SUFFIX__VERSION_2_0
{
  unsigned i;
  int p;

  POCL_MSG_PRINT_INFO("clSVMAlloc\n");

  POCL_RETURN_ERROR_COND((context == NULL), NULL);

  POCL_RETURN_ERROR_ON (
      (!context->svm_alloc_mem), NULL,
      "None of the devices in this context is SVM-capable\n");

  POCL_RETURN_ERROR_COND((size == 0), NULL);

  POCL_RETURN_ERROR_ON ((size > context->min_mem_alloc_size), NULL,
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
  POCL_RETURN_ERROR_ON((flags & (!valid_flags)), NULL, "flags argument "
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

  if (alignment == 0)
    alignment = context->min_buffer_alignment;

  /* alignment is not a power of two or the OpenCL implementation cannot support
   * the specified alignment for at least one device in context. */
  p = __builtin_popcount(alignment);
  POCL_RETURN_ERROR_ON((p > 1), NULL, "aligment argument must be a power of 2\n");

  for (i=0; i < context->num_devices; i++)
    POCL_RETURN_ERROR_ON((context->devices[i]->min_data_type_align_size < alignment),
                         NULL, "All devices must support the requested memory "
                         "aligment (%u) \n", alignment);

  void *ptr = context->svm_alloc_mem->svm_alloc (context->svm_alloc_mem, flags,
                                                 size);
  return ptr;
}
POsym(clSVMAlloc)

