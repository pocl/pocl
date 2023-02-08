/* OpenCL runtime library: clMemFreeINTEL() / clMemBlockingFreeINTEL()

   Copyright (c) 2023 Michal Babej / Intel Finland Oy

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

#include "pocl_debug.h"
#include "pocl_util.h"

extern unsigned long usm_buffer_c;

static int
pocl_mem_free_intel (cl_context context, void *usm_pointer, cl_bool blocking)
{
  POCL_RETURN_ERROR_COND (!IS_CL_OBJECT_VALID (context), CL_INVALID_CONTEXT);

  POCL_RETURN_ERROR_ON (
      (context->usm_allocdev == NULL), CL_INVALID_OPERATION,
      "None of the devices in this context is USM-capable\n");

  if (usm_pointer == NULL)
    {
      POCL_MSG_WARN ("NULL pointer passed\n");
      return CL_SUCCESS;
    }

  POCL_LOCK_OBJ (context);
  pocl_svm_ptr *tmp = NULL, *item = NULL;
  DL_FOREACH_SAFE (context->svm_ptrs, item, tmp)
  {
    if (item->svm_ptr == usm_pointer)
      {
        DL_DELETE (context->svm_ptrs, item);
        break;
      }
  }
  POCL_UNLOCK_OBJ (context);

  POCL_RETURN_ERROR_ON (
      (item == NULL), CL_INVALID_VALUE,
      "Can't find pointer in list of allocated USM pointers");

  POCL_MEM_FREE (item);

  POname (clReleaseContext) (context);

  context->svm_allocdev->ops->usm_free (context->usm_allocdev, usm_pointer,
                                        blocking);

  POCL_ATOMIC_DEC (usm_buffer_c);

  return CL_SUCCESS;
}

CL_API_ENTRY cl_int CL_API_CALL
POname (clMemFreeINTEL) (cl_context context,
                         void *usm_pointer) CL_API_SUFFIX__VERSION_2_0
{
  return pocl_mem_free_intel (context, usm_pointer, CL_FALSE);
}
POsym (clMemFreeINTEL)

    CL_API_ENTRY cl_int CL_API_CALL
    POname (clMemBlockingFreeINTEL) (cl_context context, void *usm_pointer)
        CL_API_SUFFIX__VERSION_2_0
{
  return pocl_mem_free_intel (context, usm_pointer, CL_TRUE);
}
POsym (clMemBlockingFreeINTEL)
