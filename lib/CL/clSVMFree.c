/* OpenCL runtime library: clSVMFree()

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
#include "pocl_debug.h"

extern unsigned long svm_buffer_c;

CL_API_ENTRY void CL_API_CALL
POname(clSVMFree)(cl_context context,
                  void *svm_pointer) CL_API_SUFFIX__VERSION_2_0
{
  if (!IS_CL_OBJECT_VALID (context))
    {
      POCL_MSG_ERR ("Invalid cl_context\n");
      return;
    }

  if (context->svm_allocdev == NULL)
    {
      POCL_MSG_ERR ("None of the devices in this context is SVM-capable\n");
      return;
    }

  if (svm_pointer == NULL)
    {
      POCL_MSG_ERR ("Invalid SVM pointer\n");
      return;
    }

  POCL_LOCK_OBJ (context);
  pocl_svm_ptr *tmp = NULL, *item = NULL;
  DL_FOREACH_SAFE (context->svm_ptrs, item, tmp)
  {
    if (item->svm_ptr == svm_pointer)
      {
        DL_DELETE (context->svm_ptrs, item);
        break;
      }
  }
  POCL_UNLOCK_OBJ (context);

  if (item == NULL)
    {
      POCL_MSG_ERR ("can't find pointer in list of allocated SVM pointers");
      return;
    }

  POCL_MEM_FREE (item);

  POname (clReleaseContext) (context);

  context->svm_allocdev->ops->svm_free (context->svm_allocdev, svm_pointer);

  POCL_ATOMIC_DEC (svm_buffer_c);
}

POsym (clSVMFree)
