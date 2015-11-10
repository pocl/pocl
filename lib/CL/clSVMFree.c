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

CL_API_ENTRY void CL_API_CALL
POname(clSVMFree)(cl_context context,
                  void *svm_pointer) CL_API_SUFFIX__VERSION_2_0
{
  POCL_RETURN_ERROR_COND((context == NULL), NULL);

  POCL_RETURN_ERROR_ON((!context->svm_allocdev), NULL,
                       "None of the devices in this context is SVM-capable\n");

  if (svm_pointer == NULL)
    return;

  context->svm_allocdev->ops->free_ptr(context->svm_allocdev, svm_pointer);

}
