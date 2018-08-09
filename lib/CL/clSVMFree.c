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

CL_API_ENTRY void CL_API_CALL
POname(clSVMFree)(cl_context context,
                  void *svm_pointer) CL_API_SUFFIX__VERSION_2_0
{
  if (context == NULL)
  {
    POCL_MSG_WARN("Bad cl_context");
    return;
  }

  if (context->svm_allocdev==NULL)
  {
    POCL_MSG_WARN("None of the devices in this context is SVM-capable");
    return;
  }

  if (svm_pointer == NULL)
    return;

  context->svm_allocdev->ops->svm_free (context->svm_allocdev, svm_pointer);

}
POsym(clSVMFree)

