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

#include "pocl_cl.h"
#include "devices.h"

CL_API_ENTRY void CL_API_CALL
POname(clSVMFree)(cl_context context,
                  void *svm_pointer) CL_API_SUFFIX__VERSION_2_0
{
#ifndef BUILD_HSA
  POCL_MSG_PRINT_INFO("This pocl was not built with HSA\n");
  return;
#else

  POCL_RETURN_ERROR_COND((context == NULL), NULL);

  if (svm_pointer == NULL)
    return;

  /* Find a suitable device (with SVM support) */
  cl_device_id host = NULL, svmdev = NULL, allocdev = NULL;
  for (unsigned i=0; i < context->num_devices; i++)
    {
      if (context->devices[i]->is_svm)
        svmdev = context->devices[i];
      if (!context->devices[i]->is_svm && !host)
        host = context->devices[i];
    }

  allocdev = svmdev ? svmdev : host;
  assert(allocdev);

  cl_mem mem = alloca(sizeof(struct _cl_mem));
  mem->flags = CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE;
  mem->mem_host_ptr = NULL;
  pocl_mem_identifier device_ptrs[pocl_num_devices];
  device_ptrs[allocdev->dev_id].mem_ptr = svm_pointer;
  mem->device_ptrs = device_ptrs;

  allocdev->ops->free(allocdev, mem);


#endif
}
