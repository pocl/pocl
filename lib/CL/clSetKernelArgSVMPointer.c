/* OpenCL runtime library: clSetKernelArgSVMPointer()

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

#include "config.h"
#include "pocl_cl.h"
#include "pocl_util.h"
#include "devices.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clSetKernelArgSVMPointer)(cl_kernel kernel,
                                 cl_uint arg_index,
                                 const void *arg_value) CL_API_SUFFIX__VERSION_2_0
{
  POCL_RETURN_ERROR_COND((kernel == NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_ON (
      (!kernel->context->svm_alloc_mem), CL_INVALID_CONTEXT,
      "None of the devices in this context is SVM-capable\n");

  size_t offset;

  cl_mem mem = kernel->context->svm_alloc_mem->svm_pointer_to_clmem (
      kernel->context->svm_alloc_mem, arg_value, &offset);

  POCL_RETURN_ERROR_ON ((mem == NULL), CL_INVALID_VALUE,
                        "Invalid SVM pointer in arg_value");

  POCL_MSG_PRINT_INFO ("Setting kernel ARG %i to SVM %p using cl_mem: %p\n",
                       arg_index, arg_value, (void *)mem);

  int retval
      = POname (clSetKernelArg) (kernel, arg_index, sizeof (cl_mem), &mem);
  if (retval == CL_SUCCESS && offset > 0)
    {
      kernel->dyn_arguments[arg_index].offset = offset;
    }
  return retval;
}
POsym(clSetKernelArgSVMPointer)
