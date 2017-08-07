/* OpenCL runtime library: clGetDeviceInfo()

   Copyright (c) 2012 Erik Schnetter
   
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




CL_API_ENTRY cl_int CL_API_CALL
POname(clGetKernelInfo)(cl_kernel      kernel ,
                cl_kernel_info param_name ,
                size_t         param_value_size ,
                void *         param_value ,
                size_t *       param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
  POCL_RETURN_ERROR_COND((kernel == NULL), CL_INVALID_KERNEL);

  switch (param_name) {
  case CL_KERNEL_FUNCTION_NAME:
    POCL_RETURN_GETINFO_STR(kernel->name);
  case CL_KERNEL_NUM_ARGS:
    POCL_RETURN_GETINFO(cl_uint, kernel->num_args);
  case CL_KERNEL_REFERENCE_COUNT:
    POCL_RETURN_GETINFO(cl_uint, kernel->pocl_refcount);
  case CL_KERNEL_CONTEXT:
    POCL_RETURN_GETINFO(cl_context, kernel->context);
  case CL_KERNEL_PROGRAM:
    POCL_RETURN_GETINFO(cl_program, kernel->program);
  case CL_KERNEL_ATTRIBUTES:
    if (kernel->attributes)
      POCL_RETURN_GETINFO_STR (kernel->attributes);
    else
      POCL_RETURN_GETINFO_STR ("");
  }
  return CL_INVALID_VALUE;
}
POsym(clGetKernelInfo)
