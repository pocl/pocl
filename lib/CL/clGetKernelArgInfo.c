/* OpenCL runtime library: clGetKernelArgInfo()

   Copyright (c) 2014 Michal Babej

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
#include <string.h>



#define POCL_RETURN_KERNEL_INFO(__TYPE__, __VALUE__)                    \
  {                                                                     \
    size_t const value_size = sizeof(__TYPE__);                         \
    if (param_value) {                                                  \
      if (param_value_size < value_size) return CL_INVALID_VALUE;       \
      *(__TYPE__*)param_value = __VALUE__;                              \
    }                                                                   \
    if (param_value_size_ret)                                           \
      *param_value_size_ret = value_size;                               \
    return CL_SUCCESS;                                                  \
  }

#define POCL_RETURN_KERNEL_INFO_STR(__STR__)                        \
  {                                                                 \
    size_t const value_size = strlen(__STR__) + 1;                  \
    if (param_value) {                                              \
      if (param_value_size < value_size) return CL_INVALID_VALUE;   \
      memcpy(param_value, __STR__, value_size);                     \
    }                                                               \
    if (param_value_size_ret)                                       \
      *param_value_size_ret = value_size;                           \
    return CL_SUCCESS;                                              \
  }                                                                 \



CL_API_ENTRY cl_int CL_API_CALL
POname(clGetKernelArgInfo)(cl_kernel      kernel ,
                cl_uint  arg_indx ,
                cl_kernel_arg_info  param_name ,
                size_t         param_value_size ,
                void *         param_value ,
                size_t *       param_value_size_ret) CL_API_SUFFIX__VERSION_1_2
{
  if (!kernel)
    return CL_INVALID_KERNEL;
  if (arg_indx >= kernel->num_args)
    return CL_INVALID_ARG_INDEX;

  struct pocl_argument_info *arg = &kernel->arg_info[arg_indx];
  switch (param_name) {
    case CL_KERNEL_ARG_ADDRESS_QUALIFIER:
      POCL_RETURN_KERNEL_INFO(cl_kernel_arg_address_qualifier, arg->address_qualifier);
    case CL_KERNEL_ARG_ACCESS_QUALIFIER:
      POCL_RETURN_KERNEL_INFO(cl_kernel_arg_access_qualifier, arg->access_qualifier);
    case CL_KERNEL_ARG_TYPE_NAME:
      POCL_RETURN_KERNEL_INFO_STR(arg->type_name);
    case CL_KERNEL_ARG_TYPE_QUALIFIER:
      POCL_RETURN_KERNEL_INFO(cl_kernel_arg_type_qualifier, arg->type_qualifier);
    case CL_KERNEL_ARG_NAME:
      POCL_RETURN_KERNEL_INFO_STR(arg->name);
  }
  return CL_INVALID_VALUE;
}
POsym(clGetKernelArgInfo)
