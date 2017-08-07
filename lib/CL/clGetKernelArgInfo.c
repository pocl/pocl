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

#include "pocl_util.h"


CL_API_ENTRY cl_int CL_API_CALL
POname(clGetKernelArgInfo)(cl_kernel      kernel ,
                cl_uint  arg_indx ,
                cl_kernel_arg_info  param_name ,
                size_t         param_value_size ,
                void *         param_value ,
                size_t *       param_value_size_ret) CL_API_SUFFIX__VERSION_1_2
{
  POCL_RETURN_ERROR_COND((kernel == NULL), CL_INVALID_KERNEL);

  POCL_RETURN_ERROR_ON((arg_indx >= kernel->num_args), CL_INVALID_ARG_INDEX,
    "This kernel has %u args, cannot getInfo on arg %u\n",
    (unsigned)kernel->num_args, (unsigned)arg_indx);

  /* pocl always uses -cl-kernel-arg-info because it needs the arg metadata, but
   * to the user programs we should report missing arg info in case they don't
   * request it. Piglit tests this. */
  if (kernel->program->compiler_options)
    POCL_RETURN_ERROR_ON (
        (!strstr (kernel->program->compiler_options, "cl-kernel-arg-info")),
        CL_KERNEL_ARG_INFO_NOT_AVAILABLE,
        "argument information is not available!\n");

  struct pocl_argument_info *arg = &kernel->arg_info[arg_indx];
  switch (param_name) {
    case CL_KERNEL_ARG_ADDRESS_QUALIFIER:
      if (!(kernel->has_arg_metadata & POCL_HAS_KERNEL_ARG_ADDRESS_QUALIFIER))
        return CL_KERNEL_ARG_INFO_NOT_AVAILABLE;
      POCL_RETURN_GETINFO(cl_kernel_arg_address_qualifier, arg->address_qualifier);
    case CL_KERNEL_ARG_ACCESS_QUALIFIER:
      if (!(kernel->has_arg_metadata & POCL_HAS_KERNEL_ARG_ACCESS_QUALIFIER))
        return CL_KERNEL_ARG_INFO_NOT_AVAILABLE;
      POCL_RETURN_GETINFO(cl_kernel_arg_access_qualifier, arg->access_qualifier);
    case CL_KERNEL_ARG_TYPE_NAME:
      if (!(kernel->has_arg_metadata & POCL_HAS_KERNEL_ARG_TYPE_NAME))
        return CL_KERNEL_ARG_INFO_NOT_AVAILABLE;
      POCL_RETURN_GETINFO_STR(arg->type_name);
    case CL_KERNEL_ARG_TYPE_QUALIFIER:
      if (!(kernel->has_arg_metadata & POCL_HAS_KERNEL_ARG_TYPE_QUALIFIER))
        return CL_KERNEL_ARG_INFO_NOT_AVAILABLE;
      POCL_RETURN_GETINFO(cl_kernel_arg_type_qualifier, arg->type_qualifier);
    case CL_KERNEL_ARG_NAME:
      if (!(kernel->has_arg_metadata & POCL_HAS_KERNEL_ARG_NAME))
        return CL_KERNEL_ARG_INFO_NOT_AVAILABLE;
      POCL_RETURN_GETINFO_STR(arg->name);
  }
  return CL_INVALID_VALUE;
}
POsym(clGetKernelArgInfo)
