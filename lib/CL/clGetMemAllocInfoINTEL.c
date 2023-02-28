/* OpenCL runtime library: clGetMemAllocInfoINTEL()

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

#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_CALL
POname (clGetMemAllocInfoINTEL) (
    cl_context context, const void *ptr, cl_mem_info_intel param_name,
    size_t param_value_size, void *param_value,
    size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_2_0
{

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (context)), CL_INVALID_CONTEXT);

  POCL_RETURN_ERROR_COND ((ptr == NULL), CL_INVALID_MEM_OBJECT);

  POCL_RETURN_ERROR_ON (
      (!context->usm_allocdev), CL_INVALID_OPERATION,
      "None of the devices in this context is USM-capable\n");

  cl_device_id dev = context->usm_allocdev;

  POCL_RETURN_ERROR_ON (
      (dev->ops->get_mem_info_ext == NULL), CL_INVALID_OPERATION,
      "None of the devices in this context is USM-capable\n");

  return dev->ops->get_mem_info_ext (dev, ptr, param_name, param_value_size,
                                     param_value, param_value_size_ret);
}
POsym (clGetMemAllocInfoINTEL)
