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

  pocl_raw_ptr *item = pocl_find_raw_ptr_with_vm_ptr (context, ptr);

  switch (param_name)
    {
    case CL_MEM_ALLOC_TYPE_INTEL:
      {
        if (item == NULL)
          POCL_RETURN_GETINFO (cl_uint, CL_MEM_TYPE_UNKNOWN_INTEL);
        else
          POCL_RETURN_GETINFO (cl_uint, item->usm_properties.alloc_type);
      }
    case CL_MEM_ALLOC_BASE_PTR_INTEL:
      {
        if (item == NULL)
          POCL_RETURN_GETINFO (void *, NULL);
        else
          POCL_RETURN_GETINFO (void *, item->vm_ptr);
      }
    case CL_MEM_ALLOC_SIZE_INTEL:
      {
        if (item == NULL)
          POCL_RETURN_GETINFO (size_t, 0);
        else
          POCL_RETURN_GETINFO (size_t, item->size);
      }
    case CL_MEM_ALLOC_DEVICE_INTEL:
      {
        if (item == NULL)
          POCL_RETURN_GETINFO (cl_device_id, NULL);
        else
          POCL_RETURN_GETINFO (cl_device_id, item->device);
      }
    case CL_MEM_ALLOC_FLAGS_INTEL:
      {
        if (item == NULL)
          POCL_RETURN_GETINFO (cl_mem_alloc_flags_intel, 0);
        else
          POCL_RETURN_GETINFO (cl_mem_alloc_flags_intel,
                               item->usm_properties.flags);
      }
    default:
      return CL_INVALID_VALUE;
    }
  return CL_SUCCESS;
}
POsym (clGetMemAllocInfoINTEL)
