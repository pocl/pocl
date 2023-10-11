/* OpenCL runtime library: clGetContextInfo()

   Copyright (c) 2011 Universidad Rey Juan Carlos

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

#include <assert.h>
#include "pocl_util.h"


CL_API_ENTRY cl_int CL_API_CALL
POname(clGetContextInfo)(cl_context context,
                 cl_context_info param_name,
                 size_t param_value_size,
                 void *param_value,
                 size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
  size_t value_size;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (context)),
                          CL_INVALID_COMMAND_QUEUE);

  switch (param_name) {
  case CL_CONTEXT_REFERENCE_COUNT:
    {
      cl_uint refcount = context->pocl_refcount;
      POCL_RETURN_GETINFO(cl_uint, refcount);
    }
  case CL_CONTEXT_DEVICES:
    value_size = context->num_devices * sizeof(cl_device_id);
    POCL_RETURN_GETINFO_SIZE(value_size, context->devices);
  case CL_CONTEXT_NUM_DEVICES:
    POCL_RETURN_GETINFO(cl_uint, context->num_devices);
  case CL_CONTEXT_PROPERTIES:
    if (context->properties)
      {
        value_size = (context->num_properties * 2 + 1) * sizeof(cl_context_properties);
        POCL_RETURN_GETINFO_SIZE(value_size, context->properties);
      }
    else
      {
        if (param_value_size_ret != NULL)
          *param_value_size_ret = 0;
        return CL_SUCCESS;
      }
  default:
    return CL_INVALID_VALUE;
  }
  
  return CL_SUCCESS;
}
POsym(clGetContextInfo)
