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
#include <string.h>
#include "pocl_cl.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clGetContextInfo)(cl_context context, 
                 cl_context_info param_name,
                 size_t param_value_size,
                 void *param_value,
                 size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
  size_t value_size;
  
  if (context == NULL)
    return CL_INVALID_CONTEXT;

  switch (param_name) {
    case CL_CONTEXT_REFERENCE_COUNT:
    {
      cl_uint refcount = context->pocl_refcount;
      if (param_value_size < sizeof(size_t) && param_value != NULL)
        return CL_INVALID_VALUE;
      if (param_value != NULL )
        *((size_t*)param_value)=refcount;
      if (param_value_size_ret != NULL)
        *param_value_size_ret=sizeof(size_t);
      return CL_SUCCESS;
    }
  case CL_CONTEXT_DEVICES:
    {
      value_size = context->num_devices * sizeof(cl_device_id);
      if (param_value != NULL) {
        if (param_value_size < value_size)
          return CL_INVALID_VALUE;
        memcpy(param_value, context->devices, value_size);
      }
      if (param_value_size_ret != NULL)
        *param_value_size_ret = value_size;
      return CL_SUCCESS;
    }
  case CL_CONTEXT_NUM_DEVICES:
    {
      if (param_value_size < sizeof(size_t) && param_value != NULL)
        return CL_INVALID_VALUE;
      if (param_value != NULL )
        *((size_t*)param_value)=context->num_devices;
      if (param_value_size_ret != NULL)
        *param_value_size_ret=sizeof(size_t);
      return CL_SUCCESS;
    }
  case CL_CONTEXT_PROPERTIES:
    {
      size_t properties_size = (context->num_properties * 2 + 1) * sizeof(cl_context_properties);
      if (param_value_size < properties_size && param_value != NULL)
        return CL_INVALID_VALUE;
      if (param_value != NULL )
        param_value=context->properties;
      if (param_value_size_ret != NULL)
        *param_value_size_ret=properties_size;
      return CL_SUCCESS;
  }
  default:
  return CL_INVALID_VALUE;
}

  return CL_SUCCESS;
}
POsym(clGetContextInfo)
