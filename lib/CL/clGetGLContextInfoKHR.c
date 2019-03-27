/* OpenCL runtime library: clGetGLContextInfoKHR()

   Copyright (c) 2021 Michal Babej / Tampere University

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

#include <assert.h>
#include "pocl_util.h"
#include "devices.h"

CL_API_ENTRY cl_int CL_API_CALL POname (clGetGLContextInfoKHR) (
    const cl_context_properties *properties, cl_gl_context_info param_name,
    size_t param_value_size, void *param_value,
    size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
  POCL_RETURN_ERROR_COND ((properties == NULL), CL_INVALID_OPERATION);

  switch (param_name)
    {

    case CL_DEVICES_FOR_GL_CONTEXT_KHR:
      {
        cl_device_id *dev_array
            = alloca (sizeof (cl_device_id) * pocl_num_devices);
        unsigned j = 0;
        for (unsigned i = 0; i < pocl_num_devices; ++i)
          {
            cl_device_id dev = &pocl_devices[i];
            if (dev->ops->get_gl_context_assoc != NULL
                && dev->ops->get_gl_context_assoc (
                       dev, CL_DEVICES_FOR_GL_CONTEXT_KHR, properties)
                       == CL_SUCCESS)
              {
                dev_array[j++] = dev;
              }
          }
        if (j > 0)
          POCL_RETURN_GETINFO_ARRAY (cl_device_id, j, dev_array);
        else
          {
            if (param_value_size_ret)
              *param_value_size_ret = 0;
            return CL_SUCCESS;
          }
        break;
      }

    case CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR:
      {
        cl_device_id found = NULL;
        for (unsigned i = 0; i < pocl_num_devices; ++i)
          {
            cl_device_id dev = &pocl_devices[i];
            if (dev->ops->get_gl_context_assoc != NULL
                && dev->ops->get_gl_context_assoc (
                       dev, CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR, properties)
                       == CL_SUCCESS)
              {
                found = dev;
                break;
              }
          }
        if (found)
          POCL_RETURN_GETINFO (cl_device_id, found);
        else
          {
            if (param_value_size_ret)
              *param_value_size_ret = 0;
            return CL_SUCCESS;
          }
      }
    }

  return CL_INVALID_VALUE;
}
POsym (clGetGLContextInfoKHR)
