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

#include "pocl_cl.h"



#define POCL_RETURN_QUEUE_INFO(__TYPE__, __VALUE__)                     \
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



CL_API_ENTRY cl_int CL_API_CALL
clGetCommandQueueInfo(cl_command_queue      command_queue ,
                      cl_command_queue_info param_name ,
                      size_t                param_value_size ,
                      void *                param_value ,
                      size_t *              param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
  if (!command_queue)
    return CL_INVALID_COMMAND_QUEUE;
  switch (param_name) {
  case CL_QUEUE_CONTEXT:
    POCL_RETURN_QUEUE_INFO(cl_context, command_queue->context);
  case CL_QUEUE_DEVICE:
    POCL_RETURN_QUEUE_INFO(cl_device_id, command_queue->device);
  case CL_QUEUE_REFERENCE_COUNT:
    POCL_RETURN_QUEUE_INFO(cl_uint, command_queue->pocl_refcount);
  case CL_QUEUE_PROPERTIES:
    POCL_RETURN_QUEUE_INFO(cl_command_queue_properties, command_queue->properties);
  }
  return CL_INVALID_VALUE;
}
