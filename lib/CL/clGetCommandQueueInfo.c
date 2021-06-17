/* OpenCL runtime library: clGetCommandQueueInfo

   Copyright (c) 2012 Kalle Raiskila

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
POname(clGetCommandQueueInfo)(cl_command_queue      command_queue ,
                      cl_command_queue_info param_name ,
                      size_t                param_value_size ,
                      void *                param_value ,
                      size_t *              param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_queue)),
                          CL_INVALID_COMMAND_QUEUE);

  switch (param_name)
  {
    case CL_QUEUE_CONTEXT:
      POCL_RETURN_GETINFO( cl_context, command_queue->context );
      break;
    case CL_QUEUE_DEVICE:
      POCL_RETURN_GETINFO( cl_device_id, command_queue->device );
      break;
    case CL_QUEUE_REFERENCE_COUNT:
      POCL_RETURN_GETINFO( cl_uint, (cl_uint)command_queue->pocl_refcount );
      break;
    case CL_QUEUE_PROPERTIES:
      POCL_RETURN_GETINFO( cl_command_queue_properties, 
                              command_queue->properties );
      break;
    /* Device-side enqueue specific queries */
    case CL_QUEUE_SIZE:
      return CL_INVALID_COMMAND_QUEUE;
    case CL_QUEUE_DEVICE_DEFAULT:
      POCL_RETURN_GETINFO (cl_command_queue, NULL);
      break;
  }
  return CL_INVALID_VALUE;
}
POsym(clGetCommandQueueInfo)
