/* OpenCL runtime library: clGetEventInfo()

   Copyright (c) 2011-2019 pocl developers

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

CL_API_ENTRY cl_int CL_API_CALL POname (clGetEventInfo) (
    cl_event event, cl_event_info param_name, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (event)),
                          CL_INVALID_COMMAND_QUEUE);

  POCL_LOCK_OBJ (event);
  cl_int s = event->status;
  cl_command_queue q = event->queue;
  cl_command_type t = event->command_type;
  cl_uint r = event->pocl_refcount;
  cl_context c = event->context;
  POCL_UNLOCK_OBJ (event);

  switch (param_name)
    {
    case CL_EVENT_COMMAND_EXECUTION_STATUS:
      POCL_RETURN_GETINFO (cl_int, s);
    case CL_EVENT_COMMAND_QUEUE:
      POCL_RETURN_GETINFO (cl_command_queue, q);
    case CL_EVENT_COMMAND_TYPE:
      POCL_RETURN_GETINFO (cl_command_type, t);
    case CL_EVENT_REFERENCE_COUNT:
      POCL_RETURN_GETINFO (cl_uint, r);
    case CL_EVENT_CONTEXT:
      POCL_RETURN_GETINFO (cl_context, c);
    default:
      break;
    }
  return CL_INVALID_VALUE;
}
POsym(clGetEventInfo)
