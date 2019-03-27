/* OpenCL runtime library: clCreateCommandQueueWithProperties()

   Copyright (c) 2015 Michal Babej / Tampere University of Technology

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

CL_API_ENTRY cl_command_queue CL_API_CALL
POname(clCreateCommandQueueWithProperties)(cl_context context,
                                           cl_device_id device,
                                           const cl_queue_properties *properties,
                                           cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  unsigned i = 0;
  int errcode;
  cl_bool found = CL_FALSE;
  cl_command_queue_properties queue_props = 0;
  int queue_props_set = 0;
  cl_uint queue_size = 0;
  const cl_command_queue_properties valid_prop_flags =
      (CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
       | CL_QUEUE_PROFILING_ENABLE
       | CL_QUEUE_ON_DEVICE
       | CL_QUEUE_ON_DEVICE_DEFAULT);

  POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (context)), CL_INVALID_CONTEXT);

  POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (device)), CL_INVALID_DEVICE);

  for (i=0; i<context->num_devices; i++)
    {
      if (context->devices[i] == pocl_real_dev (device))
        found = CL_TRUE;
    }

  POCL_GOTO_ERROR_ON((found == CL_FALSE), CL_INVALID_DEVICE,
                     "Could not find device in the context\n");

  i = 0;
  if (properties)
    while(properties[i])
      switch(properties[i])
        {
        case CL_QUEUE_PROPERTIES:
          queue_props = (cl_command_queue_properties)properties[i+1];
          queue_props_set = 1;
          i+=2;
          break;
        case CL_QUEUE_SIZE:
          queue_size = (cl_uint)properties[i+1];
          i+=2;
          break;
        default:
          POCL_GOTO_ERROR_ON(1, CL_INVALID_VALUE, "Invalid values it properties\n");
        }

  if (queue_props_set)
    {
      if (queue_props & CL_QUEUE_ON_DEVICE)
        {
          if (queue_size == 0)
            queue_size = device->dev_queue_pref_size;

          POCL_GOTO_ERROR_COND((queue_size > device->dev_queue_max_size),
                               CL_INVALID_QUEUE_PROPERTIES);

         // create a device side queue
         POCL_ABORT_UNIMPLEMENTED("Device side queue");
        }
      else
        POCL_GOTO_ERROR_ON((queue_size > 0), CL_INVALID_VALUE,
                           "To specify queue size, you must use CL_QUEUE_ON_DEVICE in flags\n");

      /* validate flags */
      POCL_GOTO_ERROR_ON((queue_props & (!valid_prop_flags)), CL_INVALID_VALUE,
                         "CL_QUEUE_PROPERTIES contain invalid entries");
    }

  // currently thhere's only support for host side queues.
  return POname(clCreateCommandQueue)(context, device, queue_props, errcode_ret);

ERROR:
  if(errcode_ret)
    {
      *errcode_ret = errcode;
    }
  return NULL;
}
POsym(clCreateCommandQueueWithProperties)
