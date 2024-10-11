/* OpenCL runtime library: clCreateCommandQueue()

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

#include "pocl_cl.h"
#include "pocl_cq_profiling.h"
#include "pocl_util.h"
#include "utlist.h"

CL_API_ENTRY cl_command_queue CL_API_CALL
POname(clCreateCommandQueue)(cl_context context, 
                     cl_device_id device, 
                     cl_command_queue_properties properties,
                     cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  unsigned i;
  int errcode;
  cl_bool found = CL_FALSE;

  POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (context)), CL_INVALID_CONTEXT);

  POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (device)), CL_INVALID_DEVICE);

  POCL_GOTO_ERROR_ON (((*device->available) != CL_TRUE), CL_INVALID_DEVICE,
                      "Device %d is not available\n", device->dev_id);

  /* validate flags */
  cl_command_queue_properties all_properties
      = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE
        | CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT | CL_QUEUE_HIDDEN;

  POCL_GOTO_ERROR_ON ((properties & (~all_properties)), CL_INVALID_VALUE,
                      "Unknown properties requested\n");

  cl_command_queue_properties supported_device_props;
  if (properties & (CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT))
    supported_device_props = device->on_dev_queue_props;
  else
    supported_device_props = device->on_host_queue_props | CL_QUEUE_HIDDEN;

  POCL_GOTO_ERROR_ON (((properties & supported_device_props) != properties),
                      CL_INVALID_QUEUE_PROPERTIES, "properties (%zu) are "
                      "not supported by the device (%zu)\n",
                      (size_t)properties, (size_t)supported_device_props);

  if (POCL_DEBUGGING_ON || pocl_cq_profiling_enabled)
    properties |= CL_QUEUE_PROFILING_ENABLE;

  for (i=0; i<context->num_devices; i++)
    {
      if (context->devices[i] == pocl_real_dev (device))
        found = CL_TRUE;
    }

  POCL_GOTO_ERROR_ON (
      (found == CL_FALSE), CL_INVALID_DEVICE,
      "Could not find real device of device %d in the context\n",
      device->dev_id);

  cl_command_queue command_queue
      = (cl_command_queue)calloc (1, sizeof (struct _cl_command_queue));
  if (command_queue == NULL)
  {
    errcode = CL_OUT_OF_HOST_MEMORY;
    goto ERROR;
  }

  POCL_INIT_OBJECT(command_queue);

  command_queue->context = context;
  command_queue->device = device;
  command_queue->properties = properties;

  /* hidden queues don't retain the context. */
  if ((properties & CL_QUEUE_HIDDEN) == 0)
  {
    POname (clRetainContext) (context);
    POCL_LOCK_OBJ (context);
    DL_APPEND (context->command_queues, command_queue);
    POCL_UNLOCK_OBJ (context);
  }

  TP_CREATE_QUEUE (context->id, command_queue->id);

  errcode = CL_SUCCESS;
  if (device->ops->init_queue)
    errcode = device->ops->init_queue (device, command_queue);

  POCL_ATOMIC_INC (queue_c);

  if (errcode_ret != NULL)
    *errcode_ret = errcode;

  POCL_MSG_PRINT_INFO ("Created Command Queue %" PRId64 " (%p) on device %d\n",
                       command_queue->id, command_queue, device->dev_id);

  return command_queue;

ERROR:
    if(errcode_ret)
    {
        *errcode_ret = errcode;
    }
    return NULL;
}
POsym(clCreateCommandQueue)
