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
  cl_queue_priority_khr priority = 0;
  cl_queue_throttle_khr throttle = 0;
  cl_uint queue_size = 0;
  int queue_props_set = 0, queue_size_set = 0;
  int queue_priority_set = 0, queue_throttle_set = 0;
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
    while (properties[i])
      switch (properties[i])
        {
        case CL_QUEUE_PROPERTIES:
          {
            POCL_GOTO_ERROR_ON ((queue_props_set > 0), CL_INVALID_VALUE,
                                "CL_QUEUE_PROPERTIES was already set");
            queue_props = (cl_command_queue_properties)properties[i + 1];
            queue_props_set++;
            i += 2;
            break;
          }
        case CL_QUEUE_SIZE:
          {
            POCL_GOTO_ERROR_ON ((queue_size_set > 0), CL_INVALID_VALUE,
                                "CL_QUEUE_SIZE was already set");
            queue_size = (cl_uint)properties[i + 1];
            queue_size_set++;
            i += 2;
            break;
          }
        case CL_QUEUE_PRIORITY_KHR:
          {
            cl_queue_properties value = properties[i + 1];
            POCL_GOTO_ERROR_ON ((queue_priority_set > 0), CL_INVALID_VALUE,
                                "CL_QUEUE_PRIORITY_KHR was already set");
            POCL_GOTO_ERROR_ON ((value != CL_QUEUE_PRIORITY_HIGH_KHR
                                 && value != CL_QUEUE_PRIORITY_MED_KHR
                                 && value != CL_QUEUE_PRIORITY_LOW_KHR),
                                CL_INVALID_VALUE,
                                "Invalid CL_QUEUE_PRIORITY_KHR value");
            priority = (cl_queue_priority_khr)properties[i + 1];
            queue_priority_set++;
            /* This is a hint that provides no behavior or minimum guarantees
             * so it is always safe to bring it along (no action required) */
            i += 2;
            break;
          }
        case CL_QUEUE_THROTTLE_KHR:
          {
            cl_queue_properties value = properties[i + 1];
            POCL_GOTO_ERROR_ON ((queue_throttle_set > 0), CL_INVALID_VALUE,
                                "CL_QUEUE_THROTTLE_KHR was already set");
            POCL_GOTO_ERROR_ON ((value != CL_QUEUE_THROTTLE_HIGH_KHR
                                 && value != CL_QUEUE_THROTTLE_MED_KHR
                                 && value != CL_QUEUE_THROTTLE_LOW_KHR),
                                CL_INVALID_VALUE,
                                "Invalid CL_QUEUE_THROTTLE_KHR value");
            throttle = (cl_queue_throttle_khr)properties[i + 1];
            queue_throttle_set++;
            /* This is a hint that provides no behavior or minimum guarantees
             * so it is always safe to bring it along (no action required) */
            i += 2;
            break;
          }
        default:
          POCL_GOTO_ERROR_ON (1, CL_INVALID_VALUE,
                              "Invalid values in properties: %lu\n",
                              (unsigned long)properties[i]);
        }

  if (queue_props_set)
    {
      if (queue_props & CL_QUEUE_ON_DEVICE)
        {
          if (queue_size == 0)
            queue_size = device->dev_queue_pref_size;

          POCL_GOTO_ERROR_COND((queue_size > device->dev_queue_max_size),
                               CL_INVALID_QUEUE_PROPERTIES);
          POCL_GOTO_ERROR_COND((queue_priority_set), CL_INVALID_QUEUE_PROPERTIES);
          POCL_GOTO_ERROR_COND((queue_throttle_set), CL_INVALID_QUEUE_PROPERTIES);

         // create a device side queue
          POCL_GOTO_ERROR_ON ((device->on_dev_queue_props == 0),
                              CL_INVALID_QUEUE_PROPERTIES,
                              "Device-side enqueue is not supported "
                              "by any device\n");
        }
      else
        {
          POCL_GOTO_ERROR_ON (
            (queue_size > 0), CL_INVALID_QUEUE_PROPERTIES,
            "Queue size can only be specified for on-device queues\n");
          POCL_GOTO_ERROR_ON (
            (queue_priority_set
             && (strstr (device->extensions, "cl_khr_priority_hints")
                 == NULL)),
            CL_INVALID_QUEUE_PROPERTIES,
            "device does not support cl_khr_priority_hints\n");
          POCL_GOTO_ERROR_ON (
            (queue_throttle_set
             && (strstr (device->extensions, "cl_khr_throttle_hints")
                 == NULL)),
            CL_INVALID_QUEUE_PROPERTIES,
            "device does not support cl_khr_throttle_hints\n");
        }

      /* validate flags */
      POCL_GOTO_ERROR_ON ((queue_props & (~valid_prop_flags)),
                          CL_INVALID_VALUE,
                          "CL_QUEUE_PROPERTIES contain invalid entries");
    }

  // currently thhere's only support for host side queues.
  cl_command_queue cq_ret = POname (clCreateCommandQueue) (
      context, device, queue_props, errcode_ret);
  if (cq_ret == NULL)
    return NULL;

  if (properties)
    {
      assert (i < 10);
      cq_ret->properties = queue_props;
      cq_ret->priority = priority;
      cq_ret->throttle = throttle;
      cq_ret->num_queue_properties = i + 1;
      memcpy (cq_ret->queue_properties, properties,
              cq_ret->num_queue_properties * sizeof (cl_queue_properties));
    }

  return cq_ret;

ERROR:
  if(errcode_ret)
    {
      *errcode_ret = errcode;
    }
  return NULL;
}
POsym (clCreateCommandQueueWithProperties)

  CL_API_ENTRY cl_int CL_API_CALL POname (clSetCommandQueueProperty) (
    cl_command_queue command_queue,
    cl_command_queue_properties properties,
    cl_bool enable,
    cl_command_queue_properties *old_properties)
    CL_API_SUFFIX__VERSION_1_0_DEPRECATED
{
  /* CL_INVALID_OPERATION if no devices in the context associated with
   * command_queue support modifying the properties of a command-queue */
  return CL_INVALID_OPERATION;
}
POsym (clSetCommandQueueProperty)
