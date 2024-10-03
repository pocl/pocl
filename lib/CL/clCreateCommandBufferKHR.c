/* OpenCL runtime library: clCreateCommandBufferKHR()

   Copyright (c) 2022-2024 Jan Solanti / Tampere University

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

#include <CL/cl_ext.h>

#include "pocl_cl.h"
#include "pocl_util.h"

CL_API_ENTRY cl_command_buffer_khr CL_API_CALL
POname (clCreateCommandBufferKHR) (
    cl_uint num_queues, const cl_command_queue *queues,
    const cl_command_buffer_properties_khr *properties,
    cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_2
{
  int errcode = 0;
  cl_command_buffer_khr cmdbuf = NULL;

  POCL_GOTO_ERROR_COND ((num_queues > 1
                         && strstr (queues[0]->device->extensions,
                                    "cl_khr_command_buffer_multi_device")
                              == NULL),
                        CL_INVALID_VALUE);
  POCL_GOTO_ERROR_COND ((num_queues == 0), CL_INVALID_VALUE);
  POCL_GOTO_ERROR_COND ((queues == NULL), CL_INVALID_VALUE);

  /* All queues must have the same OpenCL context */
  cl_context ref_ctx = queues[0]->context;

  for (unsigned i = 0; i < num_queues; ++i)
    {
      /* All queues must be valid Command queue objects */
      POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (queues[i])),
                            CL_INVALID_COMMAND_QUEUE);

      POCL_GOTO_ERROR_COND ((queues[i]->device == NULL),
                            CL_INVALID_COMMAND_QUEUE);

      POCL_GOTO_ERROR_COND ((queues[i]->context == NULL),
                            CL_INVALID_COMMAND_QUEUE);

      POCL_GOTO_ERROR_COND ((queues[i]->context != ref_ctx),
                            CL_INVALID_COMMAND_QUEUE);
    }

  cl_uint num_properties = 0;
  if (properties != NULL)
    {
      const cl_command_buffer_properties_khr *key = 0;
      for (key = properties; *key != 0; key += 2)
        num_properties += 1;
      POCL_GOTO_ERROR_ON (
        num_properties == 0, CL_INVALID_VALUE,
        "Properties != NULL, but zero properties in array\n");

      unsigned i = 0;
      cl_command_buffer_properties_khr seen_keys[num_properties];
      for (i = 0; i < num_properties; ++i)
        seen_keys[i] = 0;

      i = 0;
      for (key = properties; *key != 0; key += 2, ++i)
        {
          /* Duplicate keys are not allowed */
          for (unsigned j = 0; j < i; ++j)
            {
              POCL_GOTO_ERROR_ON (
                (*key == seen_keys[j]), CL_INVALID_VALUE,
                "Repeated key in cl_command_buffer_properties_khr "
                "*properties\n");
            }

          const cl_command_buffer_properties_khr *val = key + 1;
          cl_command_buffer_properties_khr tmp = *val;
          switch (*key)
            {
            case CL_COMMAND_BUFFER_FLAGS_KHR:
              /* Simultaneous use is always supported, no action needed */
              tmp &= ~CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR;

              /* If any of the devices associated with 'queues' does not
               * support a requested capability, error out with
               * CL_INVALID_PROPERTY */

              /* Any flag bits not handled above are invalid */
              POCL_GOTO_ERROR_ON (
                (tmp != 0), CL_INVALID_VALUE,
                "Unknown flags in CL_COMMAND_BUFFER_FLAGS_KHR property\n");
              seen_keys[i] = *key;
              break;
            default:
              errcode = CL_INVALID_VALUE;
              goto ERROR;
            }
        }
    }

  cmdbuf = calloc (1, sizeof (struct _cl_command_buffer_khr));
  if (cmdbuf == NULL)
    {
      errcode = CL_OUT_OF_HOST_MEMORY;
      goto ERROR;
    }

  POCL_INIT_OBJECT (cmdbuf);

  cmdbuf->state = CL_COMMAND_BUFFER_STATE_RECORDING_KHR;
  cmdbuf->num_queues = num_queues;
  cmdbuf->queues
      = (cl_command_queue *)calloc (num_queues, sizeof (cl_command_queue));
  memcpy (cmdbuf->queues, queues, num_queues * sizeof (cl_command_queue));
  cmdbuf->num_properties = num_properties;
  POCL_INIT_LOCK (cmdbuf->mutex);
  if (num_properties > 0)
    {
      cmdbuf->properties = (cl_command_buffer_properties_khr *)malloc (
          (num_properties * 2 + 1)
          * sizeof (cl_command_buffer_properties_khr));
      memcpy (cmdbuf->properties, properties,
              sizeof (cl_command_buffer_properties_khr)
                  * (num_properties * 2 + 1));
    }

  for (unsigned i = 0; i < num_queues; ++i)
    {
      POname (clRetainCommandQueue) (queues[i]);
    }

  if (errcode_ret != NULL)
    *errcode_ret = errcode;
  return cmdbuf;

ERROR:
  if (cmdbuf)
    {
      POCL_MEM_FREE (cmdbuf->queues);
      POCL_MEM_FREE (cmdbuf->properties);
    }
  POCL_MEM_FREE (cmdbuf);
  if (errcode_ret)
    *errcode_ret = errcode;
  return NULL;
}
POsym (clCreateCommandBufferKHR)
