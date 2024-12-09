/* OpenCL runtime library: clGetCommandBufferInfoKHR()

   Copyright (c) 2022 Jan Solanti / Tampere University

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

#include "pocl_cl.h"
#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_CALL
POname (clGetCommandBufferInfoKHR) (
    cl_command_buffer_khr command_buffer,
    cl_command_buffer_info_khr param_name, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_2
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_buffer)),
                          CL_INVALID_COMMAND_BUFFER_KHR);

#define PARAM_SIZE(expr)                                                      \
  {                                                                           \
    if (param_value_size_ret != NULL)                                         \
      {                                                                       \
        *param_value_size_ret = expr;                                         \
      }                                                                       \
  }
#define PARAM_VALUE(val, size)                                                \
  {                                                                           \
    if (param_value != NULL)                                                  \
      {                                                                       \
        POCL_RETURN_ERROR_COND ((param_value_size < size), CL_INVALID_VALUE); \
        memcpy (param_value, val, size);                                      \
      }                                                                       \
  }

  /* All queues must have the same OpenCL context */
  cl_context ref_ctx = command_buffer->queues[0]->context;

  switch (param_name)
    {
    case CL_COMMAND_BUFFER_NUM_QUEUES_KHR:
      PARAM_SIZE (sizeof (cl_uint));
      PARAM_VALUE (&command_buffer->num_queues, sizeof (cl_uint));
      break;
    case CL_COMMAND_BUFFER_QUEUES_KHR:
      PARAM_SIZE (sizeof (cl_command_queue) * command_buffer->num_queues);
      PARAM_VALUE (command_buffer->queues,
                   sizeof (cl_command_queue) * command_buffer->num_queues);
      break;
    case CL_COMMAND_BUFFER_REFERENCE_COUNT_KHR:
      PARAM_SIZE (sizeof (cl_uint));
      PARAM_VALUE (&command_buffer->pocl_refcount, sizeof (cl_uint));
      break;
    case CL_COMMAND_BUFFER_STATE_KHR:
      PARAM_SIZE (sizeof (cl_command_buffer_state_khr));
      PARAM_VALUE (&command_buffer->state,
                   sizeof (cl_command_buffer_state_khr));
      break;
    case CL_COMMAND_BUFFER_PROPERTIES_ARRAY_KHR:
      {
        size_t num_properties = command_buffer->num_properties > 0
                                    ? 2 * command_buffer->num_properties + 1
                                    : 0;
        POCL_RETURN_GETINFO_ARRAY (cl_command_buffer_properties_khr,
                                   num_properties, command_buffer->properties);
        break;
      }
    case CL_COMMAND_BUFFER_CONTEXT_KHR:
      PARAM_SIZE (sizeof (cl_context));
      PARAM_VALUE (&ref_ctx, sizeof (cl_context));
      break;
    default:
      return CL_INVALID_VALUE;
    }
#undef PARAM_SIZE
#undef PARAM_VALUE

  return CL_SUCCESS;
}
POsym (clGetCommandBufferInfoKHR)
