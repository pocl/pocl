/* OpenCL runtime library: clFinalizeCommandBufferKHR()

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

#include <CL/cl_ext.h>

#include "pocl_cl.h"

CL_API_ENTRY cl_int CL_API_CALL
POname (clFinalizeCommandBufferKHR) (cl_command_buffer_khr command_buffer)
    CL_API_SUFFIX__VERSION_1_2
{
  unsigned num_finalized = 0;
  int errcode_ret = CL_SUCCESS;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_buffer)),
                          CL_INVALID_COMMAND_BUFFER_KHR);

  POCL_RETURN_ERROR_COND (
      (command_buffer->state != CL_COMMAND_BUFFER_STATE_RECORDING_KHR),
      CL_INVALID_OPERATION);

  /* TODO: perform task graph optimizations here */

  /* Command buffers API is per queue but internal handling is per device */
  cl_device_id *finalized_devs
      = calloc (command_buffer->num_queues, sizeof (cl_device_id));
  POCL_RETURN_ERROR_COND ((finalized_devs == NULL), CL_OUT_OF_HOST_MEMORY);

  cl_command_queue *q = command_buffer->queues;
  for (cl_uint i = 0; i < command_buffer->num_queues; ++i, ++q)
    {
      int is_done = 0;
      for (unsigned int j = 0; j < num_finalized; ++j)
        {
          if (finalized_devs[j] == (*q)->device)
            is_done = 1;
        }
      if (is_done)
        continue;

      int errcode = CL_SUCCESS;
      if ((*q)->device->ops->create_finalized_command_buffer)
        errcode = (*q)->device->ops->create_finalized_command_buffer (
            (*q)->device, command_buffer);
      if (errcode != CL_SUCCESS)
        errcode_ret = errcode;

      finalized_devs[num_finalized++] = (*q)->device;
    }

  POCL_MEM_FREE (finalized_devs);

  command_buffer->state = CL_COMMAND_BUFFER_STATE_EXECUTABLE_KHR;

  return errcode_ret;
}
POsym (clFinalizeCommandBufferKHR)
