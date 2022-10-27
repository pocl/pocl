/* OpenCL runtime library: clRetainCommandBufferKHR()

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
POname (clRetainCommandBufferKHR) (cl_command_buffer_khr command_buffer)
    CL_API_SUFFIX__VERSION_1_2
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (command_buffer)),
                          CL_INVALID_COMMAND_BUFFER_KHR);

  POCL_RETURN_ERROR_COND (
      (command_buffer->state == CL_COMMAND_BUFFER_STATE_INVALID_KHR),
      CL_INVALID_COMMAND_BUFFER_KHR);

  int refc;
  POCL_RETAIN_OBJECT_REFCOUNT (command_buffer, refc);
  POCL_MSG_PRINT_REFCOUNTS ("Retain Command Buffer %p  : %d\n", command_buffer,
                            refc);

  return CL_SUCCESS;
}
POsym (clRetainCommandBufferKHR)
