/* OpenCL runtime library: clSetContentSizeBufferPoCL

   Copyright (c) 2021 Tampere University

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


CL_API_ENTRY cl_int CL_API_CALL
POname(clSetContentSizeBufferPoCL) (cl_mem buffer,
                                    cl_mem content_size_buffer)
CL_API_SUFFIX__VERSION_1_2
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (buffer)),
                          CL_INVALID_MEM_OBJECT);

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (content_size_buffer)),
                          CL_INVALID_MEM_OBJECT);

  POCL_RETURN_ERROR_ON ((buffer->context != content_size_buffer->context),
                        CL_INVALID_CONTEXT,
                        "Buffers are not from the same context\n");

  POCL_RETURN_ERROR_ON ((content_size_buffer->size < sizeof (uint64_t)),
                        CL_INVALID_BUFFER_SIZE,
                        "The size buffer is too small\n");

  POCL_RETURN_ERROR_ON ((content_size_buffer->parent != NULL),
                        CL_INVALID_MEM_OBJECT,
                        "The size buffer cannot be a sub-buffer\n");

  buffer->size_buffer = content_size_buffer;
  buffer->content_buffer = NULL;

  content_size_buffer->size_buffer = NULL;
  content_size_buffer->content_buffer = buffer;

  return CL_SUCCESS;
}

POsym (clSetContentSizeBufferPoCL)
