/* pocl_fill_memobj.c: helpers for FillBuffer and FillImage commands

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
#include "pocl_image_util.h"
#include "pocl_util.h"

cl_int
pocl_validate_fill_buffer (cl_command_queue command_queue, cl_mem buffer,
                           const void *pattern, size_t pattern_size,
                           size_t offset, size_t size)
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (buffer)),
                          CL_INVALID_MEM_OBJECT);

  POCL_RETURN_ERROR_ON ((buffer->type != CL_MEM_OBJECT_BUFFER),
                        CL_INVALID_MEM_OBJECT,
                        "buffer is not a CL_MEM_OBJECT_BUFFER\n");

  POCL_RETURN_ERROR_ON (
      (command_queue->context != buffer->context), CL_INVALID_CONTEXT,
      "buffer and command_queue are not from the same context\n");

  cl_int errcode = pocl_buffer_boundcheck (buffer, offset, size);
  if (errcode != CL_SUCCESS)
    return errcode;

  /* CL_INVALID_VALUE if pattern is NULL or if pattern_size is 0
   * or if pattern_size is not one of {1, 2, 4, 8, 16, 32, 64, 128}. */
  POCL_RETURN_ERROR_COND ((pattern == NULL), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND ((pattern_size == 0), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND ((pattern_size > 128), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_ON (
      (__builtin_popcount (pattern_size) > 1), CL_INVALID_VALUE,
      "pattern_size(%zu) must be a power-of-two value", pattern_size);

  /* CL_INVALID_VALUE if offset and size are not a multiple of pattern_size. */
  POCL_RETURN_ERROR_ON (
      (offset % pattern_size), CL_INVALID_VALUE,
      "offset(%zu) must be a multiple of pattern_size(%zu)\n", offset,
      pattern_size);
  POCL_RETURN_ERROR_ON ((size % pattern_size), CL_INVALID_VALUE,
                        "size(%zu) must be a multiple of pattern_size(%zu)\n",
                        size, pattern_size);

  POCL_RETURN_ON_SUB_MISALIGN (buffer, command_queue);

  return CL_SUCCESS;
}

cl_int
pocl_validate_fill_image (cl_command_queue command_queue, cl_mem image,
                          const void *fill_color, const size_t *origin,
                          const size_t *region)
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (image)),
                          CL_INVALID_MEM_OBJECT);
  POCL_RETURN_ERROR_COND ((origin == NULL), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND ((region == NULL), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND ((fill_color == NULL), CL_INVALID_VALUE);

  POCL_RETURN_ERROR_ON (
      (command_queue->context != image->context), CL_INVALID_CONTEXT,
      "image and command_queue are not from the same context\n");

  POCL_RETURN_ERROR_ON ((!image->is_image), CL_INVALID_MEM_OBJECT,
                        "image argument is not an image\n");
  POCL_RETURN_ERROR_ON ((image->is_gl_texture), CL_INVALID_MEM_OBJECT,
                        "image is a GL texture\n");
  POCL_RETURN_ON_UNSUPPORTED_IMAGE (image, command_queue->device);

  return pocl_check_image_origin_region (image, origin, region);
}
