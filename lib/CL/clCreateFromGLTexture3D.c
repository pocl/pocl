/* OpenCL runtime library: cclCreateFromGLTexture3D()

   Copyright (c) 2021 Michal Babej / Tampere University

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

#include <assert.h>
#include "pocl_util.h"
#include "pocl_shared.h"


CL_API_ENTRY cl_mem CL_API_CALL
POname(clCreateFromGLTexture3D)(cl_context      context,
                        cl_mem_flags    flags,
                        cl_GLenum       target,
                        cl_GLint        miplevel,
                        cl_GLuint       texture,
                        cl_int *        errcode_ret) 
CL_API_SUFFIX__VERSION_1_0
{
  int errcode;
  POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (context)), CL_INVALID_CONTEXT);

#ifdef ENABLE_OPENGL_INTEROP

  cl_image_format format;
  format.image_channel_data_type = CL_UNSIGNED_INT32;
  format.image_channel_order = CL_RGBA;

  cl_image_desc desc;
  desc.image_array_size = 0;
  desc.image_width = 128;
  desc.image_height = 128;
  desc.image_depth = 2;
  desc.image_row_pitch = 0;
  desc.image_slice_pitch = 0;
  desc.image_type = CL_MEM_OBJECT_IMAGE3D;
  desc.buffer = NULL;
  desc.num_mip_levels = 0;
  desc.num_samples = 0;

  return pocl_create_image_internal (context, flags, &format, &desc, NULL,
                                     errcode_ret, target, miplevel, texture);

#else

  POCL_MSG_WARN (
      "CL-GL interop is only implemented by proxy device at this point\n");
  if (errcode_ret)
    *errcode_ret = CL_INVALID_OPERATION;
  return NULL;
#endif

ERROR:
  if (*errcode_ret)
    *errcode_ret = errcode;
  return NULL;
}
POsym(clCreateFromGLTexture3D)
