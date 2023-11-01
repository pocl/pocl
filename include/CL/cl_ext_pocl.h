/*******************************************************************************
 * Copyright (c) 2021 Tampere University
 *               2023 Pekka Jääskeläinen / Intel Finland Oy
 *
 * PoCL-specific proof-of-concept (draft) or finalized OpenCL extensions.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
 * KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
 * SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
 *    https://www.khronos.org/registry/
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 ******************************************************************************/

#include <CL/cl_ext.h>

#ifndef __CL_EXT_POCL_H
#define __CL_EXT_POCL_H

#ifdef __cplusplus
extern "C"
{
#endif

/* cl_pocl_content_size should be defined in CL/cl_ext.h; however,
 * if we PoCL is built against the system headers, it's possible
 * that they have an outdated version of CL/cl_ext.h.
 * In that case, add the extension here. */

#ifndef cl_pocl_content_size

extern CL_API_ENTRY cl_int CL_API_CALL
clSetContentSizeBufferPoCL(
    cl_mem    buffer,
    cl_mem    content_size_buffer) CL_API_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int
(CL_API_CALL *clSetContentSizeBufferPoCL_fn)(
    cl_mem    buffer,
    cl_mem    content_size_buffer) CL_API_SUFFIX__VERSION_1_2;

#endif

/* cl_pocl_pinned_buffers (experimental stage) */

#ifndef cl_pocl_pinned_buffers
#define cl_pocl_pinned_buffers 1
#define CL_POCL_PINNED_BUFFERS_EXTENSION_NAME "cl_exp_pinned_buffers"

/* TODO: We need also platform/runtime extension due to the new buffer
   creation flags. */

/* clCreateBuffer(): A new cl_mem_flag CL_MEM_PINNED:

   This flag specifies that the buffer must be persistently allocated
   in the device's physical memory for its lifetime. That is, the buffer's
   device address will remain the same and the space is reserved until
   the buffer is freed. The device-specific buffer content updates are
   still performed by implicit or explicit buffer migrations performed by
   the runtime or the client code. If any of the devices in the context
   does not support pinning, an error (TO DEFINE) is returned.
*/
#define CL_MEM_PINNED (1 << 31)

/* clGetMemObjectInfo(): A new query CL_MEM_DEVICE_PTR:

Returns a list of pinned device addresses for a buffer allocated
with CL_MEM_PINNED. If the buffer was not created with CL_MEM_PINNED,
returns CL_INVALID_MEM_OBJECT.
*/
#define CL_MEM_DEVICE_PTRS 0xff01

typedef struct _cl_mem_pinning
{
  cl_device_id device;
  void *address;
} cl_mem_pinning;

/* cl_pocl_pinned_buffers */
#endif

#ifdef __cplusplus
}
#endif

#endif /* __CL_EXT_POCL_H */
