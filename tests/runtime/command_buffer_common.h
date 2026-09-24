/* Shared cl_khr_command_buffer entry point loading for the runtime tests.

   Copyright (c) 2026 PoCL developers

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

#ifndef POCL_TESTS_COMMAND_BUFFER_COMMON_H
#define POCL_TESTS_COMMAND_BUFFER_COMMON_H

#include <stdio.h>

#include <CL/cl.h>
#include <CL/cl_ext.h>

/* Union of the entry points the command-buffer tests use; each test reads the
   subset it needs. */
struct cmdbuf_ext
{
  clCreateCommandBufferKHR_fn clCreateCommandBufferKHR;
  clCommandCopyBufferKHR_fn clCommandCopyBufferKHR;
  clCommandCopyBufferRectKHR_fn clCommandCopyBufferRectKHR;
  clCommandCopyBufferToImageKHR_fn clCommandCopyBufferToImageKHR;
  clCommandCopyImageKHR_fn clCommandCopyImageKHR;
  clCommandCopyImageToBufferKHR_fn clCommandCopyImageToBufferKHR;
  clCommandFillBufferKHR_fn clCommandFillBufferKHR;
  clCommandFillImageKHR_fn clCommandFillImageKHR;
  clCommandNDRangeKernelKHR_fn clCommandNDRangeKernelKHR;
  clCommandBarrierWithWaitListKHR_fn clCommandBarrierWithWaitListKHR;
  clFinalizeCommandBufferKHR_fn clFinalizeCommandBufferKHR;
  clEnqueueCommandBufferKHR_fn clEnqueueCommandBufferKHR;
  clReleaseCommandBufferKHR_fn clReleaseCommandBufferKHR;
  clGetCommandBufferInfoKHR_fn clGetCommandBufferInfoKHR;
  clRemapCommandBufferKHR_fn clRemapCommandBufferKHR;
};

/* Resolve the entry points for `platform`. Returns 0, or 77 (the ctest skip
   code) if the platform does not implement command buffers. */
static inline int
cmdbuf_load_ext (cl_platform_id platform, struct cmdbuf_ext *ext)
{
#define CMDBUF_GET(name)                                                      \
  ext->name = clGetExtensionFunctionAddressForPlatform (platform, #name)

  CMDBUF_GET (clCreateCommandBufferKHR);
  if (ext->clCreateCommandBufferKHR == NULL)
    {
      printf ("Command buffers are not supported, skipping test\n");
      return 77;
    }

  CMDBUF_GET (clCommandCopyBufferKHR);
  CMDBUF_GET (clCommandCopyBufferRectKHR);
  CMDBUF_GET (clCommandCopyBufferToImageKHR);
  CMDBUF_GET (clCommandCopyImageKHR);
  CMDBUF_GET (clCommandCopyImageToBufferKHR);
  CMDBUF_GET (clCommandFillBufferKHR);
  CMDBUF_GET (clCommandFillImageKHR);
  CMDBUF_GET (clCommandNDRangeKernelKHR);
  CMDBUF_GET (clCommandBarrierWithWaitListKHR);
  CMDBUF_GET (clFinalizeCommandBufferKHR);
  CMDBUF_GET (clEnqueueCommandBufferKHR);
  CMDBUF_GET (clReleaseCommandBufferKHR);
  CMDBUF_GET (clGetCommandBufferInfoKHR);
  CMDBUF_GET (clRemapCommandBufferKHR);
  return 0;

#undef CMDBUF_GET
}

#endif
