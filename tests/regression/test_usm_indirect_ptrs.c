/* Regression test for the clSetKernelExecInfo(CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL)
   out-of-bounds read (reported downstream as JuliaGPU/OpenCL.jl#317).

   clSetKernelExecInfo passed the parameter's *byte size* to pocl_reset_indirect_ptrs(),
   which expects an *element count*, so it read 8x past the caller's pointer array. With a
   large indirect-USM-pointer list the over-read runs off the mapping and segfaults; with a
   small one it merely reads uninitialized memory. This test passes a large list and simply
   checks that the call returns without crashing.

   Copyright (c) 2026 Tim Besard

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

#include "pocl_opencl.h"

#include <CL/cl_ext.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_ERROR(err)                                                      \
  if (err != CL_SUCCESS)                                                      \
    {                                                                         \
      printf ("OpenCL error %d at %s:%d\n", err, __FILE__, __LINE__);         \
      return EXIT_FAILURE;                                                    \
    }

static const char *kernel_source = "__kernel void foo (void) { }";

int
main (void)
{
  cl_context context;
  cl_device_id device;
  cl_command_queue queue;
  cl_platform_id platform;

  cl_int err = poclu_get_any_device2 (&context, &device, &queue, &platform);
  CHECK_ERROR (err);

  /* Requires the USM extension. */
  char exts[4096] = { 0 };
  clGetDeviceInfo (device, CL_DEVICE_EXTENSIONS, sizeof (exts), exts, NULL);
  if (strstr (exts, "cl_intel_unified_shared_memory") == NULL)
    {
      printf ("SKIP: device does not support cl_intel_unified_shared_memory\n");
      return 77;
    }

  void *(*clDeviceMemAllocINTEL) (cl_context, cl_device_id,
                                  const cl_mem_properties_intel *, size_t,
                                  cl_uint, cl_int *)
    = clGetExtensionFunctionAddressForPlatform (platform,
                                                "clDeviceMemAllocINTEL");
  cl_int (*clMemFreeINTEL) (cl_context, void *)
    = clGetExtensionFunctionAddressForPlatform (platform, "clMemFreeINTEL");
  if (clDeviceMemAllocINTEL == NULL || clMemFreeINTEL == NULL)
    {
      printf ("SKIP: USM allocation entry points not found\n");
      return 77;
    }

  cl_program program
    = clCreateProgramWithSource (context, 1, &kernel_source, NULL, &err);
  CHECK_ERROR (err);
  err = clBuildProgram (program, 1, &device, NULL, NULL, NULL);
  CHECK_ERROR (err);
  cl_kernel kernel = clCreateKernel (program, "foo", &err);
  CHECK_ERROR (err);

  void *usm = clDeviceMemAllocINTEL (context, device, NULL, 64, 0, &err);
  CHECK_ERROR (err);

  /* A large indirect-pointer list: an 8x over-read (8 MiB -> 64 MiB) is sure to
     run off the array's mapping. The pointers are all valid USM pointers; only
     the length of the read decides whether the call faults. */
  const size_t n = 1 << 20;
  void **ptrs = (void **)malloc (n * sizeof (void *));
  for (size_t i = 0; i < n; ++i)
    ptrs[i] = usm;

  err = clSetKernelExecInfo (kernel, CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL,
                             n * sizeof (void *), ptrs);
  CHECK_ERROR (err);

  free (ptrs);
  err = clMemFreeINTEL (context, usm);
  CHECK_ERROR (err);
  clReleaseKernel (kernel);
  clReleaseProgram (program);
  clReleaseCommandQueue (queue);
  clReleaseContext (context);

  printf ("OK\n");
  return EXIT_SUCCESS;
}
