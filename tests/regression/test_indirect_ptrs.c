/* Copyright (c) 2026 PoCL Developers

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

/* The SVM and USM indirect pointer sets must not replace each other. */

#include "pocl_opencl.h"

/* pocl_cl.h includes config.h, which defines its own SRCDIR. */
#undef SRCDIR
#include "pocl_cl.h"

#include <CL/cl_ext.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utlist.h"

#define CHECK_ERROR(err)                                                      \
  if (err != CL_SUCCESS)                                                      \
    {                                                                         \
      printf ("OpenCL error %d at %s:%d\n", err, __FILE__, __LINE__);         \
      return EXIT_FAILURE;                                                    \
    }

static int
check_ptrs (cl_kernel kernel, void *svm, void *usm)
{
  unsigned expected_svm_count = svm != NULL;
  unsigned expected_usm_count = usm != NULL;
  unsigned actual_svm_count = 0;
  unsigned actual_usm_count = 0;
  pocl_ptr_list *node;
  DL_FOREACH (kernel->indirect_raw_ptrs, node)
    {
      if (node->ptr == svm)
        ++actual_svm_count;
      else if (node->ptr == usm)
        ++actual_usm_count;
      else
        return EXIT_FAILURE;
    }

  if (actual_svm_count != expected_svm_count
      || actual_usm_count != expected_usm_count)
    {
      printf ("Expected SVM/USM pointer counts %u/%u, got %u/%u\n",
              expected_svm_count, expected_usm_count, actual_svm_count,
              actual_usm_count);
      return EXIT_FAILURE;
    }
  return EXIT_SUCCESS;
}

int
main (void)
{
  cl_context context;
  cl_device_id device;
  cl_command_queue queue;
  cl_platform_id platform;

  cl_int err = poclu_get_any_device2 (&context, &device, &queue, &platform);
  CHECK_ERROR (err);

  char exts[4096] = { 0 };
  cl_device_svm_capabilities svm_caps = 0;
  err = clGetDeviceInfo (device, CL_DEVICE_EXTENSIONS, sizeof (exts), exts,
                         NULL);
  CHECK_ERROR (err);
  err = clGetDeviceInfo (device, CL_DEVICE_SVM_CAPABILITIES, sizeof (svm_caps),
                         &svm_caps, NULL);
  CHECK_ERROR (err);
  if (svm_caps == 0
      || strstr (exts, "cl_intel_unified_shared_memory") == NULL)
    {
      printf ("SKIP: device does not support both SVM and USM\n");
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

  static const char *source = "__kernel void foo (void) { }";
  cl_program program
    = clCreateProgramWithSource (context, 1, &source, NULL, &err);
  CHECK_ERROR (err);
  err = clBuildProgram (program, 1, &device, NULL, NULL, NULL);
  CHECK_ERROR (err);
  cl_kernel kernel = clCreateKernel (program, "foo", &err);
  CHECK_ERROR (err);

  void *svm = clSVMAlloc (context, CL_MEM_READ_WRITE, 64, 0);
  if (svm == NULL)
    return EXIT_FAILURE;
  void *usm = clDeviceMemAllocINTEL (context, device, NULL, 64, 0, &err);
  CHECK_ERROR (err);

  err = clSetKernelExecInfo (kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS,
                             sizeof (svm), &svm);
  CHECK_ERROR (err);
  err = clSetKernelExecInfo (kernel, CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL,
                             sizeof (usm), &usm);
  CHECK_ERROR (err);
  if (check_ptrs (kernel, svm, usm) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  err = clSetKernelExecInfo (kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS, 0, NULL);
  CHECK_ERROR (err);
  if (check_ptrs (kernel, NULL, usm) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  err = clSetKernelExecInfo (kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS,
                             sizeof (svm), &svm);
  CHECK_ERROR (err);
  err = clSetKernelExecInfo (kernel, CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL, 0,
                             NULL);
  CHECK_ERROR (err);
  if (check_ptrs (kernel, svm, NULL) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  err = clSetKernelExecInfo (kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS, 0, NULL);
  CHECK_ERROR (err);
  clSVMFree (context, svm);
  err = clMemFreeINTEL (context, usm);
  CHECK_ERROR (err);
  clReleaseKernel (kernel);
  clReleaseProgram (program);
  clReleaseCommandQueue (queue);
  clReleaseContext (context);

  printf ("OK\n");
  return EXIT_SUCCESS;
}
