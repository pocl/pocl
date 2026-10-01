/* Regression test: a USM/SVM allocation freed as soon as its last command is
   CL_COMPLETE must leave its range free for the next allocation.

   Copyright (c) 2026 pocl developers

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

#define SIZE 4096

int
main (void)
{
  cl_context context;
  cl_device_id device;
  cl_command_queue queue;
  cl_platform_id platform;
  static char host[SIZE];

  CHECK_CL_ERROR (
    poclu_get_any_device2 (&context, &device, &queue, &platform));

  clDeviceMemAllocINTEL_fn DeviceMemAlloc
    = clGetExtensionFunctionAddressForPlatform (platform,
                                                "clDeviceMemAllocINTEL");
  clMemFreeINTEL_fn MemFree
    = clGetExtensionFunctionAddressForPlatform (platform, "clMemFreeINTEL");
  clEnqueueMemcpyINTEL_fn EnqueueMemcpy
    = clGetExtensionFunctionAddressForPlatform (platform,
                                                "clEnqueueMemcpyINTEL");

  for (int svm = 0; svm < 2; ++svm)
    for (int i = 0; i < 10000; ++i)
      {
        cl_int err = CL_SUCCESS;
        void *ptr = svm ? clSVMAlloc (context, CL_MEM_READ_WRITE, SIZE, 0)
                        : DeviceMemAlloc (context, device, NULL, SIZE, 0,
                                          &err);
        CHECK_CL_ERROR (err);
        TEST_ASSERT (ptr != NULL);
        cl_event ev = NULL;
        CHECK_CL_ERROR (
          svm ? clEnqueueSVMMemcpy (queue, CL_FALSE, ptr, host, SIZE, 0, NULL,
                                    &ev)
              : EnqueueMemcpy (queue, CL_FALSE, ptr, host, SIZE, 0, NULL,
                               &ev));
        CHECK_CL_ERROR (clFlush (queue));
        /* Poll: a blocking wait usually returns after the command's cleanup,
           which hides the bug. */
        cl_int status;
        do
          CHECK_CL_ERROR (clGetEventInfo (ev,
                                          CL_EVENT_COMMAND_EXECUTION_STATUS,
                                          sizeof (status), &status, NULL));
        while (status > CL_COMPLETE);
        TEST_ASSERT (status == CL_COMPLETE);
        CHECK_CL_ERROR (clReleaseEvent (ev));
        if (svm)
          clSVMFree (context, ptr);
        else
          CHECK_CL_ERROR (MemFree (context, ptr));
      }

  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseContext (context));
  printf ("OK\n");
  return EXIT_SUCCESS;
}
