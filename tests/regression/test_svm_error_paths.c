/* Regression test: failing SVM calls must return their error.

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
  cl_int err;

  CHECK_CL_ERROR (poclu_get_any_device (&context, &device, &queue));
  void *svm = clSVMAlloc (context, CL_MEM_READ_WRITE, SIZE, 0);
  TEST_ASSERT (svm != NULL);

  /* The SVM allocation's shadow buffer already holds this device address. */
  cl_mem_properties props[]
    = { CL_MEM_DEVICE_PRIVATE_ADDRESS_EXT, CL_TRUE, 0 };
  cl_mem buf = clCreateBufferWithProperties (
    context, props, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, SIZE, svm, &err);
  TEST_ASSERT ((buf != NULL) == (err == CL_SUCCESS));
  if (buf)
    CHECK_CL_ERROR (clReleaseMemObject (buf));

  /* A wait-list event from another context must fail the enqueue. */
  static char host[SIZE];
  cl_context other = clCreateContext (NULL, 1, &device, NULL, NULL, &err);
  CHECK_CL_ERROR (err);
  cl_event foreign = clCreateUserEvent (other, &err);
  CHECK_CL_ERROR (err);
  cl_event ev = NULL;
  TEST_ASSERT (clEnqueueSVMMemcpy (queue, CL_FALSE, svm, host, SIZE, 1,
                                   &foreign, &ev)
               == CL_INVALID_CONTEXT);
  CHECK_CL_ERROR (clReleaseEvent (foreign));
  CHECK_CL_ERROR (clReleaseContext (other));

  clSVMFree (context, svm);
  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseContext (context));
  printf ("OK\n");
  return EXIT_SUCCESS;
}
