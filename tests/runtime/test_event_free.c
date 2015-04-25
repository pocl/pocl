/* Test that the associated event to a failing command isn't unduly freed

   Copyright (C) 2015 Giuseppe Bilotta <giuseppe.bilotta@gmail.com>

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include "poclu.h"
#include "pocl_tests.h"

int main(int argc, char **argv)
{
  cl_int err;
  cl_context ctx;
  cl_command_queue queue;
  cl_device_id did;

  poclu_get_any_device(&ctx, &did, &queue);
  TEST_ASSERT(ctx);
  TEST_ASSERT(did);
  TEST_ASSERT(queue);

  const size_t buf_size = sizeof(cl_int);
  cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, buf_size, NULL, &err);

  /* An invalid waiting list (e.g. a null event in it) should make
   * associated commands fail without segfaults and without touching any associated
   * event. Test that this is indeed the case.
   */

  cl_int *host_ptr = NULL;

  cl_event no_event = NULL, map_event = NULL;
  host_ptr = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_READ, 0, buf_size,
    1, &no_event, NULL, &err);
  TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);

  /* Test with map_event = NULL */
  host_ptr = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_READ, 0, buf_size,
    1, &no_event, &map_event, &err);
  TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);
  TEST_ASSERT(map_event == NULL); /* should not have been touched */

  /* Now do an actual mapping to test the unmapping */
  host_ptr = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_READ, 0, buf_size,
    0, NULL, NULL, &err);
  CHECK_OPENCL_ERROR_IN("map buffer");

  err = clEnqueueUnmapMemObject(queue, buf, host_ptr, 1, &no_event, NULL);
  TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);
  err = clEnqueueUnmapMemObject(queue, buf, host_ptr, 1, &no_event, &map_event);
  TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);
  TEST_ASSERT(map_event == NULL); /* should not have been touched */

  /* Actually unmap */
  err = clEnqueueUnmapMemObject(queue, buf, host_ptr, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("unmap buffer");
  host_ptr = NULL;

  /* Test with map_event != NULL but invalid */
  map_event = (cl_event)1;
  host_ptr = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_READ, 0, buf_size,
    1, &no_event, &map_event, &err);
  TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);
  TEST_ASSERT(map_event == (cl_event)1); /* should not have been touched */

  host_ptr = clEnqueueMapBuffer(queue, buf, CL_TRUE, CL_MAP_READ, 0, buf_size,
    0, NULL, NULL, &err);
  CHECK_OPENCL_ERROR_IN("map buffer");

  err = clEnqueueUnmapMemObject(queue, buf, host_ptr, 1, &no_event, &map_event);
  TEST_ASSERT(err == CL_INVALID_EVENT_WAIT_LIST);
  TEST_ASSERT(map_event == (cl_event)1); /* should not have been touched */

  err = clEnqueueUnmapMemObject(queue, buf, host_ptr, 0, NULL, NULL);
  CHECK_OPENCL_ERROR_IN("unmap buffer");
  host_ptr = NULL;

  return EXIT_SUCCESS;

}


