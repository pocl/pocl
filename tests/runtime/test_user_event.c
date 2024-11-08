/* Tests clCreateUserEvent(), clSetUserEventStatus() and operating on the user event

   Copyright (c) 2016 Giuseppe Bilotta
   
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

#include "pocl_opencl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_SIZE(arr) (sizeof(arr)/sizeof(*arr))

int main()
{
  cl_int err;
  cl_event user_evt = NULL;
  unsigned i;

  // An user event can be set to either complete or a negative value, indicating error;
  // additionally, no objects involved in a command that waits on the user event should
  // be released before the event status is set; however, it should be possible to release
  // everything even if the status is set to something which is NOT CL_COMPLETE. So
  // try both CL_COMPLETE and a negative value
  cl_int status[] = {CL_INVALID_EVENT, CL_COMPLETE };

  // We also query for profiling info of the event, which according to the standard
  // should return CL_PROFILING_INFO_NOT_AVAILABLE
  cl_ulong queued, submitted, started, endtime;

  cl_platform_id platform = NULL;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue queue = NULL;

  CHECK_CL_ERROR (
    poclu_get_any_device2 (&context, &device, &queue, &platform));
  TEST_ASSERT (context);
  TEST_ASSERT (device);
  TEST_ASSERT (queue);

  for (i = 0; i < ARRAY_SIZE (status); ++i)
    {

      user_evt = clCreateUserEvent (context, &err);
      CHECK_OPENCL_ERROR_IN ("clCreateUserEvent");
      TEST_ASSERT (user_evt);

      CHECK_CL_ERROR (clSetUserEventStatus (user_evt, status[i]));

      err = clGetEventProfilingInfo (user_evt, CL_PROFILING_COMMAND_QUEUED,
                                     sizeof (queued), &queued, NULL);
      TEST_ASSERT (err == CL_PROFILING_INFO_NOT_AVAILABLE);
      err = clGetEventProfilingInfo (user_evt, CL_PROFILING_COMMAND_SUBMIT,
                                     sizeof (submitted), &submitted, NULL);
      TEST_ASSERT (err == CL_PROFILING_INFO_NOT_AVAILABLE);
      err = clGetEventProfilingInfo (user_evt, CL_PROFILING_COMMAND_START,
                                     sizeof (started), &started, NULL);
      TEST_ASSERT (err == CL_PROFILING_INFO_NOT_AVAILABLE);
      err = clGetEventProfilingInfo (user_evt, CL_PROFILING_COMMAND_END,
                                     sizeof (endtime), &endtime, NULL);
      TEST_ASSERT (err == CL_PROFILING_INFO_NOT_AVAILABLE);

      CHECK_CL_ERROR (clReleaseEvent (user_evt));
    }

  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseContext (context));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (platform));

  printf ("OK\n");
  return EXIT_SUCCESS;
}
