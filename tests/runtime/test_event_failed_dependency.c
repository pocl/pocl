/* Tests that a command whose wait-list dependency has already failed is itself
   terminated with a negative status, rather than executing.

   Per the OpenCL spec, when an event in a command's wait list is set to a
   negative (error) execution status, the command is terminated and the error is
   propagated to its event. The interesting case for pocl's internals is when the
   dependency is ALREADY in a failed state at the moment the waiting command is
   enqueued: the sync edge is never created (the failed event has already
   broadcast and will not do so again), so without an explicit guard the waiter
   would be left ready with an empty wait list and would run anyway. This test
   pins that behavior down by failing a user event *before* enqueuing a command
   that waits on it, then asserting the command's event ends negative.

   Copyright (c) 2026 Brice Videau / Argonne National Laboratory

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

#define BUF_SIZE 1024

/* Enqueue a buffer command that waits on the already-failed event `dep`, then
   assert that the command's own event is terminated with a negative status
   (rather than completing). `case_name` labels the scenario in diagnostics.
   Returns 0 on the expected behavior, 1 on the bug (command ran / completed). */
static int
check_failed_dependency_propagates (cl_context context, cl_command_queue queue,
                                    cl_mem buffer, cl_event dep,
                                    const char *case_name)
{
  cl_int err;
  cl_int pattern = 0;
  cl_event dependent = NULL;

  /* The dependency is already failed at this point. Enqueue a command waiting on
     it: per spec this command must be terminated, not executed. */
  err = clEnqueueFillBuffer (queue, buffer, &pattern, sizeof (pattern), 0,
                             BUF_SIZE, 1, &dep, &dependent);

  /* Some implementations may reject the enqueue outright with an error tied to
     the bad wait-list event; that is also a correct way to refuse to run it. */
  if (err != CL_SUCCESS)
    {
      printf ("[%s] enqueue refused with %d (acceptable)\n", case_name, err);
      return 0;
    }
  TEST_ASSERT (dependent != NULL);

  /* Drain. We must not hang here: that would itself be a failure of the
     dependency handling. */
  err = clWaitForEvents (1, &dependent);
  /* clWaitForEvents returns CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST when an
     event it waits on completed with a negative status -- that is expected. */
  if (err != CL_SUCCESS
      && err != CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
    {
      printf ("[%s] FAIL: unexpected clWaitForEvents error %d\n", case_name,
              err);
      clReleaseEvent (dependent);
      return 1;
    }

  cl_int exec_status = CL_QUEUED;
  err = clGetEventInfo (dependent, CL_EVENT_COMMAND_EXECUTION_STATUS,
                        sizeof (exec_status), &exec_status, NULL);
  CHECK_OPENCL_ERROR_IN ("clGetEventInfo execution status");

  clReleaseEvent (dependent);

  printf ("[%s] dependent command execution status: %d\n", case_name,
          exec_status);

  /* The bug: the command ran to completion despite a failed dependency. */
  if (exec_status >= 0)
    {
      printf ("[%s] FAIL: command with a failed dependency completed "
              "(status %d) instead of being terminated\n",
              case_name, exec_status);
      return 1;
    }

  return 0;
}

int
main (void)
{
  cl_int err;
  cl_platform_id platform = NULL;
  cl_context context = NULL;
  cl_device_id device = NULL;
  cl_command_queue helper_queue = NULL;
  cl_command_queue queue = NULL;
  int failures = 0;

  err = poclu_get_any_device2 (&context, &device, &helper_queue, &platform);
  CHECK_OPENCL_ERROR_IN ("clCreateContext");
  TEST_ASSERT (context);
  TEST_ASSERT (device);

  /* The failed-dependency handling differs between in-order and out-of-order
     queues: on an out-of-order queue a command whose dependency is already
     failed at enqueue gets no sync edge and would be left runnable. Use an
     out-of-order queue here so we exercise that path. The helper's in-order
     queue is released immediately; we only needed the context/device. */
  CHECK_CL_ERROR (clReleaseCommandQueue (helper_queue));
  queue = clCreateCommandQueue (
      context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
  if (err == CL_INVALID_QUEUE_PROPERTIES || err == CL_INVALID_VALUE)
    {
      printf ("Device does not support out-of-order queues, skipping.\n");
      printf ("SKIP\n");
      clReleaseContext (context);
      return 77;
    }
  CHECK_OPENCL_ERROR_IN ("clCreateCommandQueue (out-of-order)");
  TEST_ASSERT (queue);

  cl_mem buffer
      = clCreateBuffer (context, CL_MEM_READ_WRITE, BUF_SIZE, NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clCreateBuffer");

  /* Case 1: dependency ALREADY failed at the time the waiter is enqueued. This
     is the case the `pocl_create_event_sync` guard covers: no sync edge is
     created, so the flag is the only thing that stops the waiter from running. */
  {
    cl_event dep = clCreateUserEvent (context, &err);
    CHECK_OPENCL_ERROR_IN ("clCreateUserEvent (already-failed)");
    CHECK_CL_ERROR (clSetUserEventStatus (dep, -1));

    failures += check_failed_dependency_propagates (
        context, queue, buffer, dep, "already-failed-at-enqueue");

    CHECK_CL_ERROR (clReleaseEvent (dep));
  }

  /* Case 2: dependency enqueued first, fails LATER. Here the normal
     broadcast -> notify failure path runs; this guards against regressions in
     that path and exercises the `|| event->failed_dependency` notify term on
     drivers where the broadcast and notify can interleave. */
  {
    cl_event dep = clCreateUserEvent (context, &err);
    CHECK_OPENCL_ERROR_IN ("clCreateUserEvent (fail-later)");

    cl_int pattern = 0;
    cl_event dependent = NULL;
    err = clEnqueueFillBuffer (queue, buffer, &pattern, sizeof (pattern), 0,
                               BUF_SIZE, 1, &dep, &dependent);
    CHECK_OPENCL_ERROR_IN ("clEnqueueFillBuffer (fail-later)");
    TEST_ASSERT (dependent != NULL);

    /* Now fail the dependency: the dependent command must not run. */
    CHECK_CL_ERROR (clSetUserEventStatus (dep, -1));

    err = clWaitForEvents (1, &dependent);
    if (err != CL_SUCCESS
        && err != CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
      {
        printf ("[fail-later] FAIL: unexpected clWaitForEvents error %d\n",
                err);
        ++failures;
      }
    else
      {
        cl_int exec_status = CL_QUEUED;
        CHECK_CL_ERROR (clGetEventInfo (dependent,
                                        CL_EVENT_COMMAND_EXECUTION_STATUS,
                                        sizeof (exec_status), &exec_status,
                                        NULL));
        printf ("[fail-later] dependent command execution status: %d\n",
                exec_status);
        if (exec_status >= 0)
          {
            printf ("[fail-later] FAIL: command with a later-failed dependency "
                    "completed (status %d) instead of being terminated\n",
                    exec_status);
            ++failures;
          }
      }

    clReleaseEvent (dependent);
    CHECK_CL_ERROR (clReleaseEvent (dep));
  }

  CHECK_CL_ERROR (clReleaseMemObject (buffer));
  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseContext (context));
  CHECK_CL_ERROR (clUnloadPlatformCompiler (platform));

  if (failures)
    {
      printf ("FAIL: %d case(s) failed\n", failures);
      return EXIT_FAILURE;
    }

  printf ("OK\n");
  return EXIT_SUCCESS;
}
