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

/* PoCL must not replace the host application's signal handlers, e.g. by
   making LLVM register its crash handlers, nor rely on them: an integer
   division by zero in a kernel must not raise SIGFPE, even when the kernel runs
   on the host application's thread (as with the basic driver). */

#include "pocl_opencl.h"

#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "poclu.h"

#define CHECK_ERROR(err)                                                      \
  if (err != CL_SUCCESS)                                                      \
    {                                                                         \
      printf ("OpenCL error %d at %s:%d\n", err, __FILE__, __LINE__);         \
      return EXIT_FAILURE;                                                    \
    }

static const int host_signals[] = { SIGSEGV, SIGBUS, SIGILL, SIGFPE };
#define NUM_HOST_SIGNALS (sizeof (host_signals) / sizeof (host_signals[0]))

static void
host_handler (int sig, siginfo_t *info, void *context)
{
  (void)sig;
  (void)info;
  (void)context;
  static const char msg[] = "host signal handler called\n";
  ssize_t res = write (STDOUT_FILENO, msg, sizeof (msg) - 1);
  (void)res;
  _exit (EXIT_FAILURE);
}

static void
install_host_handlers ()
{
  struct sigaction action;
  memset (&action, 0, sizeof (action));
  action.sa_sigaction = host_handler;
  action.sa_flags = SA_SIGINFO;
  sigemptyset (&action.sa_mask);
  for (unsigned i = 0; i < NUM_HOST_SIGNALS; ++i)
    sigaction (host_signals[i], &action, NULL);
}

static int
check_host_handlers (const char *after)
{
  for (unsigned i = 0; i < NUM_HOST_SIGNALS; ++i)
    {
      struct sigaction action;
      sigaction (host_signals[i], NULL, &action);
      if (action.sa_sigaction != host_handler)
        {
          printf ("handler for signal %d replaced after %s\n",
                  host_signals[i], after);
          return EXIT_FAILURE;
        }
    }
  return EXIT_SUCCESS;
}

int
main ()
{
  cl_context context;
  cl_device_id device;
  cl_command_queue queue;
  install_host_handlers ();

  cl_int err = poclu_get_any_device (&context, &device, &queue);
  CHECK_ERROR (err);
  if (check_host_handlers ("device initialization") != EXIT_SUCCESS)
    return EXIT_FAILURE;

  const char *source
      = "__kernel void k (__global const int *num, __global const int *den,\n"
        "                 __global int *out) {\n"
        "  size_t i = get_global_id (0);\n"
        "  out[i] = num[i] / den[i] + num[i] % den[i];\n"
        "}\n";
  cl_program program
      = clCreateProgramWithSource (context, 1, &source, NULL, &err);
  CHECK_ERROR (err);
  err = clBuildProgram (program, 1, &device, NULL, NULL, NULL);
  CHECK_ERROR (err);
  if (check_host_handlers ("building a program") != EXIT_SUCCESS)
    return EXIT_FAILURE;

  /* The results are unspecified; the kernel must merely not trap. */
  cl_int num[] = { 1, INT_MIN };
  cl_int den[] = { 0, -1 };
  cl_mem num_buf = clCreateBuffer (
      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof (num), num, &err);
  CHECK_ERROR (err);
  cl_mem den_buf = clCreateBuffer (
      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof (den), den, &err);
  CHECK_ERROR (err);
  cl_mem out_buf
      = clCreateBuffer (context, CL_MEM_WRITE_ONLY, sizeof (num), NULL, &err);
  CHECK_ERROR (err);
  cl_kernel kernel = clCreateKernel (program, "k", &err);
  CHECK_ERROR (err);
  CHECK_ERROR (clSetKernelArg (kernel, 0, sizeof (cl_mem), &num_buf));
  CHECK_ERROR (clSetKernelArg (kernel, 1, sizeof (cl_mem), &den_buf));
  CHECK_ERROR (clSetKernelArg (kernel, 2, sizeof (cl_mem), &out_buf));
  size_t global_size = 2;
  CHECK_ERROR (clEnqueueNDRangeKernel (queue, kernel, 1, NULL, &global_size,
                                       NULL, 0, NULL, NULL));
  CHECK_ERROR (clFinish (queue));
  if (check_host_handlers ("running a kernel") != EXIT_SUCCESS)
    return EXIT_FAILURE;

  CHECK_ERROR (clReleaseKernel (kernel));
  CHECK_ERROR (clReleaseMemObject (num_buf));
  CHECK_ERROR (clReleaseMemObject (den_buf));
  CHECK_ERROR (clReleaseMemObject (out_buf));

  CHECK_ERROR (clReleaseProgram (program));
  CHECK_ERROR (clReleaseCommandQueue (queue));
  CHECK_ERROR (clReleaseContext (context));

  printf ("OK\n");
  return EXIT_SUCCESS;
}
