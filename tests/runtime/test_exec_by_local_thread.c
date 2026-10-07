/* Tests for cl_intel_exec_by_local_thread: commands enqueued on a queue with
   CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL are executed by the enqueuing
   thread, during the enqueue call.

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

#define _GNU_SOURCE

#include "pocl_opencl.h"

#include <fenv.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __linux__
#include <sys/resource.h>
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <pmmintrin.h>
#endif

#define CHECK(cond)                                                           \
  do                                                                          \
    {                                                                         \
      if (!(cond))                                                            \
        {                                                                     \
          printf ("FAIL: %s at %s:%d\n", #cond, __FILE__, __LINE__);          \
          exit (1);                                                           \
        }                                                                     \
    }                                                                         \
  while (0)

#define CHECK_CL(call)                                                        \
  do                                                                          \
    {                                                                         \
      cl_int _err = (call);                                                   \
      if (_err != CL_SUCCESS)                                                 \
        {                                                                     \
          printf ("FAIL: %s returned %d at %s:%d\n", #call, _err, __FILE__,   \
                  __LINE__);                                                  \
          exit (1);                                                           \
        }                                                                     \
    }                                                                         \
  while (0)

#define N 64

static const char *source
  = "kernel void set (global int *p, int v) { p[get_global_id (0)] = v; }\n"
    "kernel void add_one (global int *p) { p[get_global_id (0)] += 1; }\n"
    "kernel void inc_at (global int *p, int i) { p[i] += 1; }\n"
    /* Mode 0 says it started and waits for mode 1 to release it. */
    "kernel void spin (global volatile int *flags, int mode) {\n"
    "  if (mode == 0) {\n"
    "    flags[1] = 1;\n"
    "    while (flags[0] == 0)\n"
    "      ;\n"
    "  } else\n"
    "    flags[0] = 1;\n"
    "}\n"
    "kernel void div (global int *p, int a, int b) { p[0] = a / b; }\n"
    "kernel void fp (global float *out, global const float *in) {\n"
    "  out[0] = in[0] + in[1];\n"
    "  out[1] = in[2] * in[3];\n"
    "  out[2] = in[4] / in[5];\n"
    "}\n";

/* Needs more automatic local memory than the device has, which fails the
   command when it runs. */
static const char *big_local_source
  = "kernel void big_local (global int *p) {\n"
    "  local int big[LOCAL_INTS];\n"
    "  big[get_local_id (0)] = 1;\n"
    "  barrier (CLK_LOCAL_MEM_FENCE);\n"
    "  p[0] = big[0];\n"
    "}\n";

static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_program program;
static int have_native_kernels;

static cl_command_queue
create_queue (cl_device_id dev, cl_command_queue_properties props)
{
  cl_int err;
  cl_queue_properties properties[]
    = { CL_QUEUE_PROPERTIES, props | CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL,
        0 };
  cl_command_queue queue
    = clCreateCommandQueueWithProperties (context, dev, properties, &err);
  CHECK_CL (err);
  return queue;
}

static cl_kernel
create_kernel (const char *name)
{
  cl_int err;
  cl_kernel kernel = clCreateKernel (program, name, &err);
  CHECK_CL (err);
  return kernel;
}

static cl_mem
create_buffer (cl_mem_flags flags, size_t size, void *host_ptr)
{
  cl_int err;
  cl_mem buf = clCreateBuffer (context, flags, size, host_ptr, &err);
  CHECK_CL (err);
  return buf;
}

static cl_int
event_status (cl_event event)
{
  cl_int status;
  CHECK_CL (clGetEventInfo (event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                            sizeof (status), &status, NULL));
  return status;
}

static void
read_ints (cl_mem buf, int *out, size_t n)
{
  cl_command_queue queue = create_queue (device, 0);
  CHECK_CL (clEnqueueReadBuffer (queue, buf, CL_FALSE, 0, n * sizeof (int),
                                 out, 0, NULL, NULL));
  CHECK_CL (clReleaseCommandQueue (queue));
}

static void
enqueue_set (cl_command_queue queue,
             cl_kernel kernel,
             cl_mem buf,
             int v,
             size_t n,
             cl_uint num_deps,
             const cl_event *deps,
             cl_event *event)
{
  CHECK_CL (clSetKernelArg (kernel, 0, sizeof (cl_mem), &buf));
  CHECK_CL (clSetKernelArg (kernel, 1, sizeof (int), &v));
  CHECK_CL (clEnqueueNDRangeKernel (queue, kernel, 1, NULL, &n, NULL, num_deps,
                                    deps, event));
}

/* A kernel and a read complete before their enqueue returns. */
static void
test_ndrange (void)
{
  cl_command_queue queue = create_queue (device, 0);
  cl_kernel kernel = create_kernel ("set");
  cl_mem buf = create_buffer (CL_MEM_READ_WRITE, N * sizeof (int), NULL);

  cl_event event;
  enqueue_set (queue, kernel, buf, 42, N, 0, NULL, &event);
  CHECK (event_status (event) == CL_COMPLETE);
  CHECK_CL (clReleaseEvent (event));

  int out[N] = { 0 };
  CHECK_CL (clEnqueueReadBuffer (queue, buf, CL_FALSE, 0, sizeof (out), out, 0,
                                 NULL, &event));
  CHECK (event_status (event) == CL_COMPLETE);
  CHECK_CL (clReleaseEvent (event));
  for (int i = 0; i < N; ++i)
    CHECK (out[i] == 42);

  /* Also without an output event, and on an out-of-order queue. */
  enqueue_set (queue, kernel, buf, 43, N, 0, NULL, NULL);
  cl_command_queue ooo_queue
    = create_queue (device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
  enqueue_set (ooo_queue, kernel, buf, 44, 1, 0, NULL, NULL);
  read_ints (buf, out, N);
  CHECK (out[0] == 44 && out[1] == 43);

  CHECK_CL (clFinish (queue));
  CHECK_CL (clFinish (ooo_queue));
  CHECK_CL (clReleaseCommandQueue (ooo_queue));
  CHECK_CL (clReleaseMemObject (buf));
  CHECK_CL (clReleaseKernel (kernel));
  CHECK_CL (clReleaseCommandQueue (queue));
  printf ("ndrange: OK\n");
}

static void
test_profiling (void)
{
  cl_command_queue queue = create_queue (device, CL_QUEUE_PROFILING_ENABLE);
  cl_kernel kernel = create_kernel ("set");
  cl_mem buf = create_buffer (CL_MEM_READ_WRITE, N * sizeof (int), NULL);

  cl_event event;
  enqueue_set (queue, kernel, buf, 1, N, 0, NULL, &event);
  cl_ulong t[4];
  const cl_profiling_info info[4]
    = { CL_PROFILING_COMMAND_QUEUED, CL_PROFILING_COMMAND_SUBMIT,
        CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END };
  for (int i = 0; i < 4; ++i)
    CHECK_CL (clGetEventProfilingInfo (event, info[i], sizeof (cl_ulong),
                                       &t[i], NULL));
  CHECK (t[0] != 0);
  CHECK (t[0] <= t[1] && t[1] <= t[2] && t[2] <= t[3]);

  CHECK_CL (clReleaseEvent (event));
  CHECK_CL (clReleaseMemObject (buf));
  CHECK_CL (clReleaseKernel (kernel));
  CHECK_CL (clReleaseCommandQueue (queue));
  printf ("profiling: OK\n");
}

typedef struct
{
  pthread_t *thread;
  int *ran;
} native_args;

static void
record_thread (void *p)
{
  native_args *args = (native_args *)p;
  *args->thread = pthread_self ();
  *args->ran = 1;
}

static void
enqueue_record_thread (cl_command_queue queue,
                       pthread_t *thread,
                       int *ran,
                       cl_uint num_deps,
                       const cl_event *deps,
                       cl_event *event)
{
  native_args args = { thread, ran };
  CHECK_CL (clEnqueueNativeKernel (queue, record_thread, &args, sizeof (args),
                                   0, NULL, NULL, num_deps, deps, event));
}

/* The command runs on the thread that enqueues it. */
static void
test_calling_thread (cl_device_id dev, const char *what)
{
  cl_command_queue queue = create_queue (dev, 0);
  pthread_t thread;
  int ran = 0;
  enqueue_record_thread (queue, &thread, &ran, 0, NULL, NULL);
  CHECK (ran);
  CHECK (pthread_equal (thread, pthread_self ()));
  CHECK_CL (clReleaseCommandQueue (queue));
  printf ("calling thread (%s): OK\n", what);
}

/* Enqueues a command on another thread, to see the enqueue block. */
typedef struct
{
  cl_command_queue queue;
  cl_kernel kernel; /* add_one, or a native kernel if NULL */
  cl_mem buf;
  cl_event dep;
  cl_event event;
  pthread_t ran_on;
  int ran;
  volatile int returned;
} blocked_enqueue;

static void *
blocked_enqueue_thread (void *p)
{
  blocked_enqueue *b = (blocked_enqueue *)p;
  if (b->kernel)
    {
      size_t n = 1;
      CHECK_CL (clSetKernelArg (b->kernel, 0, sizeof (cl_mem), &b->buf));
      CHECK_CL (clEnqueueNDRangeKernel (b->queue, b->kernel, 1, NULL, &n, NULL,
                                        1, &b->dep, &b->event));
    }
  else
    enqueue_record_thread (b->queue, &b->ran_on, &b->ran, 1, &b->dep,
                           &b->event);
  __atomic_store_n (&b->returned, 1, __ATOMIC_SEQ_CST);
  return NULL;
}

static void
start_blocked_enqueue (blocked_enqueue *b, pthread_t *thread)
{
  CHECK (pthread_create (thread, NULL, blocked_enqueue_thread, b) == 0);
  /* The enqueue can't return before the dependency completes; give it the
     chance to do so wrongly. */
  usleep (100 * 1000);
  CHECK (!__atomic_load_n (&b->returned, __ATOMIC_SEQ_CST));
}

/* An enqueue blocks until its dependencies are complete, and then runs the
   command on its own thread, not on the one completing the dependency. */
static void
test_dependencies (void)
{
  cl_int err;
  cl_command_queue queue = create_queue (device, 0);
  pthread_t thread;

  /* A user event. */
  if (have_native_kernels)
    {
      blocked_enqueue b = { queue };
      b.dep = clCreateUserEvent (context, &err);
      CHECK_CL (err);
      start_blocked_enqueue (&b, &thread);
      CHECK (!b.ran);
      CHECK_CL (clSetUserEventStatus (b.dep, CL_COMPLETE));
      CHECK (pthread_join (thread, NULL) == 0);
      CHECK (b.ran);
      CHECK (pthread_equal (b.ran_on, thread));
      CHECK (event_status (b.event) == CL_COMPLETE);
      CHECK_CL (clReleaseEvent (b.event));
      CHECK_CL (clReleaseEvent (b.dep));
    }

  /* A command on an ordinary queue, waiting for a user event itself. */
  cl_command_queue other_queue
    = clCreateCommandQueue (context, device, 0, &err);
  CHECK_CL (err);
  cl_kernel set = create_kernel ("set");
  cl_mem buf = create_buffer (CL_MEM_READ_WRITE, sizeof (int), NULL);
  cl_event gate = clCreateUserEvent (context, &err);
  CHECK_CL (err);
  blocked_enqueue b = { queue, create_kernel ("add_one"), buf };
  enqueue_set (other_queue, set, buf, 41, 1, 1, &gate, &b.dep);
  start_blocked_enqueue (&b, &thread);
  CHECK_CL (clSetUserEventStatus (gate, CL_COMPLETE));
  CHECK (pthread_join (thread, NULL) == 0);
  CHECK (event_status (b.event) == CL_COMPLETE);
  int out;
  read_ints (buf, &out, 1);
  CHECK (out == 42);
  CHECK_CL (clReleaseEvent (b.event));
  CHECK_CL (clReleaseEvent (b.dep));
  CHECK_CL (clReleaseEvent (gate));

  /* A failed dependency fails the command without running it. */
  blocked_enqueue f = { queue, b.kernel, buf };
  f.dep = clCreateUserEvent (context, &err);
  CHECK_CL (err);
  start_blocked_enqueue (&f, &thread);
  CHECK_CL (clSetUserEventStatus (f.dep, -1));
  CHECK (pthread_join (thread, NULL) == 0);
  CHECK (event_status (f.event) < 0);
  read_ints (buf, &out, 1);
  CHECK (out == 42);
  CHECK_CL (clReleaseEvent (f.event));
  CHECK_CL (clReleaseEvent (f.dep));

  /* So does one that failed before the command was enqueued. */
  cl_event failed = clCreateUserEvent (context, &err);
  CHECK_CL (err);
  CHECK_CL (clSetUserEventStatus (failed, -1));
  size_t n = 1;
  cl_event event;
  CHECK_CL (clSetKernelArg (b.kernel, 0, sizeof (cl_mem), &buf));
  CHECK_CL (clEnqueueNDRangeKernel (queue, b.kernel, 1, NULL, &n, NULL, 1,
                                    &failed, &event));
  CHECK (event_status (event) < 0);
  read_ints (buf, &out, 1);
  CHECK (out == 42);
  CHECK_CL (clReleaseEvent (event));
  CHECK_CL (clReleaseEvent (failed));

  CHECK_CL (clReleaseKernel (b.kernel));
  CHECK_CL (clReleaseKernel (set));
  CHECK_CL (clReleaseMemObject (buf));
  CHECK_CL (clReleaseCommandQueue (other_queue));
  CHECK_CL (clReleaseCommandQueue (queue));
  printf ("dependencies: OK\n");
}

/* Commands other than kernels complete before their enqueue returns too. */
static void
test_commands (void)
{
  cl_command_queue queue = create_queue (device, 0);
  cl_mem a = create_buffer (CL_MEM_READ_WRITE, N * sizeof (int), NULL);
  cl_mem b = create_buffer (CL_MEM_READ_WRITE, N * sizeof (int), NULL);
  int in[N], out[N];
  for (int i = 0; i < N; ++i)
    in[i] = i;
  cl_event events[7];

  CHECK_CL (clEnqueueWriteBuffer (queue, a, CL_FALSE, 0, sizeof (in), in, 0,
                                  NULL, &events[0]));
  memset (in, 0, sizeof (in)); /* the write must have copied it already */
  CHECK_CL (
    clEnqueueCopyBuffer (queue, a, b, 0, 0, sizeof (in), 0, NULL, &events[1]));
  int pattern = 7;
  CHECK_CL (clEnqueueFillBuffer (queue, a, &pattern, sizeof (pattern), 0,
                                 sizeof (in), 0, NULL, &events[2]));
  CHECK_CL (clEnqueueMarkerWithWaitList (queue, 0, NULL, &events[3]));
  CHECK_CL (clEnqueueBarrierWithWaitList (queue, 0, NULL, &events[4]));
  CHECK_CL (clEnqueueReadBuffer (queue, b, CL_FALSE, 0, sizeof (out), out, 0,
                                 NULL, &events[5]));
  for (int i = 0; i < N; ++i)
    CHECK (out[i] == i);

  cl_int err;
  int *mapped = clEnqueueMapBuffer (queue, a, CL_FALSE, CL_MAP_READ, 0,
                                    sizeof (in), 0, NULL, &events[6], &err);
  CHECK_CL (err);
  for (int i = 0; i < N; ++i)
    CHECK (mapped[i] == 7);
  CHECK_CL (clEnqueueUnmapMemObject (queue, a, mapped, 0, NULL, NULL));

  for (int i = 0; i < 7; ++i)
    {
      CHECK (event_status (events[i]) == CL_COMPLETE);
      CHECK_CL (clReleaseEvent (events[i]));
    }

  CHECK_CL (clReleaseMemObject (a));
  CHECK_CL (clReleaseMemObject (b));
  CHECK_CL (clReleaseCommandQueue (queue));
  printf ("commands: OK\n");
}

typedef struct
{
  cl_command_queue queue;
  cl_mem buf;
  volatile int returned;
} blocking_read_args;

static void *
blocking_read_thread (void *p)
{
  blocking_read_args *args = (blocking_read_args *)p;
  int out;
  CHECK_CL (clEnqueueReadBuffer (args->queue, args->buf, CL_TRUE, 0,
                                 sizeof (int), &out, 0, NULL, NULL));
  __atomic_store_n (&args->returned, 1, __ATOMIC_SEQ_CST);
  return NULL;
}

/* A blocking call returns once its own command is complete, like the
   non-blocking one, even while another thread's command on the same queue
   waits for something only the caller will do after that. */
static void
test_blocking_calls (void)
{
  cl_int err;
  cl_command_queue queue
    = create_queue (device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
  cl_mem buf = create_buffer (CL_MEM_READ_WRITE, sizeof (int), NULL);
  blocked_enqueue b = { queue, create_kernel ("add_one"), buf };
  b.dep = clCreateUserEvent (context, &err);
  CHECK_CL (err);
  pthread_t blocked_thread, reader_thread;
  start_blocked_enqueue (&b, &blocked_thread);

  blocking_read_args args = { queue, buf, 0 };
  CHECK (pthread_create (&reader_thread, NULL, blocking_read_thread, &args)
         == 0);
  for (int i = 0;
       i < 5000 && !__atomic_load_n (&args.returned, __ATOMIC_SEQ_CST); ++i)
    usleep (1000);
  int returned = __atomic_load_n (&args.returned, __ATOMIC_SEQ_CST);

  CHECK_CL (clSetUserEventStatus (b.dep, CL_COMPLETE));
  CHECK (pthread_join (blocked_thread, NULL) == 0);
  CHECK (pthread_join (reader_thread, NULL) == 0);
  CHECK (returned);

  CHECK_CL (clReleaseEvent (b.event));
  CHECK_CL (clReleaseEvent (b.dep));
  CHECK_CL (clReleaseKernel (b.kernel));
  CHECK_CL (clReleaseMemObject (buf));
  CHECK_CL (clReleaseCommandQueue (queue));
  printf ("blocking calls: OK\n");
}

/* Buffer contents that have to be migrated to the device first. */
static void
test_migration (void)
{
  cl_command_queue queue = create_queue (device, 0);
  cl_kernel kernel = create_kernel ("add_one");
  int in[N];
  for (int i = 0; i < N; ++i)
    in[i] = i;
  cl_mem buf = create_buffer (CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof (in), in);

  size_t n = N;
  cl_event event;
  CHECK_CL (clSetKernelArg (kernel, 0, sizeof (cl_mem), &buf));
  CHECK_CL (clEnqueueNDRangeKernel (queue, kernel, 1, NULL, &n, NULL, 0, NULL,
                                    &event));
  CHECK (event_status (event) == CL_COMPLETE);
  int out[N];
  read_ints (buf, out, N);
  for (int i = 0; i < N; ++i)
    CHECK (out[i] == i + 1);

  CHECK_CL (clReleaseEvent (event));
  CHECK_CL (clReleaseMemObject (buf));
  CHECK_CL (clReleaseKernel (kernel));
  CHECK_CL (clReleaseCommandQueue (queue));
  printf ("migration: OK\n");
}

/* A kernel that fails completes with an error status. */
static void
test_failure (void)
{
  cl_int err;
  cl_ulong local_mem;
  CHECK_CL (clGetDeviceInfo (device, CL_DEVICE_LOCAL_MEM_SIZE,
                             sizeof (local_mem), &local_mem, NULL));
  /* The CPU drivers' local memory has some slack for aligning arguments, so
     go well beyond the device's size. */
  char options[64];
  snprintf (options, sizeof (options), "-DLOCAL_INTS=%lu",
            (unsigned long)(local_mem / sizeof (int) + (1 << 20)));
  cl_program prog
    = clCreateProgramWithSource (context, 1, &big_local_source, NULL, &err);
  CHECK_CL (err);
  CHECK_CL (clBuildProgram (prog, 0, NULL, options, NULL, NULL));
  cl_kernel kernel = clCreateKernel (prog, "big_local", &err);
  CHECK_CL (err);

  cl_command_queue queue = create_queue (device, 0);
  cl_mem buf = create_buffer (CL_MEM_READ_WRITE, sizeof (int), NULL);
  CHECK_CL (clSetKernelArg (kernel, 0, sizeof (cl_mem), &buf));
  size_t n = 1;
  cl_event event;
  CHECK_CL (clEnqueueNDRangeKernel (queue, kernel, 1, NULL, &n, NULL, 0, NULL,
                                    &event));
  CHECK (event_status (event) < 0);

  CHECK_CL (clReleaseEvent (event));
  CHECK_CL (clReleaseMemObject (buf));
  CHECK_CL (clReleaseCommandQueue (queue));
  CHECK_CL (clReleaseKernel (kernel));
  CHECK_CL (clReleaseProgram (prog));
  printf ("failure: OK\n");
}

/* While a thread runs a kernel, other threads can enqueue, with the same
   cl_kernel: here, the one that lets the first kernel finish. If running the
   kernel held a lock that enqueueing needs, this would hang. */
typedef struct
{
  cl_kernel kernel;
  cl_mem flags;
  int mode;
} spin_args;

static void *
spin_thread (void *p)
{
  spin_args *args = (spin_args *)p;
  cl_command_queue queue = create_queue (device, 0);
  size_t n = 1;
  CHECK_CL (clSetKernelArg (args->kernel, 0, sizeof (cl_mem), &args->flags));
  CHECK_CL (clSetKernelArg (args->kernel, 1, sizeof (int), &args->mode));
  CHECK_CL (clEnqueueNDRangeKernel (queue, args->kernel, 1, NULL, &n, NULL, 0,
                                    NULL, NULL));
  CHECK_CL (clReleaseCommandQueue (queue));
  return NULL;
}

typedef struct
{
  cl_command_queue queue;
  cl_kernel kernel;
  cl_mem buf;
  int index;
  int iterations;
} inc_args;

static void *
inc_thread (void *p)
{
  inc_args *args = (inc_args *)p;
  size_t n = 1;
  for (int i = 0; i < args->iterations; ++i)
    {
      cl_event event;
      CHECK_CL (clEnqueueNDRangeKernel (args->queue, args->kernel, 1, NULL, &n,
                                        NULL, 0, NULL, &event));
      CHECK (event_status (event) == CL_COMPLETE);
      CHECK_CL (clReleaseEvent (event));
    }
  return NULL;
}

#define THREADS 4
#define ITERATIONS 200

static void
test_threads (void)
{
  /* The flags are read on the host while the kernel runs, so they have to be
     the host memory itself. */
  static int flags[2];
  cl_mem flags_buf = create_buffer (CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                    sizeof (flags), flags);
  cl_kernel spin = create_kernel ("spin");
  spin_args waiter = { spin, flags_buf, 0 }, releaser = { spin, flags_buf, 1 };
  pthread_t waiter_thread, releaser_thread;
  CHECK (pthread_create (&waiter_thread, NULL, spin_thread, &waiter) == 0);
  /* The waiter's arguments have been copied once its kernel runs. */
  while (__atomic_load_n (&flags[1], __ATOMIC_SEQ_CST) == 0)
    usleep (1000);
  CHECK (pthread_create (&releaser_thread, NULL, spin_thread, &releaser) == 0);
  CHECK (pthread_join (releaser_thread, NULL) == 0);
  CHECK (pthread_join (waiter_thread, NULL) == 0);
  CHECK_CL (clReleaseKernel (spin));
  CHECK_CL (clReleaseMemObject (flags_buf));

  /* Several threads with their own queues, and several threads sharing an
     in-order queue, which mustn't run their commands at the same time: those
     all increment the same element. */
  cl_mem buf
    = create_buffer (CL_MEM_READ_WRITE, 2 * THREADS * sizeof (int), NULL);
  int zero = 0;
  cl_command_queue queue = create_queue (device, 0);
  CHECK_CL (clEnqueueFillBuffer (queue, buf, &zero, sizeof (zero), 0,
                                 2 * THREADS * sizeof (int), 0, NULL, NULL));
  inc_args args[2 * THREADS];
  pthread_t threads[2 * THREADS];
  for (int i = 0; i < 2 * THREADS; ++i)
    {
      args[i].queue = i < THREADS ? create_queue (device, 0) : queue;
      args[i].kernel = create_kernel ("inc_at");
      args[i].buf = buf;
      args[i].index = i < THREADS ? i : THREADS;
      args[i].iterations = ITERATIONS;
      CHECK_CL (clSetKernelArg (args[i].kernel, 0, sizeof (cl_mem), &buf));
      CHECK_CL (
        clSetKernelArg (args[i].kernel, 1, sizeof (int), &args[i].index));
    }
  for (int i = 0; i < 2 * THREADS; ++i)
    CHECK (pthread_create (&threads[i], NULL, inc_thread, &args[i]) == 0);
  for (int i = 0; i < 2 * THREADS; ++i)
    CHECK (pthread_join (threads[i], NULL) == 0);
  int out[2 * THREADS];
  read_ints (buf, out, 2 * THREADS);
  for (int i = 0; i < THREADS; ++i)
    CHECK (out[i] == ITERATIONS);
  CHECK (out[THREADS] == THREADS * ITERATIONS);
  for (int i = 0; i < 2 * THREADS; ++i)
    {
      CHECK_CL (clReleaseKernel (args[i].kernel));
      if (i < THREADS)
        CHECK_CL (clReleaseCommandQueue (args[i].queue));
    }
  CHECK_CL (clReleaseCommandQueue (queue));
  CHECK_CL (clReleaseMemObject (buf));
  printf ("threads: OK\n");
}

/* A native kernel can enqueue on another such queue. */
typedef struct
{
  cl_command_queue queue;
  cl_kernel kernel;
  cl_int status;
} nested_args;

static void
nested_enqueue (void *p)
{
  nested_args *args = *(nested_args **)p;
  size_t n = 1;
  cl_event event;
  args->status = clEnqueueNDRangeKernel (args->queue, args->kernel, 1, NULL,
                                         &n, NULL, 0, NULL, &event);
  if (args->status == CL_SUCCESS)
    {
      args->status = event_status (event);
      clReleaseEvent (event);
    }
}

static void
test_nesting (void)
{
  cl_command_queue outer = create_queue (device, 0);
  cl_command_queue inner = create_queue (device, 0);
  cl_kernel kernel = create_kernel ("set");
  cl_mem buf = create_buffer (CL_MEM_READ_WRITE, sizeof (int), NULL);
  int v = 5;
  CHECK_CL (clSetKernelArg (kernel, 0, sizeof (cl_mem), &buf));
  CHECK_CL (clSetKernelArg (kernel, 1, sizeof (int), &v));

  /* The native kernel gets a copy of the arguments, so the status comes back
     through memory it points to. */
  nested_args *args = calloc (1, sizeof (nested_args));
  args->queue = inner;
  args->kernel = kernel;
  args->status = 1;
  nested_args *ptr = args;
  CHECK_CL (clEnqueueNativeKernel (outer, nested_enqueue, &ptr, sizeof (ptr),
                                   0, NULL, NULL, 0, NULL, NULL));
  CHECK (args->status == CL_COMPLETE);
  int out;
  read_ints (buf, &out, 1);
  CHECK (out == 5);

  free (args);
  CHECK_CL (clReleaseMemObject (buf));
  CHECK_CL (clReleaseKernel (kernel));
  CHECK_CL (clReleaseCommandQueue (inner));
  CHECK_CL (clReleaseCommandQueue (outer));
  printf ("nesting: OK\n");
}

static uint32_t
float_bits (float f)
{
  uint32_t u;
  memcpy (&u, &f, sizeof (u));
  return u;
}

/* The kernel runs in the floating-point environment it gets on the worker
   threads, and the caller's environment is left as it was. */
static void
test_fp_environment (void)
{
  cl_command_queue queue = create_queue (device, 0);
  cl_kernel kernel = create_kernel ("fp");
  /* 1 + 2^-30 rounds to 1 to nearest, and up to the next float upwards.
     FLT_MIN / 2 is denormal. 1 / 0 raises the division-by-zero exception. */
  const float in[6] = { 1.0f, 0x1p-30f, 0x1p-126f, 0.5f, 1.0f, 0.0f };
  cl_mem in_buf = create_buffer (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof (in), (void *)in);
  cl_mem out_buf = create_buffer (CL_MEM_WRITE_ONLY, 3 * sizeof (float), NULL);
  CHECK_CL (clSetKernelArg (kernel, 0, sizeof (cl_mem), &out_buf));
  CHECK_CL (clSetKernelArg (kernel, 1, sizeof (cl_mem), &in_buf));

  fenv_t saved;
  fegetenv (&saved);
  CHECK (fesetround (FE_UPWARD) == 0);
  feclearexcept (FE_ALL_EXCEPT);
  feraiseexcept (FE_INEXACT);
#if defined(__x86_64__) || defined(__i386__)
  _MM_SET_FLUSH_ZERO_MODE (_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE (_MM_DENORMALS_ZERO_ON);
#endif
#ifdef __GLIBC__
  /* Not every target can trap; then this part isn't tested. */
  int traps = feenableexcept (FE_DIVBYZERO) != -1;
#endif

  size_t n = 1;
  cl_int err
    = clEnqueueNDRangeKernel (queue, kernel, 1, NULL, &n, NULL, 0, NULL, NULL);

  int rounding = fegetround ();
  /* The kernel raises these, and the caller had raised FE_INEXACT. */
  int raised = fetestexcept (FE_DIVBYZERO | FE_UNDERFLOW | FE_INEXACT);
#ifdef __GLIBC__
  int enabled = fegetexcept ();
  fedisableexcept (FE_ALL_EXCEPT);
#endif
#if defined(__x86_64__) || defined(__i386__)
  unsigned csr = _mm_getcsr ();
#endif
  fesetenv (&saved);

  CHECK_CL (err);
  CHECK (rounding == FE_UPWARD);
  CHECK (raised == FE_INEXACT);
#ifdef __GLIBC__
  CHECK (!traps || enabled == FE_DIVBYZERO);
#endif
#if defined(__x86_64__) || defined(__i386__)
  CHECK ((csr & _MM_FLUSH_ZERO_MASK) == _MM_FLUSH_ZERO_ON);
  CHECK ((csr & _MM_DENORMALS_ZERO_MASK) == _MM_DENORMALS_ZERO_ON);
#endif

  float out[3];
  CHECK_CL (clEnqueueReadBuffer (queue, out_buf, CL_TRUE, 0, sizeof (out), out,
                                 0, NULL, NULL));
  CHECK (float_bits (out[0]) == float_bits (1.0f));
  CHECK (float_bits (out[1]) == float_bits (0x1p-127f));
  CHECK (float_bits (out[2]) == float_bits (1.0f / 0.0f));

  CHECK_CL (clReleaseMemObject (in_buf));
  CHECK_CL (clReleaseMemObject (out_buf));
  CHECK_CL (clReleaseKernel (kernel));
  CHECK_CL (clReleaseCommandQueue (queue));
  printf ("fp environment: OK\n");
}

/* Integer division by zero is undefined, but doesn't bring down the
   application either. */
static void
test_integer_division (void)
{
  cl_command_queue queue = create_queue (device, 0);
  cl_kernel kernel = create_kernel ("div");
  cl_mem buf = create_buffer (CL_MEM_READ_WRITE, sizeof (int), NULL);
  int a = 1, b = 0;
  size_t n = 1;
  CHECK_CL (clSetKernelArg (kernel, 0, sizeof (cl_mem), &buf));
  CHECK_CL (clSetKernelArg (kernel, 1, sizeof (int), &a));
  CHECK_CL (clSetKernelArg (kernel, 2, sizeof (int), &b));
  CHECK_CL (
    clEnqueueNDRangeKernel (queue, kernel, 1, NULL, &n, NULL, 0, NULL, NULL));
  CHECK_CL (clReleaseMemObject (buf));
  CHECK_CL (clReleaseKernel (kernel));
  CHECK_CL (clReleaseCommandQueue (queue));
  printf ("integer division: OK\n");
}

/* Built-in kernels get the same floating-point environment. */
static void
test_builtin_kernel (void)
{
#ifdef __GLIBC__
  cl_int err;
  clCreateProgramWithDefinedBuiltInKernelsEXP_fn create
    = (clCreateProgramWithDefinedBuiltInKernelsEXP_fn)
      clGetExtensionFunctionAddressForPlatform (
        platform, "clCreateProgramWithDefinedBuiltInKernelsEXP");
  if (create == NULL
      || !poclu_supports_extension (device, "cl_exp_defined_builtin_kernels"))
    {
      printf ("builtin kernel: skipped\n");
      return;
    }
  cl_dbk_id_exp id = CL_DBK_IMG_COLOR_CONVERT_EXP;
  const char *name = "exp_img_color_convert";
  pocl_image_attr_t in_attr = { 2, 2, POCL_COLOR_SPACE_BT709,
                                POCL_CHANNEL_RANGE_FULL, POCL_DF_IMAGE_NV12 };
  pocl_image_attr_t out_attr = in_attr;
  out_attr.format = POCL_DF_IMAGE_RGB;
  cl_dbk_attributes_img_color_convert_exp attributes = { in_attr, out_attr };
  const void *attrs = &attributes;
  cl_int supported;
  cl_program prog
    = create (context, 1, &device, 1, &id, &name, &attrs, &supported, &err);
  if (err != CL_SUCCESS)
    {
      printf ("builtin kernel: skipped\n");
      return;
    }
  CHECK_CL (clBuildProgram (prog, 1, &device, NULL, NULL, NULL));
  cl_kernel kernel = clCreateKernel (prog, name, &err);
  CHECK_CL (err);
  unsigned char in[6] = { 80, 90, 100, 110, 129, 129 };
  cl_mem in_buf
    = create_buffer (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof (in), in);
  cl_mem out_buf = create_buffer (CL_MEM_WRITE_ONLY, 12, NULL);
  CHECK_CL (clSetKernelArg (kernel, 0, sizeof (cl_mem), &in_buf));
  CHECK_CL (clSetKernelArg (kernel, 1, sizeof (cl_mem), &out_buf));

  cl_command_queue queue = create_queue (device, 0);
  size_t n = 1;
  cl_event event;
  /* The conversion is inexact, which would trap in the caller's
     environment. */
  feclearexcept (FE_ALL_EXCEPT);
  int traps = feenableexcept (FE_INEXACT) != -1;
  err = clEnqueueNDRangeKernel (queue, kernel, 1, NULL, &n, NULL, 0, NULL,
                                &event);
  int raised = fetestexcept (FE_INEXACT);
  fedisableexcept (FE_ALL_EXCEPT);
  CHECK_CL (err);
  CHECK (event_status (event) == CL_COMPLETE);
  CHECK (!traps || raised == 0);

  CHECK_CL (clReleaseEvent (event));
  CHECK_CL (clReleaseCommandQueue (queue));
  CHECK_CL (clReleaseMemObject (in_buf));
  CHECK_CL (clReleaseMemObject (out_buf));
  CHECK_CL (clReleaseKernel (kernel));
  CHECK_CL (clReleaseProgram (prog));
  printf ("builtin kernel: OK\n");
#endif
}

static void
test_subdevice (void)
{
  cl_uint max_sub_devices;
  CHECK_CL (clGetDeviceInfo (device, CL_DEVICE_PARTITION_MAX_SUB_DEVICES,
                             sizeof (max_sub_devices), &max_sub_devices,
                             NULL));
  if (max_sub_devices < 2)
    {
      printf ("subdevice: skipped\n");
      return;
    }
  const cl_device_partition_property props[]
    = { CL_DEVICE_PARTITION_BY_COUNTS, 1, 1,
        CL_DEVICE_PARTITION_BY_COUNTS_LIST_END, 0 };
  cl_device_id subdevices[2];
  CHECK_CL (clCreateSubDevices (device, props, 2, subdevices, NULL));

  cl_command_queue queue = create_queue (subdevices[1], 0);
  cl_kernel kernel = create_kernel ("set");
  cl_mem buf = create_buffer (CL_MEM_READ_WRITE, N * sizeof (int), NULL);
  cl_event event;
  enqueue_set (queue, kernel, buf, 3, N, 0, NULL, &event);
  CHECK (event_status (event) == CL_COMPLETE);
  int out[N];
  read_ints (buf, out, N);
  for (int i = 0; i < N; ++i)
    CHECK (out[i] == 3);
  if (have_native_kernels)
    test_calling_thread (subdevices[1], "subdevice");

  /* Including buffer contents that have to be migrated first. */
  int in[N];
  for (int i = 0; i < N; ++i)
    in[i] = i;
  cl_mem migrated = create_buffer (CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof (in), in);
  cl_kernel add_one = create_kernel ("add_one");
  size_t n = N;
  CHECK_CL (clSetKernelArg (add_one, 0, sizeof (cl_mem), &migrated));
  CHECK_CL (
    clEnqueueNDRangeKernel (queue, add_one, 1, NULL, &n, NULL, 0, NULL, NULL));
  read_ints (migrated, out, N);
  for (int i = 0; i < N; ++i)
    CHECK (out[i] == i + 1);
  CHECK_CL (clReleaseKernel (add_one));
  CHECK_CL (clReleaseMemObject (migrated));

  CHECK_CL (clReleaseEvent (event));
  CHECK_CL (clReleaseMemObject (buf));
  CHECK_CL (clReleaseKernel (kernel));
  CHECK_CL (clReleaseCommandQueue (queue));
  CHECK_CL (clReleaseDevice (subdevices[0]));
  CHECK_CL (clReleaseDevice (subdevices[1]));
  printf ("subdevice: OK\n");
}

/* When the enqueuing thread can't get what it needs to execute the command,
   the enqueue fails with CL_OUT_OF_HOST_MEMORY before doing anything, so that
   the command can be enqueued again. Run first in the process, before
   anything is reserved that could be reused. */
static int
test_out_of_memory (void)
{
#ifdef __linux__
  cl_int err;
  cl_command_queue ordinary = clCreateCommandQueue (context, device, 0, &err);
  CHECK_CL (err);
  cl_command_queue queue = create_queue (device, 0);
  cl_kernel kernel = create_kernel ("set");
  cl_mem buf = create_buffer (CL_MEM_READ_WRITE, sizeof (int), NULL);
  /* Compile the kernel first, and leave a value to check. */
  enqueue_set (ordinary, kernel, buf, 1, 1, 0, NULL, NULL);
  CHECK_CL (clFinish (ordinary));

  /* Leave too little address space for the kernel's stack. */
  long pages;
  FILE *statm = fopen ("/proc/self/statm", "r");
  CHECK (statm != NULL && fscanf (statm, "%ld", &pages) == 1);
  fclose (statm);
  cl_event event = (cl_event)(uintptr_t)0x1234;
  int v = 2;
  size_t n = 1;
  CHECK_CL (clSetKernelArg (kernel, 0, sizeof (cl_mem), &buf));
  CHECK_CL (clSetKernelArg (kernel, 1, sizeof (int), &v));

  struct rlimit old_limit, limit;
  CHECK (getrlimit (RLIMIT_AS, &old_limit) == 0);
  limit = old_limit;
  limit.rlim_cur = pages * sysconf (_SC_PAGESIZE) + 2 * 1024 * 1024;
  CHECK (setrlimit (RLIMIT_AS, &limit) == 0);
  err = clEnqueueNDRangeKernel (queue, kernel, 1, NULL, &n, NULL, 0, NULL,
                                &event);
  CHECK (setrlimit (RLIMIT_AS, &old_limit) == 0);

  CHECK (err == CL_OUT_OF_HOST_MEMORY);
  CHECK (event == (cl_event)(uintptr_t)0x1234);
  CHECK_CL (clFinish (queue));
  int out;
  read_ints (buf, &out, 1);
  CHECK (out == 1);

  /* The same launch works once there is memory again. */
  CHECK_CL (clEnqueueNDRangeKernel (queue, kernel, 1, NULL, &n, NULL, 0, NULL,
                                    &event));
  CHECK (event_status (event) == CL_COMPLETE);
  read_ints (buf, &out, 1);
  CHECK (out == 2);

  CHECK_CL (clReleaseEvent (event));
  CHECK_CL (clReleaseMemObject (buf));
  CHECK_CL (clReleaseKernel (kernel));
  CHECK_CL (clReleaseCommandQueue (queue));
  CHECK_CL (clReleaseCommandQueue (ordinary));
  printf ("out of memory: OK\n");
  return 0;
#else
  printf ("SKIP: needs RLIMIT_AS\n");
  return 77;
#endif
}

int
main (int argc, char **argv)
{
  cl_int err;
  cl_command_queue unused_queue;
  alarm (300);

  CHECK_CL (
    poclu_get_any_device2 (&context, &device, &unused_queue, &platform));
  CHECK_CL (clReleaseCommandQueue (unused_queue));

  if (!poclu_supports_extension (device, "cl_intel_exec_by_local_thread"))
    {
      printf ("SKIP: the device doesn't support "
              "cl_intel_exec_by_local_thread\n");
      return 77;
    }
  cl_command_queue_properties props;
  CHECK_CL (clGetDeviceInfo (device, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES,
                             sizeof (props), &props, NULL));
  CHECK (props & CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL);
  cl_device_exec_capabilities caps;
  CHECK_CL (clGetDeviceInfo (device, CL_DEVICE_EXECUTION_CAPABILITIES,
                             sizeof (caps), &caps, NULL));
  have_native_kernels = (caps & CL_EXEC_NATIVE_KERNEL) != 0;

  program = clCreateProgramWithSource (context, 1, &source, NULL, &err);
  CHECK_CL (err);
  CHECK_CL (clBuildProgram (program, 0, NULL, NULL, NULL, NULL));

  int ret = 0;
  if (argc > 1 && strcmp (argv[1], "oom") == 0)
    ret = test_out_of_memory ();
  else
    {
      /* The old API accepts the property too. */
      cl_command_queue queue = clCreateCommandQueue (
        context, device, CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL, &err);
      CHECK_CL (err);
      CHECK_CL (clReleaseCommandQueue (queue));

      test_ndrange ();
      test_profiling ();
      if (have_native_kernels)
        test_calling_thread (device, "device");
      test_dependencies ();
      test_commands ();
      test_blocking_calls ();
      test_migration ();
      test_failure ();
      test_threads ();
      if (have_native_kernels)
        test_nesting ();
      test_fp_environment ();
      test_integer_division ();
      test_builtin_kernel ();
      test_subdevice ();
    }

  CHECK_CL (clReleaseProgram (program));
  CHECK_CL (clReleaseContext (context));
  if (ret == 0)
    printf ("OK\n");
  return ret;
}
