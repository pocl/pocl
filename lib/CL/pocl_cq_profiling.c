/* OpenCL runtime library: integrated command queue profile collecting
   functionality

   Copyright (c) 2019 Pekka Jääskeläinen

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

#include "pocl_cq_profiling.h"
#include "pocl_cl.h"
#include "pocl_util.h"

/* Maximum number of events collected. */
#define POCL_CQ_PROFILING_MAX_EVENTS 1000000
#define POCL_CQ_PROFILING_MAX_KERNELS 1000

int pocl_cq_profiling_enabled = 0;
static unsigned cq_events_collected = 0;
static cl_event *profiled_cq_events = NULL;

struct kernel_stats
{
  cl_kernel kernel;
  unsigned long time;
  unsigned long launches;
};

static int
order_by_time (const void *a, const void *b)
{
  if (((struct kernel_stats *)a)->time < ((struct kernel_stats *)b)->time)
    return 1;
  else if (((struct kernel_stats *)a)->time > ((struct kernel_stats *)b)->time)
    return -1;
  else
    return 0;
}

static void
pocl_atexit ()
{
  unsigned long total_time = 0;
  unsigned long total_commands = 0;
  unsigned long different_kernels = 0;

  struct kernel_stats kernel_statistics[cq_events_collected];
  bzero (kernel_statistics, sizeof (kernel_statistics));

  /* First statistics computation round. */
  for (unsigned i = 0; i < cq_events_collected; ++i)
    {
      cl_event e = profiled_cq_events[i];
      cl_kernel kernel = e->meta_data->kernel;
      unsigned long kernel_t = e->time_end - e->time_start;
      total_time += kernel_t;
      total_commands++;

      unsigned k_i = 0;
      while (k_i < different_kernels
             && strcmp (kernel_statistics[k_i].kernel->name, kernel->name)
                    != 0)
        ++k_i;

      if (kernel_statistics[k_i].kernel == NULL)
        {
          kernel_statistics[k_i].kernel = kernel;
          different_kernels++;
        }
      kernel_statistics[k_i].time += kernel_t;
      kernel_statistics[k_i].launches++;
    }

  printf ("\n");
  printf ("     %-30s %10s %15s %3s  %10s\n", "kernel", "launches", "total us",
          "", "avg us");
  qsort (kernel_statistics, different_kernels, sizeof (struct kernel_stats),
         order_by_time);

  for (int i = 0; i < different_kernels; ++i)
    {
      printf ("%3d) %-30s %10lu %15lu %3lu%% %10lu\n", i + 1,
              kernel_statistics[i].kernel->name, kernel_statistics[i].launches,
              kernel_statistics[i].time,
              kernel_statistics[i].time * 100 / total_time,
              kernel_statistics[i].time / kernel_statistics[i].launches);
    }
  printf ("     %-30s %10s %15s %3s %10s\n", "",
          "==========", "==========", "====", "==========");

  printf ("     %-30s %10lu %15lu %4s %10lu\n", "", total_commands, total_time,
          "100%", total_time / total_commands);

  /* TODO: Critical path information of the task graph. */
}

/* Initialize the profiling data structures, if not yet done. */

void
pocl_cq_profiling_init ()
{
  profiled_cq_events
      = (cl_event *)malloc (sizeof (cl_event *) * POCL_CQ_PROFILING_MAX_EVENTS);
  atexit (pocl_atexit);
  pocl_cq_profiling_enabled = 1;
}

/* Registers the event for profiling. Retains it to keep it alive until stats
   have been collected.
*/

void
pocl_cq_profiling_register_event (cl_event event)
{
  POname(clRetainEvent) (event);
  if (event->meta_data == NULL)
    event->meta_data = (pocl_event_md *)malloc (sizeof (pocl_event_md));

  unsigned cq_events_pos = POCL_ATOMIC_INC (cq_events_collected) - 1;
  if (cq_events_pos >= POCL_CQ_PROFILING_MAX_EVENTS)
    POCL_ABORT ("CQ profiler reached the limit on tracked events.");
  profiled_cq_events[cq_events_pos] = event;
}
