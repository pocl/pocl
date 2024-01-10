/* OpenCL device using the Intel TBB library (derived from the pthread device).

   Copyright (c) 2015 Ville Korhonen, Tampere University of Technology
                 2021 Tobias Baumann, Zuse Institute Berlin

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

#ifndef POCL_TBB_SCHEDULER_H
#define POCL_TBB_SCHEDULER_H

#include "common_utils.h"
#include "pocl_cl.h"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

enum class pocl_tbb_partitioner : unsigned
{
  NONE,
  AFFINITY,
  AUTO,
  SIMPLE,
  STATIC
};

struct pocl_tbb_scheduler_data
{
  pthread_cond_t wake_meta_thread
      __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  POCL_FAST_LOCK_T wq_lock_fast
      __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  _cl_command_node *work_queue
      __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

  unsigned printf_buf_size;
  uchar *printf_buf_global_ptr;

  size_t local_mem_size;
  char *local_mem_global_ptr;

  unsigned grain_size;
  unsigned num_tbb_threads;
  pocl_tbb_partitioner selected_partitioner;
  tbb::numa_node_id numa_idx;
  tbb::task_arena arena;

  pthread_t meta_thread __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  int meta_thread_shutdown_requested;
} __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

/* Initializes scheduler. Must be called before any kernel enqueue */
void tbb_scheduler_init (cl_device_id device);

void tbb_scheduler_uninit (cl_device_id device);

/* Gives ready-to-execute command for scheduler */
void tbb_scheduler_push_command (_cl_command_node *cmd);

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif
