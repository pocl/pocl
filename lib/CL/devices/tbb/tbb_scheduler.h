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

#include "pocl_cl.h"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  typedef enum
  {
    TBB_PART_NONE,
    TBB_PART_AFFINITY,
    TBB_PART_AUTO,
    TBB_PART_SIMPLE,
    TBB_PART_STATIC
  } pocl_tbb_partitioner;

  void *TBBDriverThread (void *Dev);

  struct TBBArena;

  typedef struct
  {
    pthread_cond_t wake_meta_thread
        __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
    POCL_FAST_LOCK_T wq_lock_fast
        __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
    _cl_command_node *work_queue;

    size_t local_mem_size __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
    char *local_mem_global_ptr;

    uchar *printf_buf_global_ptr;
    unsigned printf_buf_size;

    unsigned grain_size;
    unsigned num_tbb_threads;
    pocl_tbb_partitioner selected_partitioner;

    struct TBBArena *tbb_arena;

    pthread_t meta_thread;
    int meta_thread_shutdown_requested;
  } pocl_tbb_scheduler_data
      __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

  size_t tbb_get_numa_nodes ();

  size_t tbb_get_num_threads (pocl_tbb_scheduler_data *SchedData);

  void tbb_init_arena (pocl_tbb_scheduler_data *SchedData, int OnePerNode);

  void tbb_release_arena (pocl_tbb_scheduler_data *SchedData);

#ifdef __cplusplus
}
#endif

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif
