/* OpenCL native pthreaded device implementation.

   Copyright (c) 2011-2019 pocl developers

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

#ifdef __linux__
#include <sched.h>
#endif

#include <math.h>
#include <string.h>
#include <time.h>

#include "common.h"
#include "common_driver.h"
#include "pocl-pthread.h"
#include "pocl-pthread_scheduler.h"
#include "pocl_builtin_kernels.h"
#include "pocl_cl.h"
#include "pocl_mem_management.h"
#include "pocl_timing.h"
#include "pocl_util.h"
#include "utlist.h"

#ifdef ENABLE_HOST_CPU_DEVICES_OPENMP
#include <omp.h>
#endif

// debugging help. If defined, randomize the execution order by skipping 1-3
// of the commands in the work queue.
//#define CPU_RANDOMIZE_QUEUE

static void* pocl_pthread_driver_thread (void *p);

struct pool_thread_data
{
  POCL_ALIGNAS(HOST_CPU_CACHELINE_SIZE) pocl_thread_t thread;

  unsigned long executed_commands;
  /* per-CU (= per-thread) local memory */
  void *local_mem;
  unsigned num_threads;
  /* index of this particular thread
   * [0, num_threads-1] */
  unsigned index;
  /* printf buffer*/
  void *printf_buffer;
  size_t thread_stack_size;
};

typedef struct scheduler_data_
{
  POCL_ALIGNAS(HOST_CPU_CACHELINE_SIZE) pocl_cond_t wake_pool;
  POCL_ALIGNAS(HOST_CPU_CACHELINE_SIZE) pocl_lock_t wq_lock_fast;
  POCL_ALIGNAS(HOST_CPU_CACHELINE_SIZE) _cl_command_node *work_queue;

  unsigned num_threads;
  unsigned printf_buf_size;
  size_t local_mem_size;

  int thread_pool_shutdown_requested;
  int worker_out_of_memory;

  struct pool_thread_data *thread_pool;
#ifndef ENABLE_HOST_CPU_DEVICES_OPENMP
  kernel_run_command *kernel_queue;
#endif

  POCL_ALIGNAS(HOST_CPU_CACHELINE_SIZE) pocl_barrier_t init_barrier;
} scheduler_data;

static scheduler_data scheduler;

cl_int
pthread_scheduler_init (cl_device_id device)
{
  unsigned i;
#ifdef ENABLE_HOST_CPU_DEVICES_OPENMP
  /* we still need one worker thread with OpenMP, e.g. to execute commands
    that don't use OpenMP parallel_for */
  size_t num_worker_threads = 1;
#else
  size_t num_worker_threads = device->max_compute_units;
#endif
  POCL_INIT_LOCK (scheduler.wq_lock_fast);

  POCL_INIT_COND (scheduler.wake_pool);

  POCL_LOCK (scheduler.wq_lock_fast);
  VG_ASSOC_COND_VAR (scheduler.wake_pool, scheduler.wq_lock_fast);
  POCL_UNLOCK (scheduler.wq_lock_fast);

  scheduler.thread_pool = pocl_aligned_malloc (
      HOST_CPU_CACHELINE_SIZE,
      num_worker_threads * sizeof (struct pool_thread_data));
  memset (scheduler.thread_pool, 0,
          num_worker_threads * sizeof (struct pool_thread_data));

  scheduler.num_threads = num_worker_threads;
  assert (num_worker_threads > 0);
  scheduler.printf_buf_size = device->printf_buffer_size;
  assert (device->printf_buffer_size > 0);

  /* safety margin - aligning pointers later (in kernel arg setup)
   * may require more local memory than actual local mem size.
   * TODO fix this */
  scheduler.local_mem_size = device->local_mem_size + device->max_parameter_size * MAX_EXTENDED_ALIGNMENT;

  POCL_INIT_BARRIER (scheduler.init_barrier, num_worker_threads + 1);

  scheduler.worker_out_of_memory = 0;

  for (i = 0; i < num_worker_threads; ++i)
    {
      scheduler.thread_pool[i].index = i;
      POCL_CREATE_THREAD (scheduler.thread_pool[i].thread,
                          pocl_pthread_driver_thread,
                          (void *)&scheduler.thread_pool[i]);
#ifdef ENABLE_SIGFPE_HANDLER
      pocl_ignore_sigfpe_for_thread (scheduler.thread_pool[i].thread);
#endif
    }

  POCL_WAIT_BARRIER (scheduler.init_barrier);

  if (scheduler.worker_out_of_memory)
    {
      pthread_scheduler_uninit ();
      return CL_OUT_OF_HOST_MEMORY;
    }

#ifdef HOST_CPU_ENABLE_STACK_SIZE_CHECK
  size_t min_thread_stack_size = SIZE_MAX;
  for (i = 0; i < num_worker_threads; ++i)
    min_thread_stack_size = min (scheduler.thread_pool[i].thread_stack_size,
                                 min_thread_stack_size);
  device->work_group_stack_size = min_thread_stack_size;
#endif

  return CL_SUCCESS;
}

void
pthread_scheduler_uninit ()
{
  unsigned i;

  POCL_LOCK (scheduler.wq_lock_fast);
  scheduler.thread_pool_shutdown_requested = 1;
  POCL_BROADCAST_COND (scheduler.wake_pool);
  POCL_UNLOCK (scheduler.wq_lock_fast);

  for (i = 0; i < scheduler.num_threads; ++i)
    {
      POCL_JOIN_THREAD (scheduler.thread_pool[i].thread);
    }
  scheduler.thread_pool_shutdown_requested = 0;
  pocl_aligned_free (scheduler.thread_pool);

  POCL_DESTROY_LOCK (scheduler.wq_lock_fast);
  POCL_DESTROY_COND (scheduler.wake_pool);
  POCL_DESTROY_BARRIER (scheduler.init_barrier);
}

void pthread_scheduler_push_command (_cl_command_node *cmd)
{
  POCL_LOCK (scheduler.wq_lock_fast);
  DL_APPEND (scheduler.work_queue, cmd);
  POCL_SIGNAL_COND (scheduler.wake_pool);
  POCL_UNLOCK (scheduler.wq_lock_fast);
}

#ifndef ENABLE_HOST_CPU_DEVICES_OPENMP
/* use broadcast and wake up all threads; this should only be used
   when kernel is "large" enough to reasonable load all CPU threads. */
static void
pthread_scheduler_push_kernel (kernel_run_command *run_cmd)
{
  POCL_LOCK (scheduler.wq_lock_fast);
  DL_APPEND (scheduler.kernel_queue, run_cmd);
  POCL_BROADCAST_COND (scheduler.wake_pool);
  POCL_UNLOCK (scheduler.wq_lock_fast);
}

// in nanoseconds
#define POCL_CPU_THREAD_TIME_CHUNK 120000UL

/* return 1: work to do, 0: nothing to do */
static int
get_wg_index_range (kernel_run_command *k,
                    unsigned *start_index,
                    unsigned *end_index,
                    int *last_wgs,
                    size_t scaled_wgs)
{
  assert (scaled_wgs);
  size_t max_wgs = POCL_ATOMIC_LOAD (k->wgs_total);

  size_t last_wg_idx = POCL_ATOMIC_ADD (k->wgs_dealt, scaled_wgs);
  assert ((last_wg_idx != 0) && "last wg idx == 0");
  *start_index = last_wg_idx - scaled_wgs;
  *end_index = last_wg_idx;

  if (*start_index >= max_wgs)
    return 0;

  if (*end_index >= max_wgs)
    {
      *end_index = max_wgs;
      *last_wgs = 1;
    }

  return 1;
}

inline static void translate_wg_index_to_3d_index (kernel_run_command *k,
                                                   unsigned index,
                                                   size_t *index_3d,
                                                   unsigned xy_slice,
                                                   unsigned row_size)
{
  index_3d[2] = index / xy_slice;
  index_3d[1] = (index % xy_slice) / row_size;
  index_3d[0] = (index % xy_slice) % row_size;
}

/* is_in_queue = bool
  true: is in scheduler.kernel_queue, meaning all CPU threads are participating
  in execution false: not in scheduler.kernel_queue, the calling thread
  executes all work */
static void
work_group_scheduler (kernel_run_command *k,
                      int is_in_queue,
                      struct pool_thread_data *thread_data)
{
  pocl_kernel_metadata_t *meta = k->kernel->meta;

  const size_t num_args = meta->num_args + meta->num_locals + 1;
  void *arguments = alloca (sizeof (void *) * num_args);
  void *arguments2 = alloca (sizeof (void *) * num_args);
  struct pocl_context pc;
  unsigned i;
  unsigned start_index;
  unsigned end_index;
  int last_wgs = 0;
  size_t scaled_wgs = 1;
  size_t max_wgs = POCL_ATOMIC_LOAD (k->wgs_total);
  size_t wg_size
    = k->pc.local_size[0] * k->pc.local_size[1] * k->pc.local_size[2];
  assert (wg_size);

  if (is_in_queue)
    {
      // multiple CPU threads
      if (k->timing.t.count == 0)
        {
          // unknown time / WG, with multiple CPU threads competing
          if (max_wgs <= 128 * thread_data->num_threads)
            {
              // if there are few WGs, divide evenly, since the worst case is
              // that each WG takes a very long time to execute.
              scaled_wgs = max (max_wgs / thread_data->num_threads, 1);
            }
          else
            {
              // if there are relatively many WGs, start by a small amount and
              // scale up with 32 there will be 128/32 = 4 chunks for each CPU
              // thread
              scaled_wgs = 32;
            }
        }
      else
        {
          // known time / WG, with multiple CPU threads competing
          // calculate the WG count (scaled_wgs) so that each thread takes
          // approx POCL_PTHREAD_TIME_CHUNK nanoseconds to execute
          // TODO this is inexact, we should store per-WG-size timing numbers
          // instead
          size_t max_wgs_per_thread = (max_wgs / thread_data->num_threads);
          if (!max_wgs_per_thread)
            max_wgs_per_thread = 1;
          size_t time_per_WG = (size_t)k->timing.t.cumulative_time_per_wi
                               * wg_size / k->timing.t.count;
          time_per_WG = max (time_per_WG, 1);
          scaled_wgs = max (POCL_CPU_THREAD_TIME_CHUNK / time_per_WG, 1);
          scaled_wgs = min (scaled_wgs, max_wgs_per_thread);
        }
    }
  else
    {
      // not in queue = single CPU thread executes whole range
      scaled_wgs = max_wgs;
    }

  if (!get_wg_index_range (k, &start_index, &end_index, &last_wgs, scaled_wgs))
    return;
  assert (end_index > start_index);

  pocl_setup_kernel_arg_array_with_locals (
      (void **)arguments, (void **)arguments2, k, thread_data->local_mem,
      scheduler.local_mem_size);
  memcpy (&pc, &k->pc, sizeof (struct pocl_context));

  // capacity and position already set up
  pc.printf_buffer = thread_data->printf_buffer;
  uint32_t position = 0;
  pc.printf_buffer_position = &position;
  assert (pc.printf_buffer != NULL);
  assert (pc.printf_buffer_capacity > 0);
  assert (pc.printf_buffer_position != NULL);

  pocl_cpu_setup_rm_and_ftz (k->device, k->kernel->program);

  unsigned slice_size = k->pc.num_groups[0] * k->pc.num_groups[1];
  unsigned row_size = k->pc.num_groups[0];
  unsigned execution_failed = 0;
  uint64_t total_time = 0;
  uint64_t total_wgs = 0;
  do
    {
      if (is_in_queue && last_wgs)
        {
          POCL_LOCK (scheduler.wq_lock_fast);
          DL_DELETE (scheduler.kernel_queue, k);
          POCL_UNLOCK (scheduler.wq_lock_fast);
        }

      uint64_t start_time = pocl_gettimemono_ns ();
      for (i = start_index; i < end_index; ++i)
        {
          size_t gids[3];
          translate_wg_index_to_3d_index (k, i, gids,
                                          slice_size, row_size);

#ifdef DEBUG_MT
          printf("### exec_wg: gid_x %zu, gid_y %zu, gid_z %zu\n",
                 gids[0], gids[1], gids[2]);
#endif
          k->workgroup ((uint8_t *)arguments, (uint8_t *)&pc,
                        gids[0], gids[1], gids[2]);
          execution_failed |= pc.execution_failed;
        }
      uint64_t stop_time = pocl_gettimemono_ns ();
      uint64_t time_spent_ns = stop_time - start_time;
      /* filter out potentially invalid values from clock function */
      if (stop_time > start_time && time_spent_ns < (600UL << 30))
        {
          total_time += time_spent_ns;
          total_wgs += end_index - start_index;
          if (time_spent_ns < POCL_CPU_THREAD_TIME_CHUNK)
            {
              scaled_wgs
                = scaled_wgs * POCL_CPU_THREAD_TIME_CHUNK / time_spent_ns;
            }
        }
    }
  while (
    get_wg_index_range (k, &start_index, &end_index, &last_wgs, scaled_wgs));

  if (total_time > 0)
    {
      POCL_ATOMIC_ADD (k->time_per_wg_count, total_wgs);
      POCL_ATOMIC_ADD (k->time_per_wg_total, total_time);
    }

#ifndef ENABLE_PRINTF_IMMEDIATE_FLUSH
  pocl_write_printf_buffer ((char *)pc.printf_buffer, position);
#endif

  pocl_free_kernel_arg_array_with_locals ((void **)arguments,
                                          (void **)arguments2, k);

  POCL_ATOMIC_OR (k->execution_failed, execution_failed);
}

#else /* OPENMP enabled scheduler */

static void
work_group_scheduler (kernel_run_command *k,
                      struct pool_thread_data *thread_data)
{
  cl_kernel kernel = k->kernel;
  cl_program program = kernel->program;
  pocl_kernel_metadata_t *meta = k->kernel->meta;

  omp_set_dynamic(0);
  omp_set_num_threads(k->device->max_compute_units);
#pragma omp parallel
  {
    const size_t num_args = meta->num_args + meta->num_locals + 1;
    void *arguments = alloca (sizeof (void *) * num_args);
    void *arguments2 = alloca (sizeof (void *) * num_args);
    struct pocl_context pc;
    void *local_mem = malloc (scheduler.local_mem_size);

    pocl_setup_kernel_arg_array_with_locals ((void **)arguments,
                                             (void **)arguments2, k, local_mem,
                                             scheduler.local_mem_size);
    memcpy (&pc, &k->pc, sizeof (struct pocl_context));

    assert (pc.printf_buffer_capacity > 0);
    pc.printf_buffer = malloc (pc.printf_buffer_capacity);
    assert (pc.printf_buffer != NULL);
    uint32_t position = 0;
    pc.printf_buffer_position = &position;

    pocl_cpu_setup_rm_and_ftz (k->device, k->kernel->program);

    size_t x, y, z;
    unsigned execution_failed = 0;
    /* runtime = set scheduling according to environment variable OMP_SCHEDULE
     */
#pragma omp for ordered collapse(3) schedule(runtime)
    for (z = 0; z < pc.num_groups[2]; ++z)
      for (y = 0; y < pc.num_groups[1]; ++y)
        for (x = 0; x < pc.num_groups[0]; ++x)
          {
            ((pocl_workgroup_func)k->workgroup) ((uint8_t *)arguments,
                                                 (uint8_t *)&pc, x, y, z);
            execution_failed |= pc.execution_failed;
          }
#ifndef ENABLE_PRINTF_IMMEDIATE_FLUSH
    pocl_write_printf_buffer ((char *)pc.printf_buffer, position);
#endif

    pocl_free_kernel_arg_array_with_locals ((void **)arguments,
                                            (void **)arguments2, k);

    free (local_mem);
    free (pc.printf_buffer);
#pragma omp critical
    k->execution_failed |= execution_failed;
  } // #pragma omp parallel
}

#endif

static void
finalize_kernel_command (struct pool_thread_data *thread_data,
                         kernel_run_command *k)
{
#ifdef DEBUG_MT
  printf("### kernel %s finished\n", k->cmd->command.run.kernel->name);
#endif

  pocl_free_kernel_arg_array (k);

  pocl_release_dlhandle_cache (k->cmd->command.run.device_data);

  pthread_timing_data *store_td
    = (pthread_timing_data *)k->kernel->data[k->device_i];
  assert (store_td);

  if (k->execution_failed)
    POCL_UPDATE_EVENT_FAILED_MSG (CL_FAILED, k->cmd->sync.event.event,
                                  "NDRange Kernel        ");
  else
    {
      if (k->time_per_wg_count)
        {
          size_t wg_size
            = k->pc.local_size[0] * k->pc.local_size[1] * k->pc.local_size[2];
          uint64_t time_per_WG = k->time_per_wg_total / k->time_per_wg_count;
          uint32_t time_per_WI = time_per_WG / wg_size;
          pthread_timing_data new_td, old_td, ret_td;

          do
            {
              ret_td.all = old_td.all = new_td.all
                = POCL_ATOMIC_LOAD (store_td->all);
              if (old_td.t.count < 1024
                  && old_td.t.cumulative_time_per_wi
                       < (UINT32_MAX - time_per_WI))
                {
                  new_td.t.count = old_td.t.count + 1;
                  new_td.t.cumulative_time_per_wi
                    = old_td.t.cumulative_time_per_wi + time_per_WI;

                  ret_td.all
                    = POCL_ATOMIC_CAS (&store_td->all, old_td.all, new_td.all);
                }
            }
          while (ret_td.all != old_td.all);
        }
      POCL_UPDATE_EVENT_COMPLETE_MSG (k->cmd->sync.event.event,
                                      "NDRange Kernel        ");
    }

  POCL_DESTROY_LOCK (k->lock);
  free_kernel_run_command (k);
}

static kernel_run_command *
pocl_pthread_prepare_kernel (void *data,
                             _cl_command_node *cmd,
                             struct pool_thread_data *td)
{
  kernel_run_command *run_cmd = NULL;
  cl_kernel kernel = cmd->command.run.kernel;
  struct pocl_context *pc = &cmd->command.run.pc;
  cl_program program = kernel->program;
  cl_uint dev_i = cmd->program_device_i;

  size_t num_groups = pc->num_groups[0] * pc->num_groups[1] * pc->num_groups[2];
  size_t wg_size = pc->local_size[0] * pc->local_size[1] * pc->local_size[2];

  if (num_groups == 0)
    {
      pocl_update_event_running (cmd->sync.event.event);
      POCL_UPDATE_EVENT_COMPLETE_MSG (cmd->sync.event.event,
                                      "NDRange Kernel        ");

      return NULL;
    }

  /* initialize the program gvars if required */
  if (pocl_driver_build_gvar_init_kernel (program, dev_i, cmd->device,
                                          pocl_cpu_gvar_init_callback)
      != 0)
    {
      pocl_update_event_running (cmd->sync.event.event);
      POCL_UPDATE_EVENT_FAILED_MSG (CL_FAILED, cmd->sync.event.event,
                                    "CPU: failed to compile GVar init kernel");
      return NULL;
    }

  char *saved_name = NULL;
  pocl_sanitize_builtin_kernel_name (kernel, &saved_name);
  void *ci = pocl_check_kernel_dlhandle_cache (cmd, CL_TRUE, CL_TRUE);
  cmd->command.run.device_data = ci;
  pocl_restore_builtin_kernel_name (kernel, saved_name);
  if (ci == NULL)
    {
      pocl_update_event_running (cmd->sync.event.event);
      POCL_UPDATE_EVENT_FAILED_MSG (CL_FAILED, cmd->sync.event.event,
                                    "CPU: failed to compile kernel");
      return NULL;
    }

  run_cmd = new_kernel_run_command ();
  run_cmd->data = data;
  run_cmd->kernel = kernel;
  run_cmd->device = cmd->device;
  run_cmd->pc = *pc;
  run_cmd->cmd = cmd;
  run_cmd->pc.printf_buffer = NULL;
  run_cmd->pc.printf_buffer_capacity = scheduler.printf_buf_size;
  run_cmd->pc.printf_buffer_position = NULL;
  run_cmd->pc.global_var_buffer = program->gvar_storage[dev_i];
  run_cmd->wgs_total = num_groups;
  run_cmd->wgs_dealt = 0;
  run_cmd->workgroup = cmd->command.run.wg;
  run_cmd->kernel_args = cmd->command.run.arguments;
  run_cmd->next = NULL;
  run_cmd->ref_count = 0;
  run_cmd->execution_failed = 0;
  run_cmd->time_per_wg_total = 0;
  run_cmd->time_per_wg_count = 0;
  run_cmd->device_i = cmd->program_device_i;
  pthread_timing_data *timing
    = (pthread_timing_data *)kernel->data[run_cmd->device_i];
  assert (timing);
  run_cmd->timing.all = POCL_ATOMIC_LOAD (timing->all);

  POCL_INIT_LOCK (run_cmd->lock);

  pocl_setup_kernel_arg_array (run_cmd);

  pocl_update_event_running (cmd->sync.event.event);

#ifdef ENABLE_HOST_CPU_DEVICES_OPENMP
  work_group_scheduler (run_cmd, td);
  finalize_kernel_command (td, run_cmd);
#else
  if (run_cmd->timing.t.count)
    {
      uint64_t time_per_wg
        = ((uint64_t)run_cmd->timing.t.cumulative_time_per_wi * wg_size
           / run_cmd->timing.t.count);
      uint64_t estimated_time = num_groups * time_per_wg;
      /* if the command time estimated is < single POCL_PTHREAD_TIME_CHUNK,
         don't wake up the other threads & handle the whole launch in this
         thread */
      if (estimated_time < POCL_CPU_THREAD_TIME_CHUNK)
        {
          work_group_scheduler (run_cmd, 0, td);
          finalize_kernel_command (td, run_cmd);
        }
      else
        {
          pthread_scheduler_push_kernel (run_cmd);
        }
    }
  else
    {
      pthread_scheduler_push_kernel (run_cmd);
    }
#endif
  return run_cmd;
}

static _cl_command_node *
check_cmd_queue_for_device (thread_data *td)
{
  _cl_command_node *cmd = scheduler.work_queue;
  if (cmd)
  {
    DL_DELETE (scheduler.work_queue, cmd);
    return cmd;
  }
  return NULL;
}

#ifndef ENABLE_HOST_CPU_DEVICES_OPENMP
static kernel_run_command *
check_kernel_queue_for_device (thread_data *td)
{
  return scheduler.kernel_queue;
}
#endif

static int
pthread_scheduler_get_work (thread_data *td)
{
  _cl_command_node *cmd = NULL;
  kernel_run_command *run_cmd = NULL;

  /* execute kernel if available */
  POCL_LOCK (scheduler.wq_lock_fast);
  int do_exit = 0;

RETRY:
  do_exit = scheduler.thread_pool_shutdown_requested;

#ifndef ENABLE_HOST_CPU_DEVICES_OPENMP
  run_cmd = check_kernel_queue_for_device (td);
  /* execute kernel if available */
  if (run_cmd)
    {
      ++run_cmd->ref_count;
      POCL_UNLOCK (scheduler.wq_lock_fast);

      work_group_scheduler (run_cmd, 1, td);

      POCL_LOCK (scheduler.wq_lock_fast);
      if ((--run_cmd->ref_count) == 0)
        {
          POCL_UNLOCK (scheduler.wq_lock_fast);
          finalize_kernel_command (td, run_cmd);
          POCL_LOCK (scheduler.wq_lock_fast);
        }
    }
#endif

  /* execute a command if available */
  cmd = check_cmd_queue_for_device (td);
  if (cmd)
    {
      POCL_UNLOCK (scheduler.wq_lock_fast);

      if (cmd->type == CL_COMMAND_NDRANGE_KERNEL)
        {
          cl_kernel kernel = cmd->command.run.kernel;
          cl_program program = kernel->program;
          pocl_kernel_metadata_t *meta = kernel->meta;
          cl_uint dev_i = cmd->program_device_i;
          if (program->builtin_kernel_attributes)
            {
              assert (meta->builtin_kernel_id != 0);
              pocl_update_event_running (cmd->sync.event.event);

              pocl_cpu_execute_dbk (program, kernel, meta, dev_i,
                                    cmd->command.run.arguments);

              POCL_UPDATE_EVENT_COMPLETE_MSG (cmd->sync.event.event,
                                              "Builtin Kernel        ");
            }
          else
            {
              pocl_pthread_prepare_kernel (cmd->device->data, cmd, td);
            }
        }
      else
        {
          pocl_exec_command (cmd);
        }

      POCL_LOCK (scheduler.wq_lock_fast);
      ++td->executed_commands;
    }

  /* if neither a command nor a kernel was available, sleep */
  if ((cmd == NULL) && (run_cmd == NULL) && (do_exit == 0))
    {
      POCL_WAIT_COND (scheduler.wake_pool, scheduler.wq_lock_fast);
      goto RETRY;
    }

  POCL_UNLOCK (scheduler.wq_lock_fast);

  return do_exit;
}


static
void*
pocl_pthread_driver_thread (void *p)
{
  struct pool_thread_data *td = (struct pool_thread_data*)p;
  int do_exit = 0;
  assert (td);

  td->num_threads = scheduler.num_threads;
  td->printf_buffer = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT,
                                           scheduler.printf_buf_size);

#ifdef HOST_CPU_ENABLE_STACK_SIZE_CHECK
  /* Try to set the stack size to 8MB */
  POCL_SET_THREAD_STACK_SIZE (8 * 1024 * 1024);
  size_t stack_size = POCL_GET_THREAD_STACK_SIZE ();
  /* if the call fails, set a safe minimum */
  if (stack_size == 0)
    stack_size = 512 * 1024;
  /* keep some margin for the thread's own data */
  td->thread_stack_size = stack_size * 3 / 4;
#endif

  assert (scheduler.local_mem_size > 0);
  td->local_mem = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT,
                                       scheduler.local_mem_size);
#if defined(__linux__) && !defined(__ANDROID__) && defined(PTHREAD_CHECK)
  if (pocl_get_bool_option ("POCL_AFFINITY", 0))
    {
      cpu_set_t set;
      CPU_ZERO (&set);
      CPU_SET (td->index, &set);
      PTHREAD_CHECK (
          pthread_setaffinity_np (td->thread, sizeof (cpu_set_t), &set));
    }
#endif

  if (td->printf_buffer == NULL || td->local_mem == NULL)
    {
      POCL_ATOMIC_INC (scheduler.worker_out_of_memory);
    }

  POCL_WAIT_BARRIER (scheduler.init_barrier);

#ifdef POCL_DEBUG_MESSAGES
  if (pocl_get_bool_option ("POCL_DUMP_TASK_GRAPHS", 0) == 1)
    {
      pocl_dump_dot_task_graph_wait ();
    }
#endif

  while (1)
    {
      do_exit = pthread_scheduler_get_work (td);
      if (do_exit)
        {
          pocl_aligned_free (td->printf_buffer);
          pocl_aligned_free (td->local_mem);
          return NULL;
        }
    }
}
