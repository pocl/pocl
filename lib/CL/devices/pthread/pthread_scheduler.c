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

#include <pthread.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "common.h"
#include "common_driver.h"
#include "pocl-pthread.h"
#include "pocl-pthread_scheduler.h"
#include "pocl_builtin_kernels.h"
#include "pocl_cl.h"
#include "pocl_mem_management.h"
#include "pocl_util.h"
#include "utlist.h"

#ifdef __APPLE__
#include "pthread_barrier.h"
#endif

// debugging help. If defined, randomize the execution order by skipping 1-3
// of the commands in the work queue.
//#define CPU_RANDOMIZE_QUEUE

static void* pocl_pthread_driver_thread (void *p);

struct pool_thread_data
{
  pocl_thread_t thread __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

  unsigned long executed_commands;
  /* per-CU (= per-thread) local memory */
  void *local_mem;
  unsigned current_ftz;
  unsigned num_threads;
  /* index of this particular thread
   * [0, num_threads-1]
   * used for deciding whether a particular thread should run
   * commands scheduled on a subdevice. */
  unsigned index;
  /* printf buffer*/
  void *printf_buffer;
} __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

typedef struct scheduler_data_
{
  pocl_cond_t wake_pool __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  pocl_lock_t wq_lock_fast __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  _cl_command_node *work_queue
      __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

  unsigned num_threads;
  unsigned printf_buf_size;
  size_t local_mem_size;

  int thread_pool_shutdown_requested;
  int worker_out_of_memory;

  struct pool_thread_data *thread_pool;
#ifndef ENABLE_HOST_CPU_DEVICES_OPENMP
  kernel_run_command *kernel_queue;
#endif

  pocl_barrier_t init_barrier
    __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
} scheduler_data __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

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
#if defined(__linux__) && defined(__x86_64__)
      pocl_ignore_sigfpe_for_thread (scheduler.thread_pool[i].thread);
#endif
    }

  POCL_WAIT_BARRIER (scheduler.init_barrier);

  if (scheduler.worker_out_of_memory)
    {
      pthread_scheduler_uninit (device);
      return CL_OUT_OF_HOST_MEMORY;
    }

  return CL_SUCCESS;
}

void
pthread_scheduler_uninit (cl_device_id device)
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

/* push_command and push_kernel MUST use broadcast and wake up all threads,
   because commands can be for subdevices (= not all threads) */
void pthread_scheduler_push_command (_cl_command_node *cmd)
{
  POCL_LOCK (scheduler.wq_lock_fast);
  DL_APPEND (scheduler.work_queue, cmd);
  POCL_BROADCAST_COND (scheduler.wake_pool);
  POCL_UNLOCK (scheduler.wq_lock_fast);
}

#ifndef ENABLE_HOST_CPU_DEVICES_OPENMP
static void
pthread_scheduler_push_kernel (kernel_run_command *run_cmd)
{
  POCL_LOCK (scheduler.wq_lock_fast);
  DL_APPEND (scheduler.kernel_queue, run_cmd);
  POCL_BROADCAST_COND (scheduler.wake_pool);
  POCL_UNLOCK (scheduler.wq_lock_fast);
}

/* Maximum and minimum chunk sizes for get_wg_index_range().
 * Each pthread driver's thread fetches work from a kernel's WG pool in
 * chunks, this determines the limits (scaled up by # of threads). */
#define POCL_PTHREAD_MAX_WGS 256
#define POCL_PTHREAD_MIN_WGS 32

static int
get_wg_index_range (kernel_run_command *k, unsigned *start_index,
                    unsigned *end_index, int *last_wgs, unsigned num_threads)
{
  const unsigned scaled_max_wgs = POCL_PTHREAD_MAX_WGS * num_threads;
  const unsigned scaled_min_wgs = POCL_PTHREAD_MIN_WGS * num_threads;

  unsigned limit;
  unsigned max_wgs;
  POCL_LOCK (k->lock);
  if (k->remaining_wgs == 0)
    {
      POCL_UNLOCK (k->lock);
      return 0;
    }

  /* If the work is comprised of huge number of WGs of small WIs,
   * then get_wg_index_range() becomes a problem on manycore CPUs
   * because lock contention on k->lock.
   *
   * If we have enough workgroups, scale up the requests linearly by
   * num_threads, otherwise fallback to smaller workgroups.
   */
  if (k->remaining_wgs <= (scaled_max_wgs * num_threads))
    limit = scaled_min_wgs;
  else
    limit = scaled_max_wgs;

  // divide two integers rounding up, i.e. ceil(k->remaining_wgs/num_threads)
  const unsigned wgs_per_thread = (1 + (k->remaining_wgs - 1) / num_threads);
  max_wgs = min (limit, wgs_per_thread);
  max_wgs = min (max_wgs, k->remaining_wgs);
  assert (max_wgs > 0);

  *start_index = k->wgs_dealt;
  *end_index = k->wgs_dealt + max_wgs-1;
  k->remaining_wgs -= max_wgs;
  k->wgs_dealt += max_wgs;
  if (k->remaining_wgs == 0)
    *last_wgs = 1;
  POCL_UNLOCK (k->lock);

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

static int
work_group_scheduler (kernel_run_command *k,
                      struct pool_thread_data *thread_data)
{
  pocl_kernel_metadata_t *meta = k->kernel->meta;

  void *arguments[meta->num_args + meta->num_locals + 1];
  void *arguments2[meta->num_args + meta->num_locals + 1];
  struct pocl_context pc;
  unsigned i;
  unsigned start_index;
  unsigned end_index;
  int last_wgs = 0;

  if (!get_wg_index_range (k, &start_index, &end_index, &last_wgs,
                           thread_data->num_threads))
    return 0;

  assert (end_index >= start_index);

  pocl_setup_kernel_arg_array_with_locals (
      (void **)&arguments, (void **)&arguments2, k, thread_data->local_mem,
      scheduler.local_mem_size);
  memcpy (&pc, &k->pc, sizeof (struct pocl_context));

  // capacity and position already set up
  pc.printf_buffer = thread_data->printf_buffer;
  uint32_t position = 0;
  pc.printf_buffer_position = &position;
  assert (pc.printf_buffer != NULL);
  assert (pc.printf_buffer_capacity > 0);
  assert (pc.printf_buffer_position != NULL);

  /* Flush to zero is only set once at start of kernel (because FTZ is
   * a compilation option), but we need to reset rounding mode after every
   * iteration (since it can be changed during kernel execution). */
  unsigned flush = k->kernel->program->flush_denorms;
  if (thread_data->current_ftz != flush)
    {
      pocl_set_ftz (flush);
      thread_data->current_ftz = flush;
    }

  unsigned slice_size = k->pc.num_groups[0] * k->pc.num_groups[1];
  unsigned row_size = k->pc.num_groups[0];

  do
    {
      if (last_wgs)
        {
          POCL_LOCK (scheduler.wq_lock_fast);
          DL_DELETE (scheduler.kernel_queue, k);
          POCL_UNLOCK (scheduler.wq_lock_fast);
        }

      for (i = start_index; i <= end_index; ++i)
        {
          size_t gids[3];
          translate_wg_index_to_3d_index (k, i, gids,
                                          slice_size, row_size);

#ifdef DEBUG_MT
          printf("### exec_wg: gid_x %zu, gid_y %zu, gid_z %zu\n",
                 gids[0], gids[1], gids[2]);
#endif
          pocl_set_default_rm ();
          k->workgroup ((uint8_t*)arguments, (uint8_t*)&pc,
			gids[0], gids[1], gids[2]);
        }
    }
  while (get_wg_index_range (k, &start_index, &end_index, &last_wgs,
                             thread_data->num_threads));

#ifndef ENABLE_PRINTF_IMMEDIATE_FLUSH
  pocl_write_printf_buffer ((char *)pc.printf_buffer, position);
#endif

  pocl_free_kernel_arg_array_with_locals ((void **)&arguments, (void **)&arguments2,
                                     k);

  return 1;
}

#else /* OPENMP enabled scheduler */

static int
work_group_scheduler (kernel_run_command *k,
                      struct pool_thread_data *thread_data)
{
  cl_kernel kernel = k->kernel;
  cl_program program = kernel->program;
  pocl_kernel_metadata_t *meta = k->kernel->meta;

#pragma omp parallel
  {
    void *arguments[meta->num_args + meta->num_locals + 1];
    void *arguments2[meta->num_args + meta->num_locals + 1];
    struct pocl_context pc;
    void *local_mem = malloc (scheduler.local_mem_size);

    pocl_setup_kernel_arg_array_with_locals ((void **)&arguments,
                                        (void **)&arguments2, k, local_mem,
                                        scheduler.local_mem_size);
    memcpy (&pc, &k->pc, sizeof (struct pocl_context));

    assert (pc.printf_buffer_capacity > 0);
    pc.printf_buffer = malloc (pc.printf_buffer_capacity);
    assert (pc.printf_buffer != NULL);
    uint32_t position = 0;
    pc.printf_buffer_position = &position;

    unsigned rm = pocl_save_rm ();
    pocl_set_default_rm ();
    unsigned ftz = pocl_save_ftz ();
    pocl_set_ftz (program->flush_denorms);

    size_t x, y, z;
    /* runtime = set scheduling according to environment variable OMP_SCHEDULE
     */
#pragma omp for collapse(3) schedule(runtime)
    for (z = 0; z < pc.num_groups[2]; ++z)
      for (y = 0; y < pc.num_groups[1]; ++y)
        for (x = 0; x < pc.num_groups[0]; ++x)
          ((pocl_workgroup_func)k->workgroup) ((uint8_t *)arguments,
                                               (uint8_t *)&pc, x, y, z);

    pocl_restore_rm (rm);
    pocl_restore_ftz (ftz);

#ifndef ENABLE_PRINTF_IMMEDIATE_FLUSH
    pocl_write_printf_buffer ((char *)pc.printf_buffer, position);
#endif

    pocl_free_kernel_arg_array_with_locals ((void **)&arguments,
                                       (void **)&arguments2, k);

    free (local_mem);
    free (pc.printf_buffer);
  } // #pragma omp parallel

  return 0;
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

  POCL_UPDATE_EVENT_COMPLETE_MSG (k->cmd->sync.event.event,
                                  "NDRange Kernel        ");

  POCL_DESTROY_LOCK (k->lock);
  free_kernel_run_command (k);
}

static kernel_run_command *
pocl_pthread_prepare_kernel (void *data, _cl_command_node *cmd)
{
  kernel_run_command *run_cmd;
  cl_kernel kernel = cmd->command.run.kernel;
  struct pocl_context *pc = &cmd->command.run.pc;
  cl_program program = kernel->program;
  cl_uint dev_i = cmd->program_device_i;

  size_t num_groups = pc->num_groups[0] * pc->num_groups[1] * pc->num_groups[2];

  if (num_groups == 0)
    {
      pocl_update_event_running (cmd->sync.event.event);

      POCL_UPDATE_EVENT_COMPLETE_MSG (cmd->sync.event.event,
                                      "NDRange Kernel        ");

      return NULL;
    }

  /* initialize the program gvars if required */
  pocl_driver_build_gvar_init_kernel (program, dev_i, cmd->device,
                                      pocl_cpu_gvar_init_callback);

  char *saved_name = NULL;
  pocl_sanitize_builtin_kernel_name (kernel, &saved_name);
  void *ci = pocl_check_kernel_dlhandle_cache (cmd, CL_TRUE, CL_TRUE);
  cmd->command.run.device_data = ci;
  pocl_restore_builtin_kernel_name (kernel, saved_name);

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
  run_cmd->remaining_wgs = num_groups;
  run_cmd->wgs_dealt = 0;
  run_cmd->workgroup = cmd->command.run.wg;
  run_cmd->kernel_args = cmd->command.run.arguments;
  run_cmd->next = NULL;
  run_cmd->ref_count = 0;
  POCL_INIT_LOCK (run_cmd->lock);

  pocl_setup_kernel_arg_array (run_cmd);

  pocl_update_event_running (cmd->sync.event.event);

#ifndef ENABLE_HOST_CPU_DEVICES_OPENMP
  pthread_scheduler_push_kernel (run_cmd);
#endif
  return run_cmd;
}

/*
  These two check the entire kernel/cmd queue. This is necessary
  because of commands for subdevices. The old code only checked
  the head of each queue; this can lead to a deadlock:

  two driver threads, each assigned two subdevices A, B, one
  driver queue C

  cmd A1 for A arrives in C, A starts processing
  cmd B1 for B arrives in C, B starts processing
  cmds A2, A3, B2 are pushed to C
  B finishes processing B1, checks queue head, A2 isn't for it, goes to sleep
  A finishes processing A1, processes A2 + A3 but ignores B2, it's not for it
  application calls clFinish to wait for queue

  ...now B is sleeping and not possible to wake up
  (since no new commands can arrive) and there's a B2 command
  which will never be processed.

  it's possible to workaround but it's cleaner to just check the whole queue.
 */

#ifdef ENABLE_HOST_CPU_DEVICES_OPENMP
/* with OpenMP we don't support subdevices -> run every command */
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

#else
/* if subd is not a subdevice, returns 1
 * if subd is subdevice, takes a look at the subdevice CUs
 * and if they match the current driver thread, returns 1
 * otherwise returns 0 */
static int
shall_we_run_this (thread_data *td, cl_device_id subd)
{

  if (subd && subd->parent_device)
    {
      if (!((td->index >= subd->core_start)
            && (td->index < (subd->core_start + subd->core_count))))
        {
          return 0;
        }
    }
  return 1;
}

static _cl_command_node *
check_cmd_queue_for_device (thread_data *td)
{
  _cl_command_node *cmd = NULL, *last_cmd = NULL;
  int i = 0;
#ifdef CPU_RANDOMIZE_QUEUE
  int limit = (rand() % 3) + 1;
#else
  const int limit = 1;
#endif
  DL_FOREACH (scheduler.work_queue, cmd)
  {
    cl_device_id subd = cmd->device;
    if (shall_we_run_this (td, subd))
      {
        last_cmd = cmd; ++i;
        if (i >= limit) break;
      }
  }

  if (last_cmd) {
    DL_DELETE (scheduler.work_queue, last_cmd);
  }
  return last_cmd;
}

static kernel_run_command *
check_kernel_queue_for_device (thread_data *td)
{
  kernel_run_command *cmd = NULL;
  DL_FOREACH (scheduler.kernel_queue, cmd)
  {
    cl_device_id subd = cmd->device;
    if (shall_we_run_this (td, subd))
      return cmd;
  }

  return NULL;
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

      work_group_scheduler (run_cmd, td);

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

      assert (pocl_command_is_ready (cmd->sync.event.event));

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
#ifdef ENABLE_HOST_CPU_DEVICES_OPENMP
              run_cmd = pocl_pthread_prepare_kernel (cmd->device->data, cmd);
              work_group_scheduler (run_cmd, td);
              finalize_kernel_command (td, run_cmd);
              run_cmd = NULL;
#else
              pocl_pthread_prepare_kernel (cmd->device->data, cmd);
#endif
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
  /* some random value, doesn't matter as long as it's not a valid bool - to
   * force a first FTZ setup */
  td->current_ftz = 213;
  td->num_threads = scheduler.num_threads;
  td->printf_buffer = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT,
                                           scheduler.printf_buf_size);

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
          pthread_exit (NULL);
        }
    }
}
