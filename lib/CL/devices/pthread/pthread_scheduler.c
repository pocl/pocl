#define _GNU_SOURCE

#ifdef __linux__
#include <sched.h>
#endif

#include <string.h>
#include <pthread.h>
#include <time.h>

#include "pocl-pthread_scheduler.h"
#include "pocl_cl.h"
#include "pocl-pthread.h"
#include "pocl-pthread_utils.h"
#include "utlist.h"
#include "pocl_util.h"
#include "common.h"
#include "pocl_mem_management.h" 

static void* pocl_pthread_driver_thread (void *p);

struct pool_thread_data
{
  pthread_cond_t wakeup_cond __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  pthread_mutex_t lock __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

  pthread_t thread __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

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
  void *last_cmd_ignored;

} __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

typedef struct scheduler_data_
{
  unsigned num_threads;

  struct pool_thread_data *thread_pool;
  size_t local_mem_size;

  _cl_command_node *work_queue
      __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  kernel_run_command *kernel_queue;

  pthread_cond_t wake_pool __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  pthread_mutex_t wake_lock __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  PTHREAD_FAST_LOCK_T wq_lock_fast __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

  pthread_cond_t cq_finished_cond __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  pthread_mutex_t cq_finished_lock __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

  int thread_pool_shutdown_requested;
} scheduler_data __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

static scheduler_data scheduler;

void
pthread_scheduler_init (cl_device_id device)
{
  unsigned i;
  size_t num_worker_threads = device->max_compute_units;
  PTHREAD_INIT_LOCK (&(scheduler.wake_lock));
  PTHREAD_FAST_INIT (&(scheduler.wq_lock_fast));

  PTHREAD_INIT_LOCK (&(scheduler.cq_finished_lock));
  pthread_cond_init (&(scheduler.cq_finished_cond), NULL);
  pthread_cond_init (&(scheduler.wake_pool), NULL);

  scheduler.thread_pool = pocl_aligned_malloc (
      HOST_CPU_CACHELINE_SIZE,
      num_worker_threads * sizeof (struct pool_thread_data));
  memset (scheduler.thread_pool, 0,
          num_worker_threads * sizeof (struct pool_thread_data));

  scheduler.num_threads = num_worker_threads;
  /* safety margin - aligning pointers later (in kernel arg setup)
   * may require more local memory than actual local mem size.
   * TODO fix this */
  scheduler.local_mem_size = device->local_mem_size << 4;

  for (i = 0; i < num_worker_threads; ++i)
    {
      pthread_cond_init (&scheduler.thread_pool[i].wakeup_cond, NULL);
      PTHREAD_INIT_LOCK (&scheduler.thread_pool[i].lock);
      scheduler.thread_pool[i].index = i;
      pthread_create (&scheduler.thread_pool[i].thread, NULL,
                      pocl_pthread_driver_thread,
                      (void*)&scheduler.thread_pool[i]);
    }

}

void
pthread_scheduler_uninit ()
{
  unsigned i;
  scheduler.thread_pool_shutdown_requested = 1;

  PTHREAD_LOCK (&scheduler.wake_lock);
  pthread_cond_broadcast (&scheduler.wake_pool);
  PTHREAD_UNLOCK (&scheduler.wake_lock);

  for (i = 0; i < scheduler.num_threads; ++i)
    {
      pthread_join (scheduler.thread_pool[i].thread, NULL);
      PTHREAD_DESTROY_LOCK (&scheduler.thread_pool[i].lock);
      pthread_cond_destroy (&scheduler.thread_pool[i].wakeup_cond);
    }

  pocl_aligned_free (scheduler.thread_pool);
  PTHREAD_FAST_DESTROY (&scheduler.wq_lock_fast);
  pthread_cond_destroy (&scheduler.wake_pool);
  PTHREAD_DESTROY_LOCK (&scheduler.wake_lock);

  pthread_cond_destroy (&scheduler.cq_finished_cond);
  PTHREAD_DESTROY_LOCK (&scheduler.cq_finished_lock);
}

void pthread_scheduler_push_command (_cl_command_node *cmd)
{
  PTHREAD_FAST_LOCK (&scheduler.wq_lock_fast);
  DL_APPEND (scheduler.work_queue, cmd);
  PTHREAD_FAST_UNLOCK (&scheduler.wq_lock_fast);

  PTHREAD_LOCK (&scheduler.wake_lock);
  pthread_cond_broadcast (&scheduler.wake_pool);
  PTHREAD_UNLOCK (&scheduler.wake_lock);
}

void pthread_scheduler_push_kernel (kernel_run_command *run_cmd)
{
  PTHREAD_FAST_LOCK (&scheduler.wq_lock_fast);
  LL_APPEND (scheduler.kernel_queue, run_cmd);
  PTHREAD_FAST_UNLOCK (&scheduler.wq_lock_fast);

  PTHREAD_LOCK (&scheduler.wake_lock);
  pthread_cond_broadcast (&scheduler.wake_pool);
  PTHREAD_UNLOCK (&scheduler.wake_lock);
}

void pthread_scheduler_wait_cq (cl_command_queue cq)
{
  PTHREAD_LOCK (&scheduler.cq_finished_lock);

#ifdef HAVE_CLOCK_GETTIME
  struct timespec timeout = {0, 0};
#endif

  while (1)
    {
      POCL_LOCK_OBJ (cq);
      if (cq->command_count == 0)
        {
          POCL_UNLOCK_OBJ (cq);
          PTHREAD_UNLOCK (&scheduler.cq_finished_lock);
          return;
        }
      POCL_UNLOCK_OBJ (cq);

      /* pocl_cond_timedwait() is a workaround, the pthread driver sometimes
       * gets stuck in the loop waiting for finished_cond while the CQ is
       * actually empty. With timedwait() it eventually recovers.
       */
#ifdef HAVE_CLOCK_GETTIME
      clock_gettime(CLOCK_REALTIME, &timeout);
      timeout.tv_nsec += 100000000;
      if (timeout.tv_nsec >= 1000000000)
        {
          timeout.tv_nsec -= 1000000000;
          ++timeout.tv_sec;
        }
      pthread_cond_timedwait (&scheduler.cq_finished_cond,
                              &scheduler.cq_finished_lock,
                              &timeout);
#else
       pthread_cond_wait (&scheduler.cq_finished_cond,
                          &scheduler.cq_finished_lock);
#endif

    }

  PTHREAD_UNLOCK (&scheduler.cq_finished_lock);
}

void pthread_scheduler_release_host ()
{
  PTHREAD_LOCK (&scheduler.cq_finished_lock);
  pthread_cond_signal (&scheduler.cq_finished_cond);
  PTHREAD_UNLOCK (&scheduler.cq_finished_lock);
}

static int
work_group_scheduler (kernel_run_command *k,
                      struct pool_thread_data *thread_data);

static void finalize_kernel_command (thread_data *thread_data,
                              kernel_run_command *k);

/* if subd is not a subdevice, returns 1
 * if subd is subdevice, takes a look at the subdevice CUs
 * and if they match the current driver thread, returns 1
 * otherwise set last ignored command to cmd and return 0 */
static int
shall_we_run_this (thread_data *td, cl_device_id subd, void *cmd)
{

  if (subd && subd->parent_device)
    {
      if (!((td->index >= subd->core_start)
            && (td->index < (subd->core_start + subd->core_count))))
        {
          td->last_cmd_ignored = cmd;
          return 0;
        }
    }
  td->last_cmd_ignored = NULL;
  return 1;
}

void
pthread_scheduler_get_work (thread_data *td, _cl_command_node **cmd_ptr)
{
  _cl_command_node *cmd;
  kernel_run_command *run_cmd;

  /* execute kernel if available */
  PTHREAD_FAST_LOCK (&scheduler.wq_lock_fast);
  run_cmd = scheduler.kernel_queue;

  /* execute kernel if available */
  if (run_cmd && shall_we_run_this (td, run_cmd->device, run_cmd))
    {
      ++run_cmd->ref_count;
      PTHREAD_FAST_UNLOCK (&scheduler.wq_lock_fast);

      work_group_scheduler (run_cmd, td);

      PTHREAD_FAST_LOCK (&scheduler.wq_lock_fast);
      if ((--run_cmd->ref_count) == 0)
        {
          PTHREAD_FAST_UNLOCK (&scheduler.wq_lock_fast);
          finalize_kernel_command (td, run_cmd);
          PTHREAD_FAST_LOCK (&scheduler.wq_lock_fast);
        }
    }

  /* execute a command if available */
  *cmd_ptr = NULL;
  cmd = scheduler.work_queue;
  if (cmd && shall_we_run_this (td, cmd->device, cmd))
    {
      DL_DELETE (scheduler.work_queue, cmd);
      *cmd_ptr = cmd;
    }
  PTHREAD_FAST_UNLOCK (&scheduler.wq_lock_fast);
  return;
}

static void
pthread_scheduler_sleep (thread_data *td)
{
  struct timespec time_to_wait = {0, 0};
  time_to_wait.tv_sec = time(NULL) + 5;

  PTHREAD_FAST_LOCK (&scheduler.wq_lock_fast);
  /* if the queues are empty, go to sleep.
   * if the queues are not empty, but this thread ignored the
   * last command (because it's for different subdevice CUs),
   * also go to sleep. */
  if ((scheduler.work_queue == NULL && scheduler.kernel_queue == NULL)
      || (td->last_cmd_ignored
          && (((void *)scheduler.kernel_queue == td->last_cmd_ignored)
              || ((void *)scheduler.work_queue == td->last_cmd_ignored))))
    {
      PTHREAD_FAST_UNLOCK (&scheduler.wq_lock_fast);
      PTHREAD_LOCK (&scheduler.wake_lock);
      pthread_cond_timedwait (&scheduler.wake_pool, &scheduler.wake_lock, &time_to_wait);
      PTHREAD_UNLOCK (&scheduler.wake_lock);
    }
  else
    PTHREAD_FAST_UNLOCK (&scheduler.wq_lock_fast);
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

  unsigned max_wgs;
  PTHREAD_FAST_LOCK (&k->lock);
  if (k->remaining_wgs == 0)
    {
      PTHREAD_FAST_UNLOCK (&k->lock);
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
    max_wgs = min (scaled_min_wgs, (1 + k->remaining_wgs / num_threads));
  else
    max_wgs = min (scaled_max_wgs, (1 + k->remaining_wgs / num_threads));

  max_wgs = min (max_wgs, k->remaining_wgs);
  assert (max_wgs > 0);

  *start_index = k->wgs_dealt;
  *end_index = k->wgs_dealt + max_wgs-1;
  k->remaining_wgs -= max_wgs;
  k->wgs_dealt += max_wgs;
  if (k->remaining_wgs == 0)
    *last_wgs = 1;
  PTHREAD_FAST_UNLOCK (&k->lock);

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
  void *arguments[k->kernel->num_args + k->kernel->num_locals + 1];
  void *arguments2[k->kernel->num_args + k->kernel->num_locals + 1];
  struct pocl_context pc;
  unsigned i;
  unsigned start_index;
  unsigned end_index;
  int last_wgs = 0;

  if (!get_wg_index_range (k, &start_index, &end_index, &last_wgs,
                           thread_data->num_threads))
    return 0;

  assert (end_index >= start_index);

  setup_kernel_arg_array_with_locals (
      (void **)&arguments, (void **)&arguments2, k, thread_data->local_mem,
      scheduler.local_mem_size);
  memcpy (&pc, &k->pc, sizeof (struct pocl_context));

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
          PTHREAD_FAST_LOCK (&scheduler.wq_lock_fast);
          LL_DELETE (scheduler.kernel_queue, k);
          PTHREAD_FAST_UNLOCK (&scheduler.wq_lock_fast);
        }

      for (i = start_index; i <= end_index; ++i)
        {
          translate_wg_index_to_3d_index (k, i, pc.group_id,
                                          slice_size, row_size);

#ifdef DEBUG_MT
          printf("### exec_wg: gid_x %zu, gid_y %zu, gid_z %zu\n",
                 pc.group_id[0],
                 pc.group_id[1], pc.group_id[2]);
#endif
          pocl_set_default_rm ();
          k->workgroup (arguments, &pc);
        }
    }
  while (get_wg_index_range (k, &start_index, &end_index, &last_wgs,
                             thread_data->num_threads));

  free_kernel_arg_array_with_locals ((void **)&arguments, (void **)&arguments2,
                                     k);

  return 1;
}

void finalize_kernel_command (struct pool_thread_data *thread_data,
                                     kernel_run_command *k)
{
#ifdef DEBUG_MT
  printf("### kernel %s finished\n", k->cmd->command.run.kernel->name);
#endif

  free_kernel_arg_array (k);

  pocl_release_dlhandle_cache (k->cmd);

  pocl_ndrange_node_cleanup (k->cmd);

  POCL_UPDATE_EVENT_COMPLETE_MSG (k->cmd->event, "NDRange Kernel        ");

  pocl_mem_manager_free_command (k->cmd);

  free_kernel_run_command (k);
}

static void
pocl_pthread_prepare_kernel
(void *data, 
 _cl_command_node* cmd)
{
  kernel_run_command *run_cmd;
  cl_kernel kernel = cmd->command.run.kernel;
  struct pocl_context *pc = &cmd->command.run.pc;

  if (cmd != NULL && cmd->type == CL_COMMAND_NDRANGE_KERNEL)
    pocl_check_dlhandle_cache (cmd, 1);

  int num_groups = pc->num_groups[0] * pc->num_groups[1] * pc->num_groups[2];

  run_cmd = new_kernel_run_command ();
  run_cmd->data = data;
  run_cmd->kernel = kernel;
  run_cmd->device = cmd->device;
  run_cmd->pc = *pc;
  run_cmd->cmd = cmd;
  run_cmd->pc.local_size[0] = cmd->command.run.local_x;
  run_cmd->pc.local_size[1] = cmd->command.run.local_y;
  run_cmd->pc.local_size[2] = cmd->command.run.local_z;
  run_cmd->remaining_wgs = num_groups;
  run_cmd->wgs_dealt = 0;
  run_cmd->workgroup = cmd->command.run.wg;
  run_cmd->kernel_args = cmd->command.run.arguments;
  run_cmd->next = NULL;
  run_cmd->ref_count = 0;
  PTHREAD_FAST_INIT (&run_cmd->lock);

  setup_kernel_arg_array (run_cmd);

  pthread_scheduler_push_kernel (run_cmd);
}

static void
pocl_pthread_exec_command (_cl_command_node *cmd,
                           struct pool_thread_data *td)
{
  if(cmd->type == CL_COMMAND_NDRANGE_KERNEL)
    {
      POCL_UPDATE_EVENT_RUNNING (cmd->event);
      pocl_pthread_prepare_kernel (cmd->command.run.data, cmd);
    }
  else
    {
      pocl_exec_command(cmd);
    }
}


static
void*
pocl_pthread_driver_thread (void *p)
{
  struct pool_thread_data *td = (struct pool_thread_data*)p;
  assert (td);
  _cl_command_node *cmd = NULL;
  /* some random value, doesn't matter as long as it's not a valid bool - to
   * force a first FTZ setup */
  td->current_ftz = 213;
  td->num_threads = scheduler.num_threads;
  td->last_cmd_ignored = NULL;

  assert (scheduler.local_mem_size > 0);
  td->local_mem = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT,
                                       scheduler.local_mem_size);
  assert (td->local_mem);
#ifdef __linux__
  if (pocl_get_bool_option ("POCL_AFFINITY", 0))
    {
      cpu_set_t set;
      CPU_ZERO (&set);
      CPU_SET (td->index, &set);
      pthread_setaffinity_np (td->thread, sizeof (cpu_set_t), &set);
    }
#endif

  while (1)
    {
      if (scheduler.thread_pool_shutdown_requested)
        {
          pthread_cond_destroy (&td->wakeup_cond);
          PTHREAD_DESTROY_LOCK (&td->lock);
          pthread_exit (NULL);
        }

      pthread_scheduler_get_work (td, &cmd);
      if (cmd)
        {
          assert (pocl_command_is_ready(cmd->event));
          assert (cmd->event->status == CL_SUBMITTED);
          pocl_pthread_exec_command (cmd, td);
          cmd = NULL;
          ++td->executed_commands;
        }
      // check if its time to sleep
      pthread_scheduler_sleep (td);
    }
}
