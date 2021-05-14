/* OpenCL device using the Intel TBB library (derived from the pthread device).

   Copyright (c) 2011-2021 PoCL developers

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

#include <algorithm>
#include <pthread.h>
#include <string.h>
#include <unistd.h>

#include <tbb/blocked_range3d.h>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>
#include <tbb/task_arena.h>

#include "common.h"
#include "common_utils.h"
#include "pocl_cl.h"
#include "pocl_mem_management.h"
#include "pocl_runtime_config.h"
#include "pocl_util.h"
#include "tbb_scheduler.h"
#include "utlist.h"

static void *pocl_tbb_driver_thread(void *p);

enum partitioner { NONE, AFFINITY, AUTO, SIMPLE, STATIC };

typedef struct scheduler_data_ {
  unsigned printf_buf_size;
  uchar *printf_buf_global_ptr;

  size_t local_mem_size;
  char *local_mem_global_ptr;

  size_t num_tbb_threads;
  unsigned grain_size;
  enum partitioner selected_partitioner;

  _cl_command_node *work_queue
      __attribute__((aligned(HOST_CPU_CACHELINE_SIZE)));
  POCL_FAST_LOCK_T wq_lock_fast
      __attribute__((aligned(HOST_CPU_CACHELINE_SIZE)));

  pthread_t meta_thread __attribute__((aligned(HOST_CPU_CACHELINE_SIZE)));
  pthread_cond_t wake_meta_thread
      __attribute__((aligned(HOST_CPU_CACHELINE_SIZE)));
  int meta_thread_shutdown_requested;
} scheduler_data __attribute__((aligned(HOST_CPU_CACHELINE_SIZE)));

static scheduler_data scheduler;

/* External functions declared in tbb_scheduler.h */

void tbb_scheduler_init(cl_device_id device) {
  POCL_FAST_INIT(scheduler.wq_lock_fast);

  pthread_cond_init(&(scheduler.wake_meta_thread), NULL);

  scheduler.printf_buf_size = device->printf_buffer_size;
  assert(device->printf_buffer_size > 0);

  /* safety margin - aligning pointers later (in kernel arg setup)
   * may require more local memory than actual local mem size.
   * TODO fix this */
  scheduler.local_mem_size =
      device->local_mem_size +
      device->max_parameter_size * MAX_EXTENDED_ALIGNMENT;

  scheduler.num_tbb_threads = device->max_compute_units;

  /* alloc local memory for all threads making sure the memory for each thread
   * is aligned. */
  scheduler.printf_buf_size =
      (1 + (scheduler.printf_buf_size - 1) / MAX_EXTENDED_ALIGNMENT) *
      MAX_EXTENDED_ALIGNMENT;
  scheduler.local_mem_size =
      (1 + (scheduler.local_mem_size - 1) / MAX_EXTENDED_ALIGNMENT) *
      MAX_EXTENDED_ALIGNMENT;
  scheduler.printf_buf_global_ptr =
      reinterpret_cast<uchar *>(pocl_aligned_malloc(
          MAX_EXTENDED_ALIGNMENT,
          scheduler.printf_buf_size * scheduler.num_tbb_threads));
  scheduler.local_mem_global_ptr = reinterpret_cast<char *>(pocl_aligned_malloc(
      MAX_EXTENDED_ALIGNMENT,
      scheduler.local_mem_size * scheduler.num_tbb_threads));

  /* create one meta thread to serve as an async interface thread. */
  pthread_create(&scheduler.meta_thread, NULL, pocl_tbb_driver_thread, NULL);

  scheduler.grain_size = 0;
  if (pocl_is_option_set("POCL_TBB_GRAIN_SIZE")) {
    scheduler.grain_size = pocl_get_int_option("POCL_TBB_GRAIN_SIZE", 1);
    POCL_MSG_PRINT_GENERAL("TBB: using a grain size of %u\n",
                           scheduler.grain_size);
  }

  scheduler.selected_partitioner = NONE;
  const char *ptr = pocl_get_string_option("POCL_TBB_PARTITIONER", "");
  if (strlen(ptr) > 0) {
    if (strncmp(ptr, "affinity", 8) == 0) {
      scheduler.selected_partitioner = AFFINITY;
      POCL_MSG_PRINT_GENERAL("TBB: using affinity partitioner\n");
    } else if (strncmp(ptr, "auto", 4) == 0) {
      scheduler.selected_partitioner = AUTO;
      POCL_MSG_PRINT_GENERAL("TBB: using auto partitioner\n");
    } else if (strncmp(ptr, "simple", 6) == 0) {
      scheduler.selected_partitioner = SIMPLE;
      POCL_MSG_PRINT_GENERAL("TBB: using simple partitioner\n");
    } else if (strncmp(ptr, "static", 6) == 0) {
      scheduler.selected_partitioner = STATIC;
      POCL_MSG_PRINT_GENERAL("TBB: using static partitioner\n");
    } else {
      POCL_MSG_WARN(
          "TBB: Malformed string in POCL_TBB_PARTITIONER env var: %s\n", ptr);
    }
  }
}

void tbb_scheduler_uninit() {
  unsigned i;

  POCL_FAST_LOCK(scheduler.wq_lock_fast);
  scheduler.meta_thread_shutdown_requested = 1;
  pthread_cond_broadcast(&scheduler.wake_meta_thread);
  POCL_FAST_UNLOCK(scheduler.wq_lock_fast);

  pthread_join(scheduler.meta_thread, NULL);

  POCL_FAST_DESTROY(scheduler.wq_lock_fast);
  pthread_cond_destroy(&scheduler.wake_meta_thread);

  scheduler.meta_thread_shutdown_requested = 0;

  pocl_aligned_free(scheduler.printf_buf_global_ptr);
  pocl_aligned_free(scheduler.local_mem_global_ptr);
}

/* push_command and push_kernel MUST use broadcast and wake up all threads,
   because commands can be for subdevices (= not all threads) */
void tbb_scheduler_push_command(_cl_command_node *cmd) {
  POCL_FAST_LOCK(scheduler.wq_lock_fast);
  DL_APPEND(scheduler.work_queue, cmd);
  pthread_cond_broadcast(&scheduler.wake_meta_thread);
  POCL_FAST_UNLOCK(scheduler.wq_lock_fast);
}

/* Internal functions */

class WorkGroupScheduler {
  kernel_run_command *my_k;

public:
  void operator()(const tbb::blocked_range3d<size_t> &r) const {
    kernel_run_command *k = my_k;
    pocl_kernel_metadata_t *meta = k->kernel->meta;
    const size_t my_thread_id = tbb::this_task_arena::current_thread_index();
    void *arguments[meta->num_args + meta->num_locals + 1];
    void *arguments2[meta->num_args + meta->num_locals + 1];
    char *local_mem = scheduler.local_mem_global_ptr +
                      (scheduler.local_mem_size * my_thread_id);
    uchar *printf_buffer = scheduler.printf_buf_global_ptr +
                           (scheduler.printf_buf_size * my_thread_id);
    struct pocl_context pc;
    /* some random value, doesn't matter as long as it's not a valid bool - to
     * force a first FTZ setup */
    unsigned current_ftz = 213;

    setup_kernel_arg_array_with_locals((void **)&arguments,
                                       (void **)&arguments2, k, local_mem,
                                       scheduler.local_mem_size);
    memcpy(&pc, &k->pc, sizeof(struct pocl_context));

    // capacity and position already set up
    pc.printf_buffer = printf_buffer;
    uint32_t position = 0;
    pc.printf_buffer_position = &position;
    assert(pc.printf_buffer != NULL);
    assert(pc.printf_buffer_capacity > 0);
    assert(pc.printf_buffer_position != NULL);

    /* Flush to zero is only set once at start of kernel (because FTZ is
     * a compilation option), but we need to reset rounding mode after every
     * iteration (since it can be changed during kernel execution). */
    unsigned flush = k->kernel->program->flush_denorms;
    if (current_ftz != flush) {
      pocl_set_ftz(flush);
      current_ftz = flush;
    }

    for (size_t x = r.pages().begin(); x != r.pages().end(); x++) {
      for (size_t y = r.rows().begin(); y != r.rows().end(); y++) {
        for (size_t z = r.cols().begin(); z != r.cols().end(); z++) {
          pocl_set_default_rm();
          k->workgroup((uint8_t *)arguments, (uint8_t *)&pc, x, y, z);
        }
      }
    }

    if (position > 0) {
      write(STDOUT_FILENO, pc.printf_buffer, position);
    }

    free_kernel_arg_array_with_locals((void **)&arguments, (void **)&arguments2,
                                      k);
  }
  WorkGroupScheduler(kernel_run_command *k) : my_k(k) {}
};

static void finalize_kernel_command(kernel_run_command *k) {
  free_kernel_arg_array(k);

  pocl_release_dlhandle_cache(k->cmd);

  pocl_ndrange_node_cleanup(k->cmd);

  POCL_UPDATE_EVENT_COMPLETE_MSG(k->cmd->event, "NDRange Kernel        ");

  pocl_mem_manager_free_command(k->cmd);
  free_kernel_run_command(k);
}

static kernel_run_command *pocl_tbb_prepare_kernel(void *data,
                                                   _cl_command_node *cmd) {
  kernel_run_command *run_cmd;
  cl_kernel kernel = cmd->command.run.kernel;
  struct pocl_context *pc = &cmd->command.run.pc;

  pocl_check_kernel_dlhandle_cache(cmd, 1, 1);

  run_cmd = new_kernel_run_command();
  run_cmd->data = data;
  run_cmd->kernel = kernel;
  run_cmd->device = cmd->device;
  run_cmd->pc = *pc;
  run_cmd->cmd = cmd;
  run_cmd->pc.printf_buffer = NULL;
  run_cmd->pc.printf_buffer_capacity = scheduler.printf_buf_size;
  run_cmd->pc.printf_buffer_position = NULL;
  run_cmd->workgroup =
      reinterpret_cast<pocl_workgroup_func>(cmd->command.run.wg);
  run_cmd->kernel_args = cmd->command.run.arguments;
  run_cmd->next = NULL;

  setup_kernel_arg_array(run_cmd);

  pocl_update_event_running(cmd->event);

  return (run_cmd);
}

/* Note: Using a double linked list is probably not necessary any more. */
static _cl_command_node *check_cmd_queue_for_device() {
  _cl_command_node *cmd;
  DL_FOREACH(scheduler.work_queue, cmd) {
    DL_DELETE(scheduler.work_queue, cmd)
    return cmd; // return first cmd
  }

  return NULL;
}

static void tbb_exec_command(kernel_run_command *run_cmd) {
  /* Note: Grain size variation could be allowed for each dimension
   * individually. */
  if (scheduler.grain_size) {
    if (scheduler.selected_partitioner == NONE) {
      tbb::parallel_for(tbb::blocked_range3d<size_t>(
                            0, run_cmd->pc.num_groups[0], scheduler.grain_size,
                            0, run_cmd->pc.num_groups[1], scheduler.grain_size,
                            0, run_cmd->pc.num_groups[2], scheduler.grain_size),
                        WorkGroupScheduler(run_cmd));
    } else if (scheduler.selected_partitioner == AFFINITY) {
      tbb::affinity_partitioner ap;
      tbb::parallel_for(tbb::blocked_range3d<size_t>(
                            0, run_cmd->pc.num_groups[0], scheduler.grain_size,
                            0, run_cmd->pc.num_groups[1], scheduler.grain_size,
                            0, run_cmd->pc.num_groups[2], scheduler.grain_size),
                        WorkGroupScheduler(run_cmd), ap);
    } else if (scheduler.selected_partitioner == AUTO) {
      tbb::parallel_for(tbb::blocked_range3d<size_t>(
                            0, run_cmd->pc.num_groups[0], scheduler.grain_size,
                            0, run_cmd->pc.num_groups[1], scheduler.grain_size,
                            0, run_cmd->pc.num_groups[2], scheduler.grain_size),
                        WorkGroupScheduler(run_cmd), tbb::auto_partitioner());
    } else if (scheduler.selected_partitioner == SIMPLE) {
      tbb::parallel_for(tbb::blocked_range3d<size_t>(
                            0, run_cmd->pc.num_groups[0], scheduler.grain_size,
                            0, run_cmd->pc.num_groups[1], scheduler.grain_size,
                            0, run_cmd->pc.num_groups[2], scheduler.grain_size),
                        WorkGroupScheduler(run_cmd), tbb::simple_partitioner());
    } else if (scheduler.selected_partitioner == STATIC) {
      tbb::parallel_for(tbb::blocked_range3d<size_t>(
                            0, run_cmd->pc.num_groups[0], scheduler.grain_size,
                            0, run_cmd->pc.num_groups[1], scheduler.grain_size,
                            0, run_cmd->pc.num_groups[2], scheduler.grain_size),
                        WorkGroupScheduler(run_cmd), tbb::static_partitioner());
    }
  } else { /* if (scheduler.grain_size) */
    if (scheduler.selected_partitioner == NONE) {
      tbb::parallel_for(
          tbb::blocked_range3d<size_t>(0, run_cmd->pc.num_groups[0], 0,
                                       run_cmd->pc.num_groups[1], 0,
                                       run_cmd->pc.num_groups[2]),
          WorkGroupScheduler(run_cmd));
    } else if (scheduler.selected_partitioner == AFFINITY) {
      tbb::affinity_partitioner ap;
      tbb::parallel_for(
          tbb::blocked_range3d<size_t>(0, run_cmd->pc.num_groups[0], 0,
                                       run_cmd->pc.num_groups[1], 0,
                                       run_cmd->pc.num_groups[2]),
          WorkGroupScheduler(run_cmd), ap);
    } else if (scheduler.selected_partitioner == AUTO) {
      tbb::parallel_for(
          tbb::blocked_range3d<size_t>(0, run_cmd->pc.num_groups[0], 0,
                                       run_cmd->pc.num_groups[1], 0,
                                       run_cmd->pc.num_groups[2]),
          WorkGroupScheduler(run_cmd), tbb::auto_partitioner());
    } else if (scheduler.selected_partitioner == SIMPLE) {
      tbb::parallel_for(
          tbb::blocked_range3d<size_t>(0, run_cmd->pc.num_groups[0], 0,
                                       run_cmd->pc.num_groups[1], 0,
                                       run_cmd->pc.num_groups[2]),
          WorkGroupScheduler(run_cmd), tbb::simple_partitioner());
    } else if (scheduler.selected_partitioner == STATIC) {
      tbb::parallel_for(
          tbb::blocked_range3d<size_t>(0, run_cmd->pc.num_groups[0], 0,
                                       run_cmd->pc.num_groups[1], 0,
                                       run_cmd->pc.num_groups[2]),
          WorkGroupScheduler(run_cmd), tbb::static_partitioner());
    }
  }
}

static int tbb_scheduler_get_work() {
  _cl_command_node *cmd;
  kernel_run_command *run_cmd;

  POCL_FAST_LOCK(scheduler.wq_lock_fast);
  int do_exit = 0;

RETRY:
  do_exit = scheduler.meta_thread_shutdown_requested;

  /* execute a command if available */
  cmd = check_cmd_queue_for_device();
  if (cmd) {
    POCL_FAST_UNLOCK(scheduler.wq_lock_fast);

    assert(pocl_command_is_ready(cmd->event));

    if (cmd->type == CL_COMMAND_NDRANGE_KERNEL) {
      run_cmd = pocl_tbb_prepare_kernel(cmd->device->data, cmd);
      tbb_exec_command(run_cmd);
      finalize_kernel_command(run_cmd);
    } else {
      pocl_exec_command(cmd);
    }

    POCL_FAST_LOCK(scheduler.wq_lock_fast);
  }

  /* if no command was available, sleep */
  if ((cmd == NULL) && (do_exit == 0)) {
    pthread_cond_wait(&scheduler.wake_meta_thread, &scheduler.wq_lock_fast);
    goto RETRY;
  }

  POCL_FAST_UNLOCK(scheduler.wq_lock_fast);

  return do_exit;
}

static void *pocl_tbb_driver_thread(void *p) {
  int do_exit = 0;

  while (1) {
    do_exit = tbb_scheduler_get_work();
    if (do_exit) {
      pthread_exit(NULL);
    }
  }
}
