/* OpenCL device using the Intel TBB library (derived from the pthread device).

   Copyright (c) 2011-2021 PoCL developers
                 2021 Tobias Baumann / Zuse Institute Berlin

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

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <algorithm>
#include <pthread.h>
#include <string.h>
#include <unistd.h>

#include <tbb/blocked_range3d.h>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>
#include <tbb/task_arena.h>

#include "builtin_kernels.hh"
#include "common.h"
#include "common_driver.h"
#include "common_utils.h"
#include "pocl_cl.h"
#include "pocl_mem_management.h"
#include "pocl_runtime_config.h"
#include "pocl_util.h"
#include "tbb_scheduler.h"
#include "utlist.h"

static void *pocl_tbb_driver_thread(void *dev);

/* External functions declared in tbb_dd->h */

void tbb_scheduler_init(cl_device_id device) {
  pocl_tbb_scheduler_data *dd = (pocl_tbb_scheduler_data *)device->data;
  POCL_FAST_INIT(dd->wq_lock_fast);
  dd->work_queue = NULL;

  POCL_INIT_COND(dd->wake_meta_thread);

  dd->printf_buf_size = device->printf_buffer_size;
  assert(device->printf_buffer_size > 0);

  /* safety margin - aligning pointers later (in kernel arg setup)
   * may require more local memory than actual local mem size.
   * TODO fix this */
  dd->local_mem_size = device->local_mem_size +
                       device->max_parameter_size * MAX_EXTENDED_ALIGNMENT;

  dd->num_tbb_threads = dd->arena.max_concurrency();

  /* alloc local memory for all threads making sure the memory for each thread
   * is aligned. */
  dd->printf_buf_size =
      (1 + (dd->printf_buf_size - 1) / MAX_EXTENDED_ALIGNMENT) *
      MAX_EXTENDED_ALIGNMENT;
  dd->local_mem_size = (1 + (dd->local_mem_size - 1) / MAX_EXTENDED_ALIGNMENT) *
                       MAX_EXTENDED_ALIGNMENT;
  dd->printf_buf_global_ptr = reinterpret_cast<uchar *>(pocl_aligned_malloc(
      MAX_EXTENDED_ALIGNMENT, dd->printf_buf_size * dd->num_tbb_threads));
  dd->local_mem_global_ptr = reinterpret_cast<char *>(pocl_aligned_malloc(
      MAX_EXTENDED_ALIGNMENT, dd->local_mem_size * dd->num_tbb_threads));

  /* create one meta thread per device to serve as an async interface thread. */
  pthread_create(&dd->meta_thread, NULL, pocl_tbb_driver_thread, device);

  dd->grain_size = 0;
  if (pocl_is_option_set("POCL_TBB_GRAIN_SIZE")) {
    dd->grain_size = pocl_get_int_option("POCL_TBB_GRAIN_SIZE", 1);
    POCL_MSG_PRINT_GENERAL("TBB: using a grain size of %u\n", dd->grain_size);
  }

  dd->selected_partitioner = pocl_tbb_partitioner::NONE;
  const char *ptr = pocl_get_string_option("POCL_TBB_PARTITIONER", "");
  if (strlen(ptr) > 0) {
    if (strncmp(ptr, "affinity", 8) == 0) {
      dd->selected_partitioner = pocl_tbb_partitioner::AFFINITY;
      POCL_MSG_PRINT_GENERAL("TBB: using affinity partitioner\n");
    } else if (strncmp(ptr, "auto", 4) == 0) {
      dd->selected_partitioner = pocl_tbb_partitioner::AUTO;
      POCL_MSG_PRINT_GENERAL("TBB: using auto partitioner\n");
    } else if (strncmp(ptr, "simple", 6) == 0) {
      dd->selected_partitioner = pocl_tbb_partitioner::SIMPLE;
      POCL_MSG_PRINT_GENERAL("TBB: using simple partitioner\n");
    } else if (strncmp(ptr, "static", 6) == 0) {
      dd->selected_partitioner = pocl_tbb_partitioner::STATIC;
      POCL_MSG_PRINT_GENERAL("TBB: using static partitioner\n");
    } else {
      POCL_MSG_WARN(
          "TBB: Malformed string in POCL_TBB_PARTITIONER env var: %s\n", ptr);
    }
  }
}

void tbb_scheduler_uninit(cl_device_id device) {
  pocl_tbb_scheduler_data *dd = (pocl_tbb_scheduler_data *)device->data;

  POCL_FAST_LOCK(dd->wq_lock_fast);
  dd->meta_thread_shutdown_requested = 1;
  POCL_BROADCAST_COND(dd->wake_meta_thread);
  POCL_FAST_UNLOCK(dd->wq_lock_fast);

  POCL_JOIN_THREAD(dd->meta_thread);

  POCL_FAST_DESTROY(dd->wq_lock_fast);
  POCL_DESTROY_COND(dd->wake_meta_thread);

  dd->meta_thread_shutdown_requested = 0;
  dd->work_queue = NULL;

  pocl_aligned_free(dd->printf_buf_global_ptr);
  pocl_aligned_free(dd->local_mem_global_ptr);
}

/* TBB doesn't support subdevices, so push_command can use cond_signal */
void tbb_scheduler_push_command(_cl_command_node *cmd) {
  cl_device_id device = cmd->device;
  pocl_tbb_scheduler_data *dd = (pocl_tbb_scheduler_data *)device->data;
  POCL_FAST_LOCK(dd->wq_lock_fast);
  DL_APPEND(dd->work_queue, cmd);
  POCL_SIGNAL_COND(dd->wake_meta_thread);
  POCL_FAST_UNLOCK(dd->wq_lock_fast);
}

/* Internal functions */

/* The sole purpose of this embedded class is to provide a function object that
 * can be executed on a blocked_range by TBB's parallel_for() according to
 * https://www.threadingbuildingblocks.org/docs/help/tbb_userguide/parallel_for.html */
class WorkGroupScheduler {
  kernel_run_command *my_k;
  const pocl_tbb_scheduler_data *dd;

public:
  void operator()(const tbb::blocked_range3d<size_t> &r) const {
    kernel_run_command *k = my_k;
    pocl_kernel_metadata_t *meta = k->kernel->meta;
    const size_t my_thread_id = tbb::this_task_arena::current_thread_index();
    void *arguments[meta->num_args + meta->num_locals + 1];
    void *arguments2[meta->num_args + meta->num_locals + 1];
    char *local_mem =
        dd->local_mem_global_ptr + (dd->local_mem_size * my_thread_id);
    uchar *printf_buffer =
        dd->printf_buf_global_ptr + (dd->printf_buf_size * my_thread_id);
    struct pocl_context pc;

    setup_kernel_arg_array_with_locals((void **)&arguments,
                                       (void **)&arguments2, k, local_mem,
                                       dd->local_mem_size);
    memcpy(&pc, &k->pc, sizeof(struct pocl_context));

#ifndef ENABLE_PRINTF_IMMEDIATE_FLUSH
    // capacity and position already set up
    pc.printf_buffer = printf_buffer;
    uint32_t position = 0;
    pc.printf_buffer_position = &position;
    assert(pc.printf_buffer != NULL);
    assert(pc.printf_buffer_capacity > 0);
    assert(pc.printf_buffer_position != NULL);
#else
    pc.printf_buffer = NULL;
    pc.printf_buffer_position = NULL;
#endif

    /* Flush to zero is only set once at the start of the kernel execution
     * because FTZ is a compilation option. */
    unsigned flush = k->kernel->program->flush_denorms;
    pocl_set_ftz(flush);

    for (size_t x = r.pages().begin(); x != r.pages().end(); x++) {
      for (size_t y = r.rows().begin(); y != r.rows().end(); y++) {
        for (size_t z = r.cols().begin(); z != r.cols().end(); z++) {
          /* Rounding mode must be reset after every iteration
           * since it can be changed during kernel execution. */
          pocl_set_default_rm();
          k->workgroup((uint8_t *)arguments, (uint8_t *)&pc, x, y, z);
        }
      }
    }

#ifndef ENABLE_PRINTF_IMMEDIATE_FLUSH
    if (position > 0) {
      write(STDOUT_FILENO, pc.printf_buffer, position);
    }
#endif

    free_kernel_arg_array_with_locals((void **)&arguments, (void **)&arguments2,
                                      k);
  }
  WorkGroupScheduler(kernel_run_command *k, const pocl_tbb_scheduler_data *dd)
      : my_k(k), dd(dd) {}
};

static void finalize_kernel_command(kernel_run_command *k) {
  free_kernel_arg_array(k);

  pocl_release_dlhandle_cache(k->cmd);

  POCL_UPDATE_EVENT_COMPLETE_MSG(k->cmd->sync.event.event,
                                 "NDRange Kernel        ");
  free_kernel_run_command(k);
}

static kernel_run_command *pocl_tbb_prepare_kernel(pocl_tbb_scheduler_data *dd,
                                                   _cl_command_node *cmd) {

  kernel_run_command *run_cmd;
  cl_kernel kernel = cmd->command.run.kernel;
  struct pocl_context *pc = &cmd->command.run.pc;
  cl_program program = kernel->program;
  cl_uint dev_i = cmd->program_device_i;

  size_t num_groups = pc->num_groups[0] * pc->num_groups[1] * pc->num_groups[2];

  // if WGs == 0, return early to avoid having to uninitialize anything
  if (num_groups == 0) {
    pocl_update_event_running(cmd->sync.event.event);

    POCL_UPDATE_EVENT_COMPLETE_MSG(cmd->sync.event.event,
                                   "NDRange Kernel        ");

    return NULL;
  }

  /* initialize the program gvars if required */
  pocl_driver_build_gvar_init_kernel(program, dev_i, cmd->device,
                                     pocl_cpu_gvar_init_callback);

  char *saved_name = NULL;
  pocl_sanitize_builtin_kernel_name(kernel, &saved_name);
  pocl_check_kernel_dlhandle_cache(cmd, 1, 1);
  pocl_restore_builtin_kernel_name(kernel, saved_name);

  run_cmd = new_kernel_run_command();
  run_cmd->data = dd;
  run_cmd->kernel = kernel;
  run_cmd->device = cmd->device;
  run_cmd->pc = *pc;
  run_cmd->cmd = cmd;
  run_cmd->pc.printf_buffer = NULL;
  run_cmd->pc.printf_buffer_capacity = dd->printf_buf_size;
  run_cmd->pc.printf_buffer_position = NULL;
  run_cmd->pc.global_var_buffer = (uchar *)program->gvar_storage[dev_i];
  run_cmd->workgroup =
      reinterpret_cast<pocl_workgroup_func>(cmd->command.run.wg);
  run_cmd->kernel_args = cmd->command.run.arguments;
  run_cmd->next = NULL;

  setup_kernel_arg_array(run_cmd);

  pocl_update_event_running(cmd->sync.event.event);

  return (run_cmd);
}

static void tbb_exec_command(pocl_tbb_scheduler_data *dd,
                             kernel_run_command *run_cmd) {
  /* Note: Grain size variation could be allowed for each dimension
   * individually. */
  if (dd->grain_size) {
    if (dd->selected_partitioner == pocl_tbb_partitioner::NONE) {
      tbb::parallel_for(tbb::blocked_range3d<size_t>(
                            0, run_cmd->pc.num_groups[0], dd->grain_size, 0,
                            run_cmd->pc.num_groups[1], dd->grain_size, 0,
                            run_cmd->pc.num_groups[2], dd->grain_size),
                        WorkGroupScheduler(run_cmd, dd));
    } else if (dd->selected_partitioner == pocl_tbb_partitioner::AFFINITY) {
      tbb::affinity_partitioner ap;
      tbb::parallel_for(tbb::blocked_range3d<size_t>(
                            0, run_cmd->pc.num_groups[0], dd->grain_size, 0,
                            run_cmd->pc.num_groups[1], dd->grain_size, 0,
                            run_cmd->pc.num_groups[2], dd->grain_size),
                        WorkGroupScheduler(run_cmd, dd), ap);
    } else if (dd->selected_partitioner == pocl_tbb_partitioner::AUTO) {
      tbb::parallel_for(tbb::blocked_range3d<size_t>(
                            0, run_cmd->pc.num_groups[0], dd->grain_size, 0,
                            run_cmd->pc.num_groups[1], dd->grain_size, 0,
                            run_cmd->pc.num_groups[2], dd->grain_size),
                        WorkGroupScheduler(run_cmd, dd),
                        tbb::auto_partitioner());
    } else if (dd->selected_partitioner == pocl_tbb_partitioner::SIMPLE) {
      tbb::parallel_for(tbb::blocked_range3d<size_t>(
                            0, run_cmd->pc.num_groups[0], dd->grain_size, 0,
                            run_cmd->pc.num_groups[1], dd->grain_size, 0,
                            run_cmd->pc.num_groups[2], dd->grain_size),
                        WorkGroupScheduler(run_cmd, dd),
                        tbb::simple_partitioner());
    } else if (dd->selected_partitioner == pocl_tbb_partitioner::STATIC) {
      tbb::parallel_for(tbb::blocked_range3d<size_t>(
                            0, run_cmd->pc.num_groups[0], dd->grain_size, 0,
                            run_cmd->pc.num_groups[1], dd->grain_size, 0,
                            run_cmd->pc.num_groups[2], dd->grain_size),
                        WorkGroupScheduler(run_cmd, dd),
                        tbb::static_partitioner());
    }
  } else { /* if (dd->grain_size) */
    if (dd->selected_partitioner == pocl_tbb_partitioner::NONE) {
      tbb::parallel_for(
          tbb::blocked_range3d<size_t>(0, run_cmd->pc.num_groups[0], 0,
                                       run_cmd->pc.num_groups[1], 0,
                                       run_cmd->pc.num_groups[2]),
          WorkGroupScheduler(run_cmd, dd));
    } else if (dd->selected_partitioner == pocl_tbb_partitioner::AFFINITY) {
      tbb::affinity_partitioner ap;
      tbb::parallel_for(
          tbb::blocked_range3d<size_t>(0, run_cmd->pc.num_groups[0], 0,
                                       run_cmd->pc.num_groups[1], 0,
                                       run_cmd->pc.num_groups[2]),
          WorkGroupScheduler(run_cmd, dd), ap);
    } else if (dd->selected_partitioner == pocl_tbb_partitioner::AUTO) {
      tbb::parallel_for(
          tbb::blocked_range3d<size_t>(0, run_cmd->pc.num_groups[0], 0,
                                       run_cmd->pc.num_groups[1], 0,
                                       run_cmd->pc.num_groups[2]),
          WorkGroupScheduler(run_cmd, dd), tbb::auto_partitioner());
    } else if (dd->selected_partitioner == pocl_tbb_partitioner::SIMPLE) {
      tbb::parallel_for(
          tbb::blocked_range3d<size_t>(0, run_cmd->pc.num_groups[0], 0,
                                       run_cmd->pc.num_groups[1], 0,
                                       run_cmd->pc.num_groups[2]),
          WorkGroupScheduler(run_cmd, dd), tbb::simple_partitioner());
    } else if (dd->selected_partitioner == pocl_tbb_partitioner::STATIC) {
      tbb::parallel_for(
          tbb::blocked_range3d<size_t>(0, run_cmd->pc.num_groups[0], 0,
                                       run_cmd->pc.num_groups[1], 0,
                                       run_cmd->pc.num_groups[2]),
          WorkGroupScheduler(run_cmd, dd), tbb::static_partitioner());
    }
  }
}

static int tbb_scheduler_get_work(pocl_tbb_scheduler_data *dd) {
  _cl_command_node *cmd;
  kernel_run_command *run_cmd;

  POCL_FAST_LOCK(dd->wq_lock_fast);
  int do_exit = 0;

RETRY:
  do_exit = dd->meta_thread_shutdown_requested;

  /* execute a command if available */
  cmd = dd->work_queue;
  if (cmd) {
    DL_DELETE(dd->work_queue, cmd);
    POCL_FAST_UNLOCK(dd->wq_lock_fast);

    assert(pocl_command_is_ready(cmd->sync.event.event));

    if (cmd->type == CL_COMMAND_NDRANGE_KERNEL) {
      assert((void *)dd == (void *)cmd->device->data);
      run_cmd = pocl_tbb_prepare_kernel(dd, cmd);
      if (run_cmd) {
        dd->arena.execute([run_cmd, dd]() { tbb_exec_command(dd, run_cmd); });
        finalize_kernel_command(run_cmd);
      }
    } else {
      dd->arena.execute([cmd]() { pocl_exec_command(cmd); });
    }

    POCL_FAST_LOCK(dd->wq_lock_fast);
  }

  /* if no command was available, sleep */
  if ((cmd == NULL) && (do_exit == 0)) {
    pthread_cond_wait(&dd->wake_meta_thread, &dd->wq_lock_fast);
    goto RETRY;
  }

  POCL_FAST_UNLOCK(dd->wq_lock_fast);

  return do_exit;
}

static void *pocl_tbb_driver_thread(void *dev) {
  int do_exit = 0;
  cl_device_id device = (cl_device_id)dev;
  pocl_tbb_scheduler_data *dd = (pocl_tbb_scheduler_data *)device->data;

  while (1) {
    do_exit = tbb_scheduler_get_work(dd);
    if (do_exit) {
      pthread_exit(NULL);
    }
  }
}
