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

// required for older versions of TBB
#define TBB_PREVIEW_NUMA_SUPPORT 1

#include <tbb/blocked_range3d.h>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>
#include <tbb/task_arena.h>

#include "common.h"
#include "common_driver.h"
#include "common_utils.h"
#include "pocl_builtin_kernels.h"
#include "pocl_cl.h"
#include "pocl_mem_management.h"
#include "pocl_runtime_config.h"
#include "pocl_util.h"
#include "tbb_scheduler.h"
#include "utlist.h"

/* External functions declared in tbb_dd->h */

struct TBBArena {
  tbb::numa_node_id NumaIdx;
  tbb::task_arena Arena;
};

static std::vector<tbb::numa_node_id> NumaIndexes;

static unsigned LastInitializedNumaIndex = 0;

size_t tbb_get_numa_nodes() {
  NumaIndexes = tbb::info::numa_nodes();
  return NumaIndexes.size();
}

void tbb_init_arena(pocl_tbb_scheduler_data *SchedData, int OnePerNode) {
  TBBArena *TBBA = new TBBArena;
  SchedData->tbb_arena = TBBA;
  if (OnePerNode) {
    assert(LastInitializedNumaIndex < NumaIndexes.size());
    TBBA->NumaIdx = NumaIndexes[LastInitializedNumaIndex];
    TBBA->Arena.initialize(tbb::task_arena::constraints(TBBA->NumaIdx));
  } else {
    TBBA->NumaIdx = UINT32_MAX;
    TBBA->Arena.initialize();
  }
  ++LastInitializedNumaIndex;
}

size_t tbb_get_num_threads(pocl_tbb_scheduler_data *SchedData) {
  TBBArena *TBBA = SchedData->tbb_arena;
  return TBBA->Arena.max_concurrency();
}
/* Internal functions */

/* The sole purpose of this embedded class is to provide a function object that
 * can be executed on a blocked_range by TBB's parallel_for() according to
 * https://www.threadingbuildingblocks.org/docs/help/tbb_userguide/parallel_for.html */
class WorkGroupScheduler {
  kernel_run_command *RunCmd;
  const pocl_tbb_scheduler_data *SchedData;

public:
  void operator()(const tbb::blocked_range3d<size_t> &r) const {
    kernel_run_command *K = RunCmd;
    pocl_kernel_metadata_t *Meta = K->kernel->meta;
    const size_t CurThreadID = tbb::this_task_arena::current_thread_index();
    void *Arguments[Meta->num_args + Meta->num_locals + 1];
    void *Arguments2[Meta->num_args + Meta->num_locals + 1];
    char *LocalMem = SchedData->local_mem_global_ptr +
                     (SchedData->local_mem_size * CurThreadID);
    uchar *PrintfBuffer = SchedData->printf_buf_global_ptr +
                          (SchedData->printf_buf_size * CurThreadID);
    struct pocl_context PC;

    pocl_setup_kernel_arg_array_with_locals((void **)&Arguments,
                                            (void **)&Arguments2, K, LocalMem,
                                            SchedData->local_mem_size);
    memcpy(&PC, &K->pc, sizeof(struct pocl_context));

    // capacity and position already set up
    PC.printf_buffer = PrintfBuffer;
    uint32_t Position = 0;
    PC.printf_buffer_position = &Position;
    assert(PC.printf_buffer != NULL);
    assert(PC.printf_buffer_capacity > 0);
    assert(PC.printf_buffer_position != NULL);

    /* Flush to zero is only set once at the start of the kernel execution
     * because FTZ is a compilation option. */
    unsigned Flush = K->kernel->program->flush_denorms;
    pocl_set_ftz(Flush);

    for (size_t X = r.pages().begin(); X != r.pages().end(); X++) {
      for (size_t Y = r.rows().begin(); Y != r.rows().end(); Y++) {
        for (size_t Z = r.cols().begin(); Z != r.cols().end(); Z++) {
          /* Rounding mode must be reset after every iteration
           * since it can be changed during kernel execution. */
          pocl_set_default_rm();
          K->workgroup((uint8_t *)Arguments, (uint8_t *)&PC, X, Y, Z);
        }
      }
    }

#ifndef ENABLE_PRINTF_IMMEDIATE_FLUSH
    pocl_write_printf_buffer((char *)PC.printf_buffer, Position);
#endif

    pocl_free_kernel_arg_array_with_locals((void **)&Arguments,
                                           (void **)&Arguments2, K);
  }
  WorkGroupScheduler(kernel_run_command *K, const pocl_tbb_scheduler_data *D)
      : RunCmd(K), SchedData(D) {}
};

static void finalizeKernelCommand(kernel_run_command *RunCmd) {
  pocl_free_kernel_arg_array(RunCmd);

  pocl_release_dlhandle_cache(RunCmd->cmd->command.run.device_data);

  POCL_UPDATE_EVENT_COMPLETE_MSG(RunCmd->cmd->sync.event.event,
                                 "NDRange Kernel        ");
  free_kernel_run_command(RunCmd);
}

static kernel_run_command *
prepareKernelCommand(pocl_tbb_scheduler_data *SchedData,
                     _cl_command_node *Cmd) {

  kernel_run_command *RunCmd;
  cl_kernel Kernel = Cmd->command.run.kernel;
  struct pocl_context *PC = &Cmd->command.run.pc;
  cl_program Program = Kernel->program;
  cl_uint DevI = Cmd->program_device_i;

  size_t NumGroups = PC->num_groups[0] * PC->num_groups[1] * PC->num_groups[2];

  // if WGs == 0, return early to avoid having to uninitialize anything
  if (NumGroups == 0) {
    pocl_update_event_running(Cmd->sync.event.event);

    POCL_UPDATE_EVENT_COMPLETE_MSG(Cmd->sync.event.event,
                                   "NDRange Kernel        ");

    return NULL;
  }

  /* initialize the program gvars if required */
  pocl_driver_build_gvar_init_kernel(Program, DevI, Cmd->device,
                                     pocl_cpu_gvar_init_callback);

  char *SavedName = NULL;
  pocl_sanitize_builtin_kernel_name(Kernel, &SavedName);
  void *ci = pocl_check_kernel_dlhandle_cache(Cmd, CL_TRUE, CL_TRUE);
  Cmd->command.run.device_data = ci;
  pocl_restore_builtin_kernel_name(Kernel, SavedName);

  RunCmd = new_kernel_run_command();
  RunCmd->data = SchedData;
  RunCmd->kernel = Kernel;
  RunCmd->device = Cmd->device;
  RunCmd->pc = *PC;
  RunCmd->cmd = Cmd;
  RunCmd->pc.printf_buffer = NULL;
  RunCmd->pc.printf_buffer_capacity = SchedData->printf_buf_size;
  RunCmd->pc.printf_buffer_position = NULL;
  RunCmd->pc.global_var_buffer = (uchar *)Program->gvar_storage[DevI];
  RunCmd->workgroup =
      reinterpret_cast<pocl_workgroup_func>(Cmd->command.run.wg);
  RunCmd->kernel_args = Cmd->command.run.arguments;
  RunCmd->next = NULL;

  pocl_setup_kernel_arg_array(RunCmd);

  pocl_update_event_running(Cmd->sync.event.event);

  return RunCmd;
}

static void execCommand(pocl_tbb_scheduler_data *SchedData,
                        kernel_run_command *RunCmd) {
  /* Note: Grain size variation could be allowed for each dimension
   * individually. */
  if (SchedData->grain_size) {
    if (SchedData->selected_partitioner ==
        pocl_tbb_partitioner::TBB_PART_NONE) {
      tbb::parallel_for(tbb::blocked_range3d<size_t>(
                            0, RunCmd->pc.num_groups[0], SchedData->grain_size,
                            0, RunCmd->pc.num_groups[1], SchedData->grain_size,
                            0, RunCmd->pc.num_groups[2], SchedData->grain_size),
                        WorkGroupScheduler(RunCmd, SchedData));
    } else if (SchedData->selected_partitioner ==
               pocl_tbb_partitioner::TBB_PART_AFFINITY) {
      tbb::affinity_partitioner AP;
      tbb::parallel_for(tbb::blocked_range3d<size_t>(
                            0, RunCmd->pc.num_groups[0], SchedData->grain_size,
                            0, RunCmd->pc.num_groups[1], SchedData->grain_size,
                            0, RunCmd->pc.num_groups[2], SchedData->grain_size),
                        WorkGroupScheduler(RunCmd, SchedData), AP);
    } else if (SchedData->selected_partitioner ==
               pocl_tbb_partitioner::TBB_PART_AUTO) {
      tbb::parallel_for(tbb::blocked_range3d<size_t>(
                            0, RunCmd->pc.num_groups[0], SchedData->grain_size,
                            0, RunCmd->pc.num_groups[1], SchedData->grain_size,
                            0, RunCmd->pc.num_groups[2], SchedData->grain_size),
                        WorkGroupScheduler(RunCmd, SchedData),
                        tbb::auto_partitioner());
    } else if (SchedData->selected_partitioner ==
               pocl_tbb_partitioner::TBB_PART_SIMPLE) {
      tbb::parallel_for(tbb::blocked_range3d<size_t>(
                            0, RunCmd->pc.num_groups[0], SchedData->grain_size,
                            0, RunCmd->pc.num_groups[1], SchedData->grain_size,
                            0, RunCmd->pc.num_groups[2], SchedData->grain_size),
                        WorkGroupScheduler(RunCmd, SchedData),
                        tbb::simple_partitioner());
    } else if (SchedData->selected_partitioner ==
               pocl_tbb_partitioner::TBB_PART_STATIC) {
      tbb::parallel_for(tbb::blocked_range3d<size_t>(
                            0, RunCmd->pc.num_groups[0], SchedData->grain_size,
                            0, RunCmd->pc.num_groups[1], SchedData->grain_size,
                            0, RunCmd->pc.num_groups[2], SchedData->grain_size),
                        WorkGroupScheduler(RunCmd, SchedData),
                        tbb::static_partitioner());
    }
  } else { /* if (SchedData->grain_size) */
    if (SchedData->selected_partitioner ==
        pocl_tbb_partitioner::TBB_PART_NONE) {
      tbb::parallel_for(
          tbb::blocked_range3d<size_t>(0, RunCmd->pc.num_groups[0], 0,
                                       RunCmd->pc.num_groups[1], 0,
                                       RunCmd->pc.num_groups[2]),
          WorkGroupScheduler(RunCmd, SchedData));
    } else if (SchedData->selected_partitioner ==
               pocl_tbb_partitioner::TBB_PART_AFFINITY) {
      tbb::affinity_partitioner AP;
      tbb::parallel_for(
          tbb::blocked_range3d<size_t>(0, RunCmd->pc.num_groups[0], 0,
                                       RunCmd->pc.num_groups[1], 0,
                                       RunCmd->pc.num_groups[2]),
          WorkGroupScheduler(RunCmd, SchedData), AP);
    } else if (SchedData->selected_partitioner ==
               pocl_tbb_partitioner::TBB_PART_AUTO) {
      tbb::parallel_for(
          tbb::blocked_range3d<size_t>(0, RunCmd->pc.num_groups[0], 0,
                                       RunCmd->pc.num_groups[1], 0,
                                       RunCmd->pc.num_groups[2]),
          WorkGroupScheduler(RunCmd, SchedData), tbb::auto_partitioner());
    } else if (SchedData->selected_partitioner ==
               pocl_tbb_partitioner::TBB_PART_SIMPLE) {
      tbb::parallel_for(
          tbb::blocked_range3d<size_t>(0, RunCmd->pc.num_groups[0], 0,
                                       RunCmd->pc.num_groups[1], 0,
                                       RunCmd->pc.num_groups[2]),
          WorkGroupScheduler(RunCmd, SchedData), tbb::simple_partitioner());
    } else if (SchedData->selected_partitioner ==
               pocl_tbb_partitioner::TBB_PART_STATIC) {
      tbb::parallel_for(
          tbb::blocked_range3d<size_t>(0, RunCmd->pc.num_groups[0], 0,
                                       RunCmd->pc.num_groups[1], 0,
                                       RunCmd->pc.num_groups[2]),
          WorkGroupScheduler(RunCmd, SchedData), tbb::static_partitioner());
    }
  }
}

static int runSingleCommand(pocl_tbb_scheduler_data *SchedData) {
  _cl_command_node *Cmd;
  kernel_run_command *RunCmd;
  TBBArena *TBBA = SchedData->tbb_arena;

  POCL_FAST_LOCK(SchedData->wq_lock_fast);
  int DoExit = 0;

RETRY:
  DoExit = SchedData->meta_thread_shutdown_requested;

  /* execute a command if available */
  Cmd = SchedData->work_queue;
  if (Cmd) {
    DL_DELETE(SchedData->work_queue, Cmd);
    POCL_FAST_UNLOCK(SchedData->wq_lock_fast);

    assert(pocl_command_is_ready(Cmd->sync.event.event));

    if (Cmd->type == CL_COMMAND_NDRANGE_KERNEL) {
      assert((void *)SchedData == (void *)Cmd->device->data);
      RunCmd = prepareKernelCommand(SchedData, Cmd);
      if (RunCmd) {
        TBBA->Arena.execute(
            [RunCmd, SchedData]() { execCommand(SchedData, RunCmd); });
        finalizeKernelCommand(RunCmd);
      }
    } else {
      TBBA->Arena.execute([Cmd]() { pocl_exec_command(Cmd); });
    }

    POCL_FAST_LOCK(SchedData->wq_lock_fast);
  }

  /* if no command was available, sleep */
  if ((Cmd == NULL) && (DoExit == 0)) {
    pthread_cond_wait(&SchedData->wake_meta_thread, &SchedData->wq_lock_fast);
    goto RETRY;
  }

  POCL_FAST_UNLOCK(SchedData->wq_lock_fast);

  return DoExit;
}

void *TBBDriverThread(void *Dev) {
  int DoExit = 0;
  cl_device_id Device = (cl_device_id)Dev;
  pocl_tbb_scheduler_data *SchedData = (pocl_tbb_scheduler_data *)Device->data;

  while (1) {
    DoExit = runSingleCommand(SchedData);
    if (DoExit) {
      pthread_exit(NULL);
    }
  }
}
