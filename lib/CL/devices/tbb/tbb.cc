/* OpenCL device using the Intel TBB library (derived from the pthread device).

   Copyright (c) 2011-2012 Universidad Rey Juan Carlos
                 2011-2019 Pekka Jääskeläinen
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
#define __USE_GNU

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>

#ifndef _MSC_VER
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#include <tbb/task_arena.h>

#include "common.h"
#include "common_utils.h"
#include "config.h"
#include "devices.h"
#include "pocl_mem_management.h"
#include "pocl_util.h"
#include "tbb.h"
#include "tbb_scheduler.h"

#ifdef OCS_AVAILABLE
#include "pocl_llvm.h"
#endif

void
pocl_tbb_init_device_ops(struct pocl_device_ops *ops)
{
  pocl_pthread_init_device_ops (ops);

  ops->device_name = "cpu-tbb";

  /* implementation that differs from pthread */
  ops->probe = pocl_tbb_probe;
  ops->uninit = pocl_tbb_uninit;
  ops->reinit = pocl_tbb_reinit;
  ops->init = pocl_tbb_init;

  ops->submit = pocl_tbb_submit;
  ops->notify = pocl_tbb_notify;
}


unsigned int
pocl_tbb_probe (struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count(ops->device_name);

  /* Env was not specified, default behavior was to use 1 tbb device */
  if (env_count < 0)
    return 1;

  if (env_count > 1)
    POCL_MSG_WARN ("Using more than one tbb device which is strongly discouraged.");

  return env_count;
}

char scheduler_initialized = 0;
static cl_bool tbb_available = CL_TRUE;
static cl_bool tbb_unavailable = CL_FALSE;

cl_int
pocl_tbb_init (unsigned j, cl_device_id device, const char* parameters)
{
  device->available = &tbb_unavailable;
  cl_int err = pocl_cpu_init_common (device);
  if (err) {
    POCL_MSG_ERR ("pocl_cpu_init_common failed\n");
    return CL_INVALID_DEVICE;
  }

  /* device->max_compute_units was set up by topology_detect,
     but we use the TBB library (result should be the same).
     task_area initialization is optional and max_concurrency
     can be retrieved without prior initialization. */
  tbb::task_arena ta;
  //ta.initialize ();
  device->max_compute_units = ta.max_concurrency ();
  /* subdevices not supported ATM */
  device->max_sub_devices = 0;
  device->num_partition_properties = 0;
  device->num_partition_types = 0;

  if (!scheduler_initialized)
    {
      scheduler_initialized = 1;
      pocl_init_dlhandle_cache ();
      pocl_init_kernel_run_command_manager ();
      tbb_scheduler_init (device);
    }
  /* system mem as global memory */
  device->global_mem_id = 0;
  POCL_MSG_PRINT_INFO ("TBB device %u initialized\n", j);
  device->available = &tbb_available;
  return err;
}

cl_int
pocl_tbb_uninit (unsigned j, cl_device_id device)
{
  if (scheduler_initialized)
    {
      tbb_scheduler_uninit ();
      scheduler_initialized = 0;
    }

  device->data = NULL;
  return CL_SUCCESS;
}

cl_int
pocl_tbb_reinit (unsigned j, cl_device_id device, const char *parameters)
{
  if (!scheduler_initialized)
    {
      tbb_scheduler_init (device);
      scheduler_initialized = 1;
    }

  return CL_SUCCESS;
}

void
pocl_tbb_submit (_cl_command_node *node, cl_command_queue cq)
{
  node->ready = 1;
  if (pocl_command_is_ready (node->sync.event.event))
    {
      pocl_update_event_submitted (node->sync.event.event);
      tbb_scheduler_push_command (node);
    }
  POCL_UNLOCK_OBJ (node->sync.event.event);
  return;
}

void
pocl_tbb_notify (cl_device_id device, cl_event event, cl_event finished)
{
   int wake_thread = 0;
  _cl_command_node *node = event->command;

  if (finished->status < CL_COMPLETE)
    {
      pocl_update_event_failed (event);
      return;
    }

  if (!node->ready)
    return;

  if (pocl_command_is_ready (node->sync.event.event))
    {
      if (event->status == CL_QUEUED)
        {
          pocl_update_event_submitted (event);
          wake_thread = 1;
        }
    }
  if (wake_thread)
    {
      tbb_scheduler_push_command (node);
    }
  return;
}
