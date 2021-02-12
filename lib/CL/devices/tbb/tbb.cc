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

#define _GNU_SOURCE
#define __USE_GNU
#include <sched.h>

#include <algorithm>
#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>

#ifndef _MSC_VER
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

extern "C" {
  #include "cpuinfo.h"
  #include "topology/pocl_topology.h"
}

#include <tbb/task_arena.h>

#include "common.h"
#include "config.h"
#include "devices.h"
#include "pocl_mem_management.h"
#include "pocl_runtime_config.h"
#include "pocl_util.h"
#include "tbb.h"
#include "tbb_scheduler.h"
#include "tbb_utils.h"
#include "utlist.h"

#ifndef HAVE_LIBDL
#error tbb driver requires DL library
#endif

#ifdef OCS_AVAILABLE
#include "pocl_llvm.h"
#endif

struct data {
  /* Currently loaded kernel. */
  cl_kernel current_kernel;
  volatile uint64_t total_cmd_exec_time;
};

void
pocl_tbb_init_device_ops(struct pocl_device_ops *ops)
{
  pocl_pthread_init_device_ops(ops);

  ops->device_name = "tbb";

  /* implementation that differs from pthread */
  ops->uninit = pocl_tbb_uninit;
  ops->reinit = pocl_tbb_reinit;
  ops->init = pocl_tbb_init;
  ops->submit = pocl_tbb_submit;
  ops->notify = pocl_tbb_notify;
}

char scheduler_initialized = 0;

cl_int
pocl_tbb_init (unsigned j, cl_device_id device, const char* parameters)
{
  struct data *d;
  cl_int ret = CL_SUCCESS;
  int err;

  d = (struct data *) calloc (1, sizeof (struct data));
  if (d == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  d->current_kernel = NULL;
  device->data = d;

  pocl_init_default_device_infos (device);
  device->extensions = HOST_DEVICE_EXTENSIONS;

  device->on_host_queue_props
      = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE;

  /* full memory consistency model for atomic memory and fence operations
  except CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES. see
  https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#opencl-3.0-backwards-compatibility*/
  device->atomic_memory_capabilities = CL_DEVICE_ATOMIC_ORDER_RELAXED
                                       | CL_DEVICE_ATOMIC_ORDER_ACQ_REL
                                       | CL_DEVICE_ATOMIC_ORDER_SEQ_CST
                                       | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP
                                       | CL_DEVICE_ATOMIC_SCOPE_DEVICE;
  device->atomic_fence_capabilities = CL_DEVICE_ATOMIC_ORDER_RELAXED
                                       | CL_DEVICE_ATOMIC_ORDER_ACQ_REL
                                       | CL_DEVICE_ATOMIC_ORDER_SEQ_CST
                                       | CL_DEVICE_ATOMIC_SCOPE_WORK_ITEM
                                       | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP
                                       | CL_DEVICE_ATOMIC_SCOPE_DEVICE;

  /* hwloc probes OpenCL device info at its initialization in case
     the OpenCL extension is enabled. This causes to printout
     an unimplemented property error because hwloc is used to
     initialize global_mem_size which it is not yet. Just put
     a nonzero there for now. */
  device->global_mem_size = 1;
  err = pocl_topology_detect_device_info (device);
  if (err)
    ret = CL_INVALID_DEVICE;

  /* device->max_compute_units was set up by topology_detect,
     but we use the TBB library (result should be the same).
     task_area initialization is optional and max_concurrency
     can be retrieved without prior initialization. */
  tbb::task_arena ta;
  //ta.initialize();
  device->max_compute_units = ta.max_concurrency();

  pocl_cpuinfo_detect_device_info(device);
  pocl_set_buffer_image_limits(device);

  /* in case hwloc doesn't provide a PCI ID, let's generate
     a vendor id that hopefully is unique across vendors. */
  const char *magic = "pocl";
  if (device->vendor_id == 0)
    device->vendor_id =
      magic[0] | magic[1] << 8 | magic[2] << 16 | magic[3] << 24;

  device->vendor_id += j;

  if (!scheduler_initialized)
    {
      scheduler_initialized = 1;
      pocl_init_dlhandle_cache();
      pocl_init_kernel_run_command_manager();
      tbb_scheduler_init (device);
    }
  /* system mem as global memory */
  device->global_mem_id = 0;
  return ret;
}

cl_int
pocl_tbb_uninit (unsigned j, cl_device_id device)
{
  struct data *d = (struct data*)device->data;

  if (scheduler_initialized)
    {
      tbb_scheduler_uninit ();
      scheduler_initialized = 0;
    }

  POCL_MEM_FREE(d);
  device->data = NULL;
  return CL_SUCCESS;
}

cl_int
pocl_tbb_reinit (unsigned j, cl_device_id device)
{
  struct data *d;

  d = (struct data *)calloc (1, sizeof (struct data));
  if (d == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  d->current_kernel = NULL;
  device->data = d;

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
  if (pocl_command_is_ready (node->event))
    {
      pocl_update_event_submitted (node->event);
      tbb_scheduler_push_command (node);
    }
  POCL_UNLOCK_OBJ (node->event);
  return;
}

void
pocl_tbb_notify (cl_device_id device, cl_event event, cl_event finished)
{
   int wake_thread = 0;
  _cl_command_node * volatile node = event->command;

  if (finished->status < CL_COMPLETE)
    {
      pocl_update_event_failed (event);
      return;
    }

  if (!node->ready)
    return;

  if (pocl_command_is_ready (node->event))
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
