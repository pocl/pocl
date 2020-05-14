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

#include "config.h"
#include "utlist.h"
#include "tbb.h"
#include "tbb_utils.h"
#include "tbb_scheduler.h"
#include "pocl_runtime_config.h"
#include "common.h"
#include "devices.h"
#include "pocl_util.h"
#include "pocl_mem_management.h"

#ifndef HAVE_LIBDL
#error tbb driver requires DL library
#endif

#ifdef OCS_AVAILABLE
#include "pocl_llvm.h"
#endif

/**
 * Per event data.
 */
struct event_data {
  pthread_cond_t event_cond;
};

struct data {
  /* Currently loaded kernel. */
  cl_kernel current_kernel;
  volatile uint64_t total_cmd_exec_time;
};

void
pocl_tbb_init_device_ops(struct pocl_device_ops *ops)
{
  pocl_basic_init_device_ops(ops);

  ops->device_name = "tbb";

  /* implementation that differs from basic */
  ops->probe = pocl_tbb_probe;
  ops->uninit = pocl_tbb_uninit;
  ops->reinit = pocl_tbb_reinit;
  ops->init = pocl_tbb_init;
  ops->run = pocl_tbb_run;
  ops->join = pocl_tbb_join;
  ops->submit = pocl_tbb_submit;
  ops->notify = pocl_tbb_notify;
  ops->broadcast = pocl_broadcast;
  ops->flush = pocl_tbb_flush;
  ops->wait_event = pocl_tbb_wait_event;
  ops->notify_event_finished = pocl_tbb_notify_event_finished;
  ops->notify_cmdq_finished = pocl_tbb_notify_cmdq_finished;
  ops->update_event = pocl_tbb_update_event;
  ops->free_event_data = pocl_tbb_free_event_data;
  ops->build_hash = pocl_tbb_build_hash;

  ops->init_queue = pocl_tbb_init_queue;
  ops->free_queue = pocl_tbb_free_queue;
}

char*
pocl_tbb_build_hash (cl_device_id device)
{
  char* res = reinterpret_cast<char*>(calloc(1000, sizeof(char)));
#ifdef KERNELLIB_HOST_DISTRO_VARIANTS
  char *name = get_llvm_cpu_name ();
  snprintf (res, 1000, "tbb-%s-%s", HOST_DEVICE_BUILD_HASH, name);
  POCL_MEM_FREE (name);
#else
  snprintf (res, 1000, "tbb-%s", HOST_DEVICE_BUILD_HASH);
#endif
  return res;
}

unsigned int
pocl_tbb_probe (struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count(ops->device_name);
  /* Env was not specified, default behavior was to use 1 tbb device */
  if (env_count < 0)
    return 1;

  return env_count;
}

static cl_device_partition_property tbb_partition_properties[2]
    = { CL_DEVICE_PARTITION_EQUALLY, CL_DEVICE_PARTITION_BY_COUNTS };

#define FALLBACK_MAX_THREAD_COUNT 8

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
   * but if the user requests, lower it */
  int fallback = (device->max_compute_units == 0) ? FALLBACK_MAX_THREAD_COUNT
                                                  : device->max_compute_units;
  int max_thr = pocl_get_int_option ("POCL_MAX_TBB_COUNT", fallback);

  device->max_compute_units
      = std::max ((unsigned)max_thr,
             (unsigned)pocl_get_int_option ("POCL_TBB_MIN_THREADS", 1));

  pocl_cpuinfo_detect_device_info(device);
  pocl_set_buffer_image_limits(device);

  /* in case hwloc doesn't provide a PCI ID, let's generate
     a vendor id that hopefully is unique across vendors. */
  const char *magic = "pocl";
  if (device->vendor_id == 0)
    device->vendor_id =
      magic[0] | magic[1] << 8 | magic[2] << 16 | magic[3] << 24;

  device->vendor_id += j;

<<<<<<< HEAD
  // tbb has elementary partitioning support
  device->max_sub_devices = device->max_compute_units;
  device->num_partition_properties = 2;
  device->partition_properties = tbb_partition_properties;
  device->num_partition_types = 0;
  device->partition_type = NULL;

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
pocl_tbb_run (void *data, _cl_command_node *cmd)
{
  /* not used: this device will not be told when or what to run */
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
pocl_tbb_flush(cl_device_id device, cl_command_queue cq)
{

}

void
pocl_tbb_join(cl_device_id device, cl_command_queue cq)
{
  POCL_LOCK_OBJ (cq);
  pthread_cond_t *cq_cond = (pthread_cond_t *)cq->data;
  while (1)
    {
      if (cq->command_count == 0)
        {
          POCL_UNLOCK_OBJ (cq);
          return;
        }
      else
        {
          int r = pthread_cond_wait (cq_cond, &cq->pocl_lock);
          assert (r == 0);
        }
    }
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

void
pocl_tbb_notify_cmdq_finished (cl_command_queue cq)
{
  /* must be called with CQ already locked.
   * this must be a broadcast since there could be multiple
   * user threads waiting on the same command queue
   * in tbb_scheduler_wait_cq(). */
  pthread_cond_t *cq_cond = (pthread_cond_t *)cq->data;
  int r = pthread_cond_broadcast (cq_cond);
  assert (r == 0);
}

void
pocl_tbb_notify_event_finished (cl_event event)
{
  struct event_data* e_d = reinterpret_cast<event_data*> (event->data);
  pthread_cond_broadcast (&e_d->event_cond);
}

void
pocl_tbb_update_event (cl_device_id device, cl_event event)
{
  struct event_data* e_d = NULL;
  if (event->data == NULL && event->status == CL_QUEUED)
    {
      e_d = reinterpret_cast<event_data*> (malloc(sizeof(struct event_data)));
      assert(e_d);

      pthread_cond_init(&e_d->event_cond, NULL);
      event->data = (void *) e_d;
    }
}

void pocl_tbb_wait_event (cl_device_id device, cl_event event)
{
  struct event_data *e_d = reinterpret_cast<event_data*> (event->data);

  POCL_LOCK_OBJ (event);
  while (event->status > CL_COMPLETE)
    {
      pthread_cond_wait(&e_d->event_cond, &event->pocl_lock);
    }
  POCL_UNLOCK_OBJ (event);
}


void pocl_tbb_free_event_data (cl_event event)
{
  assert(event->data != NULL);
  free(event->data);
  event->data = NULL;
}

cl_int
pocl_tbb_init_queue (cl_command_queue queue)
{
  queue->data
      = pocl_aligned_malloc (HOST_CPU_CACHELINE_SIZE, sizeof (pthread_cond_t));
  pthread_cond_t *cond = (pthread_cond_t *)queue->data;
  int r = pthread_cond_init (cond, NULL);
  assert (r == 0);
  return CL_BUILD_SUCCESS;
}

void
pocl_tbb_free_queue (cl_command_queue queue)
{
  pthread_cond_t *cond = (pthread_cond_t *)queue->data;
  int r = pthread_cond_destroy (cond);
  assert (r == 0);
  POCL_MEM_FREE (queue->data);
}
