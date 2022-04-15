/* OpenCL native pthreaded device implementation.

   Copyright (c) 2011-2012 Universidad Rey Juan Carlos
                 2011-2019 Pekka Jääskeläinen

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

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>

#ifndef _WIN32
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#include "config.h"
#include "utlist.h"
#include "pocl-pthread.h"
#include "pocl-pthread_utils.h"
#include "pocl-pthread_scheduler.h"
#include "pocl_runtime_config.h"
#include "cpuinfo.h"
#include "topology/pocl_topology.h"
#include "common.h"
#include "devices.h"
#include "pocl_util.h"
#include "pocl_mem_management.h"

#ifdef ENABLE_LLVM
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
pocl_pthread_init_device_ops(struct pocl_device_ops *ops)
{
  pocl_basic_init_device_ops(ops);

  ops->device_name = "pthread";

  /* implementation that differs from basic */
  ops->probe = pocl_pthread_probe;
  ops->uninit = pocl_pthread_uninit;
  ops->reinit = pocl_pthread_reinit;
  ops->init = pocl_pthread_init;
  ops->run = pocl_pthread_run;
  ops->join = pocl_pthread_join;
  ops->submit = pocl_pthread_submit;
  ops->notify = pocl_pthread_notify;
  ops->broadcast = pocl_broadcast;
  ops->flush = pocl_pthread_flush;
  ops->wait_event = pocl_pthread_wait_event;
  ops->notify_event_finished = pocl_pthread_notify_event_finished;
  ops->notify_cmdq_finished = pocl_pthread_notify_cmdq_finished;
  ops->update_event = pocl_pthread_update_event;
  ops->free_event_data = pocl_pthread_free_event_data;
  ops->build_hash = pocl_pthread_build_hash;

  ops->init_queue = pocl_pthread_init_queue;
  ops->free_queue = pocl_pthread_free_queue;
}

char *
pocl_pthread_build_hash (cl_device_id device)
{
  char* res = calloc(1000, sizeof(char));
#ifdef KERNELLIB_HOST_DISTRO_VARIANTS
  char *name = pocl_get_llvm_cpu_name ();
  snprintf (res, 1000, "pthread-%s-%s", HOST_DEVICE_BUILD_HASH, name);
  POCL_MEM_FREE (name);
#else
  snprintf (res, 1000, "pthread-%s", HOST_DEVICE_BUILD_HASH);
#endif
  return res;
}

unsigned int
pocl_pthread_probe (struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count(ops->device_name);
  /* Env was not specified, default behavior was to use 1 pthread device */
  if (env_count < 0)
    return 1;

  return env_count;
}

static cl_device_partition_property pthread_partition_properties[2]
    = { CL_DEVICE_PARTITION_EQUALLY, CL_DEVICE_PARTITION_BY_COUNTS };

#define FALLBACK_MAX_THREAD_COUNT 8

static int scheduler_initialized = 0;

cl_int
pocl_pthread_init (unsigned j, cl_device_id device, const char* parameters)
{
  struct data *d;
  int err;

  d = (struct data *) calloc (1, sizeof (struct data));
  if (d == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  d->current_kernel = NULL;
  device->data = d;

  pocl_init_default_device_infos (device);
  /* 0 is the host memory shared with all drivers that use it */
  device->global_mem_id = 0;

  device->version_of_latest_passed_cts = HOST_DEVICE_LATEST_CTS_PASS;
  device->extensions = HOST_DEVICE_EXTENSIONS;

#if (HOST_DEVICE_CL_VERSION_MAJOR >= 3)
  device->features = HOST_DEVICE_FEATURES_30;

  pocl_setup_opencl_c_with_version (device, CL_TRUE);
  pocl_setup_features_with_version (device);
#else
  pocl_setup_opencl_c_with_version (device, CL_FALSE);
#endif

  pocl_setup_extensions_with_version (device);

  /* builtin kernels.. skip, basic/pthread doesn't have any
  pocl_setup_builtin_kernels_with_version (device); */

  pocl_setup_ils_with_version (device);

  device->on_host_queue_props
      = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE;

#if (!defined(ENABLE_CONFORMANCE)                                             \
     || (defined(ENABLE_CONFORMANCE) && (HOST_DEVICE_CL_VERSION_MAJOR >= 3)))
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

  device->svm_allocation_priority = 1;

  /* OpenCL 2.0 properties */
  device->svm_caps = CL_DEVICE_SVM_COARSE_GRAIN_BUFFER
                     | CL_DEVICE_SVM_FINE_GRAIN_BUFFER
                     | CL_DEVICE_SVM_ATOMICS;
#endif

  /* hwloc probes OpenCL device info at its initialization in case
     the OpenCL extension is enabled. This causes to printout 
     an unimplemented property error because hwloc is used to
     initialize global_mem_size which it is not yet. Just put 
     a nonzero there for now. */
  err = pocl_topology_detect_device_info (device);
  if (err)
    return CL_INVALID_DEVICE;

  /* device->max_compute_units was set up by topology_detect,
   * but if the user requests, lower it */
  int fallback = (device->max_compute_units == 0) ? FALLBACK_MAX_THREAD_COUNT
                                                  : device->max_compute_units;
  int max_thr = pocl_get_int_option ("POCL_MAX_PTHREAD_COUNT", fallback);

  device->max_compute_units
      = max ((unsigned)max_thr,
             (unsigned)pocl_get_int_option ("POCL_PTHREAD_MIN_THREADS", 1));

  pocl_cpuinfo_detect_device_info(device);
  pocl_set_buffer_image_limits(device);

  /* in case hwloc doesn't provide a PCI ID, let's generate
     a vendor id that hopefully is unique across vendors. */
  const char *magic = "pocl";
  if (device->vendor_id == 0)
    device->vendor_id =
      magic[0] | magic[1] << 8 | magic[2] << 16 | magic[3] << 24;

  device->vendor_id += j;

  // pthread has elementary partitioning support
  device->max_sub_devices = device->max_compute_units;
  device->num_partition_properties = 2;
  device->partition_properties = pthread_partition_properties;
  device->num_partition_types = 0;
  device->partition_type = NULL;

  cl_int ret = CL_SUCCESS;
  if (!scheduler_initialized)
    {
      pocl_init_dlhandle_cache();
      pocl_init_kernel_run_command_manager();
      ret = pthread_scheduler_init (device);
      if (ret == CL_SUCCESS)
        {
          scheduler_initialized = 1;
        }
    }

  return ret;
}

cl_int
pocl_pthread_uninit (unsigned j, cl_device_id device)
{
  struct data *d = (struct data*)device->data;

  if (scheduler_initialized)
    {
      pthread_scheduler_uninit ();
      scheduler_initialized = 0;
    }

  POCL_MEM_FREE(d);
  device->data = NULL;
  return CL_SUCCESS;
}

cl_int
pocl_pthread_reinit (unsigned j, cl_device_id device)
{
  struct data *d;

  d = (struct data *)calloc (1, sizeof (struct data));
  if (d == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  d->current_kernel = NULL;
  device->data = d;

  cl_int ret = CL_SUCCESS;
  if (!scheduler_initialized)
    {
      ret = pthread_scheduler_init (device);
      if (ret == CL_SUCCESS)
        {
          scheduler_initialized = 1;
        }
    }

  return ret;
}

void
pocl_pthread_run (void *data, _cl_command_node *cmd)
{
  /* not used: this device will not be told when or what to run */
}

void
pocl_pthread_submit (_cl_command_node *node, cl_command_queue cq)
{
  node->ready = 1;
  if (pocl_command_is_ready (node->event))
    {
      pocl_update_event_submitted (node->event);
      pthread_scheduler_push_command (node);
    }
  POCL_UNLOCK_OBJ (node->event);
  return;
}

void
pocl_pthread_flush(cl_device_id device, cl_command_queue cq)
{

}

void
pocl_pthread_join(cl_device_id device, cl_command_queue cq)
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
          PTHREAD_CHECK (pthread_cond_wait (cq_cond, &cq->pocl_lock));
        }
    }
  return;
}

void
pocl_pthread_notify (cl_device_id device, cl_event event, cl_event finished)
{
  _cl_command_node *node = event->command;

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
          pthread_scheduler_push_command (node);
        }
    }

  return;
}

void
pocl_pthread_notify_cmdq_finished (cl_command_queue cq)
{
  /* must be called with CQ already locked.
   * this must be a broadcast since there could be multiple
   * user threads waiting on the same command queue
   * in pthread_scheduler_wait_cq(). */
  pthread_cond_t *cq_cond = (pthread_cond_t *)cq->data;
  PTHREAD_CHECK (pthread_cond_broadcast (cq_cond));
}

void
pocl_pthread_notify_event_finished (cl_event event)
{
  struct event_data *e_d = event->data;
  PTHREAD_CHECK (pthread_cond_broadcast (&e_d->event_cond));
}

void
pocl_pthread_update_event (cl_device_id device, cl_event event)
{
  struct event_data *e_d = NULL;
  if (event->data == NULL && event->status == CL_QUEUED)
    {
      e_d = malloc(sizeof(struct event_data));
      assert(e_d);

      PTHREAD_CHECK (pthread_cond_init (&e_d->event_cond, NULL));
      event->data = (void *) e_d;
    }
}

void pocl_pthread_wait_event (cl_device_id device, cl_event event)
{
  struct event_data *e_d = event->data;

  POCL_LOCK_OBJ (event);
  while (event->status > CL_COMPLETE)
    {
      PTHREAD_CHECK (pthread_cond_wait (&e_d->event_cond, &event->pocl_lock));
    }
  POCL_UNLOCK_OBJ (event);
}


void pocl_pthread_free_event_data (cl_event event)
{
  assert(event->data != NULL);
  free(event->data);
  event->data = NULL;
}

int
pocl_pthread_init_queue (cl_device_id device, cl_command_queue queue)
{
  queue->data
      = pocl_aligned_malloc (HOST_CPU_CACHELINE_SIZE, sizeof (pthread_cond_t));
  pthread_cond_t *cond = (pthread_cond_t *)queue->data;
  PTHREAD_CHECK (pthread_cond_init (cond, NULL));
  return CL_SUCCESS;
}

int
pocl_pthread_free_queue (cl_device_id device, cl_command_queue queue)
{
  pthread_cond_t *cond = (pthread_cond_t *)queue->data;
  PTHREAD_CHECK (pthread_cond_destroy (cond));
  POCL_MEM_FREE (queue->data);
  return CL_SUCCESS;
}
