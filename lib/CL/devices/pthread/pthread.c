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

#include "pocl-pthread.h"
#include "pocl-pthread_utils.h"
#include "pocl-pthread_scheduler.h"
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

#include "pocl_runtime_config.h"
#include "utlist.h"
#include "cpuinfo.h"
#include "topology/pocl_topology.h"
#include "common.h"
#include "config.h"
#include "devices.h"
#include "pocl_util.h"
#include "pocl_mem_management.h"

#ifndef HAVE_LTDL
#error Pthread driver requires LTDL
#endif

#ifdef OCS_AVAILABLE
#include "pocl_llvm.h"
#endif

#ifdef CUSTOM_BUFFER_ALLOCATOR

#include "bufalloc.h"

/* Instead of mallocing a buffer size for a region, try to allocate 
   this many times the buffer size to hopefully avoid mallocs for 
   the next buffer allocations.
   
   Falls back to single multiple allocation if fails to allocate a
   larger region. */
#define ALLOCATION_MULTIPLE 32

/* To avoid memory hogging in case of larger buffers, limit the
   extra allocation margin to this number of megabytes.
   
   The extra allocation should be done to avoid repetitive calls and
   memory fragmentation for smaller buffers only. 
*/
#define ADDITIONAL_ALLOCATION_MAX_MB 100

/* Always create regions with at least this size to avoid allocating
   small regions when there are lots of small buffers, which would counter 
   a purpose of having own buffer management. It would end up having a lot of
   small regions with linear searches over them.  */
#define NEW_REGION_MIN_MB 10

/* Whether to immediately free a region in case the last chunk was
   deallocated. If 0, it can reuse the same region over multiple kernels. */
#define FREE_EMPTY_REGIONS 0

/* CUSTOM_BUFFER_ALLOCATOR */
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

#ifdef CUSTOM_BUFFER_ALLOCATOR
  /* Lock for protecting the mem_regions linked list. Held when new mem_regions
     are created or old ones freed. */
  mem_regions_management* mem_regions;
#endif

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
  char *name = get_llvm_cpu_name ();
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

#ifdef CUSTOM_BUFFER_ALLOCATOR
#define INIT_MEM_REGIONS                                                      \
  do                                                                          \
    {                                                                         \
      mem_regions_management *mrm;                                            \
      mrm = malloc (sizeof (mem_regions_management));                         \
      if (mrm == NULL)                                                        \
        {                                                                     \
          free (d);                                                           \
          return CL_OUT_OF_HOST_MEMORY;                                       \
        }                                                                     \
      BA_INIT_LOCK (mrm->mem_regions_lock);                                   \
      mrm->mem_regions = NULL;                                                \
      d->mem_regions = mrm;                                                   \
    }                                                                         \
  while (0)
#else
#define INIT_MEM_REGIONS NULL
#endif

#define FALLBACK_MAX_THREAD_COUNT 8

char scheduler_initialized = 0;

cl_int
pocl_pthread_init (unsigned j, cl_device_id device, const char* parameters)
{
  struct data *d;
  cl_int ret = CL_SUCCESS;
  int err;

  d = (struct data *) calloc (1, sizeof (struct data));
  if (d == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  INIT_MEM_REGIONS;

  d->current_kernel = NULL;
  device->data = d;

  pocl_init_cpu_device_infos (device);

  device->on_host_queue_props
      = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE;

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

  if (!scheduler_initialized)
    {
      scheduler_initialized = 1;
      pocl_init_dlhandle_cache();
      pocl_init_kernel_run_command_manager();
      pthread_scheduler_init (device);
    }
  /* system mem as global memory */
  device->global_mem_id = 0;
  return ret;
}

cl_int
pocl_pthread_uninit (unsigned j, cl_device_id device)
{
  struct data *d = (struct data*)device->data;

#ifdef CUSTOM_BUFFER_ALLOCATOR
  memory_region_t *region, *temp;
  DL_FOREACH_SAFE(d->mem_regions->mem_regions, region, temp)
    {
      DL_DELETE(d->mem_regions->mem_regions, region);
      free((void*)region->chunks->start_address);
      region->chunks->start_address = 0;
      POCL_MEM_FREE(region);
    }
  d->mem_regions->mem_regions = NULL;
#endif

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

  INIT_MEM_REGIONS;

  d->current_kernel = NULL;
  device->data = d;

  if (!scheduler_initialized)
    {
      pthread_scheduler_init (device);
      scheduler_initialized = 1;
    }

  return CL_SUCCESS;
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
      POCL_UPDATE_EVENT_SUBMITTED (node->event);
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
          int r = pthread_cond_wait (cq_cond, &cq->pocl_lock);
          assert (r == 0);
        }
    }
  return;
}

void
pocl_pthread_notify (cl_device_id device, cl_event event, cl_event finished)
{
   int wake_thread = 0;
  _cl_command_node * volatile node = event->command;

  if (finished->status < CL_COMPLETE)
    {
      POCL_UPDATE_EVENT_FAILED (event);
      return;
    }

  if (!node->ready)
    return;

  if (pocl_command_is_ready (node->event))
    {
      if (event->status == CL_QUEUED)
        {
          POCL_UPDATE_EVENT_SUBMITTED (event);
          wake_thread = 1;
        }
    }
  if (wake_thread)
    {
      pthread_scheduler_push_command (node);
    }
  return;
}

static void
pthread_scheduler_release_host (cl_command_queue cq)
{
  /* must be called with CQ already locked.
   * this must be a broadcast since there could be multiple
   * user threads waiting on the same command queue
   * in pthread_scheduler_wait_cq(). */
  pthread_cond_t *cq_cond = (pthread_cond_t *)cq->data;
  int r = pthread_cond_broadcast (cq_cond);
  assert (r == 0);
}

void pocl_pthread_update_event (cl_device_id device, cl_event event, cl_int status)
{
  struct event_data *e_d = NULL;

  if(event->data == NULL && status == CL_QUEUED)
    {
      e_d = malloc(sizeof(struct event_data));
      assert(e_d);

      pthread_cond_init(&e_d->event_cond, NULL);
      event->data = (void *) e_d;
    }
  else
    {
      e_d = event->data;
    }
  /* TODO: Refactor this to another function where it can be reused.
     It should be common for most device drivers? */
  switch (status)
    {
    case CL_QUEUED:
      event->status = status;
      if (event->queue->properties & CL_QUEUE_PROFILING_ENABLE)
        event->time_queue = device->ops->get_timer_value(device->data);
      break;
    case CL_SUBMITTED:
      event->status = status;
      if (event->queue->properties & CL_QUEUE_PROFILING_ENABLE)
        event->time_submit = device->ops->get_timer_value(device->data);
      break;
    case CL_RUNNING:
      event->status = status;
      if (event->queue->properties & CL_QUEUE_PROFILING_ENABLE)
        event->time_start = device->ops->get_timer_value(device->data);
      break;
    case CL_COMPLETE:
      POCL_MSG_PRINT_EVENTS ("PTHREAD: Command complete, event %d\n",
                             event->id);
      event->status = CL_COMPLETE;

      pocl_mem_objs_cleanup (event);

      if (event->queue->properties & CL_QUEUE_PROFILING_ENABLE)
        event->time_end = device->ops->get_timer_value(device->data);

      POCL_UNLOCK_OBJ (event);
      device->ops->broadcast (event);
      pocl_update_command_queue (event, pthread_scheduler_release_host);
      POCL_LOCK_OBJ (event);

      pthread_cond_broadcast(&e_d->event_cond);
      break;

    default:
      POCL_MSG_PRINT_EVENTS ("setting FAIL status on event %u\n", event->id);

      event->status = CL_FAILED;

      pocl_mem_objs_cleanup (event);

      if (event->queue->properties & CL_QUEUE_PROFILING_ENABLE)
        event->time_end = device->ops->get_timer_value (device->data);

      POCL_UNLOCK_OBJ (event);
      device->ops->broadcast (event);
      pocl_update_command_queue (event, pthread_scheduler_release_host);
      POCL_LOCK_OBJ (event);

      pthread_cond_broadcast (&e_d->event_cond);
      break;
    }
}

void pocl_pthread_wait_event (cl_device_id device, cl_event event)
{
  struct event_data *e_d = event->data;

  POCL_LOCK_OBJ (event);
  while (event->status > CL_COMPLETE)
    {
      pthread_cond_wait(&e_d->event_cond, &event->pocl_lock);
    }
  POCL_UNLOCK_OBJ (event);
}


void pocl_pthread_free_event_data (cl_event event)
{
  assert(event->data != NULL);
  free(event->data);
  event->data = NULL;
}

cl_int
pocl_pthread_init_queue (cl_command_queue queue)
{
  queue->data
      = pocl_aligned_malloc (HOST_CPU_CACHELINE_SIZE, sizeof (pthread_cond_t));
  pthread_cond_t *cond = (pthread_cond_t *)queue->data;
  int r = pthread_cond_init (cond, NULL);
  assert (r == 0);
  return CL_BUILD_SUCCESS;
}

void
pocl_pthread_free_queue (cl_command_queue queue)
{
  pthread_cond_t *cond = (pthread_cond_t *)queue->data;
  int r = pthread_cond_destroy (cond);
  assert (r == 0);
  POCL_MEM_FREE (queue->data);
}
