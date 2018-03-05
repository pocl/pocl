/* OpenCL native pthreaded device implementation.

   Copyright (c) 2011-2012 Universidad Rey Juan Carlos and
                           Pekka Jääskeläinen / Tampere Univ. of Technology
   
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
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

#ifdef OCS_AVAILABLE
#include "pocl_llvm.h"
#endif

//#define DEBUG_MT

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

/* The name of the environment variable used to force a certain max thread count
   for the thread execution. */
#define THREAD_COUNT_ENV "POCL_MAX_PTHREAD_COUNT"

/**
 * Per event data.
 */
struct event_data {
  pthread_cond_t event_cond;
};

struct data {
  /* Currently loaded kernel. */
  cl_kernel current_kernel;
  /* Loaded kernel dynamic library handle. */
  lt_dlhandle current_dlhandle;
  volatile uint64_t total_cmd_exec_time;

#ifdef CUSTOM_BUFFER_ALLOCATOR
  /* Lock for protecting the mem_regions linked list. Held when new mem_regions
     are created or old ones freed. */
  mem_regions_management* mem_regions;
#endif

};

static size_t get_max_thread_count();

void
pocl_pthread_init_device_ops(struct pocl_device_ops *ops)
{
  pocl_basic_init_device_ops(ops);

  ops->device_name = "pthread";

  /* implementation */
  ops->probe = pocl_pthread_probe;
  ops->init_device_infos = pocl_pthread_init_device_infos;
  ops->uninit = pocl_pthread_uninit;
  ops->reinit = pocl_pthread_reinit;
  ops->init = pocl_pthread_init;
  ops->alloc_mem_obj = pocl_basic_alloc_mem_obj;
  ops->free = pocl_basic_free;
  ops->read = pocl_pthread_read;
  ops->write = pocl_pthread_write;
  ops->copy = pocl_pthread_copy;
  ops->copy_rect = pocl_basic_copy_rect;
  ops->run = pocl_pthread_run;
  ops->join = pocl_pthread_join;
  ops->submit = pocl_pthread_submit;
  ops->compile_kernel = pocl_basic_compile_kernel;
  ops->notify = pocl_pthread_notify;
  ops->broadcast = pocl_broadcast;
  ops->flush = pocl_pthread_flush;
  ops->wait_event = pocl_pthread_wait_event;
  ops->update_event = pocl_pthread_update_event;
  ops->free_event_data = pocl_pthread_free_event_data;
  ops->build_hash = pocl_pthread_build_hash;
}

char *
pocl_pthread_build_hash (cl_device_id device)
{
  char* res = calloc(1000, sizeof(char));
#ifdef KERNELLIB_HOST_DISTRO_VARIANTS
  char *name = get_cpu_name ();
  snprintf (res, 1000, "pthread-%s-%s", HOST_DEVICE_BUILD_HASH, name);
  POCL_MEM_FREE (name);
#else
  snprintf (res, 1000, "pthread-%s", HOST_DEVICE_BUILD_HASH);
#endif
  return res;
}

unsigned int
pocl_pthread_probe(struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count(ops->device_name);
  /* Env was not specified, default behavior was to use 1 pthread device */
  if(env_count < 0)
    return 1;

  return env_count;
}

void
pocl_pthread_init_device_infos(unsigned j, struct _cl_device_id* dev)
{
  pocl_basic_init_device_infos(j, dev);
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

cl_int
pocl_pthread_init (unsigned j, cl_device_id device, const char* parameters)
{
  struct data *d;
  cl_int ret = CL_SUCCESS;
  int err;
  static char scheduler_initialized = 0;

  d = (struct data *) calloc (1, sizeof (struct data));
  if (d == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  INIT_MEM_REGIONS;

  d->current_kernel = NULL;
  d->current_dlhandle = 0;
  device->data = d;

  device->address_bits = sizeof(void*) * 8;

  device->min_data_type_align_size = MAX_EXTENDED_ALIGNMENT; // this is in bytes
  device->mem_base_addr_align = MAX_EXTENDED_ALIGNMENT*8; // this is in bits

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
  device->max_compute_units
      = max (get_max_thread_count (device),
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
pocl_pthread_uninit (cl_device_id device)
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

  pthread_scheduler_uninit ();

  POCL_MEM_FREE(d);
  device->data = NULL;
  return CL_SUCCESS;
}

cl_int
pocl_pthread_reinit (cl_device_id device)
{
  struct data *d;

  d = (struct data *)calloc (1, sizeof (struct data));
  if (d == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  INIT_MEM_REGIONS;

  d->current_kernel = NULL;
  d->current_dlhandle = 0;
  device->data = d;

  pthread_scheduler_init (device);

  return CL_SUCCESS;
}

void
pocl_pthread_read (void *data, void *host_ptr, const void *device_ptr,
                   size_t offset, size_t cb)
{
  if (host_ptr == device_ptr)
    return;

  memcpy (host_ptr, (char*)device_ptr + offset, cb);
}

void
pocl_pthread_write (void *data, const void *host_ptr, void *device_ptr,
                    size_t offset, size_t cb)
{
  if (host_ptr == device_ptr)
    return;

  memcpy ((char*)device_ptr + offset, host_ptr, cb);
}

void
pocl_pthread_copy (void *data, const void *src_ptr, size_t src_offset,
                   void *__restrict__ dst_ptr, size_t dst_offset, size_t cb)
{
  if (src_ptr == dst_ptr)
    return;

  memcpy ((char*)dst_ptr + dst_offset, (char*)src_ptr + src_offset, cb);
}

#define FALLBACK_MAX_THREAD_COUNT 8
//#define DEBUG_MT
//#define DEBUG_MAX_THREAD_COUNT
/**
 * Return an estimate for the maximum thread count that should produce
 * the maximum parallelism without extra threading overheads.
 */
static size_t
get_max_thread_count (cl_device_id device)
{
  /* if return THREAD_COUNT_ENV if set,
     else return fallback or max_compute_units */
  if (device->max_compute_units == 0)
    return pocl_get_int_option (THREAD_COUNT_ENV, FALLBACK_MAX_THREAD_COUNT);
  else
    return pocl_get_int_option (THREAD_COUNT_ENV,
                                (pocl_real_dev (device))->max_compute_units);
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
  pthread_scheduler_wait_cq (cq);
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

void pocl_pthread_update_event (cl_device_id device, cl_event event, cl_int status)
{
  struct event_data *e_d = NULL;
  int cq_ready = 0;

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
      cq_ready = pocl_update_command_queue (event);
      if (cq_ready)
        pthread_scheduler_release_host ();
      POCL_LOCK_OBJ (event);

      pthread_cond_signal(&e_d->event_cond);
      break;

    default:
      POCL_MSG_PRINT_EVENTS ("setting FAIL status on event %u\n", event->id);

      event->status = CL_FAILED;

      pocl_mem_objs_cleanup (event);

      if (event->queue->properties & CL_QUEUE_PROFILING_ENABLE)
        event->time_end = device->ops->get_timer_value (device->data);

      POCL_UNLOCK_OBJ (event);
      device->ops->broadcast (event);
      cq_ready = pocl_update_command_queue (event);
      if (cq_ready)
        pthread_scheduler_release_host ();
      POCL_LOCK_OBJ (event);

      pthread_cond_signal (&e_d->event_cond);
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

