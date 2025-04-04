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

#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#include "common.h"
#include "common_utils.h"
#include "config.h"
#include "devices.h"
#include "pocl-pthread.h"
#include "pocl-pthread_scheduler.h"
#include "pocl_mem_management.h"
#include "pocl_util.h"

#ifdef ENABLE_LLVM
#include "pocl_llvm.h"
#endif

/**
 * Per event data.
 */
typedef struct event_data
{
  pocl_cond_t event_cond;
} event_data;

typedef struct queue_data
{
  pocl_cond_t cq_cond;
} queue_data;

void
pocl_pthread_init_device_ops(struct pocl_device_ops *ops)
{
  pocl_basic_init_device_ops(ops);

  ops->device_name = "cpu";

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

  ops->init_queue = pocl_pthread_init_queue;
  ops->free_queue = pocl_pthread_free_queue;
}

unsigned int
pocl_pthread_probe (struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count(ops->device_name);

  pocl_cpu_probe ();

  /* for backwards compatibility */
  if (env_count <= 0)
    env_count = pocl_device_get_env_count("pthread");

  /* Env was not specified, default behavior was to use 1 pthread device
   * unless tbb device is being built. */
  if (env_count < 0)
#ifdef BUILD_TBB
    return 0;
#else
    return 1;
#endif

  return env_count;
}

static cl_device_partition_property pthread_partition_properties[2]
    = { CL_DEVICE_PARTITION_EQUALLY, CL_DEVICE_PARTITION_BY_COUNTS };
static int scheduler_initialized = 0;

static cl_bool pthread_available = CL_TRUE;
static cl_bool pthread_unavailable = CL_FALSE;

cl_int
pocl_pthread_init (unsigned j, cl_device_id device, const char* parameters)
{
  int err;

  device->data = NULL;
  device->available = &pthread_unavailable;

  cl_int ret = pocl_cpu_init_common (device);
  if (ret != CL_SUCCESS)
    return ret;

  pocl_init_dlhandle_cache ();
  pocl_init_kernel_run_command_manager ();

  /* pthread has elementary partitioning support,
   * but only if OpenMP is disabled */
#if  defined(ENABLE_HOST_CPU_DEVICES_OPENMP) || defined(ENABLE_CONFORMANCE)
  device->max_sub_devices = 0;
  device->num_partition_properties = 0;
  device->num_partition_types = 0;
  device->partition_type = NULL;
  device->partition_properties = NULL;
#else
  device->max_sub_devices = device->max_compute_units;
  device->num_partition_properties = 2;
  device->partition_properties = pthread_partition_properties;
  device->num_partition_types = 0;
  device->partition_type = NULL;
#endif

  if (!scheduler_initialized)
    {
      ret = pthread_scheduler_init (device);
      if (ret == CL_SUCCESS)
        {
          scheduler_initialized = 1;
        }
    }

  device->available = &pthread_available;

  return ret;
}

cl_int
pocl_pthread_uninit (unsigned j, cl_device_id device)
{
  if (scheduler_initialized)
    {
      pthread_scheduler_uninit ();
      scheduler_initialized = 0;
    }

  POCL_MEM_FREE (device->data);
  return CL_SUCCESS;
}

#ifdef ENABLE_PTHREAD_FINISH_FN
void __attribute__ ((destructor)) pthread_finish_fn (void)
{
  if (scheduler_initialized)
    {
      pthread_scheduler_uninit ();
      scheduler_initialized = 0;
    }
}
#endif

cl_int
pocl_pthread_reinit (unsigned j, cl_device_id device, const char *parameters)
{
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
  node->state = POCL_COMMAND_READY;
  if (pocl_command_is_ready (node->sync.event.event))
    {
      pocl_update_event_submitted (node->sync.event.event);
      pthread_scheduler_push_command (node);
    }
  POCL_UNLOCK_OBJ (node->sync.event.event);
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
  queue_data *qdata = (queue_data *)cq->data;
  while (1)
    {
      if (cq->command_count == 0)
        {
          POCL_UNLOCK_OBJ (cq);
          return;
        }
      else
        {
          POCL_WAIT_COND (qdata->cq_cond, cq->pocl_lock);
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
      /* Unlock the finished event in order to prevent a lock order violation
       * with the command queue that will be locked during
       * pocl_update_event_failed.
       */
      pocl_unlock_events_inorder (event, finished);
      pocl_update_event_failed (CL_FAILED, NULL, 0, event, NULL);
      /* Lock events in this order to avoid a lock order violation between
       * the finished/notifier and event/wait events.
       */
      pocl_lock_events_inorder (finished, event);
      return;
    }

  if (node->state != POCL_COMMAND_READY)
    {
      POCL_MSG_PRINT_EVENTS (
        "pthread: command related to the notified event %lu not ready\n",
        event->id);
      return;
    }

  if (pocl_command_is_ready (node->sync.event.event))
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
  queue_data *qdata = (queue_data *)cq->data;
  POCL_BROADCAST_COND (qdata->cq_cond);
}

void
pocl_pthread_notify_event_finished (cl_event event)
{
  event_data *e_d = (event_data *)event->data;
  POCL_BROADCAST_COND (e_d->event_cond);
}

void
pocl_pthread_update_event (cl_device_id device, cl_event event)
{
  event_data *e_d = NULL;
  if (event->data == NULL && event->status == CL_QUEUED)
    {
      e_d = pocl_aligned_malloc (HOST_CPU_CACHELINE_SIZE, sizeof (event_data));
      assert(e_d);

      POCL_INIT_COND (e_d->event_cond);
      event->data = (void *) e_d;

      VG_ASSOC_COND_VAR (e_d->event_cond, event->pocl_lock);
    }
}

void pocl_pthread_wait_event (cl_device_id device, cl_event event)
{
  event_data *e_d = (event_data *)event->data;

  POCL_LOCK_OBJ (event);
  while (event->status > CL_COMPLETE)
    {
      POCL_WAIT_COND (e_d->event_cond, event->pocl_lock);
    }
  POCL_UNLOCK_OBJ (event);
}

void pocl_pthread_free_event_data (cl_event event)
{
  assert(event->data != NULL);
  event_data *e_d = (event_data *)event->data;
  POCL_DESTROY_COND (e_d->event_cond);
  pocl_aligned_free (event->data);
  event->data = NULL;
}

int
pocl_pthread_init_queue (cl_device_id device, cl_command_queue queue)
{
  queue_data *qdata
    = pocl_aligned_malloc (HOST_CPU_CACHELINE_SIZE, sizeof (queue_data));
  POCL_INIT_COND (qdata->cq_cond);
  queue->data = qdata;

  POCL_LOCK_OBJ (queue);
  VG_ASSOC_COND_VAR (qdata->cq_cond, queue->pocl_lock);
  POCL_UNLOCK_OBJ (queue);

  return CL_SUCCESS;
}

int
pocl_pthread_free_queue (cl_device_id device, cl_command_queue queue)
{
  queue_data *qdata = (queue_data *)queue->data;
  assert (qdata);
  POCL_DESTROY_COND (qdata->cq_cond);
  pocl_aligned_free (queue->data);
  return CL_SUCCESS;
}
