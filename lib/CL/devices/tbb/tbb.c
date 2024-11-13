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
#include <stdlib.h>
#include <string.h>

#ifndef _MSC_VER
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#include "common.h"
#include "common_utils.h"
#include "config.h"
#include "devices.h"
#include "pocl_mem_management.h"
#include "pocl_util.h"
#include "tbb.h"
#include "tbb_scheduler.h"

/* Initializes scheduler. Must be called before any kernel enqueue */
static void tbb_scheduler_init (cl_device_id device);
static void tbb_scheduler_uninit (cl_device_id device);

void pocl_tbb_init_device_ops(struct pocl_device_ops *ops) {
  pocl_pthread_init_device_ops(ops);

  ops->device_name = "cpu-tbb";

  /* implementation that differs from pthread */
  ops->probe = pocl_tbb_probe;
  ops->uninit = pocl_tbb_uninit;
  ops->reinit = pocl_tbb_reinit;
  ops->init = pocl_tbb_init;

  ops->submit = pocl_tbb_submit;
  ops->notify = pocl_tbb_notify;
}

static int one_device_per_numa_node = CL_FALSE;

unsigned int pocl_tbb_probe(struct pocl_device_ops *ops) {
  int env_count = pocl_device_get_env_count(ops->device_name);
  one_device_per_numa_node =
      pocl_get_int_option("POCL_TBB_DEV_PER_NUMA_NODE", CL_TRUE);

  if (one_device_per_numa_node) {
    /* Use one TBB device per NUMA node. */
    size_t numa_nodes = tbb_get_numa_nodes ();
    /* Env was not specified -> default to number of NUMA nodes */
    if (env_count < 0) {
      env_count = numa_nodes;
    }
    /* disallow more devices than NUMA nodes */
    if (env_count > numa_nodes)
      {
        POCL_MSG_WARN (
            "Requested more TBB devices than available NUMA nodes\n");
        env_count = numa_nodes;
      }
  } else {
    /* Use one TBB device for the whole system. */
    /* if Env was not specified -> default is 1 TBB device */
    if (env_count < 0)
      env_count = 1;
    /* disallow more than one; it's possible, but makes no sense */
    if (env_count > 1) {
      POCL_MSG_WARN("Not using redundant TBB devices\n");
      env_count = 1;
    }
  }

  return env_count;
}

static cl_bool tbb_available = CL_TRUE;
static cl_bool tbb_unavailable = CL_FALSE;

cl_int pocl_tbb_init(unsigned j, cl_device_id device, const char *parameters) {
  device->available = &tbb_unavailable;
  cl_int err = pocl_cpu_init_common(device);
  if (err) {
    POCL_MSG_ERR("pocl_cpu_init_common failed\n");
    return CL_INVALID_DEVICE;
  }

  pocl_tbb_scheduler_data *dd = calloc (1, sizeof (pocl_tbb_scheduler_data));
  device->data = (void *)dd;

  /* note: this is deliberately not using device->max_compute_units,
   * even though it has been setup by pocl_cpu_init_common() earlier.
   * The setup in that code does not take into account NUMA nodes.
   * TBD unify behaviour between drivers (make pthread NUMA aware?).
   */
  int max_threads = pocl_get_int_option ("POCL_CPU_MAX_CU_COUNT", -1);
  tbb_init_arena (dd, one_device_per_numa_node, max_threads);

  device->max_compute_units = tbb_get_num_threads (dd);
  /* subdevices not supported ATM */
  device->max_sub_devices = 0;
  device->num_partition_properties = 0;
  device->num_partition_types = 0;
  /* system mem as global memory */
  device->global_mem_id = 0;

  pocl_init_dlhandle_cache();
  pocl_init_kernel_run_command_manager();
  tbb_scheduler_init(device);

  POCL_MSG_PRINT_INFO ("TBB device %u initialized\n", j);
  device->available = &tbb_available;
  return err;
}

cl_int
pocl_tbb_uninit (unsigned J, cl_device_id Device)
{
  tbb_scheduler_uninit (Device);
  return CL_SUCCESS;
}

cl_int
pocl_tbb_reinit (unsigned J, cl_device_id Device, const char *Parameters)
{
  tbb_scheduler_init (Device);
  return CL_SUCCESS;
}

static void
tbb_scheduler_init (cl_device_id device)
{
  pocl_tbb_scheduler_data *dd = (pocl_tbb_scheduler_data *)device->data;
  POCL_INIT_LOCK (dd->wq_lock_fast);
  dd->work_queue = NULL;

  POCL_INIT_COND (dd->wake_meta_thread);

  dd->printf_buf_size = device->printf_buffer_size;
  assert (device->printf_buffer_size > 0);

  /* safety margin - aligning pointers later (in kernel arg setup)
   * may require more local memory than actual local mem size.
   * TODO fix this */
  dd->local_mem_size = device->local_mem_size
                       + device->max_parameter_size * MAX_EXTENDED_ALIGNMENT;

  dd->num_tbb_threads = tbb_get_num_threads (dd);

  /* alloc local memory for all threads making sure the memory for each thread
   * is aligned. */
  dd->printf_buf_size
      = (1 + (dd->printf_buf_size - 1) / MAX_EXTENDED_ALIGNMENT)
        * MAX_EXTENDED_ALIGNMENT;
  dd->local_mem_size = (1 + (dd->local_mem_size - 1) / MAX_EXTENDED_ALIGNMENT)
                       * MAX_EXTENDED_ALIGNMENT;
  dd->printf_buf_global_ptr = (uchar *)pocl_aligned_malloc (
      MAX_EXTENDED_ALIGNMENT, dd->printf_buf_size * dd->num_tbb_threads);
  dd->local_mem_global_ptr = (char *)pocl_aligned_malloc (
      MAX_EXTENDED_ALIGNMENT, dd->local_mem_size * dd->num_tbb_threads);

  dd->meta_thread_shutdown_requested = 0;
  /* create one meta thread per device to serve as an async interface thread. */
  POCL_CREATE_THREAD (dd->meta_thread, TBBDriverThread, device);

  dd->grain_size = 0;
  if (pocl_is_option_set ("POCL_TBB_GRAIN_SIZE"))
    {
      dd->grain_size = pocl_get_int_option ("POCL_TBB_GRAIN_SIZE", 1);
      POCL_MSG_PRINT_GENERAL ("TBB: using a grain size of %u\n",
                              dd->grain_size);
    }

  dd->selected_partitioner = TBB_PART_NONE;
  const char *ptr = pocl_get_string_option ("POCL_TBB_PARTITIONER", "");
  if (strlen (ptr) > 0)
    {
      if (strncmp (ptr, "affinity", 8) == 0)
        {
          dd->selected_partitioner = TBB_PART_AFFINITY;
          POCL_MSG_PRINT_GENERAL ("TBB: using affinity partitioner\n");
        }
      else if (strncmp (ptr, "auto", 4) == 0)
        {
          dd->selected_partitioner = TBB_PART_AUTO;
          POCL_MSG_PRINT_GENERAL ("TBB: using auto partitioner\n");
        }
      else if (strncmp (ptr, "simple", 6) == 0)
        {
          dd->selected_partitioner = TBB_PART_SIMPLE;
          POCL_MSG_PRINT_GENERAL ("TBB: using simple partitioner\n");
        }
      else if (strncmp (ptr, "static", 6) == 0)
        {
          dd->selected_partitioner = TBB_PART_STATIC;
          POCL_MSG_PRINT_GENERAL ("TBB: using static partitioner\n");
        }
      else
        {
          POCL_MSG_WARN (
              "TBB: Malformed string in POCL_TBB_PARTITIONER env var: %s\n",
              ptr);
        }
    }
}

static void
tbb_scheduler_uninit (cl_device_id device)
{
  pocl_tbb_scheduler_data *dd = (pocl_tbb_scheduler_data *)device->data;

  POCL_LOCK (dd->wq_lock_fast);
  dd->meta_thread_shutdown_requested = 1;
  POCL_BROADCAST_COND (dd->wake_meta_thread);
  POCL_UNLOCK (dd->wq_lock_fast);

  POCL_JOIN_THREAD (dd->meta_thread);

  POCL_DESTROY_LOCK (dd->wq_lock_fast);
  POCL_DESTROY_COND (dd->wake_meta_thread);

  dd->meta_thread_shutdown_requested = 0;
  dd->work_queue = NULL;

  pocl_aligned_free (dd->printf_buf_global_ptr);
  pocl_aligned_free (dd->local_mem_global_ptr);
}

/* TBB doesn't support subdevices, so push_command can use cond_signal */
static void
tbb_scheduler_push_command (_cl_command_node *cmd)
{
  cl_device_id device = cmd->device;
  pocl_tbb_scheduler_data *dd = (pocl_tbb_scheduler_data *)device->data;
  POCL_LOCK (dd->wq_lock_fast);
  DL_APPEND (dd->work_queue, cmd);
  POCL_SIGNAL_COND (dd->wake_meta_thread);
  POCL_UNLOCK (dd->wq_lock_fast);
}

void pocl_tbb_submit(_cl_command_node *node, cl_command_queue cq) {
  node->ready = 1;
  if (pocl_command_is_ready(node->sync.event.event)) {
    pocl_update_event_submitted(node->sync.event.event);
    tbb_scheduler_push_command(node);
  }
  POCL_UNLOCK_OBJ(node->sync.event.event);
  return;
}

void pocl_tbb_notify(cl_device_id device, cl_event event, cl_event finished) {
  cl_bool wake_thread = CL_FALSE;
  _cl_command_node *node = event->command;

  if (finished->status < CL_COMPLETE) {
      pocl_update_event_failed_locked (event);
      return;
  }

  if (!node->ready)
    return;

  if (pocl_command_is_ready(node->sync.event.event)) {
    if (event->status == CL_QUEUED) {
      pocl_update_event_submitted(event);
      wake_thread = CL_TRUE;
    }
  }
  if (wake_thread != CL_FALSE)
    {
      tbb_scheduler_push_command (node);
    }
  return;
}
