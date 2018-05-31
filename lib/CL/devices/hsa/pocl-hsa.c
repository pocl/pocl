/* pocl-hsa.c - driver for HSA supported devices.

   Copyright (c) 2015-2016 Pekka Jääskeläinen <pekka.jaaskelainen@tut.fi>
                 2015 Charles Chen <ccchen@pllab.cs.nthu.edu.tw>
                      Shao-chung Wang <scwang@pllab.cs.nthu.edu.tw>
                 2015-2016 Michal Babej <michal.babej@tut.fi>

   Short snippets borrowed from the MatrixMultiplication example in
   the HSA runtime library sources (c) 2014 HSA Foundation Inc.

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
/* Some example code snippets copied verbatim from vector_copy.c of HSA-Runtime-AMD: */
/* Copyright 2014 HSA Foundation Inc.  All Rights Reserved.
 *
 * HSAF is granting you permission to use this software and documentation (if
 * any) (collectively, the "Materials") pursuant to the terms and conditions
 * of the Software License Agreement included with the Materials.  If you do
 * not have a copy of the Software License Agreement, contact the  HSA Foundation for a copy.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.
 */

#ifndef _BSD_SOURCE
#define _BSD_SOURCE
#endif

#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif


#include "hsa.h"
#include "hsa_ext_finalize.h"
#include "hsa_ext_image.h"

#include "config.h"

#ifdef HAVE_HSA_EXT_AMD_H
#include "hsa_ext_amd.h"
#endif

#include "pocl-hsa.h"
#include "common.h"
#include "devices.h"
#include "pocl_file_util.h"
#include "pocl_cache.h"
#include "pocl_llvm.h"
#include "pocl_util.h"
#include "pocl_mem_management.h"

#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>

#ifndef _MSC_VER
#  include <sys/wait.h>
#  include <sys/time.h>
#  include <sys/types.h>
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#define max(a,b) (((a) > (b)) ? (a) : (b))

/* TODO:
   - fix pocl_llvm_generate_workgroup_function() to
     generate the entire linked program.bc for HSA, not just a single kernel.
   - AMD SDK samples. Requires work mainly in these areas:
      - image support
      - CL C++
   - OpenCL printf() support
   - get_global_offset() and global_work_offset param of clEnqueNDRker -
     HSA kernel dispatch packet has no offset fields -
     we must take care of it somewhere
   - clinfo of Ubuntu crashes
   - etc. etc.
*/

#define HSA_KERNEL_CACHE_SIZE 64
#define COMMAND_LIST_SIZE 4096
#define EVENT_LIST_SIZE 511

typedef struct pocl_hsa_event_data_s {
  void* actual_kernargs;
  pthread_cond_t event_cond;
} pocl_hsa_event_data_t;

/* Simple statically-sized kernel data cache */
/* for caching kernel dispatch data, binaries etc */
typedef struct pocl_hsa_kernel_cache_s {
  cl_kernel kernel;
  hsa_executable_t hsa_exe;
  uint64_t code_handle;

  uint32_t private_size;
  uint32_t static_group_size;
  uint32_t args_segment_size;
} pocl_hsa_kernel_cache_t;

/* data for driver pthread */
typedef struct pocl_hsa_device_pthread_data_s {
  /* list of running commands and their signals*/
  cl_event running_events[EVENT_LIST_SIZE];
  hsa_signal_t running_signals[EVENT_LIST_SIZE+1];
  size_t running_list_size;

  /* Queue list (for pushing work to the agent);
   * multiple queues per device */
  hsa_queue_t **queues;
  size_t num_queues, last_queue;
} pocl_hsa_device_pthread_data_t;

typedef struct pocl_hsa_device_data_s {
  /* The HSA kernel agent controlled by the device driver instance. */
  hsa_agent_t agent;
  hsa_profile_t agent_profile;

  /* mem regions */
  hsa_region_t global_region, kernarg_region, group_region;

  /* Per-program data cache to simplify program compiling stage */
  pocl_hsa_kernel_cache_t kernel_cache[HSA_KERNEL_CACHE_SIZE];
  unsigned kernel_cache_lastptr;

  /* kernel signal wait timeout hint, in HSA runtime units */
  uint64_t timeout;
  /* length of a timestamp unit expressed in nanoseconds */
  double timestamp_unit;
  /* see pocl_hsa_init for details */
  size_t hw_schedulers;

  /* list of submitted commands waiting to run later */
  cl_event wait_list[COMMAND_LIST_SIZE];
  size_t wait_list_size;

  /* list of commands ready to run */
  cl_event ready_list[COMMAND_LIST_SIZE];
  size_t ready_list_size;

  /* list manipulation mutex */
  pthread_mutex_t list_mutex;

  /* used by host thread to notify driver pthread when events change status */
  hsa_signal_t nudge_driver_thread;

  /* device pthread */
  pthread_t driver_pthread_id;

  /* device pthread data */
  pocl_hsa_device_pthread_data_t driver_data;

  /* exit signal */
  volatile int exit_driver_thread;

  /* if agent supports async handlers*/
  int have_wait_any;

  /* compilation lock */
  pocl_lock_t pocl_hsa_compilation_lock;
} pocl_hsa_device_data_t;



void
pocl_hsa_init_device_ops(struct pocl_device_ops *ops)
{
  pocl_basic_init_device_ops (ops);

  /* TODO: more descriptive name from HSA probing the device */
  ops->device_name = "hsa";
  ops->probe = pocl_hsa_probe;
  ops->uninit = pocl_hsa_uninit;
  ops->reinit = NULL;
  ops->init = pocl_hsa_init;
  ops->alloc_mem_obj = pocl_hsa_alloc_mem_obj;
  ops->free = pocl_hsa_free;
  //ops->run = pocl_hsa_run;
  ops->read = pocl_basic_read;
  ops->read_rect = pocl_basic_read_rect;
  ops->write = pocl_basic_write;
  ops->write_rect = pocl_basic_write_rect;
  ops->copy = pocl_hsa_copy;
  ops->copy_rect = pocl_basic_copy_rect;
  ops->get_timer_value = pocl_hsa_get_timer_value;

  // new driver api (out-of-order)
  ops->submit = pocl_hsa_submit;
  ops->join = pocl_hsa_join;
  ops->flush = pocl_hsa_flush;
  ops->notify = pocl_hsa_notify;
  ops->broadcast = pocl_hsa_broadcast;
  ops->wait_event = pocl_hsa_wait_event;
  ops->compile_kernel = pocl_hsa_compile_kernel;
  ops->update_event = pocl_hsa_update_event;
  ops->free_event_data = pocl_hsa_free_event_data;
  ops->init_target_machine = NULL;
  ops->run = NULL;
  ops->wait_event = pocl_hsa_wait_event;
  ops->update_event = pocl_hsa_update_event;
  ops->free_event_data = NULL;

  ops->build_hash = pocl_hsa_build_hash;
}

#define MAX_HSA_AGENTS 16

static void
pocl_hsa_abort_on_hsa_error(hsa_status_t status,
                            unsigned line,
                            const char* func,
                            const char* code)
{
  const char* str;
  if (status != HSA_STATUS_SUCCESS)
    {
      hsa_status_string(status, &str);
      POCL_MSG_PRINT2(HSA, func, line, "Error from HSA Runtime call:\n");
      POCL_ABORT ("%s\n", str);
    }
}


#define HSA_CHECK(code) pocl_hsa_abort_on_hsa_error(code,         \
                                                    __LINE__,     \
                                                    __FUNCTION__, \
                                                    #code);

static void
pocl_hsa_abort_on_pthread_error(int status,
                                unsigned line,
                                const char* func,
                                const char* code)
{
  if (status != 0)
    {
      POCL_MSG_PRINT2(HSA, func, line, "Error from pthread call:\n");
      POCL_ABORT ("%s\n", strerror (status));
    }
}

#define PTHREAD_CHECK(code) pocl_hsa_abort_on_pthread_error(code,         \
                                                            __LINE__,     \
                                                            __FUNCTION__, \
                                                            #code);

static hsa_agent_t hsa_agents[MAX_HSA_AGENTS];
static unsigned found_hsa_agents = 0;

static hsa_status_t
pocl_hsa_get_agents_callback(hsa_agent_t agent, void *data)
{
  hsa_device_type_t type;
  HSA_CHECK(hsa_agent_get_info (agent, HSA_AGENT_INFO_DEVICE, &type));

  hsa_agent_feature_t features;
  HSA_CHECK(hsa_agent_get_info(agent, HSA_AGENT_INFO_FEATURE, &features));
  if (features != HSA_AGENT_FEATURE_KERNEL_DISPATCH)
    {
      return HSA_STATUS_SUCCESS;
    }

  hsa_agents[found_hsa_agents++] = agent;
  return HSA_STATUS_SUCCESS;
}

/*
 * Sets up the memory regions in pocl_hsa_device_data for a device
 */
static
hsa_status_t
setup_agent_memory_regions_callback(hsa_region_t region, void* data)
{
  pocl_hsa_device_data_t* d = (pocl_hsa_device_data_t*)data;

  hsa_region_segment_t segment;
  hsa_region_global_flag_t flags;
  HSA_CHECK(hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment));

  if (segment == HSA_REGION_SEGMENT_GLOBAL)
    {
      d->global_region = region;
      HSA_CHECK(hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS,
                                    &flags));
      if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG)
        d->kernarg_region = region;
    }

  if (segment == HSA_REGION_SEGMENT_GROUP)
    d->group_region = region;

  return HSA_STATUS_SUCCESS;
}

// Get hsa-unsupported device features from hsa device list.
static unsigned num_hsa_device = 2;

static struct _cl_device_id
supported_hsa_devices[MAX_HSA_AGENTS] =
{
  [0] =
  {
    .long_name = "Spectre",
    .llvm_cpu = NULL,                 // native: "kaveri",
    .llvm_target_triplet = "hsail64", // native: "amdgcn--amdhsa"
    .has_64bit_long = 1,
    .vendor_id = 0x1002,
    .global_mem_cache_type = CL_READ_WRITE_CACHE,
    .max_constant_buffer_size = 65536,
    .local_mem_type = CL_LOCAL,
    .endian_little = CL_TRUE,
    .extensions = HSA_DEVICE_EXTENSIONS,
    .preferred_wg_size_multiple = 64, // wavefront size on Kaveri
    .preferred_vector_width_char = 4,
    .preferred_vector_width_short = 2,
    .preferred_vector_width_int = 1,
    .preferred_vector_width_long = 1,
    .preferred_vector_width_float = 1,
    .preferred_vector_width_double = 1,
    .native_vector_width_char = 4,
    .native_vector_width_short = 2,
    .native_vector_width_int = 1,
    .native_vector_width_long = 1,
    .native_vector_width_float = 1,
    .native_vector_width_double = 1
  },
  [1] =
  { .long_name = "phsa generic CPU agent",
    .llvm_cpu = NULL,
    .llvm_target_triplet = "hsail64",
    .has_64bit_long = 1,
    .vendor_id = 0xffff,
    .global_mem_cache_type = CL_READ_WRITE_CACHE,
    .max_constant_buffer_size = 65536,
    .local_mem_type = CL_LOCAL,
    .endian_little = !(WORDS_BIGENDIAN),
    .extensions = HSA_DEVICE_EXTENSIONS,
    .preferred_wg_size_multiple = 1,
    // We want to exploit the widest vector types in HSAIL
    // for the CPUs assuming they have some sort of SIMD ISE
    // which the finalizer than can more readily utilize.
    .preferred_vector_width_char = 16,
    .preferred_vector_width_short = 16,
    .preferred_vector_width_int = 16,
    .preferred_vector_width_long = 16,
    .preferred_vector_width_float = 16,
    .preferred_vector_width_double = 16,
    .native_vector_width_char = 16,
    .native_vector_width_short = 16,
    .native_vector_width_int = 16,
    .native_vector_width_long = 16,
    .native_vector_width_float = 16,
    .native_vector_width_double = 16
  }
};

char *
pocl_hsa_build_hash (cl_device_id device)
{
  char* res = calloc(1000, sizeof(char));
  snprintf(res, 1000, "HSA-%s-%s", device->llvm_target_triplet, device->long_name);
  return res;
}

// Detect the HSA device and populate its properties to the device
// struct.
static void
get_hsa_device_features(char* dev_name, struct _cl_device_id* dev)
{

#define COPY_ATTR(ATTR) dev->ATTR = supported_hsa_devices[i].ATTR
#define COPY_VECWIDTH(ATTR) \
     dev->preferred_vector_width_ ## ATTR = \
         supported_hsa_devices[i].preferred_vector_width_ ## ATTR; \
     dev->native_vector_width_ ## ATTR = \
         supported_hsa_devices[i].native_vector_width_ ## ATTR;

  int found = 0;
  unsigned i;
  for(i = 0; i < num_hsa_device; i++)
    {
      if (strcmp(dev_name, supported_hsa_devices[i].long_name) == 0)
        {
          dev->device_side_printf = 0;
          COPY_ATTR (llvm_cpu);
          COPY_ATTR (llvm_target_triplet);
          COPY_ATTR (has_64bit_long);
          COPY_ATTR (vendor_id);
          COPY_ATTR (global_mem_cache_type);
          COPY_ATTR (max_constant_buffer_size);
          COPY_ATTR (local_mem_type);
          COPY_ATTR (endian_little);
          COPY_ATTR (preferred_wg_size_multiple);
          COPY_ATTR (extensions);
          COPY_VECWIDTH (char);
          COPY_VECWIDTH (short);
          COPY_VECWIDTH (int);
          COPY_VECWIDTH (long);
          COPY_VECWIDTH (float);
          COPY_VECWIDTH (double);
          found = 1;
          break;
        }
    }
  if (!found)
    {
      POCL_MSG_PRINT_INFO("pocl-hsa: found unknown HSA devices '%s'.\n",
			  dev_name);
      POCL_ABORT ("We found a device for which we don't have device "
                  "OpenCL attribute information (compute unit count, "
                  "constant buffer size etc), and there's no way to get all "
                  "the required info from HSA API. Please create a "
                  "new entry with the information in supported_hsa_devices, "
                  "and send a note/patch to pocl developers. Thanks!\n");
    }
}

unsigned int
pocl_hsa_probe (struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count (ops->device_name);

  POCL_MSG_PRINT_INFO ("pocl-hsa: found %d env devices with %s.\n", env_count,
                       ops->device_name);

  /* No hsa env specified, the user did not request for HSA agents. */
  if (env_count <= 0)
    return 0;

  HSA_CHECK (hsa_init ());

  HSA_CHECK (hsa_iterate_agents (pocl_hsa_get_agents_callback, NULL));

  POCL_MSG_PRINT_INFO ("pocl-hsa: found %d agents.\n", found_hsa_agents);

  return (int)found_hsa_agents;
}

static void
hsa_queue_callback (hsa_status_t status, hsa_queue_t *q, void *data)
{
  HSA_CHECK (status);
}

/* driver pthread prototype */
void *pocl_hsa_driver_pthread (void *cldev);

cl_int
pocl_hsa_init (unsigned j, cl_device_id dev, const char *parameters)
{
  pocl_init_cpu_device_infos (dev);

  SETUP_DEVICE_CL_VERSION(HSA_DEVICE_CL_VERSION_MAJOR,
                          HSA_DEVICE_CL_VERSION_MINOR)

  dev->spmd = CL_TRUE;
  dev->autolocals_to_args = 0;

  dev->global_as_id = 1;
  dev->local_as_id = 3;
  dev->constant_as_id = 2;

  assert (found_hsa_agents > 0);
  assert (j < found_hsa_agents);
  dev->data = (void*)(uintptr_t)j;
  hsa_agent_t agent = hsa_agents[j];

  uint32_t cache_sizes[4];
  HSA_CHECK(hsa_agent_get_info (agent, HSA_AGENT_INFO_CACHE_SIZE,
                        &cache_sizes));
  // The only nonzero value on Kaveri is the first (L1)
  dev->global_mem_cache_size = cache_sizes[0];

  dev->short_name = dev->long_name = (char*)malloc (64*sizeof(char));
  HSA_CHECK(hsa_agent_get_info (agent, HSA_AGENT_INFO_NAME, dev->long_name));
  get_hsa_device_features (dev->long_name, dev);

  dev->type = CL_DEVICE_TYPE_GPU;

  // Enable when it's actually implemented AND if supported by
  // the target agent (check with hsa_agent_extension_supported).
  dev->image_support = CL_FALSE;

  dev->single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO
      | CL_FP_ROUND_TO_INF | CL_FP_FMA | CL_FP_INF_NAN;
  dev->double_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO
      | CL_FP_ROUND_TO_INF | CL_FP_FMA | CL_FP_INF_NAN;

  hsa_machine_model_t model;
  HSA_CHECK(hsa_agent_get_info (agent, HSA_AGENT_INFO_MACHINE_MODEL, &model));
  dev->address_bits = (model == HSA_MACHINE_MODEL_LARGE) ? 64 : 32;

  uint16_t wg_sizes[3];
  HSA_CHECK(hsa_agent_get_info(agent, HSA_AGENT_INFO_WORKGROUP_MAX_DIM,
                               &wg_sizes));

  int max_wg = pocl_get_int_option ("POCL_MAX_WORK_GROUP_SIZE", 0);
  if (max_wg > 0)
    {
      wg_sizes[0] = min (wg_sizes[0], max_wg);
      wg_sizes[1] = min (wg_sizes[1], max_wg);
      wg_sizes[2] = min (wg_sizes[2], max_wg);
    }

  dev->max_work_group_size = dev->max_work_item_sizes[0] = wg_sizes[0];
  dev->max_work_item_sizes[1] = wg_sizes[1];
  dev->max_work_item_sizes[2] = wg_sizes[2];

#ifdef HAVE_HSA_EXT_AMD_H
  uint32_t temp;
  HSA_CHECK(hsa_agent_get_info(agent, HSA_AMD_AGENT_INFO_CACHELINE_SIZE,
                               &temp));
  dev->global_mem_cacheline_size = temp;

  HSA_CHECK(hsa_agent_get_info(agent, HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
                               &temp));
  dev->max_compute_units = temp;

  HSA_CHECK(hsa_agent_get_info(agent, HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY,
                               &temp));
  dev->max_clock_frequency = temp;

#else
#warning "Could not use AMD headers to find out CU/frequency of your device. Using some default values which are probably wrong..."
  dev->global_mem_cacheline_size = 64;
  dev->max_compute_units = 4;
  dev->max_clock_frequency = 700;
#endif

  HSA_CHECK(hsa_agent_get_info
    (agent, HSA_AGENT_INFO_WORKGROUP_MAX_SIZE, &dev->max_work_group_size));

  /* Image features. */
  if (dev->image_support == CL_TRUE)
    {
      hsa_dim3_t image_size;
      HSA_CHECK (hsa_agent_get_info (
          agent, HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS, &image_size));
      dev->image_max_buffer_size = image_size.x;
      HSA_CHECK (hsa_agent_get_info (
          agent, HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS, &image_size));
      dev->image2d_max_height = image_size.x;
      dev->image2d_max_width = image_size.y;
      HSA_CHECK (hsa_agent_get_info (
          agent, HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS, &image_size));
      dev->image3d_max_height = image_size.x;
      dev->image3d_max_width = image_size.y;
      dev->image3d_max_depth = image_size.z;
      // is this directly the product of the dimensions?
      //stat = hsa_agent_get_info(agent, ??, &dev->image_max_array_size);
      HSA_CHECK (hsa_agent_get_info (agent,
                                     HSA_EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES,
                                     &dev->max_read_image_args));
      HSA_CHECK (hsa_agent_get_info (agent,
                                     HSA_EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES,
                                     &dev->max_read_write_image_args));
      dev->max_write_image_args = dev->max_read_write_image_args;
      HSA_CHECK (hsa_agent_get_info (
          agent, HSA_EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS, &dev->max_samplers));
    }

  dev->should_allocate_svm = 1;
  /* OpenCL 2.0 properties */
  dev->svm_caps = CL_DEVICE_SVM_COARSE_GRAIN_BUFFER
                  | CL_DEVICE_SVM_FINE_GRAIN_BUFFER
                  | CL_DEVICE_SVM_ATOMICS;
  /* This is from clinfo output ran on AMD Catalyst drivers */
  dev->max_events = 1024;
  dev->max_queues = 1;
  dev->max_pipe_args = 16;
  dev->max_pipe_active_res = 16;
  dev->max_pipe_packet_size = 1024 * 1024;
  dev->dev_queue_pref_size = 256 * 1024;
  dev->dev_queue_max_size = 512 * 1024;
  dev->on_dev_queue_props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
                               | CL_QUEUE_PROFILING_ENABLE;
  dev->on_host_queue_props = CL_QUEUE_PROFILING_ENABLE;

  pocl_hsa_device_data_t *d;

  dev->global_mem_id = 0;

  d = (pocl_hsa_device_data_t *) calloc (1, sizeof(pocl_hsa_device_data_t));

  POCL_INIT_LOCK (d->pocl_hsa_compilation_lock);

  intptr_t agent_index = (intptr_t)dev->data;
  d->agent.handle = hsa_agents[agent_index].handle;
  dev->data = d;

  HSA_CHECK(hsa_agent_iterate_regions (d->agent,
                                       setup_agent_memory_regions_callback,
                                       d));

  bool boolarg = 0;
  HSA_CHECK(hsa_region_get_info(d->global_region,
                                HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED,
                                &boolarg));
  assert(boolarg != 0);

#ifdef HAVE_HSA_EXT_AMD_H
  char booltest = 0;
  HSA_CHECK(hsa_region_get_info(d->global_region,
				HSA_AMD_REGION_INFO_HOST_ACCESSIBLE,
                                &booltest));
  assert(booltest != 0);
#endif

  size_t sizearg;
  HSA_CHECK(hsa_region_get_info(d->global_region,
				HSA_REGION_INFO_ALLOC_MAX_SIZE, &sizearg));
  dev->max_mem_alloc_size = sizearg;

  /* For some reason, the global region size returned is 128 Terabytes...
   * for now, use the max alloc size, it seems to be a much more reasonable
   * value.
   * HSA_CHECK(hsa_region_get_info(d->global_region, HSA_REGION_INFO_SIZE,
   *                               &sizearg));
   */
  HSA_CHECK(hsa_region_get_info(d->global_region,
                               HSA_REGION_INFO_SIZE, &sizearg));
  dev->global_mem_size = sizearg;
  if (dev->global_mem_size > 16 * 1024 * 1024 * (uint64_t)1024)
    dev->global_mem_size = dev->max_mem_alloc_size;

  pocl_setup_device_for_system_memory (dev);

  HSA_CHECK(hsa_region_get_info(d->group_region, HSA_REGION_INFO_SIZE,
                                &sizearg));
  dev->local_mem_size = sizearg;

  HSA_CHECK(hsa_region_get_info(d->global_region,
				HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT,
                                &sizearg));
  dev->mem_base_addr_align = sizearg * 8;

  HSA_CHECK(hsa_agent_get_info(d->agent, HSA_AGENT_INFO_PROFILE,
                               &d->agent_profile));
  dev->profile = ((d->agent_profile == HSA_PROFILE_FULL) ? "FULL_PROFILE"
                                                         : "EMBEDDED_PROFILE");

  uint64_t hsa_freq;
  HSA_CHECK(hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY,
                                &hsa_freq));
  d->timeout = hsa_freq; // 1 second in hsa units
  d->timestamp_unit = (1000000000.0 / (double)hsa_freq);
  POCL_MSG_PRINT_INFO("HSA timestamp frequency: %" PRIu64 "\n", hsa_freq);
  POCL_MSG_PRINT_INFO("HSA timeout: %" PRIu64 "\n", d->timeout);
  POCL_MSG_PRINT_INFO("HSA timestamp unit: %g\n", d->timestamp_unit);

  dev->profiling_timer_resolution = (size_t) (d->timestamp_unit) || 1;

  /* TODO proper setup */
  d->hw_schedulers = 3;

#ifdef HAVE_HSA_EXT_AMD_H
  /* TODO check at runtime */
  d->have_wait_any = 1;
#endif
  HSA_CHECK(hsa_signal_create(1, 1, &d->agent,
                              &d->nudge_driver_thread));

  pthread_mutexattr_t mattr;
  PTHREAD_CHECK(pthread_mutexattr_init(&mattr));
  PTHREAD_CHECK(pthread_mutexattr_settype(&mattr, PTHREAD_MUTEX_ERRORCHECK));
  PTHREAD_CHECK(pthread_mutex_init(&d->list_mutex, &mattr));

  d->exit_driver_thread = 0;
  PTHREAD_CHECK (pthread_create (&d->driver_pthread_id, NULL,
                                 &pocl_hsa_driver_pthread, dev));
  return CL_SUCCESS;
}

static void*
pocl_hsa_malloc_account(pocl_global_mem_t *mem, size_t size, hsa_region_t r)
{
  void *b = NULL;
  if ((mem->total_alloc_limit - mem->currently_allocated) < size)
    return NULL;

  if (hsa_memory_allocate(r, size, &b) != HSA_STATUS_SUCCESS)
    return NULL;

  mem->currently_allocated += size;
  if (mem->max_ever_allocated < mem->currently_allocated)
    mem->max_ever_allocated = mem->currently_allocated;
  assert(mem->currently_allocated <= mem->total_alloc_limit);

  if (b)
    POCL_MSG_PRINT_INFO("HSA malloc'ed : size %" PRIuS "\n", size);

  return b;
}

static void *
pocl_hsa_malloc (cl_device_id device, cl_mem_flags flags, size_t size,
                 void *host_ptr)
{
  pocl_hsa_device_data_t* d = device->data;
  void *b = NULL;
  pocl_global_mem_t *mem = device->global_memory;

  if (flags & CL_MEM_USE_HOST_PTR)
    {
      POCL_MSG_PRINT_INFO("HSA: hsa_memory_register (CL_MEM_USE_HOST_PTR)\n");
      assert(host_ptr != NULL);
      // TODO bookkeeping of mem registrations
      hsa_memory_register(host_ptr, size);
      return host_ptr;
    }

  if (flags & CL_MEM_COPY_HOST_PTR)
    {
      POCL_MSG_PRINT_INFO("HSA: hsa_memory_allocate + hsa_memory_copy"
                          " (CL_MEM_COPY_HOST_PTR)\n");
      assert(host_ptr != NULL);

      b = pocl_hsa_malloc_account(mem, size, d->global_region);
      if (b)
        hsa_memory_copy(b, host_ptr, size);
      return b;
    }

  assert(host_ptr == NULL);
  //POCL_MSG_PRINT_INFO("HSA: hsa_memory_allocate (ALLOC_HOST_PTR)\n");
  return pocl_hsa_malloc_account(mem, size, d->global_region);
}

void
pocl_hsa_free (cl_device_id device, cl_mem memobj)
{
  cl_mem_flags flags = memobj->flags;
  void* ptr = memobj->device_ptrs[device->dev_id].mem_ptr;
  size_t size = memobj->size;

  if (flags & CL_MEM_USE_HOST_PTR ||
      memobj->shared_mem_allocation_owner != device)
    hsa_memory_deregister(ptr, size);
  else
    {
      pocl_global_mem_t *mem = device->global_memory;
      assert(mem->currently_allocated >= size);
      mem->currently_allocated -= size;
      hsa_memory_free(ptr);
    }
  if (memobj->flags | CL_MEM_ALLOC_HOST_PTR)
    memobj->mem_host_ptr = NULL;
}

void
pocl_hsa_copy (void *data, const void *src_ptr, size_t src_offset,
               void *__restrict__ dst_ptr, size_t dst_offset, size_t cb)
{
  assert(src_offset == 0);
  assert(dst_offset == 0);
  HSA_CHECK(hsa_memory_copy(dst_ptr, src_ptr, cb));
}

cl_int
pocl_hsa_alloc_mem_obj(cl_device_id device, cl_mem mem_obj, void* host_ptr)
{
  void *b = NULL;
  cl_mem_flags flags = mem_obj->flags;
  unsigned i;

  /* check if some driver has already allocated memory for this mem_obj 
     in our global address space, and use that*/
  for (i = 0; i < mem_obj->context->num_devices; ++i)
    {
      if (!mem_obj->device_ptrs[i].available)
        continue;

      if (mem_obj->device_ptrs[i].global_mem_id == device->global_mem_id
          && mem_obj->device_ptrs[i].mem_ptr != NULL)
        {
          mem_obj->device_ptrs[device->dev_id].mem_ptr =
            mem_obj->device_ptrs[i].mem_ptr;
          hsa_memory_register(mem_obj->device_ptrs[device->dev_id].mem_ptr,
                              mem_obj->size);
          POCL_MSG_PRINT_INFO ("HSA: alloc_mem_obj, use already"
                               " allocated memory\n");
          return CL_SUCCESS;
        }
    }

  /* memory for this global memory is not yet allocated -> do it */
  b = pocl_hsa_malloc(device, flags, mem_obj->size, host_ptr);
  if (b == NULL)
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;

  /* take ownership if not USE_HOST_PTR */
  if (~flags & CL_MEM_USE_HOST_PTR)
    mem_obj->shared_mem_allocation_owner = device;

  mem_obj->device_ptrs[device->dev_id].mem_ptr = b;

  if (flags & CL_MEM_ALLOC_HOST_PTR)
    mem_obj->mem_host_ptr = b;

  return CL_SUCCESS;

}

static void
setup_kernel_args (pocl_hsa_device_data_t *d,
                   _cl_command_node *cmd,
                   char *arg_space,
                   size_t max_args_size,
                   uint32_t *total_group_size)
{
  char *write_pos = arg_space;
  const char *last_pos = arg_space + max_args_size;

#define CHECK_AND_ALIGN_SPACE(DSIZE)                         \
  do {                                                       \
    if (write_pos + (DSIZE) > last_pos)                      \
      POCL_ABORT("pocl-hsa: too many kernel arguments!\n");  \
    unsigned unaligned = (intptr_t)write_pos % DSIZE;        \
    if (unaligned > 0) write_pos += (DSIZE - unaligned);     \
  } while (0)

  size_t i;
  for (i = 0; i < cmd->command.run.kernel->num_args; ++i)
    {
      struct pocl_argument *al = &(cmd->command.run.arguments[i]);
      if (cmd->command.run.kernel->arg_info[i].is_local)
        {
          CHECK_AND_ALIGN_SPACE(sizeof (uint32_t));
          memcpy(write_pos, total_group_size, sizeof(uint32_t));
          *total_group_size += (uint32_t)al->size;
          write_pos += sizeof(uint32_t);
        }
      else if (cmd->command.run.kernel->arg_info[i].type
               == POCL_ARG_TYPE_POINTER)
        {
          CHECK_AND_ALIGN_SPACE(sizeof (uint64_t));
          /* Assuming the pointers are 64b (or actually the same as in
             host) due to HSA. TODO: the 32b profile. */

          if (al->value == NULL)
            {
              uint64_t temp = 0;
              memcpy (write_pos, &temp, sizeof (uint64_t));
            }
          else
            {
              cl_mem m = *(cl_mem *)al->value;
              uint64_t temp = 0;
              if (m->device_ptrs)
                temp = (uint64_t)m->device_ptrs[cmd->device->dev_id].mem_ptr;
              else
                temp = (uint64_t)m->mem_host_ptr;
              memcpy (write_pos, &temp, sizeof(uint64_t));
            }
          write_pos += sizeof(uint64_t);
#if 0
          /* TODO: It's legal to pass a NULL pointer to clSetKernelArguments.
             In that case we must pass the same NULL forward to the kernel.
             Otherwise, the user must have created a buffer with per device
             pointers stored in the cl_mem. */
          if (al->value == NULL)
            {
              arguments[i] = malloc (sizeof (void *));
              *(void **)arguments[i] = NULL;
            }
          else
            arguments[i] =
              &((*(cl_mem *) (al->value))->device_ptrs[cmd->device->dev_id]
                .mem_ptr);
#endif
        }
      else if (cmd->command.run.kernel->arg_info[i].type
               == POCL_ARG_TYPE_IMAGE)
        {
          POCL_ABORT_UNIMPLEMENTED("pocl-hsa: image arguments"
                                   " not implemented.\n");
        }
      else if (cmd->command.run.kernel->arg_info[i].type
               == POCL_ARG_TYPE_SAMPLER)
        {
          POCL_ABORT_UNIMPLEMENTED("pocl-hsa: sampler arguments"
                                   " not implemented.\n");
        }
      else
        {
          // Scalars.
          CHECK_AND_ALIGN_SPACE(al->size);
          memcpy (write_pos, al->value, al->size);
          write_pos += al->size;
        }
    }

  /* pointer to pocl_context */
  CHECK_AND_ALIGN_SPACE(sizeof (uint64_t));

  uint64_t pc_addr = (uint64_t)&cmd->command.run.pc;
  memcpy(write_pos, &pc_addr, sizeof(uint64_t));
  write_pos += sizeof(uint64_t);

#if 0
  for (size_t i = cmd->command.run.kernel->num_args;
       i < cmd->command.run.kernel->num_args
         + cmd->command.run.kernel->num_locals;
       ++i)
    {
      POCL_ABORT_UNIMPLEMENTED("hsa: automatic local buffers"
                               " not implemented.");
      al = &(cmd->command.run.arguments[i]);
      arguments[i] = malloc (sizeof (void *));
      *(void **)(arguments[i]) = pocl_hsa_malloc (data, 0, al->size, NULL);
    }
#endif
}

static int
compile_parallel_bc_to_brig(char* brigfile, cl_kernel kernel,
                            cl_device_id device) {
  int error;
  char hsailfile[POCL_FILENAME_LENGTH];
  char parallel_bc_path[POCL_FILENAME_LENGTH];

  unsigned device_i = pocl_cl_device_to_index(kernel->program, device);
  pocl_cache_work_group_function_path(parallel_bc_path, kernel->program,
                                      device_i, kernel, 0, 0, 0);

  strcpy(brigfile, parallel_bc_path);
  strncat(brigfile, ".brig", POCL_FILENAME_LENGTH-1);
  strcpy(hsailfile, parallel_bc_path);
  strncat(hsailfile, ".hsail", POCL_FILENAME_LENGTH-1);

  if (pocl_exists(brigfile))
    POCL_MSG_PRINT_INFO("pocl-hsa: using existing BRIG file: \n%s\n",
                        brigfile);
  else
    {
      // TODO call llvm via c++ interface like pocl_llvm_codegen()
      POCL_MSG_PRINT_INFO("pocl-hsa: BRIG file not found,"
                          " compiling parallel.bc to brig file: \n%s\n",
                          parallel_bc_path);


      char* args1[] = { LLVM_LLC, "-O2", "-march=hsail64", "-filetype=asm",
                        "-o", hsailfile, parallel_bc_path, NULL };
      if ((error = pocl_run_command (args1)))
        {
          POCL_MSG_PRINT_INFO("pocl-hsa: llc exit status %i\n", error);
          return error;
        }

      char* args2[] = { HSAIL_ASM, "-o", brigfile, hsailfile, NULL };
      if ((error = pocl_run_command (args2)))
        {
          POCL_MSG_PRINT_INFO("pocl-hsa: HSAILasm exit status %i\n", error);
          return error;
        }
    }

  return 0;
}

void
pocl_hsa_compile_kernel (_cl_command_node *cmd, cl_kernel kernel,
                         cl_device_id device)
{
  char brigfile[POCL_FILENAME_LENGTH];
  char *brig_blob;

  pocl_hsa_device_data_t *d =
    (pocl_hsa_device_data_t*)device->data;

  hsa_executable_t final_obj;

  POCL_LOCK (d->pocl_hsa_compilation_lock);

  int error = pocl_llvm_generate_workgroup_function (device, kernel,
                                        cmd->command.run.local_x,
                                        cmd->command.run.local_y,
                                        cmd->command.run.local_z);
  if (error)
    {
      POCL_MSG_PRINT_GENERAL ("HSA: pocl_llvm_generate_workgroup_function()"
                              " failed for kernel %s\n", kernel->name);
      assert (error == 0);
    }

  unsigned i;
  for (i = 0; i<HSA_KERNEL_CACHE_SIZE; i++)
    if (d->kernel_cache[i].kernel == kernel)
      {
        POCL_MSG_PRINT_INFO("kernel.hsa_exe found in"
                            " kernel cache, returning\n");
        POCL_UNLOCK (d->pocl_hsa_compilation_lock);
        return;
      }

  if (compile_parallel_bc_to_brig(brigfile, kernel, device))
    POCL_ABORT("Compiling LLVM IR -> HSAIL -> BRIG failed.\n");

  POCL_MSG_PRINT_INFO("pocl-hsa: loading binary from file %s.\n", brigfile);
  uint64_t filesize = 0;
  int read = pocl_read_file(brigfile, &brig_blob, &filesize);

  if (read != 0)
    POCL_ABORT("pocl-hsa: could not read the binary.\n");

  POCL_MSG_PRINT_INFO("pocl-hsa: BRIG binary size: %lu.\n", filesize);

  hsa_ext_module_t hsa_module = (hsa_ext_module_t)brig_blob;

  hsa_ext_program_t hsa_program;
  memset (&hsa_program, 0, sizeof (hsa_ext_program_t));

  HSA_CHECK(hsa_ext_program_create
    (HSA_MACHINE_MODEL_LARGE, HSA_PROFILE_FULL,
     HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL,
     &hsa_program));

  HSA_CHECK(hsa_ext_program_add_module (hsa_program, hsa_module));

  hsa_isa_t isa;
  HSA_CHECK(hsa_agent_get_info (d->agent, HSA_AGENT_INFO_ISA, &isa));

  hsa_ext_control_directives_t control_directives;
  memset (&control_directives, 0, sizeof (hsa_ext_control_directives_t));

  hsa_code_object_t code_object;
  HSA_CHECK(hsa_ext_program_finalize
    (hsa_program, isa, 0, control_directives, "",
     HSA_CODE_OBJECT_TYPE_PROGRAM, &code_object));

  HSA_CHECK(hsa_executable_create (d->agent_profile,
                                  HSA_EXECUTABLE_STATE_UNFROZEN,
                                  "", &final_obj));

  HSA_CHECK(hsa_executable_load_code_object (final_obj, d->agent,
                                            code_object, ""));

  HSA_CHECK(hsa_executable_freeze (final_obj, NULL));

  HSA_CHECK(hsa_ext_program_destroy(hsa_program));

  free(brig_blob);

  i = d->kernel_cache_lastptr;
  if (i < HSA_KERNEL_CACHE_SIZE)
    {
      d->kernel_cache[i].kernel = kernel;
      d->kernel_cache[i].hsa_exe.handle = final_obj.handle;
      d->kernel_cache_lastptr++;
    }
  else
    POCL_ABORT ("kernel cache full\n");

  hsa_executable_symbol_t kernel_symbol;

  size_t kernel_name_length = strlen (kernel->name);
  char *symbol = malloc (kernel_name_length + 2);
  symbol[0] = '&';
  symbol[1] = '\0';

  strncat (symbol, kernel->name, kernel_name_length);

  POCL_MSG_PRINT_INFO("pocl-hsa: getting kernel symbol %s.\n", symbol);

  HSA_CHECK(hsa_executable_get_symbol
    (final_obj, NULL, symbol, d->agent, 0, &kernel_symbol));

  free(symbol);

  hsa_symbol_kind_t symtype;
  HSA_CHECK(hsa_executable_symbol_get_info
    (kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &symtype));
  if(symtype != HSA_SYMBOL_KIND_KERNEL)
    POCL_ABORT ("pocl-hsa: the kernel function symbol resolves "
                "to something else than a function\n");

  uint64_t code_handle;
  HSA_CHECK(hsa_executable_symbol_get_info
    (kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &code_handle));

  d->kernel_cache[i].code_handle = code_handle;

  HSA_CHECK(hsa_executable_symbol_get_info (
       kernel_symbol,
       HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
       &d->kernel_cache[i].static_group_size));

  HSA_CHECK(hsa_executable_symbol_get_info (
       kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
       &d->kernel_cache[i].private_size));

  HSA_CHECK(hsa_executable_symbol_get_info (
       kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
       &d->kernel_cache[i].args_segment_size));

  POCL_UNLOCK (d->pocl_hsa_compilation_lock);
}

cl_int
pocl_hsa_uninit (cl_device_id device)
{
  pocl_hsa_device_data_t *d = (pocl_hsa_device_data_t*)device->data;

  if (d->driver_pthread_id)
    {
      POCL_MSG_PRINT_INFO("waiting for HSA device pthread"
                          " to finish its work...\n");
      d->exit_driver_thread = 1;
      void* ptr;
      PTHREAD_CHECK(pthread_join(d->driver_pthread_id, &ptr));
      POCL_MSG_PRINT_INFO("....done.\n");
    }

  unsigned i;
  for (i = 0; i < HSA_KERNEL_CACHE_SIZE; i++)
    if (d->kernel_cache[i].kernel)
      HSA_CHECK(hsa_executable_destroy(d->kernel_cache[i].hsa_exe));

  // TODO: destroy the executables that didn't fit to the kernel
  // cache. Also code objects are not destroyed at the moment.

  hsa_signal_destroy(d->nudge_driver_thread);

  PTHREAD_CHECK(pthread_mutex_destroy(&d->list_mutex));

  POCL_DESTROY_LOCK (d->pocl_hsa_compilation_lock);

  POCL_MEM_FREE(d);
  device->data = NULL;

  return CL_SUCCESS;
}


cl_ulong
pocl_hsa_get_timer_value(void *data)
{
  uint64_t hsa_ts;
  HSA_CHECK(hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &hsa_ts));
  cl_ulong res = (cl_ulong)(hsa_ts *
                            ((pocl_hsa_device_data_t*)data)->timestamp_unit);
  return res;
}

#define PN_ADD(array, p) \
  do { \
    if (array##_size > COMMAND_LIST_SIZE) \
      POCL_ABORT("array overload\n"); \
    array[array##_size++] = p; \
  } \
  while (0)

#define PN_REMOVE(array, index) \
  do { \
    assert(array##_size > 0); \
    array[index] = array[--array##_size]; \
    array[array##_size] = NULL; \
  } \
  while (0)

void
pocl_hsa_submit (_cl_command_node *node, cl_command_queue cq)
{
  cl_device_id device = node->device;
  pocl_hsa_device_data_t *d = device->data;
  unsigned added_to_readylist = 0;

  PTHREAD_CHECK(pthread_mutex_lock(&d->list_mutex));

  node->ready = 1;
  if (pocl_command_is_ready (node->event))
    {
      POCL_UPDATE_EVENT_SUBMITTED (node->event);
      PN_ADD(d->ready_list, node->event);
      added_to_readylist = 1;
    }
  else
    PN_ADD(d->wait_list, node->event);

  POCL_MSG_PRINT_INFO("After Event %u submit: WL : %li, RL: %li\n",
                      node->event->id, d->wait_list_size, d->ready_list_size);

  POCL_UNLOCK_OBJ (node->event);

  PTHREAD_CHECK(pthread_mutex_unlock(&d->list_mutex));

  if (added_to_readylist)
    hsa_signal_subtract_relaxed(d->nudge_driver_thread, 1);

}

void
pocl_hsa_join (cl_device_id device, cl_command_queue cq)
{
  POCL_LOCK_OBJ (cq);
  if (cq->command_count == 0)
    {
      POCL_UNLOCK_OBJ (cq);
      POCL_MSG_PRINT_INFO("pocl-hsa: device->join: empty queue\n");
      return;
    }
  cl_event event = cq->last_event.event;
  assert(event);
  POCL_LOCK_OBJ (event);
  ++event->pocl_refcount;
  POCL_UNLOCK_OBJ (cq);

  POCL_MSG_PRINT_INFO("pocl-hsa: device->join on event %u\n", event->id);

  if (event->status <= CL_COMPLETE)
    {
      POCL_MSG_PRINT_INFO("pocl-hsa: device->join: last event (%u) in queue"
                          " exists, but is complete\n", event->id);
      --event->pocl_refcount;
      POCL_UNLOCK_OBJ (event);
      return;
    }

  while (event->status > CL_COMPLETE)
    {
      pocl_hsa_event_data_t *e_d = (pocl_hsa_event_data_t *)event->data;
      PTHREAD_CHECK (pthread_cond_wait (&e_d->event_cond, &event->pocl_lock));
    }
  POCL_MSG_PRINT_INFO("pocl-hsa: device->join on event %u finished"
                      " with status: %i\n", event->id, event->status);

  assert (event->status <= CL_COMPLETE);
  --event->pocl_refcount;
  POCL_UNLOCK_OBJ (event);
}

void
pocl_hsa_flush (cl_device_id device, cl_command_queue cq)
{
  pocl_hsa_device_data_t *d = (pocl_hsa_device_data_t *)device->data;
  hsa_signal_subtract_relaxed(d->nudge_driver_thread, 1);
}

void
pocl_hsa_notify (cl_device_id device, cl_event event, cl_event finished)
{
  pocl_hsa_device_data_t *d = device->data;
  _cl_command_node *node = event->command;
  int added_to_readylist = 0;
  POCL_MSG_PRINT_INFO("pocl-hsa: notify on event %u \n", event->id);

  if (finished->status < CL_COMPLETE)
    {
      POCL_UPDATE_EVENT_FAILED (event);
      return;
    }

  if (!node->ready)
    return;

  if (pocl_command_is_ready (event))
    {
      if (event->status == CL_QUEUED)
        {
          POCL_UPDATE_EVENT_SUBMITTED (event);
          PTHREAD_CHECK(pthread_mutex_lock(&d->list_mutex));

          size_t i = 0;
          for(i = 0; i < d->wait_list_size; i++)
            if (d->wait_list[i] == event)
              break;
          if (i < d->wait_list_size)
            {
              POCL_MSG_PRINT_INFO("event %u wait_list -> ready_list\n", 
                                  event->id);
              PN_ADD(d->ready_list, event);
              PN_REMOVE(d->wait_list, i);
            }
          else
            POCL_ABORT("cant move event %u from waitlist to"
                       " readylist - not found in waitlist\n", event->id);
          added_to_readylist = 1;
          PTHREAD_CHECK(pthread_mutex_unlock(&d->list_mutex));
        }
      else
        POCL_MSG_WARN ("node->ready was 1 but event %u is"
                       " not queued: status %i!\n",
                       event->id, event->status);
    }

  if (added_to_readylist)
    hsa_signal_subtract_relaxed(d->nudge_driver_thread, 1);
}

void
pocl_hsa_broadcast (cl_event event)
{
  POCL_MSG_PRINT_INFO("pocl-hsa: broadcasting\n");
  pocl_broadcast(event);
}

void
pocl_hsa_wait_event(cl_device_id device, cl_event event)
{
  POCL_MSG_PRINT_INFO("pocl-hsa: device->wait_event on event %u\n", event->id);
  POCL_LOCK_OBJ (event);
  if (event->status <= CL_COMPLETE)
    {
      POCL_MSG_PRINT_INFO("pocl-hsa: device->wain_event: last event"
                          " (%u) in queue exists, but is complete\n", 
                          event->id);
      POCL_UNLOCK_OBJ(event);
      return;
    }
  while (event->status > CL_COMPLETE)
    {
      pocl_hsa_event_data_t *e_d = (pocl_hsa_event_data_t *)event->data;
      PTHREAD_CHECK(pthread_cond_wait(&(e_d->event_cond), &event->pocl_lock));
    }
  POCL_UNLOCK_OBJ(event);

  POCL_MSG_PRINT_INFO("event wait finished with status: %i\n", event->status);
  assert (event->status <= CL_COMPLETE);
}

/* DRIVER PTHREAD part */

/* this is array of "less than 1" conditions for signals,
 * passed to hsa_amd_signal_wait_any() as a readonly argument */
static hsa_signal_value_t signal_ones_array[EVENT_LIST_SIZE+1];
static hsa_signal_condition_t less_than_sigcond_array[EVENT_LIST_SIZE+1];
static int signal_array_initialized = 0;

static void
pocl_hsa_launch(pocl_hsa_device_data_t *d, cl_event event)
{
  POCL_LOCK_OBJ (event);
  _cl_command_node *cmd = event->command;
  cl_kernel kernel = cmd->command.run.kernel;
  struct pocl_context *pc = &cmd->command.run.pc;
  hsa_kernel_dispatch_packet_t *kernel_packet;
  pocl_hsa_device_pthread_data_t* dd = &d->driver_data;
  pocl_hsa_event_data_t *event_data = (pocl_hsa_event_data_t *)event->data;

  pocl_hsa_kernel_cache_t* cached_data = NULL;
  unsigned i;
  for (i = 0; i<HSA_KERNEL_CACHE_SIZE; i++)
    {
      if (d->kernel_cache[i].kernel == kernel)
        cached_data = &d->kernel_cache[i];
    }
  assert(cached_data);

  HSA_CHECK(hsa_memory_allocate(d->kernarg_region,
                                cached_data->args_segment_size,
                                &event_data->actual_kernargs));

  dd->last_queue = (dd->last_queue + 1) % dd->num_queues;
  hsa_queue_t* last_queue = dd->queues[dd->last_queue];
  const uint64_t queueMask = last_queue->size - 1;

  uint64_t packet_id = hsa_queue_add_write_index_relaxed(last_queue, 1);
  while ((packet_id - hsa_queue_load_read_index_acquire(last_queue))
         >= last_queue->size)
    {
      /* device queue is full. TODO this isn't the optimal solution */
      POCL_MSG_WARN("pocl-hsa: queue %" PRIuS " overloaded\n", dd->last_queue);
      usleep(2000);
    }

  kernel_packet =
      &(((hsa_kernel_dispatch_packet_t*)(last_queue->base_address))
        [packet_id & queueMask]);

  kernel_packet->workgroup_size_x = cmd->command.run.local_x;
  kernel_packet->workgroup_size_y = cmd->command.run.local_y;
  kernel_packet->workgroup_size_z = cmd->command.run.local_z;

  kernel_packet->grid_size_x = kernel_packet->grid_size_y
    = kernel_packet->grid_size_z = 1;
  kernel_packet->grid_size_x = pc->num_groups[0] * cmd->command.run.local_x;
  kernel_packet->grid_size_y = pc->num_groups[1] * cmd->command.run.local_y;
  kernel_packet->grid_size_z = pc->num_groups[2] * cmd->command.run.local_z;

  kernel_packet->kernel_object = cached_data->code_handle;
  kernel_packet->private_segment_size = cached_data->private_size;
  uint32_t total_group_size = cached_data->static_group_size;

  HSA_CHECK(hsa_signal_create(1, 1, &d->agent,
                              &kernel_packet->completion_signal));

  setup_kernel_args (d, cmd, (char*)event_data->actual_kernargs,
                     cached_data->args_segment_size, &total_group_size);

  kernel_packet->group_segment_size = total_group_size;

  POCL_MSG_PRINT_INFO("pocl-hsa: kernel's total group size: %u\n",
                      total_group_size);
  if (total_group_size > cmd->device->local_mem_size)
    POCL_ABORT ("pocl-hsa: required local memory > device local memory!\n");

  kernel_packet->kernarg_address = event_data->actual_kernargs;

  typedef union {
    uint32_t header_setup;
    struct {
      uint16_t header;
      uint16_t setup;
    } a;
  } hsa_header_union_t;

  hsa_header_union_t h;
  h.a.header = (uint16_t)HSA_FENCE_SCOPE_SYSTEM
    << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  h.a.header |= (uint16_t)HSA_FENCE_SCOPE_SYSTEM
    << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  h.a.header |= (uint16_t)HSA_PACKET_TYPE_KERNEL_DISPATCH
    << HSA_PACKET_HEADER_TYPE;
  h.a.setup = (uint16_t)cmd->command.run.pc.work_dim
    << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  __atomic_store_n((uint32_t*)(&kernel_packet->header), h.header_setup,
                   __ATOMIC_RELEASE);

  /* ring the doorbell to start execution */
  hsa_signal_store_relaxed (last_queue->doorbell_signal, packet_id);

  if (dd->running_list_size > EVENT_LIST_SIZE)
    POCL_ABORT("running events list too big\n");
  else
    {
      dd->running_events[dd->running_list_size] = event;
      dd->running_signals[dd->running_list_size++].handle
        = kernel_packet->completion_signal.handle;
    }

  POCL_UPDATE_EVENT_RUNNING_UNLOCKED (event);
  POCL_UNLOCK_OBJ (event);
}

static void
pocl_hsa_ndrange_event_finished (pocl_hsa_device_data_t *d, size_t i)
{
  pocl_hsa_device_pthread_data_t* dd = &d->driver_data;

  assert(i < dd->running_list_size);
  cl_event event = dd->running_events[i];

  POCL_LOCK_OBJ (event);
  pocl_hsa_event_data_t *event_data = (pocl_hsa_event_data_t *)event->data;

  POCL_MSG_PRINT_INFO("event %u finished, removing from running_list\n",
                      event->id);
  dd->running_events[i] = dd->running_events[--dd->running_list_size];

#ifdef HAVE_HSA_EXT_AMD_H
  /* TODO Times are reported as ticks in the domain of the HSA system clock. */
  hsa_amd_profiling_dispatch_time_t t;
  HSA_CHECK(hsa_amd_profiling_get_dispatch_time(d->agent,
                                                dd->running_signals[i], &t));
  uint64_t j = t.end - t.start;
  pocl_debug_print_duration(__func__,__LINE__,
                            "HSA NDrange Kernel (HSA clock)", j);
#endif

  hsa_signal_destroy(dd->running_signals[i]);
  dd->running_signals[i] = dd->running_signals[dd->running_list_size];

  hsa_memory_free(event_data->actual_kernargs);

  pocl_mem_manager_free_command (event->command);

  POCL_UNLOCK_OBJ (event);
  POCL_UPDATE_EVENT_COMPLETE (event);

}

static void
check_running_signals(pocl_hsa_device_data_t *d)
{
  unsigned i;
  pocl_hsa_device_pthread_data_t* dd = &d->driver_data;
  for (i = 0; i < dd->running_list_size; i++)
    {
      if (hsa_signal_load_acquire(dd->running_signals[i]) < 1)
        pocl_hsa_ndrange_event_finished(d, i);
    }
}

static int pocl_hsa_run_ready_commands(pocl_hsa_device_data_t *d)
{
  check_running_signals(d);
  int enqueued_ndrange = 0;

  PTHREAD_CHECK(pthread_mutex_lock(&d->list_mutex));
  while(d->ready_list_size)
    {
      cl_event e = d->ready_list[0];
      PN_REMOVE(d->ready_list, 0);
      PTHREAD_CHECK(pthread_mutex_unlock(&d->list_mutex));
      if (e->command->type == CL_COMMAND_NDRANGE_KERNEL)
        {
          pocl_hsa_compile_kernel (e->command,
                                   e->command->command.run.kernel,
                                   e->queue->device);
          pocl_hsa_launch(d, e);
          enqueued_ndrange = 1;
          POCL_MSG_PRINT_INFO("NDrange event %u launched, remove"
                              " from readylist\n", e->id);
        }
      else
        {
          POCL_MSG_PRINT_INFO("running non-NDrange event %u,"
                              " remove from readylist\n", e->id);
          pocl_exec_command(e->command);
        }
      check_running_signals(d);
      PTHREAD_CHECK(pthread_mutex_lock(&d->list_mutex));
    }
  PTHREAD_CHECK(pthread_mutex_unlock(&d->list_mutex));
  return enqueued_ndrange;
}


void*
pocl_hsa_driver_pthread (void * cldev)
{
  size_t i;
  if (!signal_array_initialized)
    {
      signal_array_initialized = 1;
      for (i = 0; i < (EVENT_LIST_SIZE+1); i++)
        {
          signal_ones_array[i] = 1;
          less_than_sigcond_array[i] = HSA_SIGNAL_CONDITION_LT;
        }
    }

  // TODO retain dev?
  cl_device_id device = (cl_device_id)cldev;
  pocl_hsa_device_data_t* d = device->data;
  pocl_hsa_device_pthread_data_t* dd = &d->driver_data;

  /* timeout counter, resets with each new queued kernel to 1/8, then
   * exponentially increases by 40% up to about 3/4 of d->timeout.
   * disabled for now */
  /* uint64_t kernel_timeout_ns = d->timeout >> 3; */

  dd->running_list_size = 0;
  dd->last_queue = 0;
  dd->num_queues = d->hw_schedulers;  // TODO this is somewhat arbitrary.
  POCL_MSG_PRINT_INFO("pocl-hsa: Queues: %" PRIuS "\n", dd->num_queues);

  dd->queues = (hsa_queue_t **) calloc (dd->num_queues, sizeof(hsa_queue_t*));

  uint32_t queue_min_size, queue_max_size;
  HSA_CHECK(hsa_agent_get_info(d->agent, HSA_AGENT_INFO_QUEUE_MIN_SIZE,
                               &queue_min_size));
  HSA_CHECK(hsa_agent_get_info(d->agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                               &queue_max_size));

  uint32_t queue_size = 1 << ((__builtin_ctz(queue_min_size)
                               + __builtin_ctz(queue_max_size)) / 2);
  POCL_MSG_PRINT_INFO("pocl-hsa: queue size: %" PRIu32 "\n", queue_size);

  for (i = 0; i < dd->num_queues; i++)
    {
      HSA_CHECK(hsa_queue_create(d->agent, queue_size,
                                 HSA_QUEUE_TYPE_SINGLE,
                                 hsa_queue_callback, device,
                                 -1, -1, &dd->queues[i]));
#ifdef HAVE_HSA_EXT_AMD_H
      HSA_CHECK(hsa_amd_profiling_set_profiler_enabled(dd->queues[i], 1));
#endif
    }

  while (1)
    {
      /* reset the signal. Disabled for now; see below */
#if 0
      if (pocl_hsa_run_ready_commands(d))
        kernel_timeout_ns = d->timeout >> 3;
#else
      pocl_hsa_run_ready_commands(d);
#endif
      if (d->exit_driver_thread)
        goto EXIT_PTHREAD;

      // wait for anything to happen or timeout
#ifdef HAVE_HSA_EXT_AMD_H
      // FIXME: An ABA race condition here. If there was (another) submit after
      // the previous wait returned, but before this reset, we miss the
      // notification decrement and get stuck if there are no further submits
      // to decrement the 1.
      hsa_signal_store_release(d->nudge_driver_thread, 1);

      if (d->have_wait_any)
        {
          dd->running_signals[dd->running_list_size].handle =
              d->nudge_driver_thread.handle;
          hsa_amd_signal_wait_any(dd->running_list_size+1,
                                  dd->running_signals,
                                  less_than_sigcond_array, signal_ones_array,
                                  d->timeout, HSA_WAIT_STATE_BLOCKED, NULL);
          dd->running_signals[dd->running_list_size].handle = 0;
        }
      else
        {
#endif
#if 0
          if (kernel_timeout_ns < (d->timeout >> 1))
            kernel_timeout_ns = (kernel_timeout_ns * 22937UL) >> 14;
	  // See the above comment. Busy wait for now until a proper
	  // synchronization fix is in place.
          hsa_signal_wait_acquire(d->nudge_driver_thread,
                                  HSA_SIGNAL_CONDITION_LT, 1,
                                  kernel_timeout_ns, HSA_WAIT_STATE_BLOCKED);
#endif

#ifdef HAVE_HSA_EXT_AMD_H
        }
#endif

      if (d->exit_driver_thread)
        goto EXIT_PTHREAD;
    }


EXIT_PTHREAD:
  /* TODO wait for commands to finish... */
  POCL_MSG_PRINT_INFO("pocl-hsa: driver pthread exiting, still "
                      "running evts: %" PRIuS "\n",
                      dd->running_list_size);
  assert(dd->running_list_size == 0);

  for (i = 0; i < dd->num_queues; i++)
    HSA_CHECK(hsa_queue_destroy(dd->queues[i]));
  POCL_MEM_FREE(dd->queues);

  pthread_exit(NULL);
}

void
pocl_hsa_update_event (cl_device_id device, cl_event event, cl_int status)
{
  pocl_hsa_event_data_t *e_d = NULL;

  if(event->data == NULL && status == CL_QUEUED)
    {
      pocl_hsa_event_data_t *e_d =
        (pocl_hsa_event_data_t *) malloc (sizeof(pocl_hsa_event_data_t));
      assert (e_d);
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
      POCL_MSG_PRINT_INFO("HSA: Command complete, event %d\n", event->id);
      event->status = CL_COMPLETE;

      pocl_mem_objs_cleanup (event);

      if (event->queue->properties & CL_QUEUE_PROFILING_ENABLE)
        event->time_end = device->ops->get_timer_value(device->data);

      uint64_t ns = event->time_end - event->time_start;
      pocl_debug_print_duration (__func__,__LINE__,
                                 "HSA NDrange Kernel (host clock)", ns);

      POCL_UNLOCK_OBJ (event);
      device->ops->broadcast (event);
      pocl_update_command_queue (event);
      POCL_LOCK_OBJ (event);

      pthread_cond_signal(&e_d->event_cond);
      break;
    default:
      POCL_MSG_PRINT_INFO ("HSA: EVENT FAILED, event %d\n", event->id);
      event->status = CL_FAILED;

      pocl_mem_objs_cleanup (event);

      if (event->queue->properties & CL_QUEUE_PROFILING_ENABLE)
        event->time_end = device->ops->get_timer_value (device->data);

      POCL_UNLOCK_OBJ (event);
      device->ops->broadcast (event);
      pocl_update_command_queue (event);
      POCL_LOCK_OBJ (event);

      pthread_cond_signal(&e_d->event_cond);
      break;
    }
}

void pocl_hsa_free_event_data (cl_event event)
{
  assert(event->data != NULL);
  free(event->data);
  event->data = NULL;
}
