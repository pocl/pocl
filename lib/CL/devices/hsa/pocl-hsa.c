/* pocl-hsa.c - driver for HSA supported devices.

   Copyright (c) 2015-2020 Pekka Jääskeläinen
                 2015 Charles Chen <ccchen@pllab.cs.nthu.edu.tw>
                      Shao-chung Wang <scwang@pllab.cs.nthu.edu.tw>
                 2015-2018 Michal Babej <michal.babej@tut.fi>

   Short snippets borrowed from the MatrixMultiplication example in
   the HSA runtime library sources (c) 2014 HSA Foundation Inc.

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
/* Some example code snippets copied verbatim from vector_copy.c of
 * HSA-Runtime-AMD: */
/* Copyright 2014 HSA Foundation Inc.  All Rights Reserved.
 *
 * HSAF is granting you permission to use this software and documentation (if
 * any) (collectively, the "Materials") pursuant to the terms and conditions
 * of the Software License Agreement included with the Materials.  If you do
 * not have a copy of the Software License Agreement, contact the  HSA
 * Foundation for a copy. Redistribution and use in source and binary forms,
 * with or without modification, are permitted provided that the following
 * conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * WITH THE SOFTWARE.
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
#include "config2.h"

#if defined(HAVE_HSA_EXT_AMD_H) && AMD_HSA == 1

#include "hsa_ext_amd.h"

#endif

#include "pocl-hsa.h"
#include "common.h"
#include "common_driver.h"
#include "devices.h"
#include "pocl-hsa.h"
#include "pocl_cache.h"
#include "pocl_context.h"
#include "pocl_file_util.h"
#include "pocl_llvm.h"
#include "pocl_mem_management.h"
#include "pocl_spir.h"
#include "pocl_local_size.h"
#include "pocl_util.h"

#include <assert.h>
#include <limits.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#ifndef _MSC_VER
#  include <sys/wait.h>
#  include <sys/time.h>
#  include <sys/types.h>
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#define max(a,b) (((a) > (b)) ? (a) : (b))

/* TODO: The kernel cache is never shrunk. We need a hook that is called back
   when clReleaseKernel is called to get a safe point where to release the
   kernel entry from the inmemory cache. */
#define HSA_KERNEL_CACHE_SIZE 4096
#define COMMAND_LIST_SIZE 4096
#define EVENT_LIST_SIZE 511

typedef struct pocl_hsa_event_data_s {
  /* Address of the space where this kernel launch's arguments were stored. */
  void *kernargs;
  /* The location of the pocl context struct in the Agent's global mem. */
  void *context;
  pthread_cond_t event_cond;
} pocl_hsa_event_data_t;

/* Simple statically-sized kernel data cache */
/* for caching kernel dispatch data, binaries etc */
typedef struct pocl_hsa_kernel_cache_s {
  cl_kernel kernel;

  /* use kernel hash as key */
  pocl_kernel_hash_t kernel_hash;

  hsa_executable_t hsa_exe;
  uint64_t code_handle;

  uint32_t private_size;
  uint32_t static_group_size;
  uint32_t args_segment_size;

  /* For native non-SPMD targets, we cache work-group functions specialized
     to specific work-group sizes. */
  uint64_t local_x;
  uint64_t local_y;
  uint64_t local_z;

  /* If global offset must be zero for this WG function. */
  int goffs_zero;

  /* Maximum grid dimension this WG function works with. */
  size_t max_grid_dim_width;

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
  /* The parent device struct. */
  cl_device_id device;
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

  /* printf buffer */
  void *printf_buffer;
  uint32_t *printf_write_pos;

} pocl_hsa_device_data_t;

void pocl_hsa_compile_kernel_hsail (_cl_command_node *cmd, cl_kernel kernel,
                                    cl_device_id device, int specialize);

void pocl_hsa_compile_kernel_native (_cl_command_node *cmd, cl_kernel kernel,
                                     cl_device_id device, int specialize);

static void*
pocl_hsa_malloc(pocl_global_mem_t *mem, size_t size, hsa_region_t r);

void
pocl_hsa_init_device_ops(struct pocl_device_ops *ops)
{
  /* TODO: more descriptive name from HSA probing the device */
  ops->device_name = "hsa";
  ops->probe = pocl_hsa_probe;
  ops->uninit = pocl_hsa_uninit;
  ops->reinit = pocl_hsa_reinit;
  ops->init = pocl_hsa_init;
  ops->alloc_mem_obj = pocl_hsa_alloc_mem_obj;
  ops->free = pocl_hsa_free;
  ops->run = NULL;

  ops->read = pocl_driver_read;
  ops->read_rect = pocl_driver_read_rect;
  ops->write = pocl_driver_write;
  ops->write_rect = pocl_driver_write_rect;
  ops->map_mem = pocl_driver_map_mem;
  ops->unmap_mem = pocl_driver_unmap_mem;
  ops->get_mapping_ptr = pocl_driver_get_mapping_ptr;
  ops->free_mapping_ptr = pocl_driver_free_mapping_ptr;
  ops->memfill = pocl_driver_memfill;
  ops->copy = pocl_hsa_copy;
  ops->copy_rect = pocl_driver_copy_rect;
  ops->compute_local_size = pocl_default_local_size_optimizer;

  ops->get_device_info_ext = NULL;

  ops->svm_free = pocl_hsa_svm_free;
  ops->svm_alloc = pocl_hsa_svm_alloc;
  ops->svm_copy = pocl_hsa_svm_copy;
  ops->svm_fill = pocl_basic_svm_fill;
  ops->svm_register = pocl_hsa_svm_register;
  ops->svm_unregister = pocl_hsa_svm_unregister;

  // new driver api (out-of-order)
  ops->submit = pocl_hsa_submit;
  ops->join = pocl_hsa_join;
  ops->flush = pocl_hsa_flush;
  ops->notify = pocl_hsa_notify;
  ops->broadcast = pocl_hsa_broadcast;
  ops->wait_event = pocl_hsa_wait_event;

  ops->build_source = pocl_driver_build_source;
  ops->link_program = pocl_driver_link_program;
  ops->build_binary = pocl_driver_build_binary;
  ops->free_program = pocl_driver_free_program;
  ops->setup_metadata = pocl_driver_setup_metadata;
  ops->supports_binary = pocl_driver_supports_binary;
  ops->build_poclbinary = pocl_driver_build_poclbinary;
#if HSAIL_ENABLED
  ops->compile_kernel = pocl_hsa_compile_kernel_hsail;
#else
  ops->compile_kernel = pocl_hsa_compile_kernel_native;
#endif

  ops->update_event = pocl_hsa_update_event;
  ops->notify_event_finished = pocl_hsa_notify_event_finished;
  ops->free_event_data = pocl_hsa_free_event_data;
  ops->init_target_machine = NULL;
  ops->wait_event = pocl_hsa_wait_event;
  ops->build_hash = pocl_hsa_build_hash;
  ops->init_build = pocl_hsa_init_build;
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

/* HSA unsupported device features are hard coded in a known Agent
   list and detected by the advertised agent name string. */
#define HSA_NUM_KNOWN_HSA_AGENTS 2

static const char *default_native_final_linkage_flags[] =
  {"-nostartfiles", HOST_LD_FLAGS_ARRAY, NULL};

static const char *phsa_native_device_aux_funcs[] =
  {"_pocl_run_all_wgs", "_pocl_finish_all_wgs", "_pocl_spawn_wg", NULL};

#define AMD_VENDOR_ID 0x1002

static struct _cl_device_id supported_hsa_devices[HSA_NUM_KNOWN_HSA_AGENTS]
    = { [0] = { .long_name = "Spectre",
                .llvm_cpu = (HSAIL_ENABLED ? NULL : "kaveri"),
                .llvm_target_triplet
                = (HSAIL_ENABLED ? "hsail64" : "amdgcn--amdhsa"),
                .spmd = CL_TRUE,
                .autolocals_to_args = POCL_AUTOLOCALS_TO_ARGS_NEVER,
                .device_alloca_locals = CL_FALSE,
                .context_as_id = SPIR_ADDRESS_SPACE_GLOBAL,
                .args_as_id = SPIR_ADDRESS_SPACE_GLOBAL,
                .has_64bit_long = 1,
                .vendor_id = AMD_VENDOR_ID,
                .global_mem_cache_type = CL_READ_WRITE_CACHE,
                .max_constant_buffer_size = 65536,
                .local_mem_type = CL_LOCAL,
                .endian_little = CL_TRUE,
                .extensions = HSA_DEVICE_EXTENSIONS,
                .device_side_printf = !HSAIL_ENABLED,
                .printf_buffer_size = PRINTF_BUFFER_SIZE * 1024,
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
                .native_vector_width_double = 1 },
        [1] = { .long_name = "phsa generic CPU agent",
                .llvm_cpu = NULL,
                .llvm_target_triplet = (HSAIL_ENABLED ? "hsail64" : NULL),
                .spmd = CL_FALSE,
                .autolocals_to_args
                = (HSAIL_ENABLED ? POCL_AUTOLOCALS_TO_ARGS_NEVER
                                 : POCL_AUTOLOCALS_TO_ARGS_ALWAYS),
                .device_alloca_locals = CL_TRUE,
                .context_as_id = SPIR_ADDRESS_SPACE_GLOBAL,
                .args_as_id = SPIR_ADDRESS_SPACE_GLOBAL,
                .has_64bit_long = 1,
                .vendor_id = 0xffff,
                .global_mem_cache_type = CL_READ_WRITE_CACHE,
                .max_constant_buffer_size = 65536,
                .local_mem_type = CL_LOCAL,
                .endian_little = !(WORDS_BIGENDIAN),
                .extensions = HSA_DEVICE_EXTENSIONS,
                .device_side_printf = !HSAIL_ENABLED,
                .printf_buffer_size = PRINTF_BUFFER_SIZE * 1024,
                .preferred_wg_size_multiple = 1,
                /* We want to exploit the widest vector types in HSAIL
                   for the CPUs assuming they have some sort of SIMD ISE
                   which the finalizer than can more readily utilize.  */
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
                .native_vector_width_double = 16,
                .final_linkage_flags = default_native_final_linkage_flags,
                .device_aux_functions
                = (HSAIL_ENABLED ? NULL : phsa_native_device_aux_funcs) } };

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
  for(i = 0; i < HSA_NUM_KNOWN_HSA_AGENTS; i++)
    {
      if (strcmp(dev_name, supported_hsa_devices[i].long_name) == 0)
        {
	  COPY_ATTR (llvm_cpu);
	  COPY_ATTR (llvm_target_triplet);
	  COPY_ATTR (spmd);
	  COPY_ATTR (autolocals_to_args);
          COPY_ATTR (device_alloca_locals);
          COPY_ATTR (context_as_id);
          COPY_ATTR (args_as_id);
          if (!HSAIL_ENABLED)
            {
              /* TODO: Add a CMake variable or HSA description string
                 autodetection to control these. */
              if (dev->llvm_cpu == NULL)
                dev->llvm_cpu = get_llvm_cpu_name ();
              if (dev->llvm_target_triplet == NULL)
                dev->llvm_target_triplet = OCL_KERNEL_TARGET;
              dev->arg_buffer_launcher = CL_TRUE;
            }
          COPY_ATTR (has_64bit_long);
          COPY_ATTR (vendor_id);
          COPY_ATTR (global_mem_cache_type);
          COPY_ATTR (max_constant_buffer_size);
          COPY_ATTR (local_mem_type);
          COPY_ATTR (endian_little);
          COPY_ATTR (preferred_wg_size_multiple);
          COPY_ATTR (extensions);
	  COPY_ATTR (final_linkage_flags);
	  COPY_ATTR (device_aux_functions);
	  COPY_ATTR (device_side_printf);
	  COPY_ATTR (printf_buffer_size);
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

/* (Re)initialize the device data. dev is the device driver part, count is
   the number of devices of this type initialized so far. */
static pocl_hsa_device_data_t *
init_dev_data (cl_device_id dev, int count)
{
  pocl_hsa_device_data_t *d
      = (pocl_hsa_device_data_t *)calloc (1, sizeof (pocl_hsa_device_data_t));

  dev->data = d;
  d->device = dev;
  d->hw_schedulers = 3;
  POCL_INIT_LOCK (d->pocl_hsa_compilation_lock);

  /* Before the first HSA device, re-init the runtime. */
  if (count == 0 && found_hsa_agents == 0)
    {
      HSA_CHECK (hsa_init ());
      HSA_CHECK (hsa_iterate_agents (pocl_hsa_get_agents_callback, NULL));
    }
  d->agent.handle = hsa_agents[count].handle;

  HSA_CHECK (hsa_agent_iterate_regions (
      d->agent, setup_agent_memory_regions_callback, d));
  bool boolarg = 0;
  HSA_CHECK (hsa_region_get_info (
      d->global_region, HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED, &boolarg));
  assert (boolarg != 0);

  pocl_reinit_system_memory ();

  HSA_CHECK (hsa_signal_create (1, 1, &d->agent, &d->nudge_driver_thread));
  d->exit_driver_thread = 0;
  pthread_mutexattr_t mattr;
  PTHREAD_CHECK (pthread_mutexattr_init (&mattr));
  PTHREAD_CHECK (pthread_mutexattr_settype (&mattr, PTHREAD_MUTEX_ERRORCHECK));
  PTHREAD_CHECK (pthread_mutex_init (&d->list_mutex, &mattr));

  uint64_t hsa_freq;
  HSA_CHECK (
      hsa_system_get_info (HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &hsa_freq));
  d->timeout = hsa_freq; // 1 second in hsa units
  d->timestamp_unit = (1000000000.0 / (double)hsa_freq);
  POCL_MSG_PRINT_INFO ("HSA timestamp frequency: %" PRIu64 "\n", hsa_freq);
  POCL_MSG_PRINT_INFO ("HSA timeout: %" PRIu64 "\n", d->timeout);
  POCL_MSG_PRINT_INFO ("HSA timestamp unit: %g\n", d->timestamp_unit);

#if AMD_HSA == 1
  /* TODO check at runtime */
  d->have_wait_any = 1;
#endif

#if AMD_HSA == 1
  if (dev->vendor_id == AMD_VENDOR_ID)
    {
      char booltest = 0;
      HSA_CHECK (hsa_region_get_info (
          d->global_region, HSA_AMD_REGION_INFO_HOST_ACCESSIBLE, &booltest));
      assert (booltest != 0);
    }
#endif

  size_t sizearg;
  HSA_CHECK (hsa_region_get_info (d->global_region,
                                  HSA_REGION_INFO_ALLOC_MAX_SIZE, &sizearg));
  dev->max_mem_alloc_size = sizearg;

  /* For some reason, the global region size returned is 128 Terabytes...
   * for now, use the max alloc size, it seems to be a much more reasonable
   * value.
   * HSA_CHECK(hsa_region_get_info(d->global_region, HSA_REGION_INFO_SIZE,
   *                               &sizearg));
   */
  HSA_CHECK (
      hsa_region_get_info (d->global_region, HSA_REGION_INFO_SIZE, &sizearg));
  dev->global_mem_size = sizearg;
  if (dev->global_mem_size > 16 * 1024 * 1024 * (uint64_t)1024)
    dev->global_mem_size = dev->max_mem_alloc_size;

  HSA_CHECK (
      hsa_region_get_info (d->group_region, HSA_REGION_INFO_SIZE, &sizearg));
  dev->local_mem_size = sizearg;

  HSA_CHECK (hsa_region_get_info (
      d->global_region, HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT, &sizearg));
  dev->mem_base_addr_align = max (sizearg, MAX_EXTENDED_ALIGNMENT);

  HSA_CHECK (hsa_agent_get_info (d->agent, HSA_AGENT_INFO_PROFILE,
                                 &d->agent_profile));

  if (d->agent_profile == HSA_PROFILE_FULL)
    pocl_setup_device_for_system_memory (dev);

  dev->profile = "FULL_PROFILE";
  dev->has_own_timer = CL_TRUE;

  dev->compiler_available = CL_TRUE;
  dev->linker_available = CL_TRUE;

  dev->profiling_timer_resolution = (size_t) (d->timestamp_unit) || 1;

  if (dev->device_side_printf)
    {
      d->printf_buffer = pocl_hsa_malloc
        (dev->global_memory, dev->printf_buffer_size, d->global_region);

      d->printf_write_pos = pocl_hsa_malloc
        (dev->global_memory, sizeof (size_t), d->global_region);
    }

  d->exit_driver_thread = 0;
  PTHREAD_CHECK (pthread_create (&d->driver_pthread_id, NULL,
                                 &pocl_hsa_driver_pthread, dev));

  return d;
}

cl_int
pocl_hsa_init (unsigned j, cl_device_id dev, const char *parameters)
{
  pocl_init_default_device_infos (dev);

  SETUP_DEVICE_CL_VERSION(HSA_DEVICE_CL_VERSION_MAJOR,
                          HSA_DEVICE_CL_VERSION_MINOR)

  dev->spmd = CL_TRUE;
  dev->arg_buffer_launcher = CL_FALSE;
  dev->autolocals_to_args = POCL_AUTOLOCALS_TO_ARGS_NEVER;
  dev->device_alloca_locals = CL_FALSE;

  dev->local_as_id = SPIR_ADDRESS_SPACE_LOCAL;
  dev->constant_as_id = SPIR_ADDRESS_SPACE_CONSTANT;

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
  HSA_CHECK (hsa_agent_get_info (agent, HSA_AGENT_INFO_MACHINE_MODEL, &model));
  dev->address_bits = (model == HSA_MACHINE_MODEL_LARGE) ? 64 : 32;

  uint16_t wg_sizes[3];
  HSA_CHECK (
      hsa_agent_get_info (agent, HSA_AGENT_INFO_WORKGROUP_MAX_DIM, &wg_sizes));

  int max_wg = pocl_get_int_option ("POCL_MAX_WORK_GROUP_SIZE", 0);
  if (max_wg > 0)
    {
      wg_sizes[0] = min (wg_sizes[0], max_wg);
      wg_sizes[1] = min (wg_sizes[1], max_wg);
      wg_sizes[2] = min (wg_sizes[2], max_wg);
    }

  dev->max_work_item_sizes[0] = wg_sizes[0];
  dev->max_work_item_sizes[1] = wg_sizes[1];
  dev->max_work_item_sizes[2] = wg_sizes[2];

  /* Specialize WG functions for grid dimensions of width <= USHRT_MAX. */
  dev->grid_width_specialization_limit = USHRT_MAX;

  HSA_CHECK(hsa_agent_get_info
    (agent, HSA_AGENT_INFO_WORKGROUP_MAX_SIZE, &dev->max_work_group_size));

  if (max_wg > 0)
    dev->max_work_group_size = max_wg;

  /* Assume a small maximum work-group size indicates also the desire to
     maximally utilize it. */
  if (max_wg <= 256)
    dev->ops->compute_local_size = pocl_wg_utilization_maximizer;

  if (AMD_HSA && dev->vendor_id == AMD_VENDOR_ID)
    {
#if AMD_HSA == 1
      uint32_t temp;
      HSA_CHECK (hsa_agent_get_info (agent, HSA_AMD_AGENT_INFO_CACHELINE_SIZE,
                                     &temp));
      dev->global_mem_cacheline_size = temp;

      HSA_CHECK (hsa_agent_get_info (
          agent, HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &temp));
      dev->max_compute_units = temp;

      HSA_CHECK (hsa_agent_get_info (
          agent, HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY, &temp));
      dev->max_clock_frequency = temp;
#endif
    }
  else
    {
      /* Could not use AMD extensions to find out CU/frequency of the device.
	 Using dummy values. */
      dev->global_mem_cacheline_size = 64;
      dev->max_compute_units = 4;
      dev->max_clock_frequency = 700;
    }

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

  dev->svm_allocation_priority = 2;
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
  dev->on_dev_queue_props
      = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE;
  dev->on_host_queue_props = CL_QUEUE_PROFILING_ENABLE;

  pocl_hsa_device_data_t *d = init_dev_data (dev, j);

  /* 0 is the host memory shared with all drivers that use it */
  if (d->agent_profile == HSA_PROFILE_FULL)
    dev->global_mem_id = 0;

  return CL_SUCCESS;
}


static void*
pocl_hsa_malloc (pocl_global_mem_t *mem, size_t size, hsa_region_t r)
{
  void *b = NULL;

  if (hsa_memory_allocate(r, size, &b) != HSA_STATUS_SUCCESS)
    {
      POCL_MSG_PRINT_INFO ("hsa_memory_allocate failed");
      return NULL;
    }

  if (b)
    POCL_MSG_PRINT_INFO("HSA malloc'ed : size %" PRIuS " @ %p\n", size, b);

  /* TODO: Due to lack of align parameter to the HSA allocation function, we
     should align the buffer here ourselves.  For now, let's just hope that
     the called HSA implementation wide aligns (currently to 128).  */
  if ((uint64_t)b % MAX_EXTENDED_ALIGNMENT > 0)
    POCL_MSG_WARN("HSA runtime returned a buffer with smaller alignment "
		  "than %d", MAX_EXTENDED_ALIGNMENT);

  return b;
}

void
pocl_hsa_copy (void *data,
               pocl_mem_identifier * dst_mem_id,
               cl_mem dst_buf,
               pocl_mem_identifier * src_mem_id,
               cl_mem src_buf,
               size_t dst_offset,
               size_t src_offset,
               size_t size)
{
  void *__restrict__ dst_ptr = dst_mem_id->mem_ptr;
  void *__restrict__ src_ptr = src_mem_id->mem_ptr;
  if ((src_ptr + src_offset) == (dst_ptr + dst_offset))
    return;
  HSA_CHECK (
      hsa_memory_copy (dst_ptr + dst_offset, src_ptr + src_offset, size));
}

cl_int
pocl_hsa_alloc_mem_obj (cl_device_id device, cl_mem mem_obj, void *host_ptr)
{
  /* if we share global memory with CPU, let the CPU driver allocate it */
  if (device->global_mem_id == 0)
    return pocl_basic_alloc_mem_obj (device, mem_obj, host_ptr);

  /* ... otherwise allocate it via HSA. */
  pocl_mem_identifier *p = &mem_obj->device_ptrs[device->global_mem_id];
  pocl_global_mem_t *gmem = device->global_memory;
  pocl_hsa_device_data_t* d = device->data;
  void *b = pocl_hsa_malloc (gmem, mem_obj->size, d->global_region);
  p->mem_ptr = b;
  p->version = 0;

  if (b == NULL)
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  else
    return CL_SUCCESS;
}

void
pocl_hsa_free (cl_device_id device, cl_mem memobj)
{
  /* if we share global memory with CPU, let the CPU driver free it */
  if (device->global_mem_id == 0)
    return pocl_basic_free (device, memobj);

  /* ... otherwise free it via HSA. */
  cl_mem_flags flags = memobj->flags;
  pocl_mem_identifier *p = &memobj->device_ptrs[device->global_mem_id];
  hsa_memory_free(p->mem_ptr);
  p->mem_ptr = NULL;
  p->version = 0;
}

void pocl_hsa_svm_register (cl_device_id dev, void *host_ptr, size_t size)
{
  POCL_MSG_PRINT_HSA ("hsa_memory_register()\n");
  hsa_memory_register(host_ptr, size);
}

void pocl_hsa_svm_unregister (cl_device_id dev, void *host_ptr, size_t size)
{
  POCL_MSG_PRINT_HSA ("hsa_memory_deregister()\n");
  hsa_memory_deregister(host_ptr, size);
}


static void
setup_kernel_args (pocl_hsa_device_data_t *d,
                   _cl_command_node *cmd,
                   pocl_hsa_event_data_t *event_data,
                   size_t max_args_size,
                   uint32_t *total_group_size)
{
  char *write_pos = event_data->kernargs;
  const char *last_pos = event_data->kernargs + max_args_size;
  cl_kernel kernel = cmd->command.run.kernel;
  pocl_kernel_metadata_t *meta = kernel->meta;

  POCL_MSG_PRINT_INFO ("setup_kernel_args for %s\n",
		       cmd->command.run.kernel->name);
#define CHECK_AND_ALIGN_SPACE(DSIZE)                         \
  do {                                                       \
    if (write_pos + (DSIZE) > last_pos)                      \
      POCL_ABORT("pocl-hsa: too many kernel arguments!\n");  \
    unsigned unaligned = (intptr_t)write_pos % DSIZE;        \
    if (unaligned > 0) write_pos += (DSIZE - unaligned);     \
  } while (0)

  size_t i;
  for (i = 0; i < meta->num_args + meta->num_locals; ++i)
    {
      struct pocl_argument *al = &(cmd->command.run.arguments[i]);

      if (i >= meta->num_args || ARG_IS_LOCAL (meta->arg_info[i]))
        {
	  size_t buf_size = ARG_IS_LOCAL (meta->arg_info[i]) ?
	    al->size : meta->local_sizes[i - meta->num_args];
          if (d->device->device_alloca_locals)
            {
              /* Local buffers are allocated in the device side work-group
                 launcher. Let's pass only the sizes of the local args in
                 the arg buffer. */
              assert (sizeof (size_t) == 8);
              CHECK_AND_ALIGN_SPACE (sizeof (size_t));
              memcpy (write_pos, &buf_size, sizeof (size_t));
              write_pos += sizeof (size_t);
            }
          else if (HSAIL_ENABLED)
            {
              CHECK_AND_ALIGN_SPACE (sizeof (uint32_t));
              memcpy (write_pos, total_group_size, sizeof (uint32_t));
              *total_group_size += (uint32_t)buf_size;
              write_pos += sizeof (uint32_t);
            }
          else
            assert (0 && "Unsupported local mem allocation scheme.");
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER)
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
              uint64_t dev_ptr = 0;
              if (al->is_svm)
                dev_ptr = (uint64_t) (*(void **)al->value);
              else
                {
                  cl_mem m = *(cl_mem *)al->value;
                  dev_ptr
                      = (uint64_t)m->device_ptrs[cmd->device->dev_id].mem_ptr;
                  if (m->flags & CL_MEM_USE_HOST_PTR
                      && d->agent_profile == HSA_PROFILE_BASE)
                    {
                      POCL_MSG_PRINT_INFO (
                          "HSA: Copy HOST_PTR allocated %lu byte buffer "
                          "from %p to %lx due to having a BASE profile "
                          "agent.\n",
                          m->size, m->mem_host_ptr, dev_ptr);
                      hsa_memory_copy ((void *)dev_ptr, m->mem_host_ptr,
                                       m->size);
                    }
                }

              dev_ptr += al->offset;
              memcpy (write_pos, &dev_ptr, sizeof (uint64_t));
            }
          POCL_MSG_PRINT_INFO (
              "arg %lu (global ptr) written to %lx val %lx arg offs %d\n", i,
              (uint64_t)write_pos, *(uint64_t *)write_pos,
              (int)(write_pos - (char*)event_data->kernargs));
          write_pos += sizeof (uint64_t);
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
        {
          POCL_ABORT_UNIMPLEMENTED("pocl-hsa: image arguments"
                                   " not implemented.\n");
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)
        {
          POCL_ABORT_UNIMPLEMENTED("pocl-hsa: sampler arguments"
                                   " not implemented.\n");
        }
      else
        {
          // Scalars.
          CHECK_AND_ALIGN_SPACE (al->size);
          memcpy (write_pos, al->value, al->size);
          POCL_MSG_PRINT_INFO (
              "arg %lu (scalar) written to %lx val %x offs %d\n", i,
              (uint64_t)write_pos, *(uint32_t *)al->value,
              (int)(write_pos - (char*)event_data->kernargs));
          write_pos += al->size;
        }
    }

  CHECK_AND_ALIGN_SPACE(sizeof (uint64_t));

  /* Copy the context object to HSA allocated global memory to ensure Base
     profile agents can access it. */

  event_data->context = pocl_hsa_malloc
    (d->device->global_memory, POCL_CONTEXT_SIZE (d->device->address_bits),
     d->global_region);

  if (d->device->address_bits == 64)
    memcpy (event_data->context, &cmd->command.run.pc, sizeof (struct pocl_context));
  else
    POCL_CONTEXT_COPY64TO32 (event_data->context, &cmd->command.run.pc);

  memcpy (write_pos, &event_data->context, sizeof (event_data->context));
  POCL_MSG_PRINT_INFO ("A %d-bit context object was written at %p offs %d\n",
                       d->device->address_bits, event_data->context,
                       (int)(write_pos - (char*)event_data->kernargs));
  write_pos += sizeof (uint64_t);

  /* MUST TODO: free the local buffers after finishing the kernel in case of
     host side allocation.
   */
}

static int
compile_parallel_bc_to_brig (char *brigfile, _cl_command_node *cmd,
                             int specialize)
{
  int error;
  char hsailfile[POCL_FILENAME_LENGTH];
  char parallel_bc_path[POCL_FILENAME_LENGTH];
  _cl_command_run *run_cmd = &cmd->command.run;

  pocl_cache_work_group_function_path (parallel_bc_path,
                                       run_cmd->kernel->program, cmd->device_i,
                                       run_cmd->kernel, cmd, specialize);

  strcpy (brigfile, parallel_bc_path);
  strncat (brigfile, ".brig", POCL_FILENAME_LENGTH-1);
  strcpy (hsailfile, parallel_bc_path);
  strncat (hsailfile, ".hsail", POCL_FILENAME_LENGTH-1);

  if (pocl_exists (brigfile))
    POCL_MSG_PRINT_INFO("pocl-hsa: using existing BRIG file: \n%s\n",
                        brigfile);
  else
    {
      // TODO call llvm via c++ interface like pocl_llvm_codegen()
      POCL_MSG_PRINT_HSA ("BRIG file not found,"
                          " compiling parallel.bc to brig file: \n%s\n",
                          parallel_bc_path);


      char* args1[] = { LLVM_LLC, "-O2", "-march=hsail64", "-filetype=asm",
                        "-o", hsailfile, parallel_bc_path, NULL };
      if ((error = pocl_run_command (args1)))
        {
          POCL_MSG_PRINT_HSA ("llc exit status %i\n", error);
          return error;
        }

      char* args2[] = { HSAIL_ASM, "-o", brigfile, hsailfile, NULL };
      if ((error = pocl_run_command (args2)))
        {
          POCL_MSG_PRINT_HSA ("HSAILasm exit status %i\n", error);
          return error;
        }
    }

  return 0;
}

static pocl_hsa_kernel_cache_t *
pocl_hsa_find_mem_cached_kernel (pocl_hsa_device_data_t *d,
                                 _cl_command_run *cmd)
{
  size_t i;
  for (i = 0; i < HSA_KERNEL_CACHE_SIZE; i++)
    {
      if (((d->kernel_cache[i].kernel == NULL)
           || (memcmp (d->kernel_cache[i].kernel_hash, cmd->hash,
                       sizeof (pocl_kernel_hash_t))
               != 0)))
        continue;

      if (d->kernel_cache[i].local_x == cmd->pc.local_size[0]
          && d->kernel_cache[i].local_y == cmd->pc.local_size[1]
          && d->kernel_cache[i].local_z == cmd->pc.local_size[2]
          && (!d->kernel_cache[i].goffs_zero
              || (cmd->pc.global_offset[0] == 0
                  && cmd->pc.global_offset[1] == 0
                  && cmd->pc.global_offset[2] == 0))
          && pocl_cmd_max_grid_dim_width (cmd)
                 <= d->kernel_cache[i].max_grid_dim_width)
        return &d->kernel_cache[i];
    }
  return NULL;
}

void
pocl_hsa_compile_kernel_native (_cl_command_node *cmd, cl_kernel kernel,
                                cl_device_id device, int specialize)
{
  pocl_hsa_device_data_t *d = (pocl_hsa_device_data_t*)device->data;

  _cl_command_run *run_cmd = &cmd->command.run;

  POCL_LOCK (d->pocl_hsa_compilation_lock);
  assert (cmd->command.run.kernel == kernel);
  char *binary_fn = pocl_check_kernel_disk_cache (cmd, specialize);
  if (pocl_hsa_find_mem_cached_kernel (d, &cmd->command.run) != NULL)
    {
        POCL_MSG_PRINT_INFO("built kernel found in mem cache\n");
        POCL_UNLOCK (d->pocl_hsa_compilation_lock);
        return;
    }

  POCL_MSG_PRINT_INFO ("pocl-hsa: loading native binary from file %s.\n",
                       binary_fn);

  uint64_t elf_size;
  FILE *elf_file;
  elf_file = fopen(binary_fn, "rb");
  if (elf_file == NULL)
    POCL_ABORT ("pocl-hsa: could not get the file size of the native "
                "binary\n");

  /* This assumes phsa-runtime's deserialization input format
     which stores the following data: */
  uint32_t metadata_size =
    sizeof (uint64_t) /* The ELF bin size. ELF bin follows. */ +
    sizeof (hsa_isa_t) +
    sizeof (hsa_default_float_rounding_mode_t) + sizeof (hsa_profile_t) +
    sizeof (hsa_machine_model_t);

  /* TODO: Use HSA's deserialization interface to store the final binary
     to disk so we don't need to wrap it here and fix to phsa's format.  */
  fseek (elf_file, 0, SEEK_END);
  elf_size = ftell (elf_file);
  fseek (elf_file, 0, SEEK_SET);

  uint64_t blob_size = metadata_size + elf_size;

  uint8_t *blob = malloc (blob_size);
  uint8_t *wpos = blob;

  memcpy (wpos, &elf_size, sizeof (elf_size));
  wpos += sizeof (elf_size);

  uint64_t read_size;
  if (fread (wpos, 1, elf_size, elf_file) != elf_size)
    POCL_ABORT("pocl-hsa: could not read the native ELF binary.\n");
  fclose (elf_file);

  POCL_MSG_PRINT_INFO("pocl-hsa: native binary size: %lu.\n", elf_size);

  wpos += elf_size;

  /* Assume the rest of the HSA properties are OK as zero. */
  memset (wpos, 0, metadata_size - sizeof (uint64_t));

  hsa_executable_t exe;
  hsa_code_object_t obj;

  HSA_CHECK (hsa_executable_create (d->agent_profile,
                                    HSA_EXECUTABLE_STATE_UNFROZEN, "", &exe));

  HSA_CHECK(hsa_code_object_deserialize (blob, blob_size, "", &obj));

  HSA_CHECK(hsa_executable_load_code_object (exe, d->agent, obj, ""));

  HSA_CHECK(hsa_executable_freeze (exe, NULL));

  free (blob);

  int i = d->kernel_cache_lastptr;
  if (i < HSA_KERNEL_CACHE_SIZE)
    {
      d->kernel_cache[i].kernel = kernel;
      memcpy (d->kernel_cache[i].kernel_hash, cmd->command.run.hash,
              sizeof (pocl_kernel_hash_t));
      d->kernel_cache[i].local_x = run_cmd->pc.local_size[0];
      d->kernel_cache[i].local_y = run_cmd->pc.local_size[1];
      d->kernel_cache[i].local_z = run_cmd->pc.local_size[2];
      d->kernel_cache[i].goffs_zero = run_cmd->pc.global_offset[0] == 0
                                      && run_cmd->pc.global_offset[1] == 0
                                      && run_cmd->pc.global_offset[2] == 0;

      size_t max_grid_width = pocl_cmd_max_grid_dim_width (&cmd->command.run);
      d->kernel_cache[i].max_grid_dim_width
          = max_grid_width > device->grid_width_specialization_limit
                ? SIZE_MAX
                : device->grid_width_specialization_limit;
      d->kernel_cache[i].hsa_exe.handle = exe.handle;
      d->kernel_cache_lastptr++;
    }
  else
    POCL_ABORT ("kernel cache full\n");

  hsa_executable_symbol_t kernel_symbol;

  const char *launcher_name_tmpl = "phsa_kernel.%s_grid_launcher";
  size_t launcher_name_length =
    strlen (kernel->name) + strlen (launcher_name_tmpl) + 1;
  char *symbol_name = malloc (launcher_name_length);

  snprintf (symbol_name, launcher_name_length, launcher_name_tmpl,
            kernel->name);

  POCL_MSG_PRINT_INFO ("pocl-hsa: getting kernel symbol %s.\n", symbol_name);

  HSA_CHECK (hsa_executable_get_symbol (exe, NULL, symbol_name, d->agent, 0,
                                        &kernel_symbol));
  free (symbol_name);

  hsa_symbol_kind_t symtype;
  HSA_CHECK (hsa_executable_symbol_get_info (
      kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &symtype));
  if (symtype != HSA_SYMBOL_KIND_KERNEL)
    POCL_ABORT ("pocl-hsa: the kernel function symbol resolves "
                "to something else than a function\n");

  uint64_t code_handle;
  HSA_CHECK(hsa_executable_symbol_get_info
    (kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &code_handle));

  d->kernel_cache[i].code_handle = code_handle;

  /* Group and private memory allocation is done via pocl, HSA runtime
     should not mind these.  */
  d->kernel_cache[i].static_group_size = 0;
  d->kernel_cache[i].private_size = 0;
  d->kernel_cache[i].args_segment_size = 2048;

  POCL_UNLOCK (d->pocl_hsa_compilation_lock);
  POCL_MSG_PRINT_INFO("pocl-hsa: native kernel compilation for phsa "
		      "finished\n");
}

void
pocl_hsa_compile_kernel_hsail (_cl_command_node *cmd, cl_kernel kernel,
                               cl_device_id device, int specialize)
{
  char brigfile[POCL_FILENAME_LENGTH];
  char *brig_blob;

  pocl_hsa_device_data_t *d = (pocl_hsa_device_data_t*)device->data;

  hsa_executable_t final_obj;

  _cl_command_run *run_cmd = &cmd->command.run;

  POCL_LOCK (d->pocl_hsa_compilation_lock);

  int error = pocl_llvm_generate_workgroup_function (cmd->device_i, device,
                                                     kernel, cmd, specialize);
  if (error)
    {
      POCL_MSG_PRINT_GENERAL ("HSA: pocl_llvm_generate_workgroup_function()"
                              " failed for kernel %s\n", kernel->name);
      assert (error == 0);
    }

  unsigned i;
  if (pocl_hsa_find_mem_cached_kernel (d, run_cmd) != NULL)
    {
        POCL_MSG_PRINT_INFO("built kernel found in mem cache\n");
        POCL_UNLOCK (d->pocl_hsa_compilation_lock);
        return;
    }

  if (compile_parallel_bc_to_brig (brigfile, cmd, specialize))
    POCL_ABORT("Compiling LLVM IR -> HSAIL -> BRIG failed.\n");

  POCL_MSG_PRINT_HSA ("loading binary from file %s.\n", brigfile);
  uint64_t filesize = 0;
  int read = pocl_read_file(brigfile, &brig_blob, &filesize);

  if (read != 0)
    POCL_ABORT("pocl-hsa: could not read the binary.\n");

  POCL_MSG_PRINT_HSA ("BRIG binary size: %lu.\n", filesize);

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

  free (brig_blob);

  i = d->kernel_cache_lastptr;
  if (i < HSA_KERNEL_CACHE_SIZE)
    {
      d->kernel_cache[i].kernel = kernel;
      memcpy (d->kernel_cache[i].kernel_hash, cmd->command.run.hash,
              sizeof (pocl_kernel_hash_t));
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

  POCL_MSG_PRINT_HSA ("getting kernel symbol %s.\n", symbol);

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
pocl_hsa_uninit (unsigned j, cl_device_id device)
{
  assert (found_hsa_agents > 0);
  pocl_hsa_device_data_t *d = (pocl_hsa_device_data_t *)device->data;

  if (d->driver_pthread_id)
    {
      POCL_MSG_PRINT_INFO ("waiting for HSA device pthread"
                           " to finish its work...\n");
      d->exit_driver_thread = 1;
      void* ptr;
      PTHREAD_CHECK(pthread_join(d->driver_pthread_id, &ptr));
      POCL_MSG_PRINT_INFO ("....done.\n");
    }

  if (device->device_side_printf)
    {
      hsa_memory_free (d->printf_buffer);
      hsa_memory_free (d->printf_write_pos);
    }

  unsigned i;
  for (i = 0; i < HSA_KERNEL_CACHE_SIZE; i++)
    if (d->kernel_cache[i].kernel)
      {
        HSA_CHECK (hsa_executable_destroy (d->kernel_cache[i].hsa_exe));
      }

  // TODO: destroy the executables that didn't fit to the kernel
  // cache. Also code objects are not destroyed at the moment.
  hsa_signal_destroy(d->nudge_driver_thread);

  PTHREAD_CHECK(pthread_mutex_destroy(&d->list_mutex));

  POCL_DESTROY_LOCK (d->pocl_hsa_compilation_lock);

  POCL_MEM_FREE(d);
  device->data = NULL;

  // after last device, call HSA runtime shutdown
  if (j == (found_hsa_agents - 1))
    {
      HSA_CHECK (hsa_shut_down ());
      found_hsa_agents = 0;
    }

  return CL_SUCCESS;
}

cl_int
pocl_hsa_reinit (unsigned j, cl_device_id device)
{
  assert (device->data == NULL);
  device->data = init_dev_data (device, j);
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

  PTHREAD_CHECK (pthread_mutex_lock (&d->list_mutex));

  node->ready = 1;
  if (pocl_command_is_ready (node->event))
    {
      pocl_update_event_submitted (node->event);
      PN_ADD(d->ready_list, node->event);
      added_to_readylist = 1;
    }
  else
    PN_ADD(d->wait_list, node->event);

  POCL_MSG_PRINT_INFO("After Event %" PRIu64 " submit: WL : %li, RL: %li\n",
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
      POCL_MSG_PRINT_HSA ("device->join: empty queue\n");
      return;
    }
  cl_event event = cq->last_event.event;
  assert(event);
  POCL_LOCK_OBJ (event);
  POCL_RETAIN_OBJECT_UNLOCKED (event);
  POCL_UNLOCK_OBJ (cq);

  POCL_MSG_PRINT_HSA ("device->join on event %" PRIu64 "\n", event->id);

  if (event->status <= CL_COMPLETE)
    {
      POCL_MSG_PRINT_HSA ("device->join: last event (%" PRIu64 ") in queue"
                          " exists, but is complete\n", event->id);
      goto RETURN;
    }

  while (event->status > CL_COMPLETE)
    {
      pocl_hsa_event_data_t *e_d = (pocl_hsa_event_data_t *)event->data;
      PTHREAD_CHECK (pthread_cond_wait (&e_d->event_cond, &event->pocl_lock));
    }
  POCL_MSG_PRINT_HSA ("device->join on event %" PRIu64 " finished"
                      " with status: %i\n", event->id, event->status);

RETURN:
  assert (event->status <= CL_COMPLETE);
  POCL_UNLOCK_OBJ (event);

  POname (clReleaseEvent) (event);
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
  POCL_MSG_PRINT_HSA ("notify on event %" PRIu64 " \n", event->id);

  if (finished->status < CL_COMPLETE)
    {
      pocl_update_event_failed (event);
      return;
    }

  if (!node->ready)
    return;

  if (pocl_command_is_ready (event))
    {
      if (event->status == CL_QUEUED)
        {
          pocl_update_event_submitted (event);
          PTHREAD_CHECK(pthread_mutex_lock(&d->list_mutex));

          size_t i = 0;
          for(i = 0; i < d->wait_list_size; i++)
            if (d->wait_list[i] == event)
              break;
          if (i < d->wait_list_size)
            {
              POCL_MSG_PRINT_INFO("event %" PRIu64 " wait_list -> ready_list\n",
                                  event->id);
              PN_ADD(d->ready_list, event);
              PN_REMOVE(d->wait_list, i);
            }
          else
            POCL_ABORT("cant move event %" PRIu64 " from waitlist to"
                       " readylist - not found in waitlist\n", event->id);
          added_to_readylist = 1;
          PTHREAD_CHECK(pthread_mutex_unlock(&d->list_mutex));
        }
      else
        POCL_MSG_WARN ("node->ready was 1 but event %" PRIu64 " is"
                       " not queued: status %i!\n",
                       event->id, event->status);
    }

  if (added_to_readylist)
    hsa_signal_subtract_relaxed(d->nudge_driver_thread, 1);
}

void
pocl_hsa_broadcast (cl_event event)
{
  POCL_MSG_PRINT_HSA ("broadcasting\n");
  pocl_broadcast(event);
}

void
pocl_hsa_wait_event(cl_device_id device, cl_event event)
{
  POCL_MSG_PRINT_HSA ("device->wait_event on event %" PRIu64 "\n", event->id);
  POCL_LOCK_OBJ (event);
  if (event->status <= CL_COMPLETE)
    {
      POCL_MSG_PRINT_HSA ("device->wain_event: last event"
                          " (%" PRIu64 ") in queue exists, but is complete\n",
                          event->id);
      POCL_UNLOCK_OBJ(event);
      return;
    }
  while (event->status > CL_COMPLETE)
    {
      pocl_hsa_event_data_t *e_d = (pocl_hsa_event_data_t *)event->data;
      PTHREAD_CHECK (
          pthread_cond_wait (&(e_d->event_cond), &event->pocl_lock));
    }
  POCL_UNLOCK_OBJ (event);

  POCL_MSG_PRINT_INFO("event wait finished with status: %i\n", event->status);
  assert (event->status <= CL_COMPLETE);
}

/* DRIVER PTHREAD part */

#if AMD_HSA == 1
/* this is array of "less than 1" conditions for signals,
 * passed to hsa_amd_signal_wait_any() as a readonly argument */
static hsa_signal_value_t signal_ones_array[EVENT_LIST_SIZE + 1];
static hsa_signal_condition_t less_than_sigcond_array[EVENT_LIST_SIZE + 1];
static int signal_array_initialized = 0;
#endif

static void
pocl_hsa_launch (pocl_hsa_device_data_t *d, cl_event event)
{
  POCL_LOCK_OBJ (event);
  _cl_command_node *cmd = event->command;
  _cl_command_run *run_cmd = &cmd->command.run;
  cl_kernel kernel = cmd->command.run.kernel;
  struct pocl_context *pc = &cmd->command.run.pc;
  hsa_kernel_dispatch_packet_t *kernel_packet;
  pocl_hsa_device_pthread_data_t* dd = &d->driver_data;
  pocl_hsa_event_data_t *event_data = (pocl_hsa_event_data_t *)event->data;

  unsigned i;
  pocl_hsa_kernel_cache_t *cached_data
      = pocl_hsa_find_mem_cached_kernel (d, run_cmd);
  assert (cached_data);

  HSA_CHECK(hsa_memory_allocate (d->kernarg_region,
				 cached_data->args_segment_size,
				 &event_data->kernargs));

  dd->last_queue = (dd->last_queue + 1) % dd->num_queues;
  hsa_queue_t* last_queue = dd->queues[dd->last_queue];
  const uint64_t queue_mask = last_queue->size - 1;

  uint64_t packet_id = hsa_queue_add_write_index_relaxed (last_queue, 1);
  while ((packet_id - hsa_queue_load_read_index_acquire (last_queue))
         >= last_queue->size)
    {
      /* device queue is full. TODO this isn't the optimal solution */
      POCL_MSG_WARN("pocl-hsa: queue %" PRIuS " overloaded\n", dd->last_queue);
      usleep(2000);
    }

  kernel_packet =
      &(((hsa_kernel_dispatch_packet_t*)(last_queue->base_address))
        [packet_id & queue_mask]);

  if (!HSAIL_ENABLED && !d->device->spmd)
    {
      /* For non-SPMD machines with native compilation, we produce a multi-WI
	 WG function with pocl and launch it via the HSA runtime like it was
	 a single-WI WG. */
      kernel_packet->workgroup_size_x = kernel_packet->workgroup_size_y =
	kernel_packet->workgroup_size_z = 1;
    }
  else
    {
      /* Otherwise let the target processor take care of the SPMD grid
         execution. */
      kernel_packet->workgroup_size_x = run_cmd->pc.local_size[0];
      kernel_packet->workgroup_size_y = run_cmd->pc.local_size[1];
      kernel_packet->workgroup_size_z = run_cmd->pc.local_size[2];
    }

  if (d->device->device_side_printf)
    {
      pc->printf_buffer = d->printf_buffer;
      pc->printf_buffer_capacity = d->device->printf_buffer_size;
      bzero (d->printf_write_pos, sizeof (size_t));
      pc->printf_buffer_position = d->printf_write_pos;
    }

  /* TODO: Dynamic WG sizes. */

  /* For SPMD devices we let the processor (HSA runtime) control the
     grid execution unless we are using our own WG launcher that
     uses the context struct. */
  if (!d->device->spmd || d->device->arg_buffer_launcher)
    {
      pc->local_size[0] = run_cmd->pc.local_size[0];
      pc->local_size[1] = run_cmd->pc.local_size[1];
      pc->local_size[2] = run_cmd->pc.local_size[2];
    }

  kernel_packet->grid_size_x = kernel_packet->grid_size_y
    = kernel_packet->grid_size_z = 1;
  kernel_packet->grid_size_x =
    pc->num_groups[0] * kernel_packet->workgroup_size_x;
  kernel_packet->grid_size_y =
    pc->num_groups[1] * kernel_packet->workgroup_size_y;
  kernel_packet->grid_size_z =
    pc->num_groups[2] * kernel_packet->workgroup_size_z;

  kernel_packet->kernel_object = cached_data->code_handle;
  kernel_packet->private_segment_size = cached_data->private_size;
  uint32_t total_group_size = cached_data->static_group_size;

  HSA_CHECK (
      hsa_signal_create (1, 1, &d->agent, &kernel_packet->completion_signal));

  setup_kernel_args (d, cmd, event_data,
                     cached_data->args_segment_size, &total_group_size);

  kernel_packet->group_segment_size = total_group_size;

  POCL_MSG_PRINT_HSA ("kernel's total group size: %u\n",
                      total_group_size);
  POCL_MSG_PRINT_INFO ("kernel command grid size %u x %u x %u\n",
                       kernel_packet->grid_size_x, kernel_packet->grid_size_y,
                       kernel_packet->grid_size_z);


  if (total_group_size > cmd->device->local_mem_size)
    POCL_ABORT ("pocl-hsa: required local memory > device local memory!\n");

  kernel_packet->kernarg_address = event_data->kernargs;

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
  __atomic_store_n ((uint32_t *)(&kernel_packet->header), h.header_setup,
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

  pocl_update_event_running_unlocked (event);
  POCL_UNLOCK_OBJ (event);
}

static void
pocl_hsa_ndrange_event_finished (pocl_hsa_device_data_t *d, size_t i)
{
  pocl_hsa_device_pthread_data_t* dd = &d->driver_data;

  assert(i < dd->running_list_size);
  cl_event event = dd->running_events[i];
  _cl_command_node *node = event->command;

  POCL_LOCK_OBJ (event);
  pocl_hsa_event_data_t *event_data = (pocl_hsa_event_data_t *)event->data;

  POCL_MSG_PRINT_INFO("event %" PRIu64 " finished, removing from running_list\n",
                      event->id);
  dd->running_events[i] = dd->running_events[--dd->running_list_size];

#if AMD_HSA == 1
  /* TODO Times are reported as ticks in the domain of the HSA system clock. */
  hsa_amd_profiling_dispatch_time_t t;
  HSA_CHECK(hsa_amd_profiling_get_dispatch_time(d->agent,
                                                dd->running_signals[i], &t));
  uint64_t j = t.end - t.start;
  pocl_debug_print_duration (__func__, __LINE__,
                             "HSA NDrange Kernel (HSA clock)", j);
#endif

  hsa_signal_destroy (dd->running_signals[i]);
  dd->running_signals[i] = dd->running_signals[dd->running_list_size];

  hsa_memory_free (event_data->kernargs);
  hsa_memory_free (event_data->context);

  POCL_UNLOCK_OBJ (event);

  if (d->device->device_side_printf && *d->printf_write_pos > 0)
    {
      write (STDOUT_FILENO, d->printf_buffer, *d->printf_write_pos);
      bzero (d->printf_write_pos, sizeof (size_t));
    }

  POCL_UPDATE_EVENT_COMPLETE (event);

  pocl_ndrange_node_cleanup (node);
  pocl_mem_manager_free_command (node);
}

static void
check_running_signals (pocl_hsa_device_data_t *d)
{
  unsigned i;
  pocl_hsa_device_pthread_data_t *dd = &d->driver_data;
  for (i = 0; i < dd->running_list_size; i++)
    {
      if (hsa_signal_load_acquire (dd->running_signals[i]) < 1)
        {
          pocl_hsa_ndrange_event_finished (d, i);
        }
    }
}

static int
pocl_hsa_run_ready_commands (pocl_hsa_device_data_t *d)
{
  check_running_signals (d);
  int enqueued_ndrange = 0;

  PTHREAD_CHECK (pthread_mutex_lock (&d->list_mutex));
  while (d->ready_list_size)
    {
      cl_event e = d->ready_list[0];
      PN_REMOVE (d->ready_list, 0);
      PTHREAD_CHECK (pthread_mutex_unlock (&d->list_mutex));
      if (e->command->type == CL_COMMAND_NDRANGE_KERNEL)
        {
          d->device->ops->compile_kernel (
              e->command, e->command->command.run.kernel, e->queue->device, 1);
          pocl_hsa_launch (d, e);
          enqueued_ndrange = 1;
          POCL_MSG_PRINT_INFO ("NDrange event %" PRIu64 " launched, remove"
                               " from readylist\n", e->id);
        }
      else
        {
          POCL_MSG_PRINT_INFO ("running non-NDrange event %" PRIu64 ","
                               " remove from readylist\n", e->id);
          pocl_exec_command (e->command);
        }
      check_running_signals (d);
      PTHREAD_CHECK (pthread_mutex_lock (&d->list_mutex));
    }
  PTHREAD_CHECK (pthread_mutex_unlock (&d->list_mutex));
  return enqueued_ndrange;
}

void *
pocl_hsa_driver_pthread (void *cldev)
{
  size_t i;
#if AMD_HSA == 1
  if (!signal_array_initialized)
    {
      signal_array_initialized = 1;
      for (i = 0; i < (EVENT_LIST_SIZE+1); i++)
        {
          signal_ones_array[i] = 1;
          less_than_sigcond_array[i] = HSA_SIGNAL_CONDITION_LT;
        }
    }
#endif

  cl_device_id device = (cl_device_id)cldev;
  POCL_RETAIN_OBJECT_UNLOCKED (device);
  pocl_hsa_device_data_t *d = device->data;
  pocl_hsa_device_pthread_data_t *dd = &d->driver_data;

  /* timeout counter, resets with each new queued kernel to 1/8, then
   * exponentially increases by 40% up to about 3/4 of d->timeout.
   * disabled for now */
#if 0
  uint64_t kernel_timeout_ns = d->timeout >> 3;
#endif

  dd->running_list_size = 0;
  dd->last_queue = 0;
  dd->num_queues = d->hw_schedulers;  // TODO this is somewhat arbitrary.
  POCL_MSG_PRINT_HSA ("Queues: %" PRIuS "\n", dd->num_queues);

  dd->queues = (hsa_queue_t **) calloc (dd->num_queues, sizeof(hsa_queue_t*));

  uint32_t queue_min_size, queue_max_size;
  HSA_CHECK(hsa_agent_get_info(d->agent, HSA_AGENT_INFO_QUEUE_MIN_SIZE,
                               &queue_min_size));
  HSA_CHECK(hsa_agent_get_info(d->agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                               &queue_max_size));

  uint32_t queue_size = 1 << ((__builtin_ctz(queue_min_size)
                               + __builtin_ctz(queue_max_size)) / 2);
  POCL_MSG_PRINT_HSA ("queue size: %" PRIu32 "\n", queue_size);

  for (i = 0; i < dd->num_queues; i++)
    {
      HSA_CHECK (hsa_queue_create (d->agent, queue_size, HSA_QUEUE_TYPE_SINGLE,
                                   hsa_queue_callback, device, -1, -1,
                                   &dd->queues[i]));
#if AMD_HSA == 1
      HSA_CHECK (hsa_amd_profiling_set_profiler_enabled (dd->queues[i], 1));
#endif
    }

  while (1)
    {
      /* reset the signal. Disabled for now; see below */
#if 0
      if (pocl_hsa_run_ready_commands (d))
        kernel_timeout_ns = d->timeout >> 3;
#else
      pocl_hsa_run_ready_commands (d);
#endif
      if (d->exit_driver_thread)
        goto EXIT_PTHREAD;

        // wait for anything to happen or timeout
#if AMD_HSA == 1
      // FIXME: An ABA race condition here. If there was (another) submit after
      // the previous wait returned, but before this reset, we miss the
      // notification decrement and get stuck if there are no further submits
      // to decrement the 1.
      hsa_signal_store_release (d->nudge_driver_thread, 1);

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
          hsa_signal_wait_acquire (d->nudge_driver_thread,
				   HSA_SIGNAL_CONDITION_LT, 1,
				   kernel_timeout_ns, HSA_WAIT_STATE_BLOCKED);
#endif

#if AMD_HSA == 1
        }
#endif

      if (d->exit_driver_thread)
        goto EXIT_PTHREAD;
    }

EXIT_PTHREAD:
  /* TODO wait for commands to finish... */
  POCL_MSG_PRINT_HSA ("driver pthread exiting, still "
                      "running evts: %" PRIuS "\n",
                      dd->running_list_size);
  assert(dd->running_list_size == 0);

  for (i = 0; i < dd->num_queues; i++)
    HSA_CHECK (hsa_queue_destroy (dd->queues[i]));
  POCL_MEM_FREE (dd->queues);

  POname (clReleaseDevice) (device);

  pthread_exit (NULL);
}

void
pocl_hsa_notify_event_finished (cl_event event)
{
  pocl_hsa_event_data_t *e_d = (pocl_hsa_event_data_t *)event->data;
  pthread_cond_broadcast (&e_d->event_cond);
}

void
pocl_hsa_update_event (cl_device_id device, cl_event event)
{
  pocl_hsa_event_data_t *e_d = NULL;

  if (event->data == NULL && event->status == CL_QUEUED)
    {
      pocl_hsa_event_data_t *e_d
          = (pocl_hsa_event_data_t *)malloc (sizeof (pocl_hsa_event_data_t));
      assert (e_d);
      pthread_cond_init (&e_d->event_cond, NULL);
      event->data = (void *)e_d;
    }
  else
    {
      e_d = event->data;
    }

  switch (event->status)
    {
    case CL_QUEUED:
      if (event->queue->properties & CL_QUEUE_PROFILING_ENABLE)
        event->time_queue = pocl_hsa_get_timer_value (device->data);
      break;
    case CL_SUBMITTED:
      if (event->queue->properties & CL_QUEUE_PROFILING_ENABLE)
        event->time_submit = pocl_hsa_get_timer_value (device->data);
      break;
    case CL_RUNNING:
      if (event->queue->properties & CL_QUEUE_PROFILING_ENABLE)
        event->time_start = pocl_hsa_get_timer_value (device->data);
      break;
    case CL_FAILED:
    case CL_COMPLETE:
      if (event->queue->properties & CL_QUEUE_PROFILING_ENABLE)
        event->time_end = pocl_hsa_get_timer_value (device->data);
      break;
    }
}

void pocl_hsa_free_event_data (cl_event event)
{
  assert(event->data != NULL);
  free(event->data);
  event->data = NULL;
}

/****** SVM callbacks *****/

void
pocl_hsa_svm_free (cl_device_id dev, void *svm_ptr)
{
  /* TODO we should somehow figure out the size argument
   * and call pocl_free_global_mem */
  HSA_CHECK (hsa_memory_free (svm_ptr));
}

void *
pocl_hsa_svm_alloc (cl_device_id dev, cl_svm_mem_flags flags, size_t size)
{
  if ((flags & CL_MEM_SVM_ATOMICS)
      && ((dev->svm_caps & CL_DEVICE_SVM_ATOMICS) == 0))
    {
      POCL_MSG_ERR ("This device doesn't support SVM Atomics\n");
      return NULL;
    }

  if ((flags & CL_MEM_SVM_FINE_GRAIN_BUFFER)
       && ((dev->svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) == 0))
    {
      POCL_MSG_ERR ("This device doesn't support SVM Fine grained Buffer\n");
      return NULL;
    }

  pocl_hsa_device_data_t *d = (pocl_hsa_device_data_t *)dev->data;
  void *b = NULL;
  HSA_CHECK (hsa_memory_allocate (d->global_region, size, &b));
  return b;
}

void
pocl_hsa_svm_copy (cl_device_id dev, void *__restrict__ dst,
                   const void *__restrict__ src, size_t size)
{
  HSA_CHECK (hsa_memory_copy (dst, src, size));
}

char*
pocl_hsa_init_build (void *data)
{
  if (!((pocl_hsa_device_data_t*)data)->device->device_side_printf)
    return strdup ("-DC99_PRINTF");
  else
    return NULL;
}
