/* pocl-hsa.c - driver for HSA supported devices. Currently only AMDGCN.

   Copyright (c) 2015 Pekka Jääskeläinen / Tampere University of Technology
                 2015 Charles Chen <0charleschen0@gmail.com>

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

#include "hsa.h"
#include "hsa_ext_finalize.h"
#include "hsa_ext_amd.h"
#include "hsa_ext_image.h"

#include "pocl-hsa.h"
#include "common.h"
#include "devices.h"
#include "pocl_cache.h"
#include "pocl_llvm.h"
#include "pocl_util.h"

#include <assert.h>
#include <string.h>
#include <stdlib.h>

#ifndef _MSC_VER
#  include <sys/time.h>
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#define max(a,b) (((a) > (b)) ? (a) : (b))

/* TODO:
   - allocate buffers with hsa_memory_allocate() to ensure the driver
     works with HSA Base profile agents (assuming all memory is coherent
     requires the Full profile) -- or perhaps a separate hsabase driver
     for the simpler agents.
   - check what is needed to be done only once per agent in the *run(),
     now there's _lots_ of boilerplate per kernel launch
   - local memory support
   - Share the same kernel binary function for all WG sizes as HSA is an
     SPMD target. Now it builds a new one for all WGs due to the tempdir.
   - Do not use the BRIG output of the LLVM-HSAIL branch as it's not going
     to get merged upstream. Use HSAIL text output + libHSAIL's assembler
     instead.
   - clinfo of Ubuntu crashes
   - etc. etc.
*/

struct pocl_hsa_device_data {
  /* Currently loaded kernel. */
  cl_kernel current_kernel;
  /* The HSA kernel agent controlled by the device driver instance. */
  hsa_agent_t *agent;
  /* Queue for pushing work to the agent. */
  hsa_queue_t *queue;
};

struct pocl_supported_hsa_device_properties
{
  char *dev_name;
  const char *llvm_cpu;
  const char *llvm_target_triplet;
  int has_64bit_long;
  cl_ulong max_mem_alloc_size;
  cl_ulong global_mem_size;
  cl_uint vendor_id;
  cl_device_mem_cache_type global_mem_cache_type;
  cl_uint global_mem_cacheline_size;
  cl_uint max_compute_units;
  cl_uint max_clock_frequency;
  cl_ulong max_constant_buffer_size;
  cl_ulong local_mem_size;
};

void
pocl_hsa_init_device_ops(struct pocl_device_ops *ops)
{
  pocl_basic_init_device_ops (ops);

  /* TODO: more descriptive name from HSA probing the device */
  ops->device_name = "hsa";
  ops->init_device_infos = pocl_hsa_init_device_infos;
  ops->probe = pocl_hsa_probe;
  ops->uninit = pocl_hsa_uninit;
  ops->init = pocl_hsa_init;
  ops->free = pocl_hsa_free;
  ops->compile_submitted_kernels = pocl_hsa_compile_submitted_kernels;
  ops->run = pocl_hsa_run;
  ops->read = pocl_basic_read;
  ops->read_rect = pocl_basic_read_rect;
  ops->write = pocl_basic_write;
  ops->write_rect = pocl_basic_write_rect;
  ops->copy = pocl_basic_copy;
  ops->copy_rect = pocl_basic_copy_rect;
  ops->get_timer_value = pocl_basic_get_timer_value;
}

#define MAX_HSA_AGENTS 16

static hsa_agent_t hsa_agents[MAX_HSA_AGENTS];
static int found_hsa_agents = 0;

static hsa_status_t
pocl_hsa_get_agents(hsa_agent_t agent, void *data)
{
  hsa_device_type_t type;
  hsa_status_t stat = hsa_agent_get_info (agent, HSA_AGENT_INFO_DEVICE, &type);
  if (type != HSA_DEVICE_TYPE_GPU)
    {
      return HSA_STATUS_SUCCESS;
    }

  hsa_agents[found_hsa_agents] = agent;
  ++found_hsa_agents;
  return HSA_STATUS_SUCCESS;
}

/*
 * Determines if a memory region can be used for kernarg
 * allocations.
 */
static
hsa_status_t
get_kernarg_memory_region(hsa_region_t region, void* data)
{
  hsa_region_segment_t segment;
  hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
  if (HSA_REGION_SEGMENT_GLOBAL != segment) {
    return HSA_STATUS_SUCCESS;
  }

  hsa_region_global_flag_t flags;
  hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
  if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
    hsa_region_t* ret = (hsa_region_t*) data;
    *ret = region;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

//Get hsa-unsupported device features from hsa device list.
static int num_hsa_device = 1;

static struct _cl_device_id
supported_hsa_devices[MAX_HSA_AGENTS] =
{
  [0] =
  {
    .long_name = "Spectre",
    .llvm_cpu = NULL,                 // native: "kaveri",
    .llvm_target_triplet = "hsail64", // native: "amdgcn--amdhsa"
	.has_64bit_long = 1,
	.max_mem_alloc_size =  592969728,
	.global_mem_size = 2371878912,
	.vendor_id = 0x1002,
	.global_mem_cache_type = 0x2,
	.global_mem_cacheline_size = 64,
	.max_compute_units = 8,
	.max_clock_frequency = 720,
	.max_constant_buffer_size = 65536,
	.local_mem_size = 32768
  },
};

// Detect the HSA device and populate its properties to the device
// struct.
static void
get_hsa_device_features(char* dev_name, struct _cl_device_id* dev)
{

#define COPY_ATTR(ATTR) dev->ATTR = supported_hsa_devices[i].ATTR

  int i;
  for(i = 0; i < num_hsa_device; i++)
    {
      if (strcmp(dev_name, supported_hsa_devices[i].long_name) == 0)
        {
          COPY_ATTR (llvm_cpu);
          COPY_ATTR (llvm_target_triplet);
          COPY_ATTR (has_64bit_long);
          COPY_ATTR (max_mem_alloc_size);
          COPY_ATTR (global_mem_size);
          COPY_ATTR (vendor_id);
          COPY_ATTR (global_mem_cache_type);
          COPY_ATTR (global_mem_cacheline_size);
          COPY_ATTR (max_compute_units);
          COPY_ATTR (max_clock_frequency);
          COPY_ATTR (max_constant_buffer_size);
          COPY_ATTR (local_mem_size);
	      break;
        }
    }
}

void
pocl_hsa_init_device_infos(struct _cl_device_id* dev)
{
  pocl_basic_init_device_infos (dev);
  dev->spmd = CL_TRUE;
  dev->autolocals_to_args = 0;
  hsa_agent_t agent = hsa_agents[found_hsa_agents - 1];
  hsa_status_t stat =
    hsa_agent_get_info (agent, HSA_AGENT_INFO_CACHE_SIZE,
                        &(dev->global_mem_cache_size));
  char dev_name[64] = {0};
  stat = hsa_agent_get_info (agent, HSA_AGENT_INFO_NAME, dev_name);
  dev->long_name = (char*)malloc (64*sizeof(char));
  memcpy(dev->long_name, &dev_name[0], strlen(dev_name));
  dev->short_name = dev->long_name;
  get_hsa_device_features (dev->long_name, dev);

  hsa_device_type_t dev_type;
  stat = hsa_agent_get_info (agent, HSA_AGENT_INFO_DEVICE, &dev_type);
  switch(dev_type)
    {
    case HSA_DEVICE_TYPE_GPU:
      dev->type = CL_DEVICE_TYPE_GPU;
      break;
    case HSA_DEVICE_TYPE_CPU:
	  dev->type = CL_DEVICE_TYPE_CPU;
	  break;
    case HSA_DEVICE_TYPE_DSP:
	  dev->type = CL_DEVICE_TYPE_CUSTOM;
	  break;
    default:
	  POCL_ABORT("Unsupported hsa device type!\n");
	  break;
  }

  hsa_dim3_t grid_size;
  stat = hsa_agent_get_info (agent, HSA_AGENT_INFO_GRID_MAX_DIM, &grid_size);
  dev->max_work_item_sizes[0] = grid_size.x;
  dev->max_work_item_sizes[1] = grid_size.y;
  dev->max_work_item_sizes[2] = grid_size.z;
  stat = hsa_agent_get_info
    (agent, HSA_AGENT_INFO_WORKGROUP_MAX_SIZE, &dev->max_work_group_size);
  /*Image features*/
  hsa_dim3_t image_size;
  stat = hsa_agent_get_info (agent, HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS, &image_size);
  dev->image_max_buffer_size = image_size.x;
  stat = hsa_agent_get_info (agent, HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS, &image_size);
  dev->image2d_max_height = image_size.x;
  dev->image2d_max_width = image_size.y;
  stat = hsa_agent_get_info (agent, HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS, &image_size);
  dev->image3d_max_height = image_size.x;
  dev->image3d_max_width = image_size.y;
  dev->image3d_max_depth = image_size.z;
  // is this directly the product of the dimensions?
  //stat = hsa_agent_get_info(agent, ??, &dev->image_max_array_size);
  stat = hsa_agent_get_info
    (agent, HSA_EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES, &dev->max_read_image_args);
  stat = hsa_agent_get_info
    (agent, HSA_EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES, &dev->max_write_image_args);
  stat = hsa_agent_get_info
    (agent, HSA_EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS, &dev->max_samplers);
}

unsigned int
pocl_hsa_probe(struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count(ops->device_name);

  POCL_MSG_PRINT_INFO("pocl-hsa: found %d env devices with %s.\n",
                      env_count, ops->device_name);

  /* No hsa env specified, the user did not request for HSA agents. */
  if (env_count <= 0)
    return 0;

  if (hsa_init() != HSA_STATUS_SUCCESS)
    {
      POCL_ABORT("pocl-hsa: hsa_init() failed.");
    }

  if (hsa_iterate_agents(pocl_hsa_get_agents, NULL) !=
      HSA_STATUS_SUCCESS)
    {
      assert (0 && "pocl-hsa: could not get agents.");
    }
  POCL_MSG_PRINT_INFO("pocl-hsa: found %d agents.\n", found_hsa_agents);
  return found_hsa_agents;
}

void
pocl_hsa_init (cl_device_id device, const char* parameters)
{
  struct pocl_hsa_device_data *d;
  static int global_mem_id;
  static int first_hsa_init = 1;
  hsa_device_type_t dev_type;
  hsa_status_t status;

  if (first_hsa_init)
    {
      first_hsa_init = 0;
      global_mem_id = device->dev_id;
    }
  device->global_mem_id = global_mem_id;

  d = (struct pocl_hsa_device_data *) malloc (sizeof (struct pocl_hsa_device_data));
  d->current_kernel = NULL;
  device->data = d;

  assert (found_hsa_agents > 0);

  /* TODO: support controlling multiple agents.
     Now all pocl devices control the same one. */
  d->agent = &hsa_agents[0];

  // TODO: figure out proper private_segment_size and
  // group_segment_size
  if (hsa_queue_create(*d->agent, 1, HSA_QUEUE_TYPE_MULTI, NULL, NULL,
                       4096, 4096, &d->queue) != HSA_STATUS_SUCCESS)
    {
      POCL_ABORT("pocl-hsa: could not create the queue.");
    }
}

void *
pocl_hsa_malloc (void *device_data, cl_mem_flags flags,
		    size_t size, void *host_ptr)
{
  void *b;

  if (flags & CL_MEM_COPY_HOST_PTR)
    {
      b = pocl_memalign_alloc(MAX_EXTENDED_ALIGNMENT, size);
      if (b != NULL)
	    {
	      memcpy(b, host_ptr, size);
	      hsa_memory_register(host_ptr, size);
	      return b;
	    }
      return NULL;
    }

  if (flags & CL_MEM_USE_HOST_PTR && host_ptr != NULL)
    {
      hsa_memory_register(host_ptr, size);
      return host_ptr;
    }
  b = pocl_memalign_alloc(MAX_EXTENDED_ALIGNMENT, size);
  if (b != NULL)
    return b;
  return NULL;
}

void
pocl_hsa_free (void *data, cl_mem_flags flags, void *ptr)
{
  if (flags & CL_MEM_USE_HOST_PTR)
  {
    return;
  }
  // TODO: hsa_memory_deregister() (needs size)
  POCL_MEM_FREE(ptr);
}

static hsa_status_t
symbol_printer
(hsa_executable_t exe, hsa_executable_symbol_t symbol, void *data)
{
  size_t length;
  hsa_executable_symbol_get_info (symbol, HSA_CODE_SYMBOL_INFO_NAME_LENGTH, &length);

  char *name = malloc(length);
  hsa_executable_symbol_get_info (symbol, HSA_CODE_SYMBOL_INFO_NAME, name);

  POCL_MSG_PRINT_INFO("pocl-hsa: symbol name=%s\n", name);
  free (name);

  return HSA_STATUS_SUCCESS;
}

static void
setup_kernel_args (struct pocl_hsa_device_data *d,
                   _cl_command_node *cmd,
                   char *arg_space,
                   size_t max_args_size)
{
  char *write_pos = arg_space;
  const char *last_pos = arg_space + max_args_size;

#define CHECK_SPACE(DSIZE)                                   \
  do {                                                       \
    if (write_pos + (DSIZE) > last_pos)                      \
      POCL_ABORT("pocl-hsa: too many kernel arguments!\n");  \
  } while (0)

  for (size_t i = 0; i < cmd->command.run.kernel->num_args; ++i)
    {
      struct pocl_argument *al = &(cmd->command.run.arguments[i]);
      if (cmd->command.run.kernel->arg_info[i].is_local)
        {
          POCL_ABORT_UNIMPLEMENTED("pocl-hsa: local buffers not implemented.");
#if 0
    	  //For further info,
    	  //Please refer to https://github.com/HSAFoundation/HSA-Runtime-AMD/issues/8
    	  /*
    	  if(args_offset%8 !=0 )
			  args_offset = (args_offset+8)/8;
		  */
          memcpy(write_pos, &dynamic_local_address, sizeof(uint64_t));
    	  kernel_packet.group_segment_size += al->size;
          write_pos += sizeof(uint64_t);
    	  dynamic_local_address += sizeof(uint64_t);
#endif
        }
      else if (cmd->command.run.kernel->arg_info[i].type == POCL_ARG_TYPE_POINTER)
        {
          CHECK_SPACE (sizeof (uint64_t));
          /* Assuming the pointers are 64b (or actually the same as in
             host) due to HSA. TODO: the 32b profile. */

          if (al->value == NULL)
            {
              uint64_t temp = 0;
              memcpy (write_pos, &temp, sizeof (uint64_t));
            }
          else
            {
        	  uint64_t temp = (uint64_t)(*(cl_mem *)
				  (al->value))->device_ptrs[cmd->device->dev_id].mem_ptr;
              memcpy (write_pos, &temp, sizeof(uint64_t));
            }
          write_pos += sizeof(uint64_t);
#if 0
          /* TODO: It's legal to pass a NULL pointer to clSetKernelArguments. In
             that case we must pass the same NULL forward to the kernel.
             Otherwise, the user must have created a buffer with per device
             pointers stored in the cl_mem. */
          if (al->value == NULL)
            {
              arguments[i] = malloc (sizeof (void *));
              *(void **)arguments[i] = NULL;
            }
          else
            arguments[i] =
              &((*(cl_mem *) (al->value))->device_ptrs[cmd->device->dev_id].mem_ptr);
#endif
        }
      else if (cmd->command.run.kernel->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
        {
          POCL_ABORT_UNIMPLEMENTED("pocl-hsa: image arguments not implemented.\n");
        }
      else if (cmd->command.run.kernel->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)
        {
          POCL_ABORT_UNIMPLEMENTED("pocl-hsa: sampler arguments not implemented.\n");
        }
      else
        {
          // Scalars.
          CHECK_SPACE (al->size);
          memcpy (write_pos, al->value, al->size);
          write_pos += al->size;
        }
    }

  for (size_t i = cmd->command.run.kernel->num_args;
       i < cmd->command.run.kernel->num_args + cmd->command.run.kernel->num_locals;
       ++i)
    {
      POCL_ABORT_UNIMPLEMENTED("hsa: automatic local buffers not implemented.");
#if 0
      al = &(cmd->command.run.arguments[i]);
      arguments[i] = malloc (sizeof (void *));
      *(void **)(arguments[i]) = pocl_hsa_malloc (data, 0, al->size, NULL);
#endif
    }
}

void
pocl_hsa_run(void *data, _cl_command_node* cmd)
{
  struct pocl_hsa_device_data *d;
  unsigned i;
  cl_kernel kernel = cmd->command.run.kernel;
  struct pocl_context *pc = &cmd->command.run.pc;
  hsa_signal_value_t initial_value = 1;
  hsa_kernel_dispatch_packet_t *kernel_packet;
  hsa_signal_t kernel_completion_signal;
  hsa_region_t region;
  int status;

  assert (data != NULL);
  d = data;
  d->current_kernel = kernel;

  const uint32_t queueMask = d->queue->size - 1;
  uint64_t queue_index =
    hsa_queue_load_write_index_relaxed (d->queue);
  kernel_packet =
    &(((hsa_kernel_dispatch_packet_t*)(d->queue->base_address))[queue_index & queueMask]);

  kernel_packet->setup |= 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;

  /* Process the kernel arguments. Convert the opaque buffer
     pointers to real device pointers, allocate dynamic local
     memory buffers, etc. */
#if 0
  uint64_t dynamic_local_address = kernel_packet.group_segment_size;
#else
  // TODO
  uint64_t dynamic_local_address = 2048;
#endif

  kernel_packet->workgroup_size_x = cmd->command.run.local_x;
  kernel_packet->workgroup_size_y = cmd->command.run.local_y;
  kernel_packet->workgroup_size_z = cmd->command.run.local_z;

  kernel_packet->grid_size_x = kernel_packet->grid_size_y = kernel_packet->grid_size_z = 1;
  kernel_packet->grid_size_x = pc->num_groups[0] * cmd->command.run.local_x;
  kernel_packet->grid_size_y = pc->num_groups[1] * cmd->command.run.local_y;
  kernel_packet->grid_size_z = pc->num_groups[2] * cmd->command.run.local_z;

//  kernel_packet.header.type = HSA_PACKET_TYPE_DISPATCH;
//  kernel_packet.header.acquire_fence_scope = HSA_FENCE_SCOPE_SYSTEM;
//  kernel_packet.header.release_fence_scope = HSA_FENCE_SCOPE_SYSTEM;
//  kernel_packet.header.barrier = 1;

  hsa_executable_t *exe = (hsa_executable_t *)cmd->command.run.device_data[0];
  hsa_code_object_t *code_obj = (hsa_code_object_t *)cmd->command.run.device_data[1];

  status = hsa_executable_load_code_object (*exe, *d->agent, *code_obj, "");
  if (status != HSA_STATUS_SUCCESS)
    POCL_ABORT ("pocl-hsa: error while loading the code object.\n");

  status = hsa_executable_freeze (*exe, NULL);
  if (status != HSA_STATUS_SUCCESS)
    POCL_ABORT ("pocl-hsa: error while loading the code object.\n");

  hsa_executable_symbol_t kernel_symbol;

  //hsa_executable_iterate_symbols (*exe, symbol_printer, NULL);

  size_t kernel_name_length = strlen (kernel->name);
  char *symbol = malloc (kernel_name_length + 2);
  symbol[0] = '&';
  symbol[1] = '\0';

  strncat (symbol, kernel->name, kernel_name_length);

  POCL_MSG_PRINT_INFO("pocl-hsa: getting kernel symbol %s.\n", symbol);

  status = hsa_executable_get_symbol
    (*exe, NULL, symbol, *d->agent, 0, &kernel_symbol);
  if (status != HSA_STATUS_SUCCESS)
    POCL_ABORT ("pocl-hsa: unable to get the kernel function symbol\n");

  uint64_t code_handle;
  status = hsa_executable_symbol_get_info
    (kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &code_handle);
  if (status != HSA_STATUS_SUCCESS)
    POCL_ABORT ("pocl-hsa: unable to get the code handle for the kernel function.\n");

  kernel_packet->kernel_object = code_handle;

  status = hsa_signal_create(initial_value, 0, NULL, &kernel_completion_signal);
  assert (status == HSA_STATUS_SUCCESS);

  kernel_packet->completion_signal = kernel_completion_signal;

  /*
   * Allocate the kernel argument buffer from the correct region.
   */
  uint32_t args_segment_size;
  status = hsa_executable_symbol_get_info
    (kernel_symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
     &args_segment_size);

  hsa_region_t kernarg_region;
  kernarg_region.handle = (uint64_t)-1;
  hsa_agent_iterate_regions (*d->agent, get_kernarg_memory_region, &kernarg_region);

  void *args;

  status = hsa_memory_allocate(kernarg_region, args_segment_size, &args);
  if (status != HSA_STATUS_SUCCESS)
    POCL_ABORT ("pocl-hsa: unable to allocate argument memory.\n");

  setup_kernel_args (d, cmd, (char*)args, args_segment_size);

  kernel_packet->kernarg_address = args;

  uint64_t header = 0;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  header |= HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
   __atomic_store_n((uint16_t*)(&kernel_packet->header), header, __ATOMIC_RELEASE);

   /*
    * Increment the write index and ring the doorbell to dispatch the kernel.
    */
   hsa_queue_store_write_index_relaxed (d->queue, queue_index + 1);
   hsa_signal_store_relaxed (d->queue->doorbell_signal, queue_index);

  /* Launch the kernel by allocating a slot in the queue, writing the
     command to it, signaling the update with a door bell and finally,
     block waiting until finish signalled with the completion_signal. */

#if 0
  const uint32_t queue_mask = d->queue->size - 1;
  uint64_t queue_index = hsa_queue_load_write_index_relaxed(d->queue);
  hsa_signal_value_t sigval;
  ((hsa_kernel_dispatch_packet_t*)(d->queue->base_address))[queue_index & queue_mask] =
    kernel_packet;
  hsa_queue_store_write_index_relaxed(d->queue, queue_index + 1);
  hsa_signal_store_relaxed(d->queue->doorbell_signal, queue_index);
#endif

  hsa_signal_value_t sigval =
    hsa_signal_wait_acquire
    (kernel_completion_signal, HSA_SIGNAL_CONDITION_LT, 1,
     (uint64_t)(-1), HSA_WAIT_STATE_ACTIVE);

  for (i = 0; i < kernel->num_args; ++i)
    {
      if (kernel->arg_info[i].is_local)
        {
#if 0
          pocl_hsa_free (data, 0, *(void **)(arguments[i]));
          POCL_MEM_FREE(arguments[i]);
#endif
        }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
        {
#if 0
          pocl_hsa_free (data, 0, *(void **)(arguments[i]));
          POCL_MEM_FREE(arguments[i]);
#endif
        }
#if 0
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_SAMPLER ||
               (kernel->arg_info[i].type == POCL_ARG_TYPE_POINTER &&
                *(void**)args->kernel_args[i] == NULL))
        {
          POCL_MEM_FREE(arguments[i]);
        }
#endif
    }
  for (i = kernel->num_args;
       i < kernel->num_args + kernel->num_locals;
       ++i)
    {
#if 0
      pocl_hsa_free(data, 0, *(void **)(arguments[i]));
      POCL_MEM_FREE(arguments[i]);
#endif
    }
  free(args);
}

static void
compile (_cl_command_node *cmd)
{
  int error, status;
  char bytecode[POCL_FILENAME_LENGTH];
  char objfile[POCL_FILENAME_LENGTH];
  FILE *file;
  char *blob;
  size_t file_size, got_size;

  struct pocl_hsa_device_data *d =
    (struct pocl_hsa_device_data*)cmd->device->data;

  error = snprintf (bytecode, POCL_FILENAME_LENGTH,
                    "%s/%s", cmd->command.run.tmp_dir,
                    POCL_PARALLEL_BC_FILENAME);
  assert (error >= 0);

  error = snprintf (objfile, POCL_FILENAME_LENGTH,
                    "%s/%s.o", cmd->command.run.tmp_dir,
                    POCL_PARALLEL_BC_FILENAME);
  assert (error >= 0);

  error = pocl_llvm_codegen (cmd->command.run.kernel, cmd->device, bytecode, objfile);
  assert (error == 0);

  POCL_MSG_PRINT_INFO("pocl-hsa: loading binary from file %s.\n", objfile);
  file = fopen (objfile, "rb");
  assert (file != NULL);

  cmd->command.run.device_data = malloc (sizeof(void*)*2);

  file_size = pocl_file_size (file);
  blob = malloc (file_size);
  got_size = fread (blob, 1, file_size, file);

  if (file_size != got_size)
    POCL_ABORT ("pocl-hsa: could not read the binary.\n");

  hsa_ext_module_t module = (hsa_ext_module_t)blob;

  cmd->command.run.device_data[1] = blob;

  hsa_ext_program_t program;
  memset (&program, 0, sizeof (hsa_ext_program_t));

  status = hsa_ext_program_create
    (HSA_MACHINE_MODEL_LARGE, HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL,
     &program);

  if (status != HSA_STATUS_SUCCESS)
    POCL_ABORT ("pocl-hsa: error while building the HSA program.\n");

  status = hsa_ext_program_add_module (program, module);
  if (status != HSA_STATUS_SUCCESS)
    POCL_ABORT ("pocl-hsa: error while adding the BRIG module to the HSA program.\n");

  hsa_isa_t isa;
  status = hsa_agent_get_info (*d->agent, HSA_AGENT_INFO_ISA, &isa);

  if (status != HSA_STATUS_SUCCESS)
    POCL_ABORT ("pocl-hsa: error while getting the ISA info for the HSA AGENT.\n");

  hsa_ext_control_directives_t control_directives;
  memset (&control_directives, 0, sizeof (hsa_ext_control_directives_t));

  hsa_code_object_t *code_object = malloc (sizeof (hsa_code_object_t));
  status = hsa_ext_program_finalize
    (program, isa, 0, control_directives, "", HSA_CODE_OBJECT_TYPE_PROGRAM, code_object);

  if (status != HSA_STATUS_SUCCESS)
    POCL_ABORT ("pocl-hsa: error finalizing the program.\n");

  status = hsa_ext_program_destroy (program);
  if (status != HSA_STATUS_SUCCESS)
    POCL_ABORT ("pocl-hsa: error destroying the program.\n");

  hsa_executable_t *exe = malloc(sizeof(hsa_executable_t));
  status = hsa_executable_create (HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN, "", exe);
  if (status != HSA_STATUS_SUCCESS)
    POCL_ABORT ("pocl-hsa: error while creating an executable.\n");

  cmd->command.run.device_data[0] = exe;
  cmd->command.run.device_data[1] = code_object;
  fclose (file);
}

void
pocl_hsa_uninit (cl_device_id device)
{
  struct pocl_hsa_device_data *d = (struct pocl_hsa_device_data*)device->data;
  POCL_MEM_FREE(d);
  device->data = NULL;
}

void
pocl_hsa_compile_submitted_kernels (_cl_command_node *cmd)
{
  if (cmd->type == CL_COMMAND_NDRANGE_KERNEL)
    compile (cmd);
}
