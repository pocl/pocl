﻿/* pocl-vulkan.c - driver for Vulkan Compute API devices.

   Copyright (c) 2018-2021 Michal Babej / Tampere University

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

/********** What works:
 *
 * (hopefully) reasonable memory-type handling with both iGPUs and dGPUs
 *
 * buffer arguments (cl_mem)
 *
 * global offsets
 *
 * unlimited WG counts (implemented via global offsets)
 *
 * POD (plain old data) arguments = integers + structs
 *
 * automatic (hidden) locals; local arguments;
 *
 *
 * pocl-binaries
 *
 * CL_MEM_USE_HOST_PTR (partially)
 *
 * module scope constants support
 *
 * compilation arguments
 *
 ********* Doesn't work / unfinished / non-optimal:
 *
 * VMA (virtual memory allocator) on device memory, currently
 * driver just calls vkAllocateMemory for each cl_mem allocation
 *
 * CL_MEM_ALLOC_HOST_PTR is ignored
 *
 * properly cleanup objects, check for memory leaks
 *
 * fix statically sized data structs
 *
 * descriptor set should be cached (setup once per kernel, then just update)
 *
 * image / sampler support support missing
 *
 * some things that are stored per-kernel should be stored per-program,
 * and v-v (e.g. compiled shader)
 *
 * do transfers on transfer queues not compute queues
 *
 * kernel library - clspv documentation says:
 * "OpenCL C language built-in functions are mapped,
 * where possible, onto their GLSL 4.5 built-in equivalents"
 * - this is not the best, they have unknown precision
 *
 * maybe use push constants for POD arguments instead of UBO
 *
 * stop using deprecated clspv-reflection, instead extract the
 * kernel metadata from the SPIR-V file itself, clspv now puts it there
 *
 * bunch of TODOs all over the place
 */

/********************************************************************/

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <vulkan/vulkan.h>

#ifndef VK_API_VERSION_1_1
#error this program needs Vulkan headers to be at least version 1.1
#endif

#include "common.h"
#include "common_driver.h"
#include "config.h"
#include "devices.h"
#include "pocl_cl.h"
#include "pocl_local_size.h"
#include "utlist.h"

#include "pocl_cache.h"
#include "pocl_debug.h"
#include "pocl_file_util.h"
#include "pocl_llvm.h"
#include "pocl_timing.h"
#include "pocl_util.h"
#include "pocl_version.h"

#include "pocl-vulkan.h"

#include "bufalloc.h"

#define MAX_CMD_BUFFERS 8
#define MAX_BUF_DESCRIPTORS 4096
#define MAX_UBO_DESCRIPTORS 1024
#define MAX_DESC_SETS 1024

#include "memfill64.h"
#include "memfill128.h"

#define PAGE_SIZE 4096

static void *pocl_vulkan_driver_pthread (void *cldev);

typedef struct VkStructBase {
    VkStructureType    sType;
    void*              pNext;
} VkStructBase;

typedef struct pushc_data
{
  uint32_t ord;
  uint32_t offset;
  uint32_t size;
  uint32_t unused;
} pushc_data;

typedef struct local_data
{
  uint32_t ord;
  uint32_t elem_size;
  uint32_t spec_id;
  uint32_t unused;
} local_data;

typedef struct buf_data
{
  uint32_t ord;
  uint32_t dset;
  uint32_t binding;
  uint32_t offset;
} buf_data;

#define MAX_LOCALS 128
#define MAX_BUFS 128
#define MAX_PODS 128
#define MAX_PUSHC 128

typedef struct pocl_vulkan_kernel_data_s
{
  /* linked list */
  struct pocl_vulkan_kernel_data_s *next;
  struct pocl_vulkan_kernel_data_s *prev;

  /* to match kernel */
  char *name;

  /* these are parsed from clspv provided descriptor map.
   * most are used just for verification (clspv assigns descriptors
   * quite predictably), except locals.elem_size which is required for
   * setting the Specialization constant correctly. */
  size_t num_locals;
  size_t num_bufs;
  size_t num_pods;
  size_t num_pushc;
  local_data locals[MAX_LOCALS];
  buf_data bufs[MAX_BUFS];
  buf_data pods[MAX_PODS];
  pushc_data pushc[MAX_PUSHC];

  /* since POD arguments are pushed via a buffer (kernarg_buf),
   * total size helps with preallocating kernarg buffer */
  size_t num_pod_bytes;

  /* size of push constant bytes used for arguments.
   * (doesn't include goffset_pushc_size) */
  size_t num_pushc_arg_bytes;

  /* offset for goffset push constants */
  uint32_t goffset_pushc_offset;
  /* size should always be 12 (3x uint32) */
  uint32_t goffset_pushc_size;

  /* kernarg buffer, for POD arguments */
  VkBuffer kernarg_buf;
  VkDeviceSize kernarg_buf_offset;
  VkDeviceSize kernarg_buf_size;
  chunk_info_t *kernarg_chunk;

} pocl_vulkan_kernel_data_t;

typedef struct pocl_vulkan_event_data_s
{
  pthread_cond_t event_cond;
} pocl_vulkan_event_data_t;

typedef struct pocl_vulkan_mem_data_s
{
  /* For devices which can't directly transfer to/from host memory */
  VkBuffer staging_buf;
  VkDeviceMemory staging_mem;
  /* Device-local memory buffer */
  VkBuffer device_buf;
  VkDeviceMemory device_mem;
} pocl_vulkan_mem_data_t;

typedef struct pocl_vulkan_device_data_s
{
  VkDevice device;
  cl_device_id dev;

  VkQueue compute_queue;
  uint32_t compute_queue_fam_index;

  VkCommandPool command_pool;
  VkDescriptorPool buf_descriptor_pool;

  VkSubmitInfo submit_info;
  VkCommandBufferBeginInfo cmd_buf_begin_info;

  /* device memory reserved for kernel arguments */
  VkDeviceMemory kernarg_mem;
  size_t kernarg_size;
  memory_region_t kernarg_region;

  chunk_info_t *memfill_chunk;
  VkBuffer memfill_buf;

  /* device memory reserved for constant arguments */
  VkDeviceMemory constant_mem;
  size_t constant_size;
  memory_region_t constant_region;

  /* staging area, for copying kernarg_mem & constant_mem to device-local
   * memory, if we can't do it more efficiently (kernarg_is_mappable == 0) */
  VkDeviceMemory staging_mem;
  VkBuffer staging_buf;
  size_t staging_size;
  void *staging_mapped;

  VkPhysicalDeviceProperties dev_props;
  VkPhysicalDeviceMemoryProperties mem_props;

  /* Unlike in OpenCL, Vulkan devices have WG count limits (-> grid size
   * limits).
   * TODO pocl must launch multiple times if WG count > this limit */
  uint32_t max_wg_count[4];

  /* device limits */
  VkDeviceSize host_staging_mem_size, device_mem_size;
  /* memory types*/
  uint32_t host_staging_read_type, host_staging_write_type, device_mem_type,
      kernarg_mem_type, constant_mem_type;
  uint32_t min_ubo_align, min_stor_align, min_map_align;

  VkCommandBuffer command_buffer;
  VkCommandBuffer tmp_command_buffer;

  VkPipelineCache cache;

  /* integrated GPUs have different Vulkan memory layout */
  int device_is_iGPU;
  /* device needs staging buffers for memory transfers
   * TODO this might be equal to "device_is_iGPU" */
  int needs_staging_mem;
  /* 1 if kernarg memory is equal to device-local memory */
  int kernarg_is_device_mem;
  /* 1 if kernarg memory is directly mappable to host AS */
  int kernarg_is_mappable;
  /* capabilities */
  int have_atom_64bit;

  int have_i8_shader;
  int have_i16_shader;
  int have_i64_shader;

  int have_f16_shader;
  int have_f64_shader;

  int have_8b_ssbo;
  int have_8b_ubo;
  int have_8b_pushc;
  int have_16b_ssbo;
  int have_16b_ubo;
  int have_16b_pushc;
  int have_ext_host_mem;
  /* minimal alignment of imported host pointers */
  uint32_t min_ext_host_mem_align;
  /* it's an extension function so we need to get its pointer */
  PFN_vkGetMemoryHostPointerPropertiesEXT vkGetMemoryHostPointerProperties;

  uint32_t max_pushc_size;
  uint32_t max_ubo_size;

  /* kernels for clEnqueueFillBuffer. Should be converted to builtin kernels */
  cl_program memfill64_prog;
  cl_program memfill128_prog;
  cl_kernel memfill64_ker;
  cl_kernel memfill128_ker;

  /*************************************************************************/
  _cl_command_node *work_queue;

  /* driver wake + lock */
  pthread_cond_t wakeup_cond
      __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  POCL_FAST_LOCK_T wq_lock_fast
      __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

  size_t driver_thread_exit_requested
      __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  /* device pthread */
  pthread_t driver_pthread_id;

  cl_bool available;

} pocl_vulkan_device_data_t;

typedef struct pocl_vulkan_program_data_s
{
  char *clspv_map_filename;
  pocl_vulkan_kernel_data_t *vk_kernel_meta_list;
  pocl_kernel_metadata_t *kernel_meta;
  unsigned num_kernels;
  int has_wg_spec_constants;
  VkShaderModule shader;

  /* module constant data */
  size_t constant_data_size;
  uint32_t *constant_data;
  uint32_t constant_dset, constant_binding;

  /* buffer for holding module constant data */
  VkBuffer constant_buf;
  VkDeviceSize constant_buf_offset;
  VkDeviceSize constant_buf_size;
  chunk_info_t *constant_chunk;

} pocl_vulkan_program_data_t;

static void
pocl_vulkan_abort_on_vk_error (VkResult status, unsigned line,
                               const char *func, const char *code)
{
  const char *str = code;
  if (status != VK_SUCCESS)
    {
      /* TODO convert vulkan errors to strings */
      POCL_MSG_PRINT2 (VULKAN, func, line,
                       "Error %i from Vulkan Runtime call:\n", (int)status );
      POCL_ABORT ("Code:\n%s\n", str);
    }
}

#define VULKAN_CHECK_ABORT(code)                                              \
  pocl_vulkan_abort_on_vk_error (code, __LINE__, __FUNCTION__, #code)

#define VULKAN_CHECK_RET(RETVAL, CODE)                                        \
  do                                                                          \
    {                                                                         \
      VkResult res = CODE;                                                    \
      if (res != VK_SUCCESS)                                                  \
        {                                                                     \
          POCL_MSG_PRINT2 (ERROR, __FUNCTION__, __LINE__,                     \
                           "Error %i from Vulkan Runtime call:\n", (int)res); \
          return RETVAL;                                                      \
        }                                                                     \
    }                                                                         \
  while (0)

static VkResult
pocl_vulkan_get_best_compute_queue (VkPhysicalDevice dev,
                                    uint32_t *compute_queue_index,
                                    uint32_t *compute_queue_count)
{
  VkQueueFamilyProperties queue_preps[128];
  uint32_t queue_prep_count = 0;

  vkGetPhysicalDeviceQueueFamilyProperties (dev, &queue_prep_count, NULL);
  assert (queue_prep_count < 128);
  vkGetPhysicalDeviceQueueFamilyProperties (dev, &queue_prep_count,
                                            queue_preps);

  uint32_t comp_only_i = UINT32_MAX;
  uint32_t comp_i = UINT32_MAX;
  uint32_t i;

  /* Prefer compute-only queue */
  for (i = 0; i < queue_prep_count; i++)
    {
      VkQueueFlags flags = queue_preps[i].queueFlags;

      if (flags & VK_QUEUE_COMPUTE_BIT)
        {
          comp_i = i;
          if (!(flags & VK_QUEUE_GRAPHICS_BIT))
            {
              comp_only_i = i;
              break;
            }
        }
    }

  uint32_t q_i = (comp_only_i < UINT32_MAX) ? comp_only_i : comp_i;
  if (q_i == UINT32_MAX)
    return VK_ERROR_INITIALIZATION_FAILED;

  *compute_queue_index = q_i;
  *compute_queue_count = queue_preps[q_i].queueCount;
  return VK_SUCCESS;
}

/* Memory for OpenCL constant memory and kernel arguments */
#define KERNARG_BUFFER_SIZE (2 << 20)
#define CONSTANT_BUFFER_SIZE (8 << 20)
/* larger of the previous two */
#define STAGING_BUF_SIZE (8 << 20)

static int
pocl_vulkan_setup_memory_types (cl_device_id dev, pocl_vulkan_device_data_t *d,
                                VkPhysicalDevice pd)
{
  size_t i, heap_i, staging_i, gart_i;
  /* Find memory types and save them */
  vkGetPhysicalDeviceMemoryProperties (pd, &d->mem_props);

  d->host_staging_read_type = UINT32_MAX;
  d->host_staging_write_type = UINT32_MAX;
  d->device_mem_type = UINT32_MAX;
  d->kernarg_mem_type = UINT32_MAX;
  uint32_t gart_mem_type = UINT32_MAX;
  VkDeviceSize gart_mem_size = 0;

  if (d->device_is_iGPU || (d->mem_props.memoryHeapCount == 1))
    {
      /* integrated GPU */
      heap_i = UINT32_MAX;
      for (i = 0; i < d->mem_props.memoryHeapCount; ++i)
        {
          if (d->mem_props.memoryHeaps[i].flags
              & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            {
              heap_i = i;
              break;
            }
        }
      if (heap_i == UINT32_MAX)
        {
          POCL_MSG_ERR ("Vulkan: can't find device local memory\n");
          return CL_FAILED;
        }

      d->device_mem_type = UINT32_MAX;
      for (i = 0; i < d->mem_props.memoryTypeCount; ++i)
        {
          if (d->mem_props.memoryTypes[i].heapIndex != heap_i)
            continue;
          VkMemoryPropertyFlags f = d->mem_props.memoryTypes[i].propertyFlags;
          if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
              && (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
              && (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
            {
              d->device_mem_type = i;
              break;
            }
        }
      if (d->device_mem_type == UINT32_MAX)
        {
          POCL_MSG_ERR ("Vulkan: can't find device memory type\n");
          return CL_FAILED;
        }

      d->device_mem_size = d->mem_props.memoryHeaps[heap_i].size;

      d->needs_staging_mem = 0;
      d->host_staging_mem_size = 0;

      d->kernarg_is_device_mem = 1;
      d->kernarg_is_mappable = 1;

      d->kernarg_mem_type = d->device_mem_type;
      d->kernarg_size = KERNARG_BUFFER_SIZE;
      d->device_mem_size -= d->kernarg_size;

      d->constant_mem_type = d->device_mem_type;
      d->constant_size = CONSTANT_BUFFER_SIZE;
      d->device_mem_size -= d->constant_size;
    }
  else
    {
      d->needs_staging_mem = 1;

      for (uint32_t i = 0; i < d->mem_props.memoryTypeCount; ++i)
        {
          VkMemoryPropertyFlags f = d->mem_props.memoryTypes[i].propertyFlags;

          if ((f == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
              && (d->device_mem_type == UINT32_MAX))
            d->device_mem_type = i;

          /* older Vulkan implementation on Nvidia don't expose this.
 *           A small memory heap accessible by both CPU and GPU,
*            useful for kernel args and constant mem */
          if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
              && (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
              && (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
              && gart_mem_type == UINT32_MAX)
            gart_mem_type = i;

          if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
              && (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
              && (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
              && d->host_staging_read_type == UINT32_MAX)
            d->host_staging_read_type = i;

          if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
              && (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
              && ((f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) == 0)
              && d->host_staging_write_type == UINT32_MAX)
            d->host_staging_write_type = i;
        }

      if (d->device_mem_type == UINT32_MAX)
        {
          POCL_MSG_ERR ("Vulkan: can't find device memory type \n");
          return CL_FAILED;
        }

      heap_i = d->mem_props.memoryTypes[d->device_mem_type].heapIndex;
      d->device_mem_size = d->mem_props.memoryHeaps[heap_i].size;

      /* if we have only one staging, make the other identical.
       * This is merely a performance issue */
      if (d->host_staging_read_type == UINT32_MAX)
        d->host_staging_read_type = d->host_staging_write_type;
      if (d->host_staging_write_type == UINT32_MAX)
        d->host_staging_write_type = d->host_staging_read_type;

      assert ((d->host_staging_read_type < UINT32_MAX)
              && (d->host_staging_write_type < UINT32_MAX));

      staging_i
          = d->mem_props.memoryTypes[d->host_staging_read_type].heapIndex;
      d->host_staging_mem_size = d->mem_props.memoryHeaps[staging_i].size;

      if (gart_mem_type < UINT32_MAX)
        {
          gart_i = d->mem_props.memoryTypes[gart_mem_type].heapIndex;
          gart_mem_size = d->mem_props.memoryHeaps[gart_i].size;
          assert (gart_mem_size > KERNARG_BUFFER_SIZE);
        }
      else
        gart_i = heap_i;

      /* if we have separate heap for GART memory, use it for kernel arguments */
      if (gart_i != heap_i && gart_i != staging_i)
        {
          d->kernarg_is_device_mem = 0;
          d->kernarg_is_mappable = 1;

          d->kernarg_mem_type = gart_mem_type;
          d->kernarg_size = KERNARG_BUFFER_SIZE;

          d->constant_mem_type = d->device_mem_type;
          d->constant_size = CONSTANT_BUFFER_SIZE;
          d->device_mem_size -= d->constant_size;
        }
      else
        {
          d->kernarg_is_device_mem = 1;
          d->kernarg_is_mappable = 0;

          d->kernarg_mem_type = d->device_mem_type;
          d->kernarg_size = KERNARG_BUFFER_SIZE;
          d->device_mem_size -= d->kernarg_size;

          d->constant_mem_type = d->device_mem_type;
          d->constant_size = CONSTANT_BUFFER_SIZE;
          d->device_mem_size -= d->constant_size;
        }
    }

  POCL_MSG_PRINT_VULKAN ("Device %s MEMORY: DEVICE %zu M | GART %zu M | "
                         "STAGING %zu M | KERNARG %zu M | CONSTANT %zu M |\n",
                         dev->short_name, (size_t) (d->device_mem_size >> 20),
                         (size_t) (gart_mem_size >> 20),
                         (size_t) (d->host_staging_mem_size >> 20),
                         (size_t) (d->kernarg_size >> 20),
                         (size_t) (d->constant_size >> 20));

  /* preallocate kernarg memory */

  VkMemoryAllocateInfo allocate_info
      = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, d->kernarg_size,
          d->kernarg_mem_type };
  VULKAN_CHECK_ABORT (
      vkAllocateMemory (d->device, &allocate_info, NULL, &d->kernarg_mem));

  pocl_init_mem_region (&d->kernarg_region, 0, d->kernarg_size);
  d->kernarg_region.strategy = BALLOCS_TIGHT;
  d->kernarg_region.alignment = PAGE_SIZE;

  {
    VkBufferCreateInfo buffer_info = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                       NULL,
                                       0,
                                       MAX_EXTENDED_ALIGNMENT,
                                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                                           | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                           | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       VK_SHARING_MODE_EXCLUSIVE,
                                       1,
                                       &d->compute_queue_fam_index };

    VULKAN_CHECK_ABORT (
        vkCreateBuffer (d->device, &buffer_info, NULL, &d->memfill_buf));
    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements (d->device, d->memfill_buf, &mem_req);
    assert (mem_req.size >= MAX_EXTENDED_ALIGNMENT);

    d->memfill_chunk
        = pocl_alloc_buffer_from_region (&d->kernarg_region, mem_req.size);
    assert (d->memfill_chunk);
    assert (d->memfill_chunk->start_address % 4 == 0);
    VULKAN_CHECK_ABORT (vkBindBufferMemory (d->device, d->memfill_buf,
                                            d->kernarg_mem,
                                            d->memfill_chunk->start_address));
  }
  POCL_MSG_PRINT_VULKAN ("Allocated %zu memory for kernel arguments\n",
                         d->kernarg_size);

  /* preallocate constant memory */

  allocate_info.allocationSize = d->constant_size;
  allocate_info.memoryTypeIndex = d->constant_mem_type;

  VULKAN_CHECK_ABORT (
      vkAllocateMemory (d->device, &allocate_info, NULL, &d->constant_mem));

  pocl_init_mem_region (&d->constant_region, 0, d->constant_size);
  d->constant_region.strategy = BALLOCS_TIGHT;
  d->constant_region.alignment = PAGE_SIZE;

  POCL_MSG_PRINT_VULKAN ("Allocated %zu memory for constant memory\n",
                         d->constant_size);

  /* create staging buf, if needed */

  if ((!d->kernarg_is_mappable) && (d->kernarg_is_device_mem))
    {
      d->staging_size = STAGING_BUF_SIZE;
      assert (d->host_staging_write_type < UINT32_MAX);
      VkBufferCreateInfo buffer_info
          = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
              NULL,
              0, /* TODO flags */
              d->staging_size,
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                  | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                  | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
              VK_SHARING_MODE_EXCLUSIVE,
              1,
              &d->compute_queue_fam_index };

      VkMemoryAllocateInfo allocate_info
          = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, d->staging_size,
              d->host_staging_write_type };

      VULKAN_CHECK_ABORT (
          vkCreateBuffer (d->device, &buffer_info, NULL, &d->staging_buf));
      VULKAN_CHECK_ABORT (
          vkAllocateMemory (d->device, &allocate_info, NULL, &d->staging_mem));
      VkMemoryRequirements memReq;
      vkGetBufferMemoryRequirements (d->device, d->staging_buf, &memReq);
      assert (d->staging_size == memReq.size);
      VULKAN_CHECK_ABORT (
          vkBindBufferMemory (d->device, d->staging_buf, d->staging_mem, 0));

      /* TODO track available host_staging_mem_bytes */
      VULKAN_CHECK_ABORT (vkMapMemory (d->device, d->staging_mem, 0,
                                       d->staging_size, 0,
                                       &d->staging_mapped));
    }

  dev->max_constant_buffer_size = d->constant_size;

  return CL_SUCCESS;
}

static void
pocl_vulkan_enqueue_staging_buffer_copy (pocl_vulkan_device_data_t *d,
                                         VkBuffer dest_buf,
                                         VkDeviceSize dest_size)
{
  dest_size = pocl_align_value(dest_size, d->dev_props.limits.nonCoherentAtomSize);
  VkMappedMemoryRange mem_range = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                                    NULL, d->staging_mem, 0, dest_size };
  /* TODO only if non-coherent */
  VULKAN_CHECK_ABORT (vkFlushMappedMemoryRanges (d->device, 1, &mem_range));

  /* copy staging mem -> dev mem */
  VkCommandBuffer cb = d->tmp_command_buffer;
  VULKAN_CHECK_ABORT (vkResetCommandBuffer (cb, 0));
  VULKAN_CHECK_ABORT (vkBeginCommandBuffer (cb, &d->cmd_buf_begin_info));
  VkBufferCopy copy;
  copy.srcOffset = 0;
  copy.dstOffset = 0;
  copy.size = dest_size;

  vkCmdCopyBuffer (cb, d->staging_buf, dest_buf, 1, &copy);
  VULKAN_CHECK_ABORT (vkEndCommandBuffer (cb));
  d->submit_info.pCommandBuffers = &cb;
  VULKAN_CHECK_ABORT (
      vkQueueSubmit (d->compute_queue, 1, &d->submit_info, NULL));

  VkMemoryBarrier memory_barrier;
  memory_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  memory_barrier.pNext = NULL;
  memory_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier (cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                        &memory_barrier, 0, 0, 0, 0);
}

void
pocl_vulkan_init_device_ops (struct pocl_device_ops *ops)
{
  ops->device_name = "vulkan";

  ops->probe = pocl_vulkan_probe;
  ops->init = pocl_vulkan_init;
  ops->uninit = pocl_vulkan_uninit;
  ops->reinit = pocl_vulkan_reinit;

  ops->read = pocl_vulkan_read;
  ops->read_rect = pocl_vulkan_read_rect;
  ops->write = pocl_vulkan_write;
  ops->write_rect = pocl_vulkan_write_rect;
  ops->copy = pocl_vulkan_copy;
  ops->copy_rect = pocl_vulkan_copy_rect;
  ops->memfill = pocl_vulkan_memfill;
  ops->map_mem = pocl_vulkan_map_mem;
  ops->unmap_mem = pocl_vulkan_unmap_mem;
  ops->get_mapping_ptr = pocl_vulkan_get_mapping_ptr;
  ops->free_mapping_ptr = pocl_vulkan_free_mapping_ptr;
  ops->can_migrate_d2d = NULL;
  ops->migrate_d2d = NULL;
  ops->compute_local_size = pocl_wg_utilization_maximizer;

  ops->run = pocl_vulkan_run;
  ops->run_native = NULL;

  ops->alloc_mem_obj = pocl_vulkan_alloc_mem_obj;
  ops->free = pocl_vulkan_free;

  ops->build_source = pocl_vulkan_build_source;
  ops->build_binary = pocl_vulkan_build_binary;
  ops->link_program = NULL;
  ops->free_program = pocl_vulkan_free_program;
  ops->setup_metadata = pocl_vulkan_setup_metadata;
  ops->supports_binary = pocl_vulkan_supports_binary;
  ops->build_poclbinary = pocl_vulkan_build_poclbinary;
  ops->compile_kernel = NULL;

  ops->join = pocl_vulkan_join;
  ops->submit = pocl_vulkan_submit;
  ops->broadcast = pocl_broadcast;
  ops->notify = pocl_vulkan_notify;
  ops->flush = pocl_vulkan_flush;
  ops->build_hash = pocl_vulkan_build_hash;

  /* TODO get timing data from Vulkan API */
  /* ops->get_timer_value = pocl_vulkan_get_timer_value; */

  ops->wait_event = pocl_vulkan_wait_event;
  ops->notify_event_finished = pocl_vulkan_notify_event_finished;
  ops->notify_cmdq_finished = pocl_vulkan_notify_cmdq_finished;
  ops->free_event_data = pocl_vulkan_free_event_data;
  ops->wait_event = pocl_vulkan_wait_event;
  ops->update_event = pocl_vulkan_update_event;

  ops->init_queue = pocl_vulkan_init_queue;
  ops->free_queue = pocl_vulkan_free_queue;

  /* ########### IMAGES ############ */
  /*
  ops->create_image = NULL;
  ops->free_image = NULL;
  ops->create_sampler = NULL;
  ops->free_sampler = NULL;
  ops->copy_image_rect = pocl_vulkan_copy_image_rect;
  ops->write_image_rect = pocl_vulkan_write_image_rect;
  ops->read_image_rect = pocl_vulkan_read_image_rect;
  ops->map_image = pocl_vulkan_map_image;
  ops->unmap_image = pocl_vulkan_unmap_image;
  ops->fill_image = pocl_vulkan_fill_image;
  */
}

/* The binary format version that this driver code can read. */
#define VULKAN_BINARY_FORMAT "1"

char *
pocl_vulkan_build_hash (cl_device_id device)
{
  char *res = (char *)malloc (32);
  snprintf (res, 32, "pocl-vulkan-clspv " VULKAN_BINARY_FORMAT);
  return res;
}

static const VkApplicationInfo pocl_vulkan_application_info
    = { VK_STRUCTURE_TYPE_APPLICATION_INFO,
        NULL,
        "PoCL OpenCL application",
        0,
        "PoCL " POCL_VERSION_BASE,
        120,
#ifdef VK_MAKE_API_VERSION
        VK_MAKE_API_VERSION(0, 1, 2, 0)
#else
        VK_MAKE_VERSION(1, 2, 0)
#endif
};

#define MAX_VULKAN_DEVICES 32

/* TODO replace with dynamic arrays */
static VkInstance pocl_vulkan_instance;
static unsigned pocl_vulkan_device_count;
static unsigned pocl_vulkan_initialized_dev_count = 0;
static VkPhysicalDevice pocl_vulkan_devices[MAX_VULKAN_DEVICES];

static int pocl_vulkan_enable_validation;
static int pocl_vulkan_debug_report_available;
static int pocl_vulkan_debug_utils_available;

static VkDebugReportCallbackEXT debug_report_callback_ext;

/* Vulkan debug callback */
static VKAPI_ATTR VkBool32 VKAPI_CALL
debug_callback (VkDebugReportFlagsEXT flags,
                VkDebugReportObjectTypeEXT objType, uint64_t obj,
                size_t location, int32_t code, const char *layerPrefix,
                const char *msg, void *userData)
{

  POCL_MSG_WARN ("VALIDATION LAYER %s ::\n%s\n", layerPrefix, msg);

  return VK_FALSE;
}

unsigned int
pocl_vulkan_probe (struct pocl_device_ops *ops)
{
  VkResult res;
  int env_count = pocl_device_get_env_count (ops->device_name);

  pocl_vulkan_enable_validation = pocl_is_option_set ("POCL_VULKAN_VALIDATE");

  if (env_count <= 0)
    return 0;

  size_t i;
  VkInstanceCreateInfo cinfo = { 0 };
  cinfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  cinfo.pApplicationInfo = &pocl_vulkan_application_info;

#ifdef HAVE_CLSPV
  if (!pocl_exists (CLSPV))
    {
      POCL_MSG_ERR ("Vulkan: can't find CLSPV compiler!\n");
      return 0;
    }
#endif

  /* extensions */
  uint32_t ext_prop_count = 128;
  VkExtensionProperties properties[128];
  VULKAN_CHECK_RET (0, vkEnumerateInstanceExtensionProperties (
                           NULL, &ext_prop_count, properties));
  assert (ext_prop_count < 128);

  cinfo.enabledExtensionCount = 0;
  const char *ins_extensions[2] = { 0 };
  cinfo.ppEnabledExtensionNames = ins_extensions;

  if (pocl_vulkan_enable_validation)
    {
      for (i = 0; i < ext_prop_count; ++i)
        {
          if (strncmp ("VK_EXT_debug_report", properties[i].extensionName,
                       VK_MAX_EXTENSION_NAME_SIZE)
              == 0)
            {
              pocl_vulkan_debug_report_available = 1;
              ins_extensions[cinfo.enabledExtensionCount++]
                  = "VK_EXT_debug_report";
            }
          if (strncmp ("VK_EXT_debug_utils", properties[i].extensionName,
                       VK_MAX_EXTENSION_NAME_SIZE)
              == 0)
            {
              pocl_vulkan_debug_utils_available = 1;
              ins_extensions[cinfo.enabledExtensionCount++]
                  = "VK_EXT_debug_utils";
            }
        }
    }

  /* layers */
  uint32_t layer_count = 128;
  VkLayerProperties layers[128];
  VULKAN_CHECK_RET (0,
                    vkEnumerateInstanceLayerProperties (&layer_count, layers));
  assert (layer_count < 128);

  cinfo.enabledLayerCount = 0;
  const char *ins_layers[3] = { 0 };
  cinfo.ppEnabledLayerNames = ins_layers;

  if (pocl_vulkan_enable_validation)
    {
      for (i = 0; i < layer_count; ++i)
        {
          if (strncmp ("VK_LAYER_LUNARG_core_validation", layers[i].layerName,
                       VK_MAX_EXTENSION_NAME_SIZE)
              == 0)
            {
              ins_layers[cinfo.enabledLayerCount++]
                  = "VK_LAYER_LUNARG_core_validation";
            }
          if (strncmp ("VK_LAYER_LUNARG_standard_validation",
                       layers[i].layerName, VK_MAX_EXTENSION_NAME_SIZE)
              == 0)
            {
              ins_layers[cinfo.enabledLayerCount++]
                  = "VK_LAYER_LUNARG_standard_validation";
            }
          if (strncmp ("VK_LAYER_LUNARG_api_dump", layers[i].layerName,
                       VK_MAX_EXTENSION_NAME_SIZE)
              == 0)
            {
              ins_layers[cinfo.enabledLayerCount++]
                  = "VK_LAYER_LUNARG_api_dump";
            }
        }
    }

  VULKAN_CHECK_RET (0, vkCreateInstance (&cinfo, NULL, &pocl_vulkan_instance));

  if (pocl_vulkan_enable_validation && pocl_vulkan_debug_report_available)
    {
      VkDebugReportCallbackCreateInfoEXT cb_create_info;
      cb_create_info.sType
          = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
      cb_create_info.pNext = NULL;
      cb_create_info.flags
          = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
      cb_create_info.pfnCallback = debug_callback;

      PFN_vkCreateDebugReportCallbackEXT func
          = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr (
              pocl_vulkan_instance, "vkCreateDebugReportCallbackEXT");
      if (func != NULL)
        {
          func (pocl_vulkan_instance, &cb_create_info, NULL,
                &debug_report_callback_ext);
        }
      else
        {
          POCL_MSG_WARN ("Can't enable Vulkan debug report extension.\n");
        }
    }

  /* TODO ignore llvmpipe -type devices */
  VULKAN_CHECK_RET (0,
                    vkEnumeratePhysicalDevices (pocl_vulkan_instance,
                                                &pocl_vulkan_device_count, 0));
  if (pocl_vulkan_device_count > MAX_VULKAN_DEVICES)
    {
      POCL_MSG_ERR ("Ignoring > %i devices\n", MAX_VULKAN_DEVICES);
      pocl_vulkan_device_count = MAX_VULKAN_DEVICES;
    }
  VULKAN_CHECK_RET (0, vkEnumeratePhysicalDevices (pocl_vulkan_instance,
                                                   &pocl_vulkan_device_count,
                                                   pocl_vulkan_devices));

  POCL_MSG_PRINT_VULKAN ("%u Vulkan devices found.\n",
                         pocl_vulkan_device_count);

  /* TODO: clamp pocl_vulkan_device_count to env_count */

  return pocl_vulkan_device_count;
}

static int
pocl_vulkan_setup_memfill_kernels (cl_device_id dev,
                                   pocl_vulkan_device_data_t *d)
{
  int err;

  d->memfill64_prog = (cl_program)calloc (1, sizeof (struct _cl_program));
  assert (d->memfill64_prog);
  d->memfill64_prog->pocl_binaries = calloc (1, sizeof (char *));
  d->memfill64_prog->binaries = calloc (1, sizeof (char *));
  d->memfill64_prog->binary_sizes = calloc (1, sizeof (size_t));
  d->memfill64_prog->devices = calloc (1, sizeof (cl_device_id));
  d->memfill64_prog->data = calloc (1, sizeof (void *));
  d->memfill64_prog->build_hash = calloc (1, sizeof (SHA1_digest_t));
  d->memfill64_prog->build_log = calloc (1, sizeof (char *));

  d->memfill64_prog->num_devices = 1;
  d->memfill64_prog->devices[0] = dev;
  d->memfill64_prog->binaries[0] = (unsigned char *)bin2c_memfill64_spv;
  d->memfill64_prog->binary_sizes[0] = sizeof(bin2c_memfill64_spv);

  err = pocl_vulkan_build_binary (d->memfill64_prog, 0, 1, 0);
  if (err != CL_SUCCESS)
    return err;
  pocl_vulkan_setup_metadata (dev, d->memfill64_prog, 0);

  d->memfill64_ker = (cl_kernel)calloc (1, sizeof (struct _cl_kernel));
  assert (d->memfill64_ker);
  d->memfill64_ker->meta = &d->memfill64_prog->kernel_meta[0];
  d->memfill64_ker->data = (void **)calloc (1, sizeof (void *));
  d->memfill64_ker->name = d->memfill64_ker->meta->name;
  d->memfill64_ker->context = NULL;
  d->memfill64_ker->program = d->memfill64_prog;

  d->memfill64_ker->dyn_arguments = (pocl_argument *)calloc (
      (d->memfill64_ker->meta->num_args), sizeof (struct pocl_argument));

  d->memfill128_prog = (cl_program)calloc (1, sizeof (struct _cl_program));
  assert (d->memfill128_prog);

  d->memfill128_prog->pocl_binaries = calloc (1, sizeof (char *));
  d->memfill128_prog->binaries = calloc (1, sizeof (char *));
  d->memfill128_prog->binary_sizes = calloc (1, sizeof (size_t));
  d->memfill128_prog->devices = calloc (1, sizeof (cl_device_id));
  d->memfill128_prog->data = calloc (1, sizeof (void *));
  d->memfill128_prog->build_hash = calloc (1, sizeof (SHA1_digest_t));
  d->memfill128_prog->build_log = calloc (1, sizeof (char *));

  d->memfill128_prog->num_devices = 1;
  d->memfill128_prog->devices[0] = dev;
  d->memfill128_prog->binaries[0] = (unsigned char *)bin2c_memfill128_spv;
  d->memfill128_prog->binary_sizes[0] = sizeof(bin2c_memfill128_spv);

  err = pocl_vulkan_build_binary (d->memfill128_prog, 0, 1, 0);
  if (err != CL_SUCCESS)
    return err;
  pocl_vulkan_setup_metadata (dev, d->memfill128_prog, 0);

  d->memfill128_ker = (cl_kernel)calloc (1, sizeof (struct _cl_kernel));
  assert (d->memfill128_ker);
  d->memfill128_ker->meta = &d->memfill128_prog->kernel_meta[0];
  d->memfill128_ker->data = (void **)calloc (1, sizeof (void *));
  d->memfill128_ker->name = d->memfill128_ker->meta->name;
  d->memfill128_ker->context = NULL;
  d->memfill128_ker->program = d->memfill128_prog;

  d->memfill128_ker->dyn_arguments = (pocl_argument *)calloc (
      (d->memfill128_ker->meta->num_args), sizeof (struct pocl_argument));
  return CL_SUCCESS;
}

static const char* VULKAN_SERIALIZE_ENTRIES[2] = { "/program.spv", "/program.map" };

cl_int
pocl_vulkan_init (unsigned j, cl_device_id dev, const char *parameters)
{
  assert (j < pocl_vulkan_device_count);
  POCL_MSG_PRINT_VULKAN ("Initializing device %u\n", j);
  size_t i;
  int err;

  SETUP_DEVICE_CL_VERSION (dev, HOST_DEVICE_CL_VERSION_MAJOR,
                           HOST_DEVICE_CL_VERSION_MINOR)

  pocl_vulkan_device_data_t *d;

  d = (pocl_vulkan_device_data_t *)calloc (1,
                                           sizeof (pocl_vulkan_device_data_t));
  dev->data = d;
  d->dev = dev;
  dev->available = &d->available;

  VkPhysicalDevice pd = pocl_vulkan_devices[j];

  uint32_t comp_queue_fam, comp_queue_count;
  VULKAN_CHECK_RET (CL_FAILED, pocl_vulkan_get_best_compute_queue (
                                   pd, &comp_queue_fam, &comp_queue_count));
  POCL_MSG_PRINT_VULKAN (
      "Vulkan Dev %u using Compute Queue Fam: %u, Count: %u\n", j,
      comp_queue_fam, comp_queue_count);

  float comp_queue_prio = 1.0f;
  VkDeviceQueueCreateInfo queue_fam_cinfo
      = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
          0,
          0,
          comp_queue_fam,
          1,
          &comp_queue_prio };

  /* clspv:
   * The implementation must support extensions:
   * VK_KHR_storage_buffer_storage_class
   * VK_KHR_variable_pointers
   * VK_KHR_shader_non_semantic_info
   */

  const char *requested_exts[32];
  uint32_t requested_ext_count = 0;

  uint32_t dev_ext_count = 0;
  VkExtensionProperties dev_exts[256];
  VULKAN_CHECK_RET (CL_FAILED, vkEnumerateDeviceExtensionProperties (
                                   pd, NULL, &dev_ext_count, NULL));
  assert (dev_ext_count < 256);
  VULKAN_CHECK_RET (CL_FAILED, vkEnumerateDeviceExtensionProperties (
                                   pd, NULL, &dev_ext_count, dev_exts));

  int have_amd_shader_core_properties = 0;
  int have_needed_extensions = 0;
  int ext_atom_64bit = 0;
  int ext_i8_f16_shader = 0;
  int ext_8b_store = 0;
  int ext_16b_store = 0;
  int ext_memory_host_ptr = 0;
  for (i = 0; i < dev_ext_count; ++i)
    {
#ifdef VK_AMD_shader_core_properties
      if (strncmp ("VK_AMD_shader_core_properties", dev_exts[i].extensionName,
                   VK_MAX_EXTENSION_NAME_SIZE)
          == 0)
        {
          have_amd_shader_core_properties = 1;
          requested_exts[requested_ext_count++]
              = "VK_AMD_shader_core_properties";
        }
#endif
      if (strncmp ("VK_KHR_variable_pointers", dev_exts[i].extensionName,
                   VK_MAX_EXTENSION_NAME_SIZE)
          == 0)
        {
          ++have_needed_extensions;
          requested_exts[requested_ext_count++] = "VK_KHR_variable_pointers";
        }
      if (strncmp ("VK_KHR_storage_buffer_storage_class",
                   dev_exts[i].extensionName, VK_MAX_EXTENSION_NAME_SIZE)
          == 0)
        {
          ++have_needed_extensions;
          requested_exts[requested_ext_count++]
              = "VK_KHR_storage_buffer_storage_class";
        }

/* TODO this will be required once we get rid of clspv-reflection
      if (strncmp ("VK_KHR_shader_non_semantic_info",
                   dev_exts[i].extensionName, VK_MAX_EXTENSION_NAME_SIZE)
          == 0)
        {
          ++have_needed_extensions;
          requested_exts[requested_ext_count++]
              = "VK_KHR_shader_non_semantic_info";
        }
*/

      if (strncmp ("VK_KHR_shader_atomic_int64", dev_exts[i].extensionName,
                   VK_MAX_EXTENSION_NAME_SIZE)
          == 0)
        {
          ext_atom_64bit = 1;
          requested_exts[requested_ext_count++] = "VK_KHR_shader_atomic_int64";
        }

      if (strncmp ("VK_KHR_shader_float16_int8", dev_exts[i].extensionName,
                   VK_MAX_EXTENSION_NAME_SIZE)
          == 0)
        {
          ext_i8_f16_shader = 1;
          requested_exts[requested_ext_count++] = "VK_KHR_shader_float16_int8";
        }

      if (strncmp ("VK_KHR_8bit_storage", dev_exts[i].extensionName,
                   VK_MAX_EXTENSION_NAME_SIZE)
          == 0)
        {
          ext_8b_store = 1;
          requested_exts[requested_ext_count++] = "VK_KHR_8bit_storage";
        }

      if (strncmp ("VK_KHR_16bit_storage", dev_exts[i].extensionName,
                   VK_MAX_EXTENSION_NAME_SIZE)
          == 0)
        {
          ext_16b_store = 1;
          requested_exts[requested_ext_count++] = "VK_KHR_16bit_storage";
        }

      if (strncmp ("VK_EXT_external_memory_host", dev_exts[i].extensionName,
                   VK_MAX_EXTENSION_NAME_SIZE)
          == 0)
        {
          ++ext_memory_host_ptr;
          requested_exts[requested_ext_count++]
              = "VK_EXT_external_memory_host";
        }

      if (strncmp ("VK_KHR_external_memory", dev_exts[i].extensionName,
                   VK_MAX_EXTENSION_NAME_SIZE)
          == 0)
        {
          ++ext_memory_host_ptr;
          requested_exts[requested_ext_count++] = "VK_KHR_external_memory";
        }
    }

  if (have_needed_extensions < 2)
    {
      POCL_MSG_ERR ("pocl-vulkan requires a device that supports: "
                    "VK_KHR_variable_pointers + "
                    "VK_KHR_storage_buffer_storage_class + "
                    "VK_KHR_shader_non_semantic_info;\n"
                    "disabling device %s\n", dev->short_name);
      return CL_FAILED;
    }

    /* get device properties */
  if (have_amd_shader_core_properties || ext_memory_host_ptr == 2)
    {
      VkPhysicalDeviceProperties2 general_props
          = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
      VkStructBase *p = (VkStructBase *)&general_props;
#ifdef VK_AMD_shader_core_properties
      VkPhysicalDeviceShaderCorePropertiesAMD shader_core_properties
          = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD };
      p->pNext = &shader_core_properties;
      p = (VkStructBase *)&shader_core_properties;
#endif
#ifdef VK_EXT_external_memory_host
      VkPhysicalDeviceExternalMemoryHostPropertiesEXT ext_mem_properties = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT
      };
      p->pNext = &ext_mem_properties;
      p = (VkStructBase *)&ext_mem_properties;
#endif
      p->pNext = NULL;

      vkGetPhysicalDeviceProperties2 (pocl_vulkan_devices[j], &general_props);

      memcpy (&d->dev_props, &general_props.properties,
              sizeof (VkPhysicalDeviceProperties));

#ifdef VK_EXT_external_memory_host
      if (ext_memory_host_ptr == 2)
        d->min_ext_host_mem_align
            = ext_mem_properties.minImportedHostPointerAlignment;
      else
#endif
        d->min_ext_host_mem_align = 0;

#ifdef VK_AMD_shader_core_properties
      if (have_amd_shader_core_properties)
        dev->max_compute_units
            = shader_core_properties.shaderEngineCount
              * shader_core_properties.shaderArraysPerEngineCount
              * shader_core_properties.computeUnitsPerShaderArray;
      else
#endif
        dev->max_compute_units = 1;
    }
  else
    {
      vkGetPhysicalDeviceProperties (pd, &d->dev_props);
      dev->max_compute_units = 1;
      d->min_ext_host_mem_align = 0;
    }

  /* TODO get this from Vulkan API */
  dev->max_clock_frequency = 1000;

  /* clspv:
    If the short/ushort types are used in the OpenCL C:
    The shaderInt16 field of VkPhysicalDeviceFeatures must be set to true.
          shaderFloat64                           = 1
          shaderInt64                             = 1
          shaderInt16                             = 0
  */

  VkPhysicalDeviceFeatures2 dev_features
      = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, 0 };
#ifdef VK_VERSION_1_2

  VkPhysicalDeviceShaderAtomicInt64Features atomic64b_features
      = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES, 0 };
  VkPhysicalDeviceShaderFloat16Int8Features f16_i8_features
      = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES, 0 };
  VkPhysicalDevice8BitStorageFeatures storage8b_features
      = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES, 0 };
  VkPhysicalDevice16BitStorageFeatures storage16b_features
      = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, 0 };

  VkStructBase *p = (VkStructBase *)&dev_features;
  if (ext_atom_64bit)
    {
      p->pNext = &atomic64b_features;
      p  = (VkStructBase *)&atomic64b_features;
    }
  if (ext_i8_f16_shader)
    {
      p->pNext = &f16_i8_features;
      p  = (VkStructBase *)&f16_i8_features;
    }
  if (ext_8b_store)
    {
      p->pNext = &storage8b_features;
      p = (VkStructBase *)&storage8b_features;
    }
  if (ext_16b_store)
    {
      p->pNext = &storage16b_features;
      p = (VkStructBase *)&storage16b_features;
    }
  p->pNext = NULL;
#endif
  vkGetPhysicalDeviceFeatures2 (pd, &dev_features);

#ifdef VK_VERSION_1_2
  d->have_atom_64bit
      = (ext_atom_64bit && atomic64b_features.shaderBufferInt64Atomics
         && atomic64b_features.shaderSharedInt64Atomics);

  d->have_8b_ssbo
      = (ext_8b_store && storage8b_features.storageBuffer8BitAccess);
  d->have_8b_ubo
      = (ext_8b_store && storage8b_features.uniformAndStorageBuffer8BitAccess);
  d->have_8b_pushc = (ext_8b_store && storage8b_features.storagePushConstant8);
  d->have_16b_ssbo
      = (ext_16b_store && storage16b_features.storageBuffer16BitAccess);
  d->have_16b_ubo
      = (ext_16b_store
         && storage16b_features.uniformAndStorageBuffer16BitAccess);
  d->have_16b_pushc
      = (ext_16b_store && storage16b_features.storagePushConstant16);

  d->have_f16_shader = (ext_i8_f16_shader && f16_i8_features.shaderFloat16);

  d->have_i8_shader = (ext_i8_f16_shader && f16_i8_features.shaderInt8);
#endif
  d->have_f64_shader = dev_features.features.shaderFloat64;

  d->have_i16_shader = dev_features.features.shaderInt16;
  d->have_i64_shader = dev_features.features.shaderInt64;

  d->max_pushc_size = d->dev_props.limits.maxPushConstantsSize;
  d->max_ubo_size = d->dev_props.limits.maxUniformBufferRange;

#ifdef VK_EXT_external_memory_host
  d->have_ext_host_mem
      = (ext_memory_host_ptr == 2 && d->min_ext_host_mem_align > 0);
  if (d->have_ext_host_mem)
    {
      d->vkGetMemoryHostPointerProperties
          = (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetInstanceProcAddr (
              pocl_vulkan_instance, "vkGetMemoryHostPointerPropertiesEXT");
      if (d->vkGetMemoryHostPointerProperties == NULL)
        {
          POCL_MSG_WARN ("couldn't get ext address of function "
                         "vkGetMemoryHostPointerPropertiesEXT\n");
          d->have_ext_host_mem = 0;
        }
    }
#endif

  if (d->have_f64_shader)
    {
      dev->double_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO
                              | CL_FP_ROUND_TO_INF | CL_FP_FMA | CL_FP_INF_NAN
                              | CL_FP_DENORM;
    }
  else
    {
      dev->double_fp_config = 0;
    }
  if (d->have_f16_shader)
    {
      dev->half_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO
                            | CL_FP_ROUND_TO_INF | CL_FP_FMA | CL_FP_INF_NAN
                            | CL_FP_DENORM;
    }
  else
    {
      dev->half_fp_config = 0;
    }
  dev->single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO
                          | CL_FP_ROUND_TO_INF | CL_FP_FMA | CL_FP_INF_NAN
                          | CL_FP_DENORM;

  dev->has_64bit_long = d->have_i64_shader;

  /*
   If images are used in the OpenCL C:
   The shaderStorageImageReadWithoutFormat field of VkPhysicalDeviceFeatures
   must be set to true. The shaderStorageImageWriteWithoutFormat field of
   VkPhysicalDeviceFeatures must be set to true.
  */

  /* TODO: Get images working */
  dev->image_support = CL_FALSE;

  VkDeviceCreateInfo dev_cinfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                   dev_features.pNext,
                                   0,
                                   1,
                                   &queue_fam_cinfo,
                                   0, /* deprecated */
                                   0, /* deprecated */
                                   requested_ext_count,
                                   requested_exts,
                                   &dev_features.features };

  /* create logical device */
  VULKAN_CHECK_RET (CL_FAILED, vkCreateDevice (pocl_vulkan_devices[j],
                                               &dev_cinfo, 0, &d->device));
  pocl_vulkan_initialized_dev_count++;

  dev->num_serialize_entries = 2;
  dev->serialize_entries = VULKAN_SERIALIZE_ENTRIES;

  dev->profile = "FULL_PROFILE";
  dev->vendor_id = d->dev_props.vendorID;
  char extensions[1024];
  extensions[0] = 0;
  strcat (extensions, "cl_khr_byte_addressable_store"
                      " cl_khr_global_int32_base_atomics"
                      " cl_khr_global_int32_extended_atomics"
                      " cl_khr_local_int32_base_atomics"
                      " cl_khr_local_int32_extended_atomics");
  if (d->have_atom_64bit)
    strcat (extensions, " cl_khr_int64_base_atomics"
                        " cl_khr_int64_extended_atomics");
  if (dev->half_fp_config)
    strcat (extensions, " cl_khr_fp16");
  if (dev->double_fp_config)
    strcat (extensions, " cl_khr_fp64");

  dev->extensions = strdup (extensions);

  if (dev->vendor_id == 0x10de)
    {
      dev->vendor = "NVIDIA Corporation";
    }
  else if (dev->vendor_id == 0x1002)
    {
      dev->vendor = "AMD Corporation";
    }
  else if (dev->vendor_id == 0x8086)
    {
      dev->vendor = "Intel Corporation";
    }
  else if (dev->vendor_id == 0x14e4)
    {
      dev->vendor = "Broadcom Inc.";
    }
  else
    {
      dev->vendor = "Unknown";
    }
  /* TODO get from API */
  dev->preferred_wg_size_multiple = 64;

  VkPhysicalDeviceType dtype = d->dev_props.deviceType;
  dev->short_name = dev->long_name = d->dev_props.deviceName;

  if (dtype == VK_PHYSICAL_DEVICE_TYPE_CPU)
    dev->type = CL_DEVICE_TYPE_CPU;
  else if (dtype == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
    {
      dev->type = CL_DEVICE_TYPE_GPU;
      d->device_is_iGPU = 0;
    }
  else if (dtype == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
    {
      dev->type = CL_DEVICE_TYPE_GPU;
      d->device_is_iGPU = 1;
    }
  else
    {
      dev->type = CL_DEVICE_TYPE_CUSTOM;
      POCL_MSG_ERR ("ignoring Vulkan device %s because of unsupported device type", dev->short_name);
      return CL_FAILED;
    }

  dev->execution_capabilities = CL_EXEC_KERNEL;
  dev->address_bits = 64;
  dev->mem_base_addr_align
      = max ((cl_uint)MAX_EXTENDED_ALIGNMENT,
             (cl_uint)d->dev_props.limits.minStorageBufferOffsetAlignment);
  if (d->min_ext_host_mem_align)
    dev->mem_base_addr_align
        = max (d->min_ext_host_mem_align, dev->mem_base_addr_align);

#ifdef HAVE_CLSPV
  dev->compiler_available = CL_TRUE;
  dev->linker_available = CL_TRUE;
#else
  dev->compiler_available = CL_FALSE;
  dev->linker_available = CL_FALSE;
  /* TODO enable the following once the build callbacks
   * are fixed to extract kernel metadata from SPIR-V
   * directly instead of using clspv-reflection
   */
#endif

  dev->preferred_vector_width_char = 1;
  dev->preferred_vector_width_short = 1;
  dev->preferred_vector_width_int = 1;
  dev->preferred_vector_width_long = 1;
  dev->preferred_vector_width_float = 1;
  dev->preferred_vector_width_double = 1;
  dev->preferred_vector_width_half = 1;
  dev->native_vector_width_char = 1;
  dev->native_vector_width_short = 1;
  dev->native_vector_width_int = 1;
  dev->native_vector_width_long = 1;
  dev->native_vector_width_float = 1;
  dev->native_vector_width_double = 1;
  dev->native_vector_width_half = 1;

  dev->device_side_printf = 0;
  dev->printf_buffer_size = 1024*1024;

  dev->endian_little = CL_TRUE;
  dev->parent_device = NULL;
  dev->max_sub_devices = 0;
  dev->num_partition_properties = 0;
  dev->partition_properties = NULL;
  dev->num_partition_types = 0;
  dev->partition_type = NULL;
  dev->max_constant_args = 8;
  dev->host_unified_memory = CL_FALSE;
  dev->min_data_type_align_size = MAX_EXTENDED_ALIGNMENT;
  dev->max_parameter_size = 1024;

  dev->local_mem_type = CL_LOCAL;
  /* maxComputeSharedMemorySize is the maximum total storage size, in bytes,
   * of all variables declared with the WorkgroupLocal storage class in shader
   * modules (or with the shared storage qualifier in GLSL) in the compute
   * shader stage.
   */
  dev->local_mem_size = d->dev_props.limits.maxComputeSharedMemorySize;
  /* maxComputeWorkGroupInvocations is the maximum total number of compute
   * shader invocations in a single local workgroup. The product of the X, Y,
   * and Z sizes as specified by the LocalSize execution mode in shader modules
   * and by the object decorated by the WorkgroupSize decoration must be less
   * than or equal to this limit.
   */
  dev->max_work_item_dimensions = 3;
  dev->max_work_item_sizes[0] = d->dev_props.limits.maxComputeWorkGroupSize[0];
  dev->max_work_item_sizes[1] = d->dev_props.limits.maxComputeWorkGroupSize[1];
  dev->max_work_item_sizes[2] = d->dev_props.limits.maxComputeWorkGroupSize[2];
  dev->max_work_group_size
      = d->dev_props.limits.maxComputeWorkGroupInvocations;

  /* Vulkan devices typically don't have unlimited number of groups per
   * command, unlike OpenCL */
  d->max_wg_count[0] = d->dev_props.limits.maxComputeWorkGroupCount[0];
  d->max_wg_count[1] = d->dev_props.limits.maxComputeWorkGroupCount[1];
  d->max_wg_count[2] = d->dev_props.limits.maxComputeWorkGroupCount[2];

  d->min_ubo_align = d->dev_props.limits.minUniformBufferOffsetAlignment;
  d->min_stor_align = d->dev_props.limits.minStorageBufferOffsetAlignment;
  d->min_map_align = d->dev_props.limits.minMemoryMapAlignment;

  d->compute_queue_fam_index = comp_queue_fam;
  vkGetDeviceQueue (d->device, comp_queue_fam, 0, &d->compute_queue);

  err = pocl_vulkan_setup_memory_types (dev, d, pd);
  if (err != CL_SUCCESS)
    {
      POCL_MSG_ERR ("Vulkan: failed to setup memory types for device %s\n",
                    dev->long_name);
      return CL_FAILED;
    }

  dev->global_mem_size = d->device_mem_size;
  dev->global_mem_cacheline_size = HOST_CPU_CACHELINE_SIZE;
  dev->global_mem_cache_size = 32768; /* TODO we should detect somehow.. */
  dev->global_mem_cache_type = CL_READ_WRITE_CACHE;

  /* TODO VkPhysicalDeviceVulkan11Properties . maxMemoryAllocationSize */
  dev->max_mem_alloc_size
      = min (dev->global_mem_size / 2,
             d->dev_props.limits.maxStorageBufferRange & (~4095U));
  dev->max_mem_alloc_size = max (dev->max_mem_alloc_size, 128 * 1024 * 1024);

  err = pocl_vulkan_setup_memfill_kernels (dev, d);
  if (err != CL_SUCCESS)
    {
      POCL_MSG_ERR ("Vulkan: failed to setup memfill kernels for device %s\n",
                    dev->long_name);
      return CL_FAILED;
    }
  /************************************************************************/

  VkPipelineCacheCreateInfo cache_create_info = {
    VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
    NULL,
    0, // flags
    0, NULL
  };
  VULKAN_CHECK_ABORT (
      vkCreatePipelineCache (d->device, &cache_create_info, NULL, &d->cache));

  VkCommandPoolCreateInfo pool_cinfo;
  pool_cinfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_cinfo.pNext = NULL;
  pool_cinfo.queueFamilyIndex = comp_queue_fam;
  pool_cinfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
                     | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VULKAN_CHECK_ABORT (
      vkCreateCommandPool (d->device, &pool_cinfo, NULL, &d->command_pool));

  VkCommandBuffer tmp[2];
  VkCommandBufferAllocateInfo alloc_cinfo;
  alloc_cinfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_cinfo.pNext = NULL;
  alloc_cinfo.commandPool = d->command_pool;
  alloc_cinfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_cinfo.commandBufferCount = 2;
  VULKAN_CHECK_ABORT (vkAllocateCommandBuffers (d->device, &alloc_cinfo, tmp));
  d->command_buffer = tmp[0];
  d->tmp_command_buffer = tmp[1];

  d->submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  d->submit_info.pNext = NULL;
  d->submit_info.waitSemaphoreCount = 0;
  d->submit_info.pWaitSemaphores = NULL;
  d->submit_info.signalSemaphoreCount = 0;
  d->submit_info.pSignalSemaphores = NULL;
  d->submit_info.commandBufferCount = 1;
  d->submit_info.pCommandBuffers = NULL;
  d->submit_info.pWaitDstStageMask = NULL;

  d->cmd_buf_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  d->cmd_buf_begin_info.pNext = NULL;
  d->cmd_buf_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  d->cmd_buf_begin_info.pInheritanceInfo = NULL;

  VkDescriptorPoolSize descriptor_pool_size[2]
      = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MAX_BUF_DESCRIPTORS },
          { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_UBO_DESCRIPTORS } };

  VkDescriptorPoolCreateInfo descriptor_pool_create_info
      = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
          NULL,
          VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
          MAX_DESC_SETS, /* maxSets */
          2,             /* size of next array */
          descriptor_pool_size };

  VULKAN_CHECK_ABORT (vkCreateDescriptorPool (
      d->device, &descriptor_pool_create_info, NULL, &d->buf_descriptor_pool));

  POCL_INIT_COND (d->wakeup_cond);

  POCL_FAST_INIT (d->wq_lock_fast);

  d->work_queue = NULL;

  PTHREAD_CHECK (pthread_create (&d->driver_pthread_id, NULL,
                                 pocl_vulkan_driver_pthread, dev));

  d->available = CL_TRUE;
  return CL_SUCCESS;
}

/* TODO finish implementation */
cl_int
pocl_vulkan_uninit (unsigned j, cl_device_id device)
{
  pocl_vulkan_device_data_t *d = (pocl_vulkan_device_data_t*)device->data;

  if (device->available != CL_FALSE)
    {

      POCL_FAST_LOCK (d->wq_lock_fast);
      d->driver_thread_exit_requested = 1;
      POCL_SIGNAL_COND (d->wakeup_cond);
      POCL_FAST_UNLOCK (d->wq_lock_fast);

      POCL_JOIN_THREAD (d->driver_pthread_id);

      assert (pocl_vulkan_initialized_dev_count > 0);
      --pocl_vulkan_initialized_dev_count;
    }

  /* this must be called after all devices !! */
  if (pocl_vulkan_initialized_dev_count == 0)
    vkDestroyInstance (pocl_vulkan_instance, NULL);

  return CL_SUCCESS;
}

cl_int
pocl_vulkan_reinit (unsigned j, cl_device_id device, const char *parameters)
{
  return 0;
}

#if 0
cl_ulong
pocl_vulkan_get_timer_value(void *data)
{
  return pocl_gettimemono_ns();
}
#endif

int
run_and_append_output_to_build_log (cl_program program, unsigned device_i,
                                    char *const *args)
{
  int errcode = CL_SUCCESS;

  char *capture_string = NULL;
  size_t capture_capacity = 0;
  if (program->build_log[device_i] != NULL)
    {
      size_t len = strlen (program->build_log[device_i]);
      capture_string = program->build_log[device_i] + len;
      if (len + 1 < 128 * 1024)
        {
          capture_capacity = (128 * 1024) - len - 1;
          capture_string[0] = 0;
        }
      else
        {
          capture_string = NULL;
          capture_capacity = 0;
        }
    }
  else
    {
      capture_string = (char *)malloc (128 * 1024);
      POCL_RETURN_ERROR_ON ((capture_string == NULL), CL_OUT_OF_HOST_MEMORY,
                            "Error while allocating temporary memory\n");
      program->build_log[device_i] = capture_string;
      capture_capacity = (128 * 1024) - 1;
      capture_string[0] = 0;
    }

  char cmdline[8192];
  char *p = cmdline;
  *p = 0;
  for (unsigned i = 0; args[i] != NULL; ++i)
    {
      size_t len = strlen (args[i]);
      if (p+len+2 > cmdline+8192) break;
      memcpy (p, args[i], len);
      p += len;
      *p++ = ' ';
    }
  *p = 0;
  POCL_MSG_PRINT_VULKAN ("launching command: \n#### %s\n", cmdline);

  if (capture_string)
    {
      char launch[1024];
      int len = snprintf (launch, 1023, "Output of %s:\n", args[0]);
      if (len > 0 && len < capture_capacity)
        {
          strcat (capture_string, launch);
          capture_string += len;
          capture_capacity -= len;
        }
    }

  errcode = pocl_run_command_capture_output (capture_string, &capture_capacity,
                                             args);
  if (capture_string && capture_capacity > 0)
    capture_string[capture_capacity] = 0;

  return errcode;
}

static int
extract_clspv_map_metadata (pocl_vulkan_program_data_t *vulkan_program_data,
                            cl_uint program_num_devices);

static int
compile_shader (cl_program program, cl_uint device_i,
                const char *program_spv_path,
                const char *program_map_path)
{
  cl_device_id dev = program->devices[device_i];
  pocl_vulkan_device_data_t *d = (pocl_vulkan_device_data_t *)dev->data;

  if (program->binaries[device_i] == NULL)
    {
      assert (program_spv_path);
      uint64_t bin_size;
      char *binary;
      int res = pocl_read_file (program_spv_path, &binary, &bin_size);
      POCL_RETURN_ERROR_ON ((res != 0),
                            CL_BUILD_PROGRAM_FAILURE,
                            "Failed to read file %s\n", program_spv_path);
      program->binaries[device_i] = binary;
      program->binary_sizes[device_i] = bin_size;
    }

  assert (program->binaries[device_i] != NULL);
  assert (program->binary_sizes[device_i] != 0);

  pocl_vulkan_program_data_t *vpd
      = calloc (1, sizeof (pocl_vulkan_program_data_t));
  POCL_RETURN_ERROR_COND ((vpd == NULL), CL_OUT_OF_HOST_MEMORY);

  vpd->clspv_map_filename = strdup (program_map_path);

  int err = extract_clspv_map_metadata (vpd, program->num_devices);
  POCL_RETURN_ERROR_ON ((err != CL_SUCCESS),
                        CL_BUILD_PROGRAM_FAILURE,
                        "Failed to parse program map file\n");
  POCL_RETURN_ERROR_ON ((vpd->num_kernels == 0),
                        CL_BUILD_PROGRAM_FAILURE,
                        "No kernels found in the program\n");

  VkShaderModuleCreateInfo shader_info;
  shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_info.pNext = NULL;
  shader_info.pCode = (const uint32_t *)program->binaries[device_i];
  shader_info.codeSize = program->binary_sizes[device_i];
  shader_info.flags = 0;
  VkShaderModule tempShader = NULL;
  int res = (vkCreateShaderModule (d->device, &shader_info, NULL,
                                   &tempShader));
  POCL_RETURN_ERROR_ON ((res != VK_SUCCESS),
                        CL_BUILD_PROGRAM_FAILURE,
                        "Failed to create Vulkan shader from the Spir-V\n");

  vpd->shader = tempShader;

  program->data[device_i] = vpd;
  return CL_SUCCESS;
}

#define MAX_COMPILATION_ARGS 2048
#define MAX_COMPILATION_ARGS_LEN (256*1024)
int
pocl_vulkan_build_source (cl_program program, cl_uint device_i,
                          cl_uint num_input_headers,
                          const cl_program *input_headers,
                          const char **header_include_names, int link_program)
{
#ifdef HAVE_CLSPV
  int errcode = CL_BUILD_PROGRAM_FAILURE;
  int failed_to_add_compiler_options = CL_FALSE;
  uint64_t bin_size;
  char *binary;
  int err;
  char *strings_to_free[MAX_COMPILATION_ARGS];
  unsigned num_strings_to_free = 0;
  char *compilation_args_concated = NULL;
  unsigned compilation_args_concated_len = 0;
  char* hash_source = NULL;

  cl_device_id dev = program->devices[device_i];
  pocl_vulkan_device_data_t *d = (pocl_vulkan_device_data_t *)dev->data;

  assert (dev->compiler_available == CL_TRUE);
  assert (dev->linker_available == CL_TRUE);
  assert (program->source);
  size_t source_len = strlen (program->source);

  POCL_MSG_PRINT_VULKAN ("building with CLSPV from sources for device %d\n",
                         device_i);

  POCL_GOTO_ERROR_ON ((num_input_headers > 0),
                      CL_BUILD_PROGRAM_FAILURE,
                      "Vulkan compilation with "
                      "headers is not implemented yet\n");

  POCL_GOTO_ERROR_ON ((!link_program),
                      CL_BUILD_PROGRAM_FAILURE,
                      "clCompileProgram() for Vulkan "
                      "is not yet implemented\n");

  char program_cl_path_temp[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_tempname (program_cl_path_temp, ".cl", NULL);
  pocl_write_file (program_cl_path_temp, program->source, source_len, 0);

  char program_map_path_temp[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_tempname (program_map_path_temp, ".map", NULL);

  char program_spv_path_temp[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_tempname (program_spv_path_temp, ".spv", NULL);

  char *COMPILATION[MAX_COMPILATION_ARGS]
      = { CLSPV,
          "-x=cl",
          "--spv-version=1.0",
          "--cl-kernel-arg-info",
          "--keep-unused-arguments",
          "--uniform-workgroup-size",
          "--global-offset",
          "--long-vector",
          "--global-offset-push-constant", // goffs as push constant
          "--module-constants-in-storage-buffer",
          /* push constants should be faster,
           * but currently don't work with goffs */
          /* "--pod-pushconstant",*/
          "--pod-ubo",
          "--cluster-pod-kernel-args",
          NULL };

  unsigned last_arg_idx = 12;

  if (d->have_i8_shader)
    COMPILATION[last_arg_idx++] = "--int8";

  if (d->have_f16_shader)
    COMPILATION[last_arg_idx++] = "--fp16";

  if (d->have_f64_shader)
    COMPILATION[last_arg_idx++] = "--fp64";

  char max_ubo_size_str[128] = { 0 };
  if (d->max_ubo_size != UINT32_MAX)
    {
      snprintf (max_ubo_size_str, 128, "--max-ubo-size=%u", d->max_ubo_size);
      COMPILATION[last_arg_idx++] = max_ubo_size_str;
    }

  char max_pushc_size_str[128] = { 0 };
  if (d->max_pushc_size != UINT32_MAX)
    {
      snprintf (max_pushc_size_str, 128, "--max-pushconstant-size=%u",
                d->max_pushc_size);
      COMPILATION[last_arg_idx++] = max_pushc_size_str;
    }

  char disable_16b_stor[128] = { 0 };
  if (!d->have_16b_pushc || !d->have_16b_ssbo || !d->have_16b_ubo)
    {
      int need_comma = 0;
      strcat (disable_16b_stor, "-no-16bit-storage=");
      if (!d->have_16b_pushc)
        {
          if (need_comma)
            strcat (disable_16b_stor, ",");
          need_comma = 1;
          strcat (disable_16b_stor, "pushconstant");
        }
      if (!d->have_16b_ssbo)
        {
          if (need_comma)
            strcat (disable_16b_stor, ",");
          need_comma = 1;
          strcat (disable_16b_stor, "ssbo");
        }
      if (!d->have_16b_ubo)
        {
          if (need_comma)
            strcat (disable_16b_stor, ",");
          strcat (disable_16b_stor, "ubo");
        }
    }

  char disable_8b_stor[128] = { 0 };
  if (!d->have_8b_pushc || !d->have_8b_ssbo || !d->have_8b_ubo)
    {
      int need_comma = 0;
      strcat (disable_8b_stor, "-no-8bit-storage=");
      if (!d->have_8b_pushc)
        {
          if (need_comma)
            strcat (disable_8b_stor, ",");
          need_comma = 1;
          strcat (disable_8b_stor, "pushconstant");
        }
      if (!d->have_8b_ssbo)
        {
          if (need_comma)
            strcat (disable_8b_stor, ",");
          need_comma = 1;
          strcat (disable_8b_stor, "ssbo");
        }
      if (!d->have_8b_ubo)
        {
          if (need_comma)
            strcat (disable_8b_stor, ",");
          strcat (disable_8b_stor, "ubo");
        }
    }

  if (disable_8b_stor[0])
    COMPILATION[last_arg_idx++] = disable_8b_stor;

  if (disable_16b_stor[0])
    COMPILATION[last_arg_idx++] = disable_16b_stor;

  compilation_args_concated = malloc(MAX_COMPILATION_ARGS_LEN);
  POCL_GOTO_ERROR_COND (compilation_args_concated == NULL,
                        CL_OUT_OF_HOST_MEMORY);
  compilation_args_concated[0] = 0;
  for (unsigned i = 0; i < last_arg_idx; ++i) {
    strcpy (compilation_args_concated + compilation_args_concated_len,
            COMPILATION[i]);
    compilation_args_concated_len += strlen(COMPILATION[i]);
    assert (compilation_args_concated_len < MAX_COMPILATION_ARGS_LEN);
  }

  if (program->compiler_options)
    {
      /* TODO options are not exactly same for clspv */
      char* temp_opts = strdup (program->compiler_options);
      POCL_GOTO_ERROR_COND ( (temp_opts == NULL), CL_OUT_OF_HOST_MEMORY );
      char space_replacement_char = 0;
      if (pocl_escape_quoted_whitespace (temp_opts,
                                         &space_replacement_char) != 0)
        {
          POCL_MSG_ERR ("could not find an unused char in options\n");
          failed_to_add_compiler_options = CL_TRUE;
        }
      else
        {
          char *token = NULL;
          char delim[] = { ' ', 0 };
          token = strtok (temp_opts, delim);
          while (last_arg_idx < MAX_COMPILATION_ARGS && token != NULL)
            {
              char *tok = strdup (token);

              strcpy (compilation_args_concated + compilation_args_concated_len,
                      tok);
              compilation_args_concated_len += strlen(tok);
              assert (compilation_args_concated_len < MAX_COMPILATION_ARGS_LEN);

              for (char* p = tok; *p; ++p)
                if (*p == space_replacement_char)
                  *p = ' ';
              COMPILATION[last_arg_idx++] = tok;
              strings_to_free[num_strings_to_free++] = tok;

              token = strtok (NULL, delim);
            }
          if (last_arg_idx >= MAX_COMPILATION_ARGS)
            failed_to_add_compiler_options = CL_TRUE;
        }
      free (temp_opts);
    }
  POCL_GOTO_ERROR_ON( (failed_to_add_compiler_options != CL_FALSE),
                      CL_BUILD_PROGRAM_FAILURE,
                      "failed to process build options\n");

  unsigned hash_source_len = source_len + compilation_args_concated_len;
  hash_source = malloc (hash_source_len);
  POCL_GOTO_ERROR_COND ((hash_source == NULL), CL_OUT_OF_HOST_MEMORY);
  memcpy (hash_source, program->source, source_len);
  memcpy (hash_source+source_len, compilation_args_concated,
          compilation_args_concated_len);

  char program_bc_path[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_create_program_cachedir (program, device_i, hash_source,
                                      hash_source_len, program_bc_path);
  size_t len = strlen (program_bc_path);
  assert (len > 3);
  len -= 2;
  program_bc_path[len] = 0;
  //  char program_cl_path[POCL_MAX_PATHNAME_LENGTH];
  //  strncpy (program_cl_path, program_bc_path, POCL_MAX_PATHNAME_LENGTH);
  //  strcat (program_cl_path, "cl");
  char program_spv_path[POCL_MAX_PATHNAME_LENGTH];
  strncpy (program_spv_path, program_bc_path, POCL_MAX_PATHNAME_LENGTH);
  strcat (program_spv_path, "spv");
  char program_map_path[POCL_MAX_PATHNAME_LENGTH];
  strncpy (program_map_path, program_bc_path, POCL_MAX_PATHNAME_LENGTH);
  strcat (program_map_path, "map");

  if (pocl_exists (program_spv_path) && pocl_exists (program_map_path))
    goto FINISH;

  COMPILATION[last_arg_idx++] = "-o";
  COMPILATION[last_arg_idx++] = program_spv_path_temp;
  COMPILATION[last_arg_idx++] = program_cl_path_temp;
  COMPILATION[last_arg_idx++] = NULL;

  err = run_and_append_output_to_build_log (program, device_i, COMPILATION);
  POCL_GOTO_ERROR_ON ((err != 0), CL_BUILD_PROGRAM_FAILURE,
                      "clspv exited with nonzero code\n");
  POCL_GOTO_ERROR_ON (!pocl_exists (program_spv_path_temp),
                      CL_BUILD_PROGRAM_FAILURE, "clspv produced no output\n");

  char *REFLECTION[] = { CLSPV_REFLECTION, program_spv_path_temp, "-o",
                         program_map_path_temp, NULL };
  err = run_and_append_output_to_build_log (program, device_i, REFLECTION);

  POCL_GOTO_ERROR_ON ((err != 0), CL_BUILD_PROGRAM_FAILURE,
                      "clspv-reflection exited with nonzero code\n");
  POCL_GOTO_ERROR_ON (!pocl_exists (program_map_path_temp),
                      CL_BUILD_PROGRAM_FAILURE,
                      "clspv-reflection produced no output\n");

  pocl_rename (program_spv_path_temp, program_spv_path);
  pocl_rename (program_map_path_temp, program_map_path);

FINISH:
  errcode = compile_shader (program, device_i, program_spv_path, program_map_path);

ERROR:
  for (unsigned i = 0; i < num_strings_to_free; ++i)
    free (strings_to_free[i]);

  free (compilation_args_concated);
  free (hash_source);

  return errcode;
#else
  POCL_MSG_ERR ("vulkan driver without clspv can't compile anything\n");
  return CL_BUILD_PROGRAM_FAILURE;
#endif
}

int
pocl_vulkan_supports_binary (cl_device_id device, size_t length,
                             const char *binary)
{
/* TODO remove ifdef once the build callbacks
* are fixed to extract kernel metadata from SPIR-V directly
* instead of using clspv-reflection
*/
#ifdef HAVE_CLSPV
  return pocl_bitcode_is_spirv_execmodel_shader (binary, length);
#else
  return 0;
#endif
}

int
pocl_vulkan_build_binary (cl_program program, cl_uint device_i,
                          int link_program, int spir_build)
{

  if (program->pocl_binaries[device_i])
    {
      /* clCreateProgramWithBinary has unpacked the poclbin,
         but it didn't read the binary yet, b/c it's not called program.bc */
      assert (program->binaries[device_i] == NULL);

      char program_bc_path[POCL_MAX_PATHNAME_LENGTH];
      pocl_cache_program_bc_path (program_bc_path, program, device_i);

      size_t len = strlen (program_bc_path);
      assert (len > 3);
      len -= 2;
      program_bc_path[len] = 0;
      char program_spv_path[POCL_MAX_PATHNAME_LENGTH];
      strncpy (program_spv_path, program_bc_path, POCL_MAX_PATHNAME_LENGTH);
      strcat (program_spv_path, "spv");
      char program_map_path[POCL_MAX_PATHNAME_LENGTH];
      strncpy (program_map_path, program_bc_path, POCL_MAX_PATHNAME_LENGTH);
      strcat (program_map_path, "map");

      POCL_RETURN_ERROR_ON (
          (!pocl_exists (program_spv_path)), CL_BUILD_PROGRAM_FAILURE,
          "PoCL binary doesn't contain %s\n", program_spv_path);

      POCL_RETURN_ERROR_ON (
          (!pocl_exists (program_map_path)), CL_BUILD_PROGRAM_FAILURE,
          "PoCL binary doesn't contain %s\n", program_map_path);

      return compile_shader (program, device_i,
                             program_spv_path, program_map_path);
    }

#ifdef HAVE_CLSPV
  /* we have program->binaries[] which is SPIR-V */
  assert (program->binaries[device_i]);
  int is_spirv = pocl_bitcode_is_spirv_execmodel_shader (
      program->binaries[device_i], program->binary_sizes[device_i]);
  POCL_RETURN_ERROR_ON ((is_spirv == 0), CL_BUILD_PROGRAM_FAILURE,
                        "the binary supplied to vulkan driver is not SPIR-V, "
                        "or it's not using execution model shader\n");

  char program_bc_path[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_create_program_cachedir (program, device_i,
                                      program->binaries[device_i],
                                      program->binary_sizes[device_i],
                                      program_bc_path);
  size_t len = strlen (program_bc_path);
  assert (len > 3);
  len -= 2;
  program_bc_path[len] = 0;
  char program_cl_path[POCL_MAX_PATHNAME_LENGTH];
  strncpy (program_cl_path, program_bc_path, POCL_MAX_PATHNAME_LENGTH);
  strcat (program_cl_path, "cl");
  char program_spv_path[POCL_MAX_PATHNAME_LENGTH];
  strncpy (program_spv_path, program_bc_path, POCL_MAX_PATHNAME_LENGTH);
  strcat (program_spv_path, "spv");
  char program_map_path[POCL_MAX_PATHNAME_LENGTH];
  strncpy (program_map_path, program_bc_path, POCL_MAX_PATHNAME_LENGTH);
  strcat (program_map_path, "map");

  if (!pocl_exists (program_spv_path) || !pocl_exists (program_map_path))
    {
      char program_spv_path_temp[POCL_MAX_PATHNAME_LENGTH];
      pocl_cache_tempname (program_spv_path_temp, ".spv", NULL);
      char program_map_path_temp[POCL_MAX_PATHNAME_LENGTH];
      pocl_cache_tempname (program_map_path_temp, ".map", NULL);
      int err = CL_SUCCESS;

      pocl_write_file (program_spv_path_temp, program->binaries[device_i],
                       program->binary_sizes[device_i], 0);
      POCL_RETURN_ERROR_ON (
          !pocl_exists (program_spv_path_temp), CL_BUILD_PROGRAM_FAILURE,
          "failed to write SPIR-V file %s\n", program_spv_path_temp);

      char *REFLECTION[] = { CLSPV "-reflection", program_spv_path_temp, "-o",
                             program_map_path_temp, NULL };

      err = run_and_append_output_to_build_log (program, device_i, REFLECTION);
      POCL_RETURN_ERROR_ON ((err != 0), CL_BUILD_PROGRAM_FAILURE,
                            "clspv-reflection exited with nonzero code\n");
      POCL_RETURN_ERROR_ON (!pocl_exists (program_map_path_temp),
                            CL_BUILD_PROGRAM_FAILURE,
                            "clspv-reflection produced no output\n");
      pocl_rename (program_spv_path_temp, program_spv_path);
      pocl_rename (program_map_path_temp, program_map_path);
    }

  return compile_shader (program, device_i, program_spv_path, program_map_path);

#else
  return CL_BUILD_PROGRAM_FAILURE;
#endif
}

#if 0
/* TODO implement */
int
pocl_vulkan_link_program (cl_program program, cl_uint device_i,
                          cl_uint num_input_programs,
                          const cl_program *input_programs, int
create_library);
#endif

int
pocl_vulkan_free_program (cl_device_id device, cl_program program,
                          unsigned program_device_i)
{
  pocl_vulkan_device_data_t *d = (pocl_vulkan_device_data_t *)device->data;
  pocl_vulkan_program_data_t *vpd = program->data[program_device_i];
  /* vpd can be NULL if compilation fails */
  if (vpd)
    {
      POCL_MEM_FREE (vpd->clspv_map_filename);
      pocl_vulkan_kernel_data_t *el, *tmp;
      DL_FOREACH_SAFE (vpd->vk_kernel_meta_list, el, tmp)
      {
        if (el->kernarg_chunk)
          pocl_free_chunk (el->kernarg_chunk);
        if (el->kernarg_buf)
          vkDestroyBuffer (d->device, el->kernarg_buf, NULL);
        POCL_MEM_FREE (el->name);
        POCL_MEM_FREE (el);
      }

      if (vpd->shader)
        vkDestroyShaderModule (d->device, vpd->shader, NULL);
      if (vpd->kernel_meta)
        {
          pocl_kernel_metadata_t *meta = vpd->kernel_meta;
          POCL_MEM_FREE (meta->build_hash);
          POCL_MEM_FREE (meta->attributes);
          POCL_MEM_FREE (meta->name);
          for (unsigned j = 0; j < meta->num_args; ++j)
            {
              POCL_MEM_FREE (meta->arg_info[j].name);
              POCL_MEM_FREE (meta->arg_info[j].type_name);
            }
          POCL_MEM_FREE (meta->arg_info);
          POCL_MEM_FREE (vpd->kernel_meta);
        }
      if (vpd->constant_data_size)
        POCL_MEM_FREE (vpd->constant_data);
      if (vpd->constant_chunk)
        pocl_free_chunk( vpd->constant_chunk);
      if (vpd->constant_buf)
         vkDestroyBuffer(d->device, vpd->constant_buf, NULL);
      POCL_MEM_FREE (vpd);
    }
  return 0;
}


#define MAX_ARGS 128

static int
parse_new_kernel (pocl_kernel_metadata_t *p, char *line)
{
  char delim[2] = { ',', 0x0 };
  /* kernel_decl */
  char *token = strtok (line, delim);
  /* name */
  token = strtok (NULL, delim);

  p->name = strdup (token);
  p->num_args = 0;
  p->num_locals = 0;
  p->local_sizes = NULL;
  p->attributes = NULL;
  p->reqd_wg_size[0] = p->reqd_wg_size[1] = p->reqd_wg_size[2] = 0;
  p->has_arg_metadata
      = POCL_HAS_KERNEL_ARG_ADDRESS_QUALIFIER | POCL_HAS_KERNEL_ARG_NAME;
  p->arg_info = calloc (MAX_ARGS, sizeof (pocl_argument_info));
  p->total_argument_storage_size = 0;
  p->data = NULL;
  p->build_hash = NULL;
  p->builtin_kernel = 0;

  if (p->arg_info == NULL || p->name == NULL) return -1;
  return 0;
}

static int
parse_arg_line (pocl_kernel_metadata_t *p, pocl_vulkan_kernel_data_t *pp,
                char *line)
{
  char delim[2] = { ',', 0x0 };

  char *token = strtok (line, delim);
  char *tokens[1024];
  unsigned num_tokens = 0;
  while (num_tokens < 1024 && token != NULL)
    {
      tokens[num_tokens] = strdup (token);
      num_tokens++;
      token = strtok (NULL, delim);
    }
  if (num_tokens == 1024 && token != NULL)
    return -1;
  token = NULL;

  char *arg_name = NULL;
  int is_pushc = CL_FALSE;
  unsigned ord, dSet, binding, offset, kind = UINT32_MAX, size, elemSize,
                                       specID = UINT32_MAX;
  for (size_t j = 0; j < num_tokens; j += 2)
    {
      if (strcmp (tokens[j], "kernel") == 0)
        {
          /* skip */
        }
      if (strcmp (tokens[j], "arg") == 0) /* name */
        {
          arg_name = strdup (tokens[j + 1]);
        }
      if (strcmp (tokens[j], "argOrdinal") == 0)
        {
          errno = 0;
          ord = strtol (tokens[j + 1], NULL, 10);
          if (errno != 0) return -1;
        }
      if (strcmp (tokens[j], "descriptorSet") == 0)
        {
          errno = 0;
          dSet = strtol (tokens[j + 1], NULL, 10);
          if (errno != 0) return -1;
        }
      if (strcmp (tokens[j], "binding") == 0)
        {
          errno = 0;
          binding = strtol (tokens[j + 1], NULL, 10);
          if (errno != 0) return -1;
        }
      if (strcmp (tokens[j], "offset") == 0)
        {
          errno = 0;
          offset = strtol (tokens[j + 1], NULL, 10);
          if (errno != 0) return -1;
        }
      if (strcmp (tokens[j], "argKind") == 0)
        {
          /* kernel,matrix_transpose,arg,output,argOrdinal,0,descriptorSet,0,binding,0,offset,0,argKind,buffer */
          if (strcmp (tokens[j + 1], "buffer") == 0)
            {
              kind = POCL_ARG_TYPE_POINTER;
            }
          /* kernel,boxadd,arg,SZ,argOrdinal,5,descriptorSet,0,binding,3,offset,8,argKind,pod_ubo,argSize,4 */
          else if (strcmp (tokens[j + 1], "pod_ubo") == 0)
            {
              kind = POCL_ARG_TYPE_NONE;
            }
          /* kernel,matrix_transpose,arg,tile,argOrdinal,2,argKind,local,arrayElemSize,4,arrayNumElemSpecId,3 */
          else if (strcmp (tokens[j + 1], "local") == 0)
            {
              kind = POCL_ARG_TYPE_POINTER;
            }
          /* kernel,test22,arg,size,argOrdinal,3,offset,20,argKind,pod_pushconstant,argSize,4
           */
          else if (strcmp (tokens[j + 1], "pod_pushconstant") == 0)
            {
              kind = POCL_ARG_TYPE_NONE;
              is_pushc = CL_TRUE;
            }
          else
            return -1;
        }
      if (strcmp (tokens[j], "argSize") == 0)
        {
          errno = 0;
          size = strtol (tokens[j + 1], NULL, 10);
          if (errno != 0) return -1;
        }
      if (strcmp (tokens[j], "arrayElemSize") == 0)
        {
          errno = 0;
          elemSize = strtol (tokens[j + 1], NULL, 10);
          if (errno != 0) return -1;
        }
      if (strcmp (tokens[j], "arrayNumElemSpecId") == 0)
        {
          errno = 0;
          specID = strtol (tokens[j + 1], NULL, 10);
          if (errno != 0) return -1;
        }
    }

  for (size_t i = 0; i < num_tokens; ++i)
    free (tokens[i]);

  assert (arg_name != NULL);
  p->arg_info[ord].name = arg_name;
  p->arg_info[ord].type = kind;

  /* local */
  if (kind == POCL_ARG_TYPE_POINTER && specID != UINT32_MAX)
    {
      p->arg_info[ord].address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL;
      p->arg_info[ord].type_size = sizeof (cl_mem);
      pp->locals[pp->num_locals].elem_size = elemSize;
      pp->locals[pp->num_locals].spec_id = specID;
      pp->locals[pp->num_locals].ord = ord;
      ++pp->num_locals;
    }

  /* buffer */
  if (kind == POCL_ARG_TYPE_POINTER && specID == UINT32_MAX)
    {
      p->arg_info[ord].address_qualifier
          = CL_KERNEL_ARG_ADDRESS_GLOBAL;
      p->arg_info[ord].type_size = sizeof (cl_mem);
      pp->bufs[pp->num_bufs].binding = binding;
      pp->bufs[pp->num_bufs].dset = dSet;
      pp->bufs[pp->num_bufs].offset = offset;
      pp->bufs[pp->num_bufs].ord = ord;
      ++pp->num_bufs;
    }

  /* POD */
  if (kind == POCL_ARG_TYPE_NONE)
    {
      p->arg_info[ord].address_qualifier
          = CL_KERNEL_ARG_ADDRESS_PRIVATE;
      if (size == 0) return -1;
      p->arg_info[ord].type_size = size;

      if (is_pushc)
        {
          pp->pushc[pp->num_pushc].size = size;
          pp->pushc[pp->num_pushc].offset = offset;
          pp->pushc[pp->num_pushc].ord = ord;
          ++pp->num_pushc;
          pp->num_pushc_arg_bytes += size;
        }
      else
        {
          pp->pods[pp->num_pods].binding = binding;
          pp->pods[pp->num_pods].dset = dSet;
          pp->pods[pp->num_pods].offset = offset;
          pp->pods[pp->num_pods].ord = ord;
          ++pp->num_pods;
          pp->num_pod_bytes += size;
        }
    }

  /* TODO constants !!! */

  ++p->num_args;
  if (p->num_args > MAX_ARGS) return -1;

  return 0;
}

int
parse_goffs_pushc_line (pocl_kernel_metadata_t *p,
                        pocl_vulkan_kernel_data_t *pp, char *line,
                        unsigned *goffs_pushc_offset,
                        unsigned *goffs_pushc_size)
{
  char delim[2] = { ',', 0x0 };

  char *token = strtok (line, delim);
  char *tokens[32];
  unsigned num_tokens = 0;
  while (num_tokens < 32 && token != NULL)
    {
      tokens[num_tokens] = strdup (token);
      num_tokens++;
      token = strtok (NULL, delim);
    }
  if (num_tokens == 32 && token != NULL)
    return -1;
  token = NULL;

  char *const_name = NULL;
  unsigned offset = UINT32_MAX;
  unsigned size = UINT32_MAX;
  for (size_t j = 1; j < num_tokens; j += 2)
    {
      if (strcmp (tokens[j], "name") == 0)
        {
          if (strcmp (tokens[j + 1], "global_offset") != 0) return -1;
          continue;
        }
      if (strcmp (tokens[j], "offset") == 0)
        {
          errno = 0;
          offset = strtol (tokens[j + 1], NULL, 10);
          if (errno != 0) return -1;
          continue;
        }
      if (strcmp (tokens[j], "size") == 0)
        {
          errno = 0;
          size = strtol (tokens[j + 1], NULL, 10);
          if (errno != 0) return -1;
          continue;
        }
    }

  for (size_t i = 0; i < num_tokens; ++i)
    free (tokens[i]);

  *goffs_pushc_offset = offset;
  *goffs_pushc_size = size;
  return 0;
}

int
parse_constant_line (pocl_kernel_metadata_t *p,
                     pocl_vulkan_kernel_data_t *pp,
                     unsigned char** constant_data,
                     unsigned *constant_data_size,
                     unsigned *constant_dset,
                     unsigned *constant_binding,
                     char *line)
{
  char delim[2] = { ',', 0x0 };

  char *token = strtok (line, delim);
  char *tokens[32];
  unsigned num_tokens = 0;
  while (num_tokens < 32 && token != NULL)
    {
      tokens[num_tokens] = strdup (token);
      num_tokens++;
      token = strtok (NULL, delim);
    }
  if (num_tokens == 32 && token != NULL)
    return -1;
  token = NULL;

  uint32_t descriptor_set = UINT32_MAX;
  uint32_t ds_binding = UINT32_MAX;
  unsigned char* bytes = NULL;
  unsigned len = 0;
  for (size_t j = 1; j < num_tokens; j += 2)
    {
      if (strcmp (tokens[j], "descriptorSet") == 0)
        {
          errno = 0;
          descriptor_set = strtol (tokens[j + 1], NULL, 10);
          if (errno != 0) return -1;
        }
      if (strcmp (tokens[j], "binding") == 0)
        {
          errno = 0;
          ds_binding = strtol (tokens[j + 1], NULL, 10);
          if (errno != 0) return -1;
        }
      if (strcmp (tokens[j], "hexbytes") == 0)
        {
          errno = 0;
          size_t i = 0;
          char* hexbytes = tokens[j + 1];
          len = strlen (hexbytes);
          bytes = malloc(len + 1);
          if (bytes == NULL) return -1;

          while (hexbytes[i] != 0)
            {
              char tmp[3];
              tmp[0] = hexbytes[i];
              tmp[1] = hexbytes[i+1];
              tmp[2] = 0;
              unsigned long converted = strtol (tmp, NULL, 16);
              if (errno != 0) return -1;
              bytes[i/2] = converted;
              i += 2;
            }
          continue;
        }
    }

  for (size_t i = 0; i < num_tokens; ++i)
    free (tokens[i]);

  if (len == 0 || bytes == NULL)
    return -1;

  *constant_data = bytes;
  *constant_data_size = len;
  *constant_dset = descriptor_set;
  *constant_binding = ds_binding;
  return 0;
}

static int
extract_clspv_map_metadata (pocl_vulkan_program_data_t *vulkan_program_data,
                            cl_uint program_num_devices)
{
  /* read map file from clspv-reflection */
  char *content = NULL;
  int errcode = CL_SUCCESS;
  uint64_t content_size = 0;
  int r = pocl_read_file (vulkan_program_data->clspv_map_filename, &content,
                          &content_size);
  POCL_GOTO_ERROR_ON ((r != 0),
                      CL_BUILD_PROGRAM_FAILURE,
                      "can't read map file\n");

  char *lines[4096];
  unsigned num_lines = 0;
  char delim[2] = { 0x0A, 0x0 };
  char *token = strtok (content, delim);
  while (num_lines < 4096 && token != NULL)
    {
      lines[num_lines] = strdup (token);
      num_lines++;
      token = strtok (NULL, delim);
    }
  POCL_GOTO_ERROR_ON ((num_lines >= 4096),
                      CL_BUILD_PROGRAM_FAILURE,
                      "too many lines in the map file\n");

  POCL_MEM_FREE (content);

  pocl_kernel_metadata_t kernel_meta_array[1024];
  pocl_kernel_metadata_t *p = NULL;
  vulkan_program_data->num_kernels = 0;
  pocl_vulkan_kernel_data_t *current_kernel_metadata = NULL;
  int has_wg_spec_const = 0;
  int has_goffs_pushc = 0;
  int has_constants = 0;
  unsigned go_pushc_offset = 0;
  unsigned go_pushc_size = 0;

  unsigned constant_size = 0;
  unsigned char* constant_bytes = NULL;
  unsigned constant_binding = 0;
  unsigned constant_dset = 0;

  for (size_t i = 0; i < num_lines; ++i)
    {
      if (memcmp (lines[i], "kernel_decl,", 12) == 0)
        {
          current_kernel_metadata = NULL;
          p = &kernel_meta_array[vulkan_program_data->num_kernels];
          POCL_GOTO_ERROR_ON (parse_new_kernel (p, lines[i]) != 0,
                              CL_BUILD_PROGRAM_FAILURE,
                              "failed to parse kernel_decl line\n");
          current_kernel_metadata
              = calloc (1, sizeof (pocl_vulkan_kernel_data_t));
          POCL_GOTO_ERROR_COND (current_kernel_metadata == NULL,
                                CL_OUT_OF_HOST_MEMORY);
          DL_APPEND (vulkan_program_data->vk_kernel_meta_list,
                     current_kernel_metadata);
          current_kernel_metadata->name = strdup (p->name);
          POCL_GOTO_ERROR_COND (current_kernel_metadata->name == NULL,
                                CL_OUT_OF_HOST_MEMORY);
          ++vulkan_program_data->num_kernels;
          continue;
        }
      if (memcmp (lines[i], "kernel,", 7) == 0)
        {
          assert (p != NULL);
          assert (current_kernel_metadata != NULL);
          parse_arg_line (p, current_kernel_metadata, lines[i]);
          continue;
        }
      if (memcmp (lines[i], "spec_constant", 12) == 0)
        {
          if (strstr (lines[i], "workgroup_size") != NULL)
            has_wg_spec_const++;
          continue;
        }
      if (memcmp (lines[i], "pushconstant", 12) == 0)
        {
          has_goffs_pushc++;
          int r = parse_goffs_pushc_line (p, current_kernel_metadata, lines[i],
                                          &go_pushc_offset, &go_pushc_size);
          POCL_GOTO_ERROR_ON (r != 0,
                              CL_BUILD_PROGRAM_FAILURE,
                              "failed to parse pushconstant line\n");
          continue;
        }
      if (memcmp (lines[i], "constant", 8) == 0)
        {
          has_constants++;
          int r = parse_constant_line (p, current_kernel_metadata,
                                       &constant_bytes, &constant_size,
                                       &constant_dset, &constant_binding,
                                       lines[i]);
          POCL_GOTO_ERROR_ON (r != 0,
                              CL_BUILD_PROGRAM_FAILURE,
                              "failed to parse constant line\n");
          continue;
        }
    }

  for (size_t i = 0; i < num_lines; ++i)
  {
    free (lines[i]);
    lines[i] = NULL;
  }

  if (vulkan_program_data->num_kernels > 0)
    {
      // should have 3 spec constants for local WG sizes
      if (has_wg_spec_const > 0) {
        assert (has_wg_spec_const == 3);
      }

      if (go_pushc_size > 0)
        assert (go_pushc_size == 12); // (3 * uint32) for global offsets

      pocl_vulkan_kernel_data_t *el = NULL;
      DL_FOREACH (vulkan_program_data->vk_kernel_meta_list, el)
      {
        el->goffset_pushc_offset = go_pushc_offset;
        el->goffset_pushc_size = go_pushc_size;
      }

      vulkan_program_data->has_wg_spec_constants = (has_wg_spec_const > 0);

      if (constant_size > 0)
      {
          vulkan_program_data->constant_binding = constant_binding;
          vulkan_program_data->constant_dset = constant_dset;
          vulkan_program_data->constant_data = (uint32_t *)constant_bytes;
          vulkan_program_data->constant_data_size = constant_size;
      }

      vulkan_program_data->kernel_meta = calloc (
          vulkan_program_data->num_kernels, sizeof (pocl_kernel_metadata_t));
      memcpy (vulkan_program_data->kernel_meta, kernel_meta_array,
              sizeof (pocl_kernel_metadata_t)
                  * vulkan_program_data->num_kernels);
    }
  else
    POCL_MSG_WARN ("vulkan compilation: zero kernels found\n");

  return CL_SUCCESS;

ERROR:
  if (content != NULL)
    POCL_MEM_FREE (content);
  for (size_t i = 0; i < num_lines; ++i)
    free (lines[i]);
  if (current_kernel_metadata)
    {
      POCL_MEM_FREE (current_kernel_metadata->name);
    }

  return CL_BUILD_PROGRAM_FAILURE;
}

int
pocl_vulkan_setup_metadata (cl_device_id device, cl_program program,
                            unsigned program_device_i)
{
  assert (program->data[program_device_i] != NULL);
  pocl_vulkan_program_data_t *vpd = program->data[program_device_i];

  program->num_kernels = vpd->num_kernels;
  program->kernel_meta = vpd->kernel_meta;
  vpd->kernel_meta = NULL;

  return 1;
}

int
pocl_vulkan_build_poclbinary (cl_program program, cl_uint device_i)
{
  unsigned i;
  _cl_command_node cmd;
  cl_device_id device = program->devices[device_i];

  assert (program->build_status == CL_BUILD_SUCCESS);
  if (program->num_kernels == 0)
    return CL_SUCCESS;

  /* For binaries of other than Executable type (libraries, compiled but
   * not linked programs, etc), do not attempt to compile the kernels. */
  if (program->binary_type != CL_PROGRAM_BINARY_TYPE_EXECUTABLE)
    return CL_SUCCESS;

  POCL_LOCK_OBJ (program);

  assert (program->binaries[device_i]);

  POCL_UNLOCK_OBJ (program);

  return CL_SUCCESS;
}

/********************************************************************/

static void
vulkan_push_command (cl_device_id dev, _cl_command_node *cmd)
{
  pocl_vulkan_device_data_t *d = (pocl_vulkan_device_data_t *)dev->data;
  POCL_FAST_LOCK (d->wq_lock_fast);
  DL_APPEND (d->work_queue, cmd);
  POCL_SIGNAL_COND (d->wakeup_cond);
  POCL_FAST_UNLOCK (d->wq_lock_fast);
}

void
pocl_vulkan_submit (_cl_command_node *node, cl_command_queue cq)
{
  node->ready = 1;
  if (pocl_command_is_ready (node->sync.event.event))
    {
      pocl_update_event_submitted (node->sync.event.event);
      vulkan_push_command (cq->device, node);
    }
  POCL_UNLOCK_OBJ (node->sync.event.event);
  return;
}

int
pocl_vulkan_init_queue (cl_device_id dev, cl_command_queue queue)
{
  queue->data
      = pocl_aligned_malloc (HOST_CPU_CACHELINE_SIZE, sizeof (pthread_cond_t));
  pthread_cond_t *cond = (pthread_cond_t *)queue->data;
  PTHREAD_CHECK (pthread_cond_init (cond, NULL));
  return CL_SUCCESS;
}

int
pocl_vulkan_free_queue (cl_device_id dev, cl_command_queue queue)
{
  pthread_cond_t *cond = (pthread_cond_t *)queue->data;
  PTHREAD_CHECK (pthread_cond_destroy (cond));
  POCL_MEM_FREE (queue->data);
  return CL_SUCCESS;
}

void
pocl_vulkan_notify_cmdq_finished (cl_command_queue cq)
{
  /* must be called with CQ already locked.
   * this must be a broadcast since there could be multiple
   * user threads waiting on the same command queue
   * in pthread_scheduler_wait_cq(). */
  pthread_cond_t *cq_cond = (pthread_cond_t *)cq->data;
  PTHREAD_CHECK (pthread_cond_broadcast (cq_cond));
}

void
pocl_vulkan_notify_event_finished (cl_event event)
{
  pocl_vulkan_event_data_t *e_d = event->data;
  POCL_BROADCAST_COND (e_d->event_cond);
}

void
pocl_vulkan_free_event_data (cl_event event)
{
  assert (event->data != NULL);
  pocl_vulkan_event_data_t *e_d = (pocl_vulkan_event_data_t *)event->data;
  POCL_DESTROY_COND (e_d->event_cond);
  POCL_MEM_FREE (event->data);
}

void
pocl_vulkan_join (cl_device_id device, cl_command_queue cq)
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
pocl_vulkan_flush (cl_device_id device, cl_command_queue cq)
{
}

void
pocl_vulkan_notify (cl_device_id device, cl_event event, cl_event finished)
{
  _cl_command_node *node = event->command;

  if (finished->status < CL_COMPLETE)
    {
      pocl_update_event_failed (event);
      return;
    }

  if (!node->ready)
    return;

  POCL_MSG_PRINT_VULKAN ("notify on event %zu \n", event->id);

  if (pocl_command_is_ready (node->sync.event.event))
    {
      pocl_update_event_submitted (event);
      vulkan_push_command (device, node);
    }

  return;
}

void
pocl_vulkan_update_event (cl_device_id device, cl_event event)
{
  pocl_vulkan_event_data_t *e_d = NULL;
  if (event->data == NULL && event->status == CL_QUEUED)
    {
      e_d = (pocl_vulkan_event_data_t *)malloc (
          sizeof (pocl_vulkan_event_data_t));
      assert (e_d);

      POCL_INIT_COND (e_d->event_cond);
      event->data = (void *)e_d;
    }
}

void
pocl_vulkan_wait_event (cl_device_id device, cl_event event)
{
  POCL_MSG_PRINT_VULKAN (" device->wait_event on event %zu\n", event->id);
  pocl_vulkan_event_data_t *e_d = (pocl_vulkan_event_data_t *)event->data;

  POCL_LOCK_OBJ (event);
  while (event->status > CL_COMPLETE)
    {
      POCL_WAIT_COND (e_d->event_cond, event->pocl_lock);
    }
  POCL_UNLOCK_OBJ (event);
}

/****************************************************************************************/

static void submit_CB (pocl_vulkan_device_data_t *d, VkCommandBuffer *cmdbuf_p)
{
  VkFence fence;

  VkFenceCreateInfo fCreateInfo = {
    VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0 };

  VULKAN_CHECK_ABORT (vkCreateFence (d->device, &fCreateInfo, NULL, &fence));

  d->submit_info.pCommandBuffers = cmdbuf_p;
  VULKAN_CHECK_ABORT (
      vkQueueSubmit (d->compute_queue, 1, &d->submit_info, fence));

  VkResult res = vkWaitForFences (d->device, 1, &fence, VK_TRUE, 1000000000U);
  if (res == VK_TIMEOUT)
    {
      while (res == VK_TIMEOUT)
        {
          res = vkWaitForFences (d->device, 1, &fence, VK_TRUE, 0U);
          if (res == VK_TIMEOUT)
            usleep(5000);
        }
    }
  VULKAN_CHECK_ABORT (res);

  vkDestroyFence (d->device, fence, NULL);
}

static void
pocl_vulkan_dev2host (pocl_vulkan_device_data_t *d,
                      pocl_vulkan_mem_data_t *memdata,
                      pocl_mem_identifier *mem_id, void *restrict host_ptr,
                      size_t offset, size_t size)
{
  void *mapped_ptr = (char *)mem_id->extra_ptr + offset;

  /* TODO: vkFlushMappedMemoryRanges and vkInvalidateMappedMemoryRanges */

  if (d->needs_staging_mem)
    {
      size_t size2 = pocl_align_value(size, d->dev_props.limits.nonCoherentAtomSize);
      size_t offset2 = offset;
      if (offset2 + size2 > mem_id->extra)
        offset2 = mem_id->extra - size2;

      /* copy dev mem -> staging mem */
      VkCommandBuffer cb = d->command_buffer;
      VULKAN_CHECK_ABORT (vkResetCommandBuffer (cb, 0));
      VULKAN_CHECK_ABORT (vkBeginCommandBuffer (cb, &d->cmd_buf_begin_info));
      VkBufferCopy copy;
      copy.srcOffset = offset2;
      copy.dstOffset = offset2;
      copy.size = size2;

      /* POCL_MSG_ERR ("DEV2HOST : %zu / %zu \n", offset, size); */
      vkCmdCopyBuffer (cb, memdata->device_buf, memdata->staging_buf, 1,
                       &copy);
      VULKAN_CHECK_ABORT (vkEndCommandBuffer (cb));
      submit_CB (d, &cb);

      /* copy staging mem -> host_ptr */
      VkMappedMemoryRange mem_range
          = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL,
              memdata->staging_mem, offset2, size2 };
      /* TODO only if non-coherent */
      VULKAN_CHECK_ABORT (
          vkInvalidateMappedMemoryRanges (d->device, 1, &mem_range));
    }

  if (mapped_ptr != host_ptr)
    {
      memcpy (host_ptr, mapped_ptr, size);
    }
}

static void
pocl_vulkan_host2dev (pocl_vulkan_device_data_t *d,
                      pocl_vulkan_mem_data_t *memdata,
                      pocl_mem_identifier *mem_id,
                      const void *restrict host_ptr, size_t offset,
                      size_t size)
{
  void *mapped_ptr = (char *)mem_id->extra_ptr + offset;

  if (mapped_ptr != host_ptr)
    {
      memcpy (mapped_ptr, host_ptr, size);
    }

  if (d->needs_staging_mem)
    {
      size_t size2 = pocl_align_value(size, d->dev_props.limits.nonCoherentAtomSize);
      size_t offset2 = offset;
      if (offset2 + size2 > mem_id->extra)
        offset2 = mem_id->extra - size2;

      /* POCL_MSG_ERR ("HOST2DEV : %zu / %zu\n", offset, size); */
      VkMappedMemoryRange mem_range
          = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL,
              memdata->staging_mem, offset2, size2 };
      /* TODO only if non-coherent */
      VULKAN_CHECK_ABORT (
          vkFlushMappedMemoryRanges (d->device, 1, &mem_range));

      /* copy staging mem -> dev mem */
      VkCommandBuffer cb = d->command_buffer;
      VULKAN_CHECK_ABORT (vkResetCommandBuffer (cb, 0));
      VULKAN_CHECK_ABORT (vkBeginCommandBuffer (cb, &d->cmd_buf_begin_info));
      VkBufferCopy copy;
      copy.srcOffset = offset2;
      copy.dstOffset = offset2;
      copy.size = size2;

      vkCmdCopyBuffer (cb, memdata->staging_buf, memdata->device_buf, 1,
                       &copy);
      VULKAN_CHECK_ABORT (vkEndCommandBuffer (cb));
      submit_CB (d, &cb);
    }
}

void
pocl_vulkan_read (void *data, void *__restrict__ host_ptr,
                  pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                  size_t offset, size_t size)
{
  POCL_MSG_PRINT_VULKAN ("INSIDE READ OFF %zu SIZE %zu \n", offset, size);
  pocl_vulkan_device_data_t *d = (pocl_vulkan_device_data_t *)data;
  pocl_vulkan_mem_data_t *memdata
      = (pocl_vulkan_mem_data_t *)src_mem_id->mem_ptr;

  pocl_vulkan_dev2host (d, memdata, src_mem_id, host_ptr, offset, size);
}

void
pocl_vulkan_write (void *data, const void *__restrict__ host_ptr,
                   pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                   size_t offset, size_t size)
{
  POCL_MSG_PRINT_VULKAN ("INSIDE WRITE to %p OFF %zu SIZE %zu\n", host_ptr,
                         offset, size);
  pocl_vulkan_device_data_t *d = (pocl_vulkan_device_data_t *)data;
  pocl_vulkan_mem_data_t *memdata
      = (pocl_vulkan_mem_data_t *)dst_mem_id->mem_ptr;

  pocl_vulkan_host2dev (d, memdata, dst_mem_id, host_ptr, offset, size);
}

void
pocl_vulkan_copy (void *data, pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
                  pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                  size_t dst_offset, size_t src_offset, size_t size)
{
  POCL_MSG_PRINT_VULKAN ("INSIDE COPY\n");
  pocl_vulkan_device_data_t *d = (pocl_vulkan_device_data_t *)data;

  pocl_vulkan_mem_data_t *src = src_mem_id->mem_ptr;
  pocl_vulkan_mem_data_t *dst = dst_mem_id->mem_ptr;

  /* copy dev mem -> dev mem */
  VkCommandBuffer cb = d->command_buffer;
  VULKAN_CHECK_ABORT (vkResetCommandBuffer (cb, 0));
  VULKAN_CHECK_ABORT (vkBeginCommandBuffer (cb, &d->cmd_buf_begin_info));
  VkBufferCopy copy;
  copy.srcOffset = src_offset;
  copy.dstOffset = dst_offset;
  copy.size = size;
  vkCmdCopyBuffer (cb, src->device_buf, dst->device_buf, 1, &copy);
  VULKAN_CHECK_ABORT (vkEndCommandBuffer (cb));
  submit_CB (d, &cb);
}

void
pocl_vulkan_copy_rect (void *data,
                      pocl_mem_identifier * dst_mem_id,
                      cl_mem dst_buf,
                      pocl_mem_identifier * src_mem_id,
                      cl_mem src_buf,
                      const size_t *__restrict__ const dst_origin,
                      const size_t *__restrict__ const src_origin,
                      const size_t *__restrict__ const region,
                      size_t const dst_row_pitch,
                      size_t const dst_slice_pitch,
                      size_t const src_row_pitch,
                      size_t const src_slice_pitch)
{
  pocl_vulkan_device_data_t *d = (pocl_vulkan_device_data_t *)data;
  pocl_vulkan_mem_data_t *srcdata
      = (pocl_vulkan_mem_data_t *)src_mem_id->mem_ptr;
  pocl_vulkan_mem_data_t *dstdata
      = (pocl_vulkan_mem_data_t *)dst_mem_id->mem_ptr;

  size_t src_start_offset = src_origin[0] + src_row_pitch * src_origin[1]
                            + src_slice_pitch * src_origin[2];
  size_t dst_start_offset = dst_origin[0] + dst_row_pitch * dst_origin[1]
                            + dst_slice_pitch * dst_origin[2];

  size_t j, k;
  VkBufferCopy *copy_regions = malloc (sizeof (VkBufferCopy) * region[1]);
  assert (copy_regions);
  for (k = 0; k < region[2]; ++k)
    {
      /* copy dev mem -> staging mem */
      VkCommandBuffer cb = d->command_buffer;
      VULKAN_CHECK_ABORT (vkResetCommandBuffer (cb, 0));
      VULKAN_CHECK_ABORT (vkBeginCommandBuffer (cb, &d->cmd_buf_begin_info));
      for (j = 0; j < region[1]; ++j)
        {
          copy_regions[j].srcOffset
              = src_start_offset + src_row_pitch * j + src_slice_pitch * k;
          copy_regions[j].dstOffset
              = dst_start_offset + dst_row_pitch * j + dst_slice_pitch * k;
          copy_regions[j].size = region[0];
        }
      vkCmdCopyBuffer (cb, srcdata->device_buf, dstdata->device_buf, region[1],
                       copy_regions);
      VULKAN_CHECK_ABORT (vkEndCommandBuffer (cb));
      submit_CB (d, &cb);
    }
  free (copy_regions);
}

void
pocl_vulkan_read_rect (void *data,
                      void *__restrict__ host_ptr,
                      pocl_mem_identifier * src_mem_id,
                      cl_mem src_buf,
                      const size_t *__restrict__ const buffer_origin,
                      const size_t *__restrict__ const host_origin,
                      const size_t *__restrict__ const region,
                      size_t const buffer_row_pitch,
                      size_t const buffer_slice_pitch,
                      size_t const host_row_pitch,
                      size_t const host_slice_pitch)
{
  pocl_vulkan_device_data_t *d = (pocl_vulkan_device_data_t *)data;
  pocl_vulkan_mem_data_t *memdata
      = (pocl_vulkan_mem_data_t *)src_mem_id->mem_ptr;

  void *mapped_ptr = (char *)src_mem_id->extra_ptr;
  assert (mapped_ptr != host_ptr);

  if (d->needs_staging_mem)
    {
      size_t src_start_offset = buffer_origin[0]
                                + buffer_row_pitch * buffer_origin[1]
                                + buffer_slice_pitch * buffer_origin[2];
      size_t dst_start_offset = host_origin[0]
                                + host_row_pitch * host_origin[1]
                                + host_slice_pitch * host_origin[2];

      size_t j, k;
      VkBufferCopy *copy_regions = malloc (sizeof (VkBufferCopy) * region[1]);
      assert (copy_regions);
      for (k = 0; k < region[2]; ++k)
        {
          /* copy dev mem -> staging mem */
          VkCommandBuffer cb = d->command_buffer;
          VULKAN_CHECK_ABORT (vkResetCommandBuffer (cb, 0));
          VULKAN_CHECK_ABORT (
              vkBeginCommandBuffer (cb, &d->cmd_buf_begin_info));
          for (j = 0; j < region[1]; ++j)
            {
              copy_regions[j].srcOffset = src_start_offset + host_row_pitch * j
                                          + host_slice_pitch * k;
              copy_regions[j].dstOffset = dst_start_offset
                                          + buffer_row_pitch * j
                                          + buffer_slice_pitch * k;
              copy_regions[j].size = region[0];
            }
          vkCmdCopyBuffer (cb, memdata->device_buf, memdata->staging_buf,
                           region[1], copy_regions);
          VULKAN_CHECK_ABORT (vkEndCommandBuffer (cb));
          submit_CB (d, &cb);
        }
      free (copy_regions);

      /* staging mem -> host_ptr */
      VkMappedMemoryRange mem_range
          = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL,
              memdata->staging_mem, 0, src_mem_id->extra };
      /* TODO only if non-coherent */
      VULKAN_CHECK_ABORT (
          vkInvalidateMappedMemoryRanges (d->device, 1, &mem_range));
    }

  pocl_mem_identifier src;
  src.mem_ptr = mapped_ptr;
  pocl_driver_read_rect (NULL, host_ptr, &src, src_buf, buffer_origin,
                         host_origin, region, buffer_row_pitch,
                         buffer_slice_pitch, host_row_pitch, host_slice_pitch);
}

void
pocl_vulkan_write_rect (void *data,
                      const void *__restrict__ host_ptr,
                      pocl_mem_identifier * dst_mem_id,
                      cl_mem dst_buf,
                      const size_t *__restrict__ const buffer_origin,
                      const size_t *__restrict__ const host_origin,
                      const size_t *__restrict__ const region,
                      size_t const buffer_row_pitch,
                      size_t const buffer_slice_pitch,
                      size_t const host_row_pitch,
                      size_t const host_slice_pitch)
{
  pocl_vulkan_device_data_t *d = (pocl_vulkan_device_data_t *)data;
  pocl_vulkan_mem_data_t *memdata
      = (pocl_vulkan_mem_data_t *)dst_mem_id->mem_ptr;

  void *mapped_ptr = (char *)dst_mem_id->extra_ptr;
  assert (mapped_ptr != host_ptr);

  pocl_mem_identifier dst;
  dst.mem_ptr = mapped_ptr;
  pocl_driver_write_rect (
      NULL, host_ptr, &dst, dst_buf, buffer_origin, host_origin, region,
      buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch);

  if (d->needs_staging_mem)
    {
      size_t dst_start_offset = buffer_origin[0]
                                + buffer_row_pitch * buffer_origin[1]
                                + buffer_slice_pitch * buffer_origin[2];
      size_t src_start_offset = host_origin[0]
                                + host_row_pitch * host_origin[1]
                                + host_slice_pitch * host_origin[2];

      /* POCL_MSG_ERR ("HOST2DEV : %zu / %zu\n", offset, size); */
      VkMappedMemoryRange mem_range
          = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL,
              memdata->staging_mem, 0, dst_mem_id->extra };
      /* TODO only if non-coherent */
      VULKAN_CHECK_ABORT (
          vkFlushMappedMemoryRanges (d->device, 1, &mem_range));

      size_t j, k;
      VkBufferCopy *copy_regions = malloc (sizeof (VkBufferCopy) * region[1]);
      assert (copy_regions);
      for (k = 0; k < region[2]; ++k)
        {
          /* staging mem -> dev mem */
          VkCommandBuffer cb = d->command_buffer;
          VULKAN_CHECK_ABORT (vkResetCommandBuffer (cb, 0));
          VULKAN_CHECK_ABORT (
              vkBeginCommandBuffer (cb, &d->cmd_buf_begin_info));
          for (j = 0; j < region[1]; ++j)
            {
              copy_regions[j].srcOffset = src_start_offset + host_row_pitch * j
                                          + host_slice_pitch * k;
              copy_regions[j].dstOffset = dst_start_offset
                                          + buffer_row_pitch * j
                                          + buffer_slice_pitch * k;
              copy_regions[j].size = region[0];
            }
          vkCmdCopyBuffer (cb, memdata->staging_buf, memdata->device_buf,
                           region[1], copy_regions);
          VULKAN_CHECK_ABORT (vkEndCommandBuffer (cb));
          submit_CB (d, &cb);
        }
      free (copy_regions);
    }
}


void pocl_vulkan_memfill(void *data,
                        pocl_mem_identifier * dst_mem_id,
                        cl_mem dst_buf,
                        size_t size,
                        size_t offset,
                        const void *__restrict__  pattern,
                        size_t pattern_size)
{
  pocl_vulkan_device_data_t *d = (pocl_vulkan_device_data_t *)data;
  pocl_vulkan_mem_data_t *memdata
      = (pocl_vulkan_mem_data_t *)dst_mem_id->mem_ptr;

  _cl_command_node cmd;
  _cl_command_run *co = &cmd.command.run;
  cmd.device = d->dev;
  cmd.program_device_i = 0;
  cmd.type = CL_COMMAND_NDRANGE_KERNEL;
  cmd.sync.event.event = NULL;
  cmd.next = NULL;
  cmd.prev = NULL;
  cmd.ready = 1;

  co->wg = NULL;
  co->hash = NULL;

  /* start recording commands. */
  unsigned cb_recorded_commands = 0;
  VkCommandBuffer cb = d->command_buffer;
  VULKAN_CHECK_ABORT (vkResetCommandBuffer (cb, 0));
  VkCommandBufferBeginInfo begin_info;
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.pNext = NULL;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  begin_info.pInheritanceInfo = NULL;
  VULKAN_CHECK_ABORT (vkBeginCommandBuffer (cb, &begin_info));

/* largest pattern size in bytes */
#define NDR_PATTERN_SIZE 128U

/* for sizes above NDR_LIMIT, enqueue a NDRange kernel that copies
 * the pattern; for sizes below, copy using vkCmdCopyBuffer
 * must be multiple of NDR_PATTERN_SIZE */
#define NDR_LIMIT (size_t)32768

  /* prepare a buffer with pattern expanded to NDR_PATTERN_SIZE bytes */
  void *tmp_pattern = pocl_aligned_malloc (NDR_PATTERN_SIZE, NDR_PATTERN_SIZE);
  pocl_fill_aligned_buf_with_pattern (tmp_pattern, 0, NDR_PATTERN_SIZE,
                                      pattern, pattern_size);
  vkCmdUpdateBuffer (cb, d->memfill_buf, 0, NDR_PATTERN_SIZE, tmp_pattern);

  size_t ndr_command_offset = 0, ndr_command_size = 0;
  /** the "head" & "tail" of dst_buf are updated with vkCmdUpdateBuffer **/
  /* in theory we could use vkCmdUpdateBuffer directly instead of
   * vkCmdUpdateBuffer+vkCmdCopyBuffer, but vkCmdUpdateBuffer has
   * alignment 4 requirement on all arguments, so it doesn't work
   * with 1&2-byte sized patterns. */
  if (offset % NDR_PATTERN_SIZE)
    {
      size_t filled_until
          = min (offset + size, (offset | NDR_PATTERN_SIZE - 1) + 1);
      size_t tmp_size = filled_until - offset;

      VkBufferCopy region = { 0, offset, tmp_size };
      vkCmdCopyBuffer (cb, d->memfill_buf, memdata->device_buf, 1, &region);
      cb_recorded_commands++;

      offset = offset + tmp_size;
      size = size - tmp_size;
    }

  if (size >= NDR_LIMIT)
    {
      assert (offset % NDR_PATTERN_SIZE == 0);
      ndr_command_offset = offset;
      ndr_command_size = size & ~(NDR_LIMIT - 1);
      offset = offset + ndr_command_size;
      size = size - ndr_command_size;
    }

  if (size > 0)
    {
      assert (size < NDR_LIMIT);
      assert (offset % NDR_PATTERN_SIZE == 0);
      unsigned repeats = size / NDR_PATTERN_SIZE;
      if (repeats)
        {
          for (unsigned i = 0; i < repeats; ++i)
            {
              VkBufferCopy region
                  = { 0, offset + i * NDR_PATTERN_SIZE, NDR_PATTERN_SIZE };
              vkCmdCopyBuffer (cb, d->memfill_buf, memdata->device_buf, 1,
                               &region);
              cb_recorded_commands++;
            }
          offset += repeats * NDR_PATTERN_SIZE;
          size -= repeats * NDR_PATTERN_SIZE;
        }

      if (size > 0)
        {
          VkBufferCopy region = { 0, offset, size };
          vkCmdCopyBuffer (cb, d->memfill_buf, memdata->device_buf, 1,
                           &region);
          cb_recorded_commands++;
        }

      offset = offset + size;
      size = 0;
    }

  /* submit the CB now, because pocl_vulkan_run will reset CB */
  VULKAN_CHECK_ABORT (vkEndCommandBuffer (cb));
  if (cb_recorded_commands)
    submit_CB (d, &cb);
  /**** end of "head & tail" commands ***/

  /**** now submit the bulk update with a 64B or 128B pattern kernel ***/
  if (ndr_command_size > 0)
    {
      size = ndr_command_size;
      offset = ndr_command_offset;
      assert (offset % NDR_PATTERN_SIZE == 0);
      assert (size % NDR_LIMIT == 0);
      pattern_size = (pattern_size <= 64 ? 64 : 128);
      size_t ngroups = size / pattern_size;
      size_t wgsize = 1;
      while (ngroups % 2 == 0 && wgsize * 2 < d->max_wg_count[0])
        {
          ngroups /= 2;
          wgsize *= 2;
        }
      co->pc.local_size[0] = wgsize;
      co->pc.local_size[1] = 1;
      co->pc.local_size[2] = 1;
      co->pc.num_groups[0] = ngroups;
      co->pc.num_groups[1] = 1;
      co->pc.num_groups[2] = 1;
      co->pc.global_offset[0] = offset / pattern_size;
      co->pc.global_offset[1] = 0;
      co->pc.global_offset[2] = 0;

      co->kernel = (pattern_size > 64 ? d->memfill128_ker : d->memfill64_ker);
      co->arguments = (struct pocl_argument *)malloc (
          co->kernel->meta->num_args * sizeof (struct pocl_argument));
      assert (co->kernel->meta->num_args >= 2);

      co->arguments[0].size = sizeof (cl_mem);
      co->arguments[0].value = &dst_buf;

      co->arguments[1].size = pattern_size;
      co->arguments[1].value = tmp_pattern;

      // enqueue kernel
      pocl_vulkan_run (data, &cmd);
    }

  pocl_aligned_free (tmp_pattern);
}

cl_int
pocl_vulkan_map_mem (void *data, pocl_mem_identifier *src_mem_id,
                     cl_mem src_buf, mem_mapping_t *map)
{
  pocl_vulkan_mem_data_t *memdata
      = (pocl_vulkan_mem_data_t *)src_mem_id->mem_ptr;
  pocl_vulkan_device_data_t *d = (pocl_vulkan_device_data_t *)data;

  POCL_MSG_PRINT_VULKAN ("MAP MEM: %p FLAGS %zu\n", memdata, map->map_flags);

  if (map->map_flags & CL_MAP_WRITE_INVALIDATE_REGION)
    return CL_SUCCESS;

  pocl_vulkan_dev2host (d, memdata, src_mem_id, map->host_ptr, map->offset,
                        map->size);

  return CL_SUCCESS;
}

cl_int
pocl_vulkan_unmap_mem (void *data, pocl_mem_identifier *dst_mem_id,
                       cl_mem dst_buf, mem_mapping_t *map)
{
  pocl_vulkan_mem_data_t *memdata
      = (pocl_vulkan_mem_data_t *)dst_mem_id->mem_ptr;
  pocl_vulkan_device_data_t *d = (pocl_vulkan_device_data_t *)data;

  POCL_MSG_PRINT_VULKAN ("UNMAP MEM: %p FLAGS %zu\n", memdata, map->map_flags);

  /* for read mappings, don't copy anything */
  if (map->map_flags == CL_MAP_READ)
    return CL_SUCCESS;

  pocl_vulkan_host2dev (d, memdata, dst_mem_id, map->host_ptr, map->offset,
                        map->size);

  return CL_SUCCESS;
}

/****************************************************************************************/

static void
pocl_vulkan_setup_kernel_arguments (
    pocl_vulkan_device_data_t *d, cl_device_id dev, unsigned program_device_i,
    _cl_command_run *co, VkShaderModule *compute_shader,
    VkDescriptorSet *ds, VkDescriptorSet *const_ds,
    VkDescriptorSetLayout *dsl, VkDescriptorSetLayout *const_dsl,
    VkSpecializationInfo *spec_info, uint32_t *spec_data,
    VkPushConstantRange *pushc_range, char *pushc_data, char **goffs_start,
    VkSpecializationMapEntry *entries, VkDescriptorSetLayoutBinding *bindings,
    VkDescriptorBufferInfo *descriptor_buffer_info,
    VkDescriptorBufferInfo *pod_ubo_info,
    VkDescriptorSetLayoutBinding *const_binding,
    VkDescriptorBufferInfo *const_descriptor_buffer_info)
{
  cl_kernel kernel = co->kernel;
  struct pocl_argument *pa = co->arguments;
  unsigned dev_i = program_device_i;

  pocl_vulkan_program_data_t *pdata = kernel->program->data[dev_i];
  pocl_vulkan_kernel_data_t *el = NULL, *kdata = NULL;
  DL_FOREACH (pdata->vk_kernel_meta_list, el)
  {
    if (strcmp (el->name, kernel->name) == 0)
      {
        kdata = el;
        break;
      }
  }
  assert (kdata != NULL);
  assert (pdata->shader != NULL);
  *compute_shader = pdata->shader;

  /* setup specialization constants for local size. */
  spec_data[0] = (uint32_t)co->pc.local_size[0];
  spec_data[1] = (uint32_t)co->pc.local_size[1];
  spec_data[2] = (uint32_t)co->pc.local_size[2];

  POCL_MSG_PRINT_VULKAN ("LOC X %zu Y %zu Z %zu \n", co->pc.local_size[0],
                         co->pc.local_size[1], co->pc.local_size[2]);

  entries[0].constantID = 0;
  entries[0].offset = 0;
  entries[0].size = 4;

  entries[1].constantID = 1;
  entries[1].offset = 4;
  entries[1].size = 4;

  entries[2].constantID = 2;
  entries[2].offset = 8;
  entries[2].size = 4;

  uint32_t spec_entries = 0;
  uint32_t spec_offset = 0;
  if (pdata->has_wg_spec_constants)
  {
    spec_entries = 3;
    spec_offset = 12;
  }

  uint32_t pod_entries = 0;

  // num entries, excluding goffset variables
  uint32_t pushc_entries = 0;
  // size, including goffset and all variables
  uint32_t pushc_total_size = 0;

  uint32_t locals = 0;
  uint32_t bufs = 0;

  unsigned current = 0;
  unsigned pod_binding = 0;

  if (kdata->goffset_pushc_size > 0)
    {
      pushc_total_size = kdata->goffset_pushc_size;
      *goffs_start = pushc_data + kdata->goffset_pushc_offset;
    }

  /****************************************************************************************/

  /* preallocate buffers if needed */
  VkMemoryRequirements mem_req;

  if (kdata->kernarg_buf == NULL && kdata->num_pods > 0)
    {
      assert (kdata->num_pod_bytes > 0);

      size_t size = kdata->num_pod_bytes;

      VkBufferCreateInfo buffer_info
          = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
              NULL,
              0,
              size,
              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                  | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                  | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
              VK_SHARING_MODE_EXCLUSIVE,
              1,
              &d->compute_queue_fam_index };

      VULKAN_CHECK_ABORT (
          vkCreateBuffer (d->device, &buffer_info, NULL, &kdata->kernarg_buf));

      vkGetBufferMemoryRequirements (d->device, kdata->kernarg_buf, &mem_req);
      assert (mem_req.size >= size);

      kdata->kernarg_chunk
          = pocl_alloc_buffer_from_region (&d->kernarg_region, mem_req.size);
      assert (kdata->kernarg_chunk);

      kdata->kernarg_buf_offset = kdata->kernarg_chunk->start_address;
      kdata->kernarg_buf_size = mem_req.size;

      VULKAN_CHECK_ABORT (vkBindBufferMemory (d->device, kdata->kernarg_buf,
                                              d->kernarg_mem,
                                              kdata->kernarg_buf_offset));
    }

  if (pdata->constant_buf == NULL && pdata->constant_data_size > 0)
    {
      VkBufferCreateInfo buffer_info
          = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
              NULL,
              0,
              pdata->constant_data_size,
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                  | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                  | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
              VK_SHARING_MODE_EXCLUSIVE,
              1,
              &d->compute_queue_fam_index };

      VULKAN_CHECK_ABORT (vkCreateBuffer (d->device, &buffer_info, NULL,
                                          &pdata->constant_buf));

      vkGetBufferMemoryRequirements (d->device, pdata->constant_buf, &mem_req);
      assert (mem_req.size >= pdata->constant_data_size);

      pdata->constant_chunk
          = pocl_alloc_buffer_from_region (&d->constant_region, mem_req.size);
      assert (pdata->constant_chunk);

      pdata->constant_buf_offset = pdata->constant_chunk->start_address;
      pdata->constant_buf_size = mem_req.size;

      VULKAN_CHECK_ABORT (vkBindBufferMemory (d->device, pdata->constant_buf,
                                              d->constant_mem,
                                              pdata->constant_buf_offset));

      /* copy data to contant mem */
      if (d->kernarg_is_mappable)
        {
          void *constant_pod_ptr;
          VULKAN_CHECK_ABORT (vkMapMemory (
              d->device, d->constant_mem, pdata->constant_buf_offset,
              pdata->constant_buf_size, 0, &constant_pod_ptr));
          memcpy (constant_pod_ptr, pdata->constant_data,
                  pdata->constant_data_size);
          vkUnmapMemory (d->device, d->constant_mem);
        }
      else
        {
          memcpy (d->staging_mapped, pdata->constant_data,
                  pdata->constant_data_size);
          pocl_vulkan_enqueue_staging_buffer_copy (d, pdata->constant_buf,
                                                   pdata->constant_buf_size);
          VULKAN_CHECK_ABORT (vkQueueWaitIdle (d->compute_queue));
        }
    }

  /****************************************************************************************/

  /*
   *
   * https: *github.com/google/clspv/blob/master/docs/OpenCLCOnVulkan.md

   * The default way to map an OpenCL C language kernel to a Vulkan SPIR-V
   * compute shader is as follows:

   * If a sampler map file is specified, all literal samplers use descriptor set 0.

   * By default, all kernels in the translation unit use the same descriptor set number,
   * either 0, 1, or 2. (The particular value depends on whether a sampler map is used,
   * and how __constant variables are mapped.) This is new default behaviour.

   * The compiler can report the descriptor set and bindings used for samplers
   * in the sampler map and for the kernel arguments, and also array sizing
   * information for pointer-to-local arguments. Use option -descriptormap
   *  to name a file that should contain the mapping information.

   * Except for pointer-to-local arguments, each kernel argument is assigned
   * a descriptor binding in that kernel's corresponding DescriptorSet.
   */

  char *kernarg_pod_ptr;

  if (kdata->num_pods > 0)
    {
      if (d->kernarg_is_mappable)
        {
          VULKAN_CHECK_ABORT (vkMapMemory (
              d->device, d->kernarg_mem, kdata->kernarg_buf_offset,
              kdata->kernarg_buf_size, 0, (void **)(&kernarg_pod_ptr)));
        }
      else
        {
          kernarg_pod_ptr = d->staging_mapped;
        }
    }

  cl_uint i;
  for (i = 0; i < kernel->meta->num_args; ++i)
    {
      /* POCL_MSG_ERR ("ARGUMENT: %u | SIZE %lu | VAL %p \n",
                        i, pa[i].size, pa[i].value); */
      if (ARG_IS_LOCAL (kernel->meta->arg_info[i]))
        {
          /* If the argument to the kernel is a pointer to type T in __local
           * storage, then no descriptor is generated. Instead, that argument
           * is mapped to a variable in Workgroup storage class, of type
           * array-of-T. The array size is specified by an integer
           * specialization constant. The specialization ID is reported in the
           * descriptor map file, generated via the -descriptormap option.
           */
          assert (pa[i].size > 0);

          assert (kdata->locals[locals].ord == i);
          uint32_t elems = pa[i].size / kdata->locals[locals].elem_size;
          POCL_MSG_PRINT_VULKAN ("setting LOCAL argument %i TO"
                                 "SIZE %zu ELEMS: %u\n; ",
                                  i, pa[i].size, elems);
          uint32_t spec_id = kdata->locals[locals].spec_id;
          entries[spec_entries].constantID = spec_id;
          entries[spec_entries].offset = spec_offset;
          entries[spec_entries].size = 4;
          spec_data[spec_entries] = elems;
          ++spec_entries;
          ++locals;
          spec_offset += 4;
        }
      else if (kernel->meta->arg_info[i].type == POCL_ARG_TYPE_POINTER)
        {
          /* If the argument to the kernel is a global or constant pointer,
           * it is placed into a SPIR-V OpTypeStruct that is decorated
           * with Block, and an OpVariable of this structure type is
           * created and decorated with the corresponding DescriptorSet
           * and Binding, using the StorageBuffer storage class.
           */
          assert (pa[i].value != NULL);
          assert (pa[i].size == sizeof (void *));

          cl_mem arg_buf = (*(cl_mem *)(pa[i].value));
          pocl_mem_identifier *memid
              = &arg_buf->device_ptrs[dev->global_mem_id];
          pocl_vulkan_mem_data_t *memdata
              = (pocl_vulkan_mem_data_t *)memid->mem_ptr;
          POCL_MSG_PRINT_VULKAN (
              "ARG BUF: %p MEMDATA: %p VULKANBUF: %zu CURRENT: %u \n", arg_buf,
              memdata, (size_t)memdata->device_buf, current);

          descriptor_buffer_info[current].buffer = memdata->device_buf;
          descriptor_buffer_info[current].offset = 0;
          descriptor_buffer_info[current].range = memid->extra;

          bindings[current].binding = kdata->bufs[bufs].binding;
          bindings[current].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          bindings[current].descriptorCount = 1;
          bindings[current].pImmutableSamplers = 0;
          bindings[current].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
          assert (current < 128);
          ++current;

          assert (kdata->bufs[bufs].dset == 0);
          assert (kdata->bufs[bufs].ord == i);
          ++bufs;
        }
      else if (kernel->meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
        {
          /* If the argument to the kernel is an image or sampler,
           * an OpVariable of the OpTypeImage or OpTypeSampler type
           * is created and decorated with the corresponding DescriptorSet
           * and Binding, using the UniformConstant storage class.
           */

          POCL_ABORT_UNIMPLEMENTED ("images in vulkan");
        }
      else if (kernel->meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)
        {
          POCL_ABORT_UNIMPLEMENTED ("samplers in vulkan");
        }
      else
        {
          /* Normally plain-old-data arguments are passed into the kernel via a
           * storage buffer. Use option -pod-ubo to pass these parameters in
           * via a uniform buffer. These can be faster to read in the shader.
           * When option -pod-ubo is used, the descriptor map list the argKind
           * of a plain-old-data argument as pod_ubo rather than the default of
           * pod.
           */

          /*
           * Clustering plain-old-data kernel arguments to save descriptors
           *
           * Descriptors can be scarce. So the compiler also has an option
           * -cluster-pod-kernel-args which can be used to reduce the number
           * of descriptors. When the option is used:

           * All plain-old-data (POD) kernel arguments are collected into a
           single struct
           * and passed into the compute shader via a single storage buffer
           resource.

           * The binding number for the struct containing the POD arguments
           * is one more than the highest non-POD argument.
           */

          assert (pa[i].value != NULL);
          assert (pa[i].size > 0);
          assert (pa[i].size == kernel->meta->arg_info[i].type_size);

          if (kdata->num_pushc_arg_bytes > 0)
            {
              assert (kdata->pushc[pushc_entries].size == pa[i].size);
              assert (kdata->pushc[pushc_entries].ord == i);
              uint32_t offs = kdata->pushc[pushc_entries].offset;

              memcpy (pushc_data + offs, pa[i].value, pa[i].size);
              if (offs + pa[i].size > pushc_total_size)
                pushc_total_size = offs + pa[i].size;

              ++pushc_entries;
            }
          else
            {
              assert (kdata->pods[pod_entries].dset == 0);
              assert (kdata->pods[pod_entries].ord == i);
              uint32_t offs = kdata->pods[pod_entries].offset;

              memcpy (kernarg_pod_ptr + offs, pa[i].value, pa[i].size);
              ++pod_entries;
            }
        }
    }

  pushc_range->offset = 0;
  pushc_range->size = pushc_total_size;
  pushc_range->stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  if (kdata->num_pods > 0)
    {
      if (d->kernarg_is_mappable)
        {
          vkUnmapMemory (d->device, d->kernarg_mem);
        }
      else
        {
          pocl_vulkan_enqueue_staging_buffer_copy (d, kdata->kernarg_buf,
                                                   kdata->kernarg_buf_size);
        }
    }

  /* PODs: setup descriptor & bindings for PODs; last binding in DS 0 */
  if (kdata->num_pods > 0)
    {
      /* add the kernarg memory */
      pod_binding = kdata->pods[0].binding;

      pod_ubo_info->buffer = kdata->kernarg_buf; /* the POD buffer */
      pod_ubo_info->offset = 0;
      pod_ubo_info->range = kdata->num_pod_bytes;

      bindings[current].binding = pod_binding;
      bindings[current].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      bindings[current].descriptorCount = 1;
      bindings[current].pImmutableSamplers = 0;
      bindings[current].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      assert (current < 128);
      ++current;
    }

  /* static locals are taken care of in clspv. */
  assert (kernel->meta->num_locals == 0);

  spec_info->mapEntryCount = spec_entries;
  spec_info->pMapEntries = (spec_entries > 0) ? entries : NULL;
  spec_info->dataSize = spec_offset;
  spec_info->pData = (spec_entries > 0) ? spec_data : NULL;

  POCL_MSG_PRINT_VULKAN (
      "SPECINFO  entry count: %u ENTRIES %p DATASIZE %lu DATA %p\n",
      spec_entries, entries, spec_info->dataSize, spec_data);
  for (i = 0; i < spec_entries; ++i)
    POCL_MSG_PRINT_VULKAN (
        "SPECENTRY %u ||| ID %u | OFF %u | SIZE %zu | DATA %u \n", i,
        entries[i].constantID, entries[i].offset, entries[i].size,
        spec_data[i]);

  VkDescriptorSetLayoutCreateInfo dslCreateInfo
      = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
          NULL,
          0,
          current,
          bindings };
  VULKAN_CHECK_ABORT (
      vkCreateDescriptorSetLayout (d->device, &dslCreateInfo, NULL, dsl));

  VkDescriptorSetAllocateInfo descriptorSetallocate_info
      = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, 0,
          d->buf_descriptor_pool, 1, dsl };
  VULKAN_CHECK_ABORT (
      vkAllocateDescriptorSets (d->device, &descriptorSetallocate_info, ds));

  /* write descriptors  */
  for (i = 0; i < current; ++i)
    {

      if (bindings[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
        {
          VkWriteDescriptorSet writeDescriptorSet
              = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                  0,                                 /* next */
                  *ds,                               /* dstSet */
                  bindings[i].binding,               /* dstBinding */
                  0,                                 /* dstArrayElement */
                  1,                                 /* descriptorCount */
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, /* descriptorType */
                  0,
                  &descriptor_buffer_info[i],
                  0 };
          vkUpdateDescriptorSets (d->device, 1, &writeDescriptorSet, 0, 0);
        }

      if (bindings[i].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
        {
          VkWriteDescriptorSet writeDescriptorSet
              = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                  0,
                  *ds,                               /* dstSet */
                  pod_binding,                       /* dstBinding */
                  0,                                 /* dstArrayElement */
                  1,                                 /* descriptorCount */
                  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, /* descriptorType */
                  0,
                  pod_ubo_info,
                  0 };
          vkUpdateDescriptorSets (d->device, 1, &writeDescriptorSet, 0, 0);
        }
    }

  /* setup descriptor & bindings for constants   */
  if (pdata->constant_data_size > 0)
    {
      /* add the constant memory */
      const_descriptor_buffer_info->buffer
          = pdata->constant_buf; /* the constant buffer */
      const_descriptor_buffer_info->offset = 0;
      const_descriptor_buffer_info->range = VK_WHOLE_SIZE;

      const_binding->binding = pdata->constant_binding;
      const_binding->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      const_binding->descriptorCount = 1;
      const_binding->pImmutableSamplers = 0;
      const_binding->stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

      VkDescriptorSetLayoutCreateInfo dslCreateInfo
          = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, 0, 0, 1,
              const_binding };
      VULKAN_CHECK_ABORT (vkCreateDescriptorSetLayout (
          d->device, &dslCreateInfo, NULL, const_dsl));

      VkDescriptorSetAllocateInfo descriptorSetallocate_info
          = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, 0,
              d->buf_descriptor_pool, 1, const_dsl };
      VULKAN_CHECK_ABORT (vkAllocateDescriptorSets (
          d->device, &descriptorSetallocate_info, const_ds));

      VkWriteDescriptorSet writeDescriptorSet
          = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
              0,                                  /* pNext */
              *const_ds,                          /* dstSet */
              pdata->constant_binding,            /* dstBinding */
              0,                                  /* dstArrayElement */
              1,                                  /* descriptorCount */
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  /* descriptorType */
              0,                                  /* pImageInfo*/
              const_descriptor_buffer_info,       /* pBufferInfo */
              0 };                                /* pTexelBufferView */
      vkUpdateDescriptorSets (d->device, 1, &writeDescriptorSet, 0, 0);
    }
  else
    {
      *const_ds = NULL;
      *const_dsl = NULL;
    }
}

void
pocl_vulkan_run (void *data, _cl_command_node *cmd)
{
  pocl_vulkan_device_data_t *d = data;
  _cl_command_run *co = &cmd->command.run;
  cl_device_id dev = cmd->device;
  assert (cmd->type == CL_COMMAND_NDRANGE_KERNEL);
  cl_kernel kernel = cmd->command.run.kernel;

  struct pocl_context *pc = &co->pc;
  uint32_t total_wg_x = pc->num_groups[0];
  uint32_t total_wg_y = pc->num_groups[1];
  uint32_t total_wg_z = pc->num_groups[2];
  uint32_t start_offs_x = pc->global_offset[0];
  uint32_t start_offs_y = pc->global_offset[1];
  uint32_t start_offs_z = pc->global_offset[2];

  size_t total_wgs = total_wg_x * total_wg_y * total_wg_z;
  if (total_wgs == 0)
    return;

  POCL_MSG_PRINT_VULKAN ("WG X %u Y %u Z %u ||| OFFS X %u Y %u Z %u\n",
                         total_wg_x, total_wg_y, total_wg_z,
                         start_offs_x, start_offs_y, start_offs_z);

  VkDescriptorSet descriptor_sets[2] = { NULL, NULL };
  VkDescriptorSetLayout descriptor_set_layouts[2];
  VkSpecializationInfo specInfo;
  uint32_t spec_data[128];
  VkSpecializationMapEntry entries[128];
  VkDescriptorSetLayoutBinding bindings[128];
  VkDescriptorBufferInfo descriptor_buffer_info[128];
  VkDescriptorBufferInfo pod_ubo_info = { NULL, 0, 0 };
  VkDescriptorSetLayoutBinding const_binding = { 0, 0, 0, 0, NULL };
  VkDescriptorBufferInfo const_descriptor_buffer_info = { 0 };

  VkPipelineLayout pipeline_layout;
  VkPipeline pipeline;
  VkShaderModule compute_shader = NULL;
  VkPushConstantRange pushc_range;
  char pushc_data[d->max_pushc_size];
  char *goffs_start = NULL;

  pocl_vulkan_setup_kernel_arguments (
      d, cmd->device, cmd->program_device_i, co, &compute_shader,
      descriptor_sets, descriptor_sets + 1,
      descriptor_set_layouts, descriptor_set_layouts + 1,
      &specInfo, spec_data,
      &pushc_range, pushc_data, &goffs_start,
      entries, bindings,
      descriptor_buffer_info, &pod_ubo_info,
      &const_binding, &const_descriptor_buffer_info);
  VkPipelineShaderStageCreateInfo shader_stage_info;
  shader_stage_info.sType
      = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shader_stage_info.pNext = NULL;
  shader_stage_info.flags = 0;
  shader_stage_info.pSpecializationInfo = &specInfo;
  shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  shader_stage_info.module = compute_shader;
  shader_stage_info.pName = kernel->name;

  VkPipelineLayoutCreateInfo pipeline_layout_create_info;
  pipeline_layout_create_info.sType
      = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_create_info.pNext = NULL;
  pipeline_layout_create_info.flags = 0;
  if (pushc_range.size > 0)
    {
      pipeline_layout_create_info.pPushConstantRanges = &pushc_range;
      pipeline_layout_create_info.pushConstantRangeCount = 1;
    }
  else
    {
      pipeline_layout_create_info.pPushConstantRanges = 0;
      pipeline_layout_create_info.pushConstantRangeCount = 0;
    }
  if (descriptor_sets[1])
    pipeline_layout_create_info.setLayoutCount = 2;
  else
    pipeline_layout_create_info.setLayoutCount = 1;
  pipeline_layout_create_info.pSetLayouts = descriptor_set_layouts;
  VULKAN_CHECK_ABORT (vkCreatePipelineLayout (
      d->device, &pipeline_layout_create_info, NULL, &pipeline_layout));

  VkComputePipelineCreateInfo pipeline_create_info;
  pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_create_info.pNext = NULL;
  pipeline_create_info.flags = 0;
  pipeline_create_info.stage = shader_stage_info;
  pipeline_create_info.layout = pipeline_layout;
  pipeline_create_info.basePipelineIndex = 0;
  pipeline_create_info.basePipelineHandle = 0;

  VULKAN_CHECK_ABORT (vkCreateComputePipelines (
      d->device, d->cache, 1, &pipeline_create_info, NULL, &pipeline));

  VkCommandBuffer cb = d->command_buffer;
  uint32_t commands_recorded = 0;
  uint32_t wg_x = 0, wg_y = 0, wg_z = 0;
  uint32_t goffs_x = 0, goffs_y = 0, goffs_z = 0;
  if (goffs_start != NULL)
    {
      assert (goffs_start >= pushc_data);
      assert (goffs_start < pushc_data + d->max_pushc_size);
    }

  VkCommandBufferBeginInfo begin_info;
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.pNext = NULL;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  begin_info.pInheritanceInfo = NULL;

  VULKAN_CHECK_ABORT (vkResetCommandBuffer (cb, 0));
  VULKAN_CHECK_ABORT (vkBeginCommandBuffer (cb, &begin_info));
  vkCmdBindPipeline (cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets (cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout,
                           0, ((descriptor_sets[1] ? 2 : 1)), descriptor_sets,
                           0, NULL);

  for (uint32_t wg_z_offs = 0; wg_z_offs < total_wg_z;
       wg_z_offs += d->max_wg_count[2])
    {
      wg_z = min (d->max_wg_count[2], total_wg_z - wg_z_offs);
      goffs_z = wg_z_offs * pc->local_size[2];
      {
        for (uint32_t wg_y_offs = 0; wg_y_offs < total_wg_y;
             wg_y_offs += d->max_wg_count[1])
          {
            wg_y = min (d->max_wg_count[1], total_wg_y - wg_y_offs);
            goffs_y = wg_y_offs * pc->local_size[1];
            {
              for (uint32_t wg_x_offs = 0; wg_x_offs < total_wg_x;
                   wg_x_offs += d->max_wg_count[0])
                {
                  wg_x = min (d->max_wg_count[0], total_wg_x - wg_x_offs);
                  goffs_x = wg_x_offs * pc->local_size[0];

                  if (pushc_range.size > 0)
                    {
                      // global offset sometimes is optimized out, even
                      // if we compile with --global-offsets option
                      if (goffs_start != NULL)
                        {
                          memcpy (goffs_start + 0, &goffs_x,
                                  sizeof (uint32_t));
                          memcpy (goffs_start + 4, &goffs_y,
                                  sizeof (uint32_t));
                          memcpy (goffs_start + 8, &goffs_z,
                                  sizeof (uint32_t));
                        }
                      // upload the arg data to the GPU via push constants
                      vkCmdPushConstants (cb, pipeline_layout,
                                          VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                          pushc_range.size, pushc_data);
                    }

                  vkCmdDispatch (cb, wg_x, wg_y, wg_z);

                  /* TODO find out what's the limit of submit commands in a
                   * single command buffer.
                   */
                  if (commands_recorded == 2048)
                    {
                      commands_recorded = 0;
                      VULKAN_CHECK_ABORT (vkEndCommandBuffer (cb));
                      submit_CB (d, &cb);
                      VULKAN_CHECK_ABORT (vkResetCommandBuffer (cb, 0));
                      VULKAN_CHECK_ABORT (
                          vkBeginCommandBuffer (cb, &begin_info));
                      vkCmdBindPipeline (cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                         pipeline);
                      vkCmdBindDescriptorSets (
                          cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout,
                          0, ((descriptor_sets[1] ? 2 : 1)), descriptor_sets,
                          0, NULL);
                    }
                  else
                    ++commands_recorded;
                }
            }
          }
      }
    }

  if (commands_recorded > 0)
    {
      VULKAN_CHECK_ABORT (vkEndCommandBuffer (cb));
      submit_CB (d, &cb);
    }

  if (descriptor_sets[0]) {
      VULKAN_CHECK_ABORT (vkFreeDescriptorSets (
          d->device, d->buf_descriptor_pool, 1, descriptor_sets));
      vkDestroyDescriptorSetLayout (d->device, descriptor_set_layouts[0],
                                    NULL);
  }
  if (descriptor_sets[1]) {
      VULKAN_CHECK_ABORT (vkFreeDescriptorSets (
          d->device, d->buf_descriptor_pool, 1, descriptor_sets + 1));
      vkDestroyDescriptorSetLayout (d->device, descriptor_set_layouts[1],
                                    NULL);
  }

  vkDestroyPipelineLayout (d->device, pipeline_layout, NULL);
  vkDestroyPipeline (d->device, pipeline, NULL);
}

static size_t
vulkan_process_work (pocl_vulkan_device_data_t *d)
{
  _cl_command_node *cmd;

  POCL_FAST_LOCK (d->wq_lock_fast);
  size_t do_exit = 0;

RETRY:
  do_exit = d->driver_thread_exit_requested;

  cmd = d->work_queue;
  if (cmd)
    {
      DL_DELETE (d->work_queue, cmd);
      POCL_FAST_UNLOCK (d->wq_lock_fast);

      assert (pocl_command_is_ready (cmd->sync.event.event));
      assert (cmd->sync.event.event->status == CL_SUBMITTED);

      pocl_exec_command (cmd);

      POCL_FAST_LOCK (d->wq_lock_fast);
    }

  if ((cmd == NULL) && (do_exit == 0))
    {
      POCL_WAIT_COND (d->wakeup_cond, d->wq_lock_fast);
      /* since cond_wait returns with locked mutex, might as well retry */
      goto RETRY;
    }

  POCL_FAST_UNLOCK (d->wq_lock_fast);

  return do_exit;
}

static void *
pocl_vulkan_driver_pthread (void *cldev)
{
  cl_device_id device = (cl_device_id)cldev;
  pocl_vulkan_device_data_t *d = (pocl_vulkan_device_data_t *)device->data;

  while (1)
    {
      if (vulkan_process_work (d))
        goto EXIT_PTHREAD;
    }

EXIT_PTHREAD:

  vkDestroyCommandPool (d->device, d->command_pool, NULL);
  vkDestroyDescriptorPool (d->device, d->buf_descriptor_pool, NULL);
  /* destroy logical device */
  vkDestroyDevice (d->device, NULL);

  pthread_exit (NULL);
}

/****************************************************************************************/

/* assumes alignment is pow-of-2 */
size_t
pocl_vulkan_actual_memobj_size (pocl_vulkan_device_data_t *d, cl_mem mem,
                                pocl_mem_identifier *p,
                                VkMemoryRequirements *mem_req)
{
  if (p->extra != 0)
    return p->extra;

  VkBuffer buffer;

  VkBufferCreateInfo buffer_info
      = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
          NULL,
          0, /* TODO flags */
          mem->size,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
              | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
          VK_SHARING_MODE_EXCLUSIVE,
          1,
          &d->compute_queue_fam_index };

  VULKAN_CHECK_ABORT (vkCreateBuffer (d->device, &buffer_info, NULL, &buffer));

  if (p->extra == 0)
    {
      vkGetBufferMemoryRequirements (d->device, buffer, mem_req);
      assert (mem_req->size > 0);
      p->extra = mem_req->size;
    }

  vkDestroyBuffer (d->device, buffer, NULL);

  return p->extra;
}

static int
should_import_external_memory (cl_mem mem, pocl_vulkan_device_data_t *d,
                               VkMemoryAllocateInfo *allocate_info,
                               VkImportMemoryHostPointerInfoEXT *import_mem)
{
  int res = CL_FALSE;
  if ((mem->flags & CL_MEM_USE_HOST_PTR) && (d->needs_staging_mem == CL_FALSE))
    {
      assert (mem->mem_host_ptr != NULL);

      if ((uintptr_t)mem->mem_host_ptr
          & (uintptr_t) (d->min_ext_host_mem_align - 1))
        POCL_MSG_WARN (
            "Vulkan: pointer %p given to CL_MEM_USE_HOST_PTR is not "
            "aligned properly; using an extra staging buffer\n",
            mem->mem_host_ptr);
      else
        {
          VkMemoryHostPointerPropertiesEXT host_mem_props
              = { VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT, NULL,
                  0 };
          VkResult r = d->vkGetMemoryHostPointerProperties (
              d->device,
              VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
              mem->mem_host_ptr, &host_mem_props);

          if (r == VK_SUCCESS)
            {
              POCL_MSG_WARN ("memory Type Bits: %u\n",
                             host_mem_props.memoryTypeBits);
              unsigned memtype = 0;
              unsigned val = host_mem_props.memoryTypeBits;
              while (val != 1)
                {
                  ++memtype;
                  val >>= 1;
                }
              allocate_info->pNext = import_mem;
              allocate_info->memoryTypeIndex = memtype;
              res = CL_TRUE;

              size_t imported_mem_size = allocate_info->allocationSize;
              /* round up the buffer size to the nearest multiple of
               * minImportedHostPointerAlignment. This is required for some
               * drivers like AMD RADV, otherwise they return
               *  -2 VK_ERROR_OUT_OF_DEVICE_MEMORY.
               * in theory should be safe, but cache
               * invalidation could be a problem */
              if (imported_mem_size & (size_t) (d->min_ext_host_mem_align - 1))
                {
                  imported_mem_size
                      = (imported_mem_size
                         | (size_t) (d->min_ext_host_mem_align - 1))
                        + 1;
                }
              allocate_info->allocationSize = imported_mem_size;
            }
        }
    }
  return res;
}

int
pocl_vulkan_alloc_mem_obj (cl_device_id device, cl_mem mem, void *host_ptr)
{
  pocl_vulkan_device_data_t *d = (pocl_vulkan_device_data_t *)device->data;
  pocl_mem_identifier *p = &mem->device_ptrs[device->global_mem_id];
  VkBuffer b;
  VkDeviceMemory m;

  assert (p->mem_ptr == NULL);

  /* TODO driver doesn't preallocate YET, but we should, when requested */
  if ((mem->flags & CL_MEM_ALLOC_HOST_PTR) && (mem->mem_host_ptr == NULL))
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;

  VkMemoryRequirements memReq;
  size_t actual_mem_size = pocl_vulkan_actual_memobj_size (d, mem, p, &memReq);
  if (actual_mem_size == 0)
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  p->extra = actual_mem_size;

  p->version = 0;
  /* either mapped staging buffer (dGPU), mapped device buffer (iGPU),
   * or host memory (imported pointer) when CL_MEM_USE_HOST_PTR */
  p->extra_ptr = NULL;

  pocl_vulkan_mem_data_t *memdata
      = (pocl_vulkan_mem_data_t *)calloc (1, sizeof (pocl_vulkan_mem_data_t));
  if (memdata == NULL)
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;
  p->mem_ptr = memdata;

  VkMemoryAllocateInfo allocate_info
      = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, actual_mem_size,
          d->device_mem_type };
  VkImportMemoryHostPointerInfoEXT import_mem
      = { VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, NULL,
          VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
          mem->mem_host_ptr };

  int import_success = CL_FALSE;

  /* if CL_MEM_USE_HOST_PTR is in flags, try to import host pointer */
  if (should_import_external_memory (mem, d, &allocate_info, &import_mem))
    {
      if (vkAllocateMemory (d->device, &allocate_info, NULL, &m) == VK_SUCCESS)
        import_success = CL_TRUE;
    }

  /* host pointer import failed / not supported / no CL_USE_MEM_HOST_PTR */
  if (import_success == CL_FALSE)
    {
      allocate_info.pNext = NULL;
      allocate_info.memoryTypeIndex = d->device_mem_type;
      allocate_info.allocationSize = actual_mem_size;
      int res = (vkAllocateMemory (d->device, &allocate_info, NULL, &m));
      if (res != VK_SUCCESS)
        {
          goto ERROR;
        }
    }

  actual_mem_size = allocate_info.allocationSize;
  VkBufferCreateInfo buffer_info
      = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
          NULL,
          0,
          actual_mem_size,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
              | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
          VK_SHARING_MODE_EXCLUSIVE,
          1,
          &d->compute_queue_fam_index };
  int res = (vkCreateBuffer (d->device, &buffer_info, NULL, &b));
  if (res != VK_SUCCESS)
    goto ERROR;

  memdata->device_buf = b;
  memdata->device_mem = m;

  vkGetBufferMemoryRequirements (d->device, memdata->device_buf, &memReq);
  assert (actual_mem_size == memReq.size);
  if (vkBindBufferMemory (d->device, memdata->device_buf, memdata->device_mem,
                          0)
      != VK_SUCCESS)
    goto ERROR;

  /* STAGING MEM */
  if (d->needs_staging_mem)
    {
      if (mem->flags & (CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY))
        allocate_info.memoryTypeIndex = d->host_staging_write_type;
      else
        allocate_info.memoryTypeIndex = d->host_staging_read_type;

      if (vkCreateBuffer (d->device, &buffer_info, NULL, &b) != VK_SUCCESS)
        goto ERROR;
      memdata->staging_buf = b;
      if (vkAllocateMemory (d->device, &allocate_info, NULL, &m) != VK_SUCCESS)
        goto ERROR;
      memdata->staging_mem = m;
      vkGetBufferMemoryRequirements (d->device, memdata->staging_buf, &memReq);
      assert (actual_mem_size == memReq.size);
      if (vkBindBufferMemory (d->device, memdata->staging_buf,
                              memdata->staging_mem, 0)
          != VK_SUCCESS)
        goto ERROR;

      if (vkMapMemory (d->device, memdata->staging_mem, 0, actual_mem_size, 0,
                       &p->extra_ptr)
          != VK_SUCCESS)
        goto ERROR;
    }
  else
    {
      memdata->staging_buf = 0;
      memdata->staging_mem = 0;
      if (import_success)
        p->extra_ptr = mem->mem_host_ptr;
      else if (vkMapMemory (d->device, memdata->device_mem, 0, actual_mem_size,
                            0, &p->extra_ptr)
               != VK_SUCCESS)
        goto ERROR;
    }

  POCL_MSG_PRINT_MEMORY ("VULKAN DEVICE ALLOC PTR %p ALLOC %zu | "
                         "VULKAN DEV BUF %p | STA BUF %p | EXTRA_PTR %p \n",
                         p->mem_ptr, p->extra, (void *)memdata->device_buf,
                         (void *)memdata->staging_buf, p->extra_ptr);

  return CL_SUCCESS;

ERROR:
  pocl_vulkan_free (device, mem);
  return CL_MEM_OBJECT_ALLOCATION_FAILURE;
}

void
pocl_vulkan_free (cl_device_id device, cl_mem mem)
{
  pocl_vulkan_device_data_t *d = (pocl_vulkan_device_data_t *)device->data;
  pocl_mem_identifier *p = &mem->device_ptrs[device->global_mem_id];
  pocl_vulkan_mem_data_t *memdata = (pocl_vulkan_mem_data_t *)p->mem_ptr;

  assert (p->extra > 0);
  assert (memdata != NULL);

  if (d->needs_staging_mem)
    {
      if (memdata->staging_mem)
        vkUnmapMemory (d->device, memdata->staging_mem);

      if (memdata->staging_buf)
        vkDestroyBuffer (d->device, memdata->staging_buf, NULL);
      if (memdata->staging_mem)
        vkFreeMemory (d->device, memdata->staging_mem, NULL);

      if (memdata->device_buf)
        vkDestroyBuffer (d->device, memdata->device_buf, NULL);
      if (memdata->device_mem)
        vkFreeMemory (d->device, memdata->device_mem, NULL);
    }
  else
    {
      if (p->extra_ptr != mem->mem_host_ptr && memdata->device_mem)
        vkUnmapMemory (d->device, memdata->device_mem);

      if (memdata->device_buf)
        vkDestroyBuffer (d->device, memdata->device_buf, NULL);
      if (memdata->device_mem)
        vkFreeMemory (d->device, memdata->device_mem, NULL);
    }

  POCL_MSG_PRINT_MEMORY ("VULKAN DEVICE FREE PTR %p SIZE %zu \n", p->mem_ptr,
                         mem->size);

  POCL_MEM_FREE (p->mem_ptr);

  p->mem_ptr = NULL;
  p->version = 0;
  p->extra_ptr = NULL;
  p->extra = 0;
}


cl_int
pocl_vulkan_get_mapping_ptr (void *data, pocl_mem_identifier *mem_id,
                             cl_mem mem, mem_mapping_t *map)
{
  pocl_vulkan_device_data_t *d = (pocl_vulkan_device_data_t *)data;
  /* assume buffer is allocated */
  assert (mem_id->mem_ptr != NULL);
  assert (mem_id->extra_ptr != NULL);
  assert (mem->size > 0);
  assert (map->size > 0);

  if (mem->flags & CL_MEM_USE_HOST_PTR)
    map->host_ptr = (char *)mem->mem_host_ptr + map->offset;
  else
    map->host_ptr = (char *)mem_id->extra_ptr + map->offset;
  /* POCL_MSG_ERR ("map HOST_PTR: %p | SIZE %zu | OFFS %zu | DEV PTR: %p \n",
                  map->host_ptr, map->size, map->offset, mem_id->mem_ptr); */
  assert (map->host_ptr);
  return CL_SUCCESS;
}

cl_int
pocl_vulkan_free_mapping_ptr (void *data, pocl_mem_identifier *mem_id,
                              cl_mem mem, mem_mapping_t *map)
{
  pocl_vulkan_device_data_t *d = (pocl_vulkan_device_data_t *)data;
  map->host_ptr = NULL;
  return CL_SUCCESS;
}
