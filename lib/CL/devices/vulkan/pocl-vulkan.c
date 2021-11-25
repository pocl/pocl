/* pocl-vulkan.c - driver for Vulkan Compute API devices.

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

/********************************************************************/

/* What works:
 *
 * (hopefully) reasonable memory-type handling with both iGPUs and dGPUs
 * buffer arguments (cl_mem)
 * POD (plain old data) arguments, by which i mean integers; structs / unions
 * are untested automatic (hidden) locals local arguments simple kernels like
 * matrix multiplication
 *
 * Doesnt work / unfinished / non-optimal:
 *
 * VMA (virtual memory allocator) on device memory, currently
 * driver just calls vkAllocateMemory for each cl_mem allocation
 *
 * CL_MEM_USE_HOST_PTR is broken,
 * CL_MEM_ALLOC_HOST_PTR is ignored
 *
 * properly cleanup objects, check for memory leaks
 *
 * fix statically sized data structs
 *
 * descriptor set should be cached (setup once per kernel, then just update)
 *
 * there's a device limit on max WG count
 *    (the amount of local WGs that can be executed by a single command)
 *    - need to handle global size > than that
 *    - need offset in get_global_id()
 *
 * image / sampler support support missing
 *
 * global offsets of kernel enqueue are ignored (should be solved by
 * compiling two versions of each program, one with goffsets and one
 * without, then select at runtime which to use)
 *
 * some things that are stored per-kernel should be stored per-program,
 * and v-v (e.g. compiled shader)
 *
 * module scope constants support missing
 *
 * rectangular read/copy/write commands (clEnqueueReadBufferRect etc)
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

#include <vulkan/vulkan.h>

#include "config.h"
#include "pocl_cl.h"

#include "common.h"
#include "common_driver.h"
#include "devices.h"
#include "utlist.h"

#include "pocl_cache.h"
#include "pocl_debug.h"
#include "pocl_file_util.h"
#include "pocl_llvm.h"
#include "pocl_timing.h"
#include "pocl_util.h"

#include "pocl-vulkan.h"

#include "bufalloc.h"

#define MAX_CMD_BUFFERS 8
#define MAX_BUF_DESCRIPTORS 4096
#define MAX_UBO_DESCRIPTORS 1024
#define MAX_DESC_SETS 1024

#define PAGE_SIZE 4096

static void *pocl_vulkan_driver_pthread (void *cldev);

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

typedef struct pocl_vulkan_kernel_data_s
{
  /* these are parsed from clspv provided descriptor map.
   * most are used just for verification (clspv assigns descriptors
   * quite predictably), except locals.elem_size which is required for
   * setting the Specialization constant correctly. */
  size_t num_locals;
  size_t num_bufs;
  size_t num_pods;
  local_data locals[MAX_LOCALS];
  buf_data bufs[MAX_BUFS];
  buf_data pods[MAX_PODS];

  /* since POD arguments are pushed via a buffer (kernarg_buf),
   * total size helps with preallocating kernarg buffer */
  size_t num_pod_bytes;

  /* constants are also preallocated, this is the total size */
  size_t num_constant_bytes;

  uint32_t constant_dset, constant_binding;
  uint32_t *constant_data;

  /* kernarg buffer, for POD arguments */
  VkBuffer kernarg_buf;
  VkDeviceSize kernarg_buf_offset;
  VkDeviceSize kernarg_buf_size;
  chunk_info_t *kernarg_chunk;

  /* compiled shader binary */
  VkShaderModule shader;

  /* buffer for holding Constant data per kernel
   * TODO this should be per-program not per-kernel */
  VkBuffer constant_buf;
  VkDeviceSize constant_buf_offset;
  VkDeviceSize constant_buf_size;
  chunk_info_t *constant_chunk;
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

  /* integrated GPUs have different Vulkan memory layout */
  int device_is_iGPU;
  /* device needs staging buffers for memory transfers
   * TODO this might be equal to "device_is_iGPU" */
  int needs_staging_mem;
  /* 1 if kernarg memory is equal to device-local memory */
  int kernarg_is_device_mem;
  /* 1 if kernarg memory is directly mappable to host AS */
  int kernarg_is_mappable;

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

} pocl_vulkan_device_data_t;

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

#define VULKAN_CHECK(code)                                                    \
  pocl_vulkan_abort_on_vk_error (code, __LINE__, __FUNCTION__, #code)


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

#define KERNARG_BUFFER_SIZE (8 << 20)
#define CONSTANT_BUFFER_SIZE (8 << 20)
#define STAGING_BUF_SIZE (4 << 20)

static void
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

  if (d->device_is_iGPU)
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
      assert (heap_i != UINT32_MAX);

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
      assert (d->device_mem_type != UINT32_MAX);

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
        POCL_ABORT ("Vulkan: can't find device memory type \n");

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
  VULKAN_CHECK (
      vkAllocateMemory (d->device, &allocate_info, NULL, &d->kernarg_mem));

  init_mem_region (&d->kernarg_region, 0, d->kernarg_size);
  d->kernarg_region.strategy = BALLOCS_TIGHT;
  d->kernarg_region.alignment = PAGE_SIZE;

  POCL_MSG_PRINT_VULKAN ("Allocated %zu memory for kernel arguments\n",
                         d->kernarg_size);

  /* preallocate constant memory */

  allocate_info.allocationSize = d->constant_size;
  allocate_info.memoryTypeIndex = d->constant_mem_type;

  VULKAN_CHECK (
      vkAllocateMemory (d->device, &allocate_info, NULL, &d->constant_mem));

  init_mem_region (&d->constant_region, 0, d->constant_size);
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

      VULKAN_CHECK (
          vkCreateBuffer (d->device, &buffer_info, NULL, &d->staging_buf));
      VULKAN_CHECK (
          vkAllocateMemory (d->device, &allocate_info, NULL, &d->staging_mem));
      VkMemoryRequirements memReq;
      vkGetBufferMemoryRequirements (d->device, d->staging_buf, &memReq);
      assert (d->staging_size == memReq.size);
      VULKAN_CHECK (
          vkBindBufferMemory (d->device, d->staging_buf, d->staging_mem, 0));

      /* TODO track available host_staging_mem_bytes */
      VULKAN_CHECK (vkMapMemory (d->device, d->staging_mem, 0, d->staging_size,
                                 0, &d->staging_mapped));
    }

  dev->max_constant_buffer_size = d->constant_size;

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
  VULKAN_CHECK (vkFlushMappedMemoryRanges (d->device, 1, &mem_range));

  /* copy staging mem -> dev mem */
  VkCommandBuffer cb = d->tmp_command_buffer;
  VULKAN_CHECK (vkResetCommandBuffer (cb, 0));
  VULKAN_CHECK (vkBeginCommandBuffer (cb, &d->cmd_buf_begin_info));
  VkBufferCopy copy;
  copy.srcOffset = 0;
  copy.dstOffset = 0;
  copy.size = dest_size;

  vkCmdCopyBuffer (cb, d->staging_buf, dest_buf, 1, &copy);
  VULKAN_CHECK (vkEndCommandBuffer (cb));
  d->submit_info.pCommandBuffers = &cb;
  VULKAN_CHECK (vkQueueSubmit (d->compute_queue, 1, &d->submit_info, NULL));

  VkMemoryBarrier memory_barrier;
  memory_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  memory_barrier.pNext = NULL;
  memory_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memory_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier (cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                        &memory_barrier, 0, 0, 0, 0);
}

/********************************************************************/

void
pocl_vulkan_init_device_ops (struct pocl_device_ops *ops)
{
  ops->device_name = "VULKAN";

  ops->probe = pocl_vulkan_probe;
  ops->init = pocl_vulkan_init;
  /*  ops->uninit = pocl_vulkan_uninit; */
  /*  ops->reinit = pocl_vulkan_reinit; */

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
        VK_MAKE_VERSION (1, 2, 0),
        VK_API_VERSION_1_1 };

#define MAX_VULKAN_DEVICES 32

/* TODO replace with dynamic arrays */
static VkInstance pocl_vulkan_instance;
static unsigned pocl_vulkan_device_count;
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

  POCL_MSG_PRINT_VULKAN ("VALIDATION LAYER %s ::\n%s\n", layerPrefix, msg);

  return VK_FALSE;
}

unsigned int
pocl_vulkan_probe (struct pocl_device_ops *ops)
{
  VkResult res;
  int env_count = pocl_device_get_env_count (ops->device_name);

  pocl_vulkan_enable_validation = pocl_is_option_set ("POCL_VULKAN_VALIDATE");

  if (env_count < 0)
    return 0;

  size_t i;
  VkInstanceCreateInfo cinfo;
  cinfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  cinfo.pApplicationInfo = &pocl_vulkan_application_info;

#ifdef HAVE_CLSPV
  if (!pocl_exists (CLSPV))
    POCL_ABORT ("Can't find CLSPV compiler\n");
#endif

  /* extensions */
  uint32_t ext_prop_count = 128;
  VkExtensionProperties properties[128];
  VULKAN_CHECK (vkEnumerateInstanceExtensionProperties (NULL, &ext_prop_count,
                                                        properties));
  assert (ext_prop_count < 128);

  cinfo.enabledExtensionCount = 0;
  const char *ins_extensions[2];
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
  VULKAN_CHECK (vkEnumerateInstanceLayerProperties (&layer_count, layers));
  assert (layer_count < 128);

  cinfo.enabledLayerCount = 0;
  const char *ins_layers[3];
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

  res = vkCreateInstance (&cinfo, NULL, &pocl_vulkan_instance);
  if (res != VK_SUCCESS)
    {
      POCL_MSG_PRINT_VULKAN ("No Vulkan devices found.\n");
      return 0;
    }

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
  VULKAN_CHECK (vkEnumeratePhysicalDevices (pocl_vulkan_instance,
                                            &pocl_vulkan_device_count, 0));
  assert (pocl_vulkan_device_count < MAX_VULKAN_DEVICES);
  VULKAN_CHECK (vkEnumeratePhysicalDevices (
      pocl_vulkan_instance, &pocl_vulkan_device_count, pocl_vulkan_devices));

  POCL_MSG_PRINT_VULKAN ("%u Vulkan devices found.\n",
                         pocl_vulkan_device_count);
  return pocl_vulkan_device_count;
}

cl_int
pocl_vulkan_init (unsigned j, cl_device_id dev, const char *parameters)
{
  assert (j < pocl_vulkan_device_count);
  POCL_MSG_PRINT_VULKAN ("Initializing device %u\n", j);
  size_t i;

  SETUP_DEVICE_CL_VERSION (HOST_DEVICE_CL_VERSION_MAJOR,
                           HOST_DEVICE_CL_VERSION_MINOR)

  pocl_vulkan_device_data_t *d;

  d = (pocl_vulkan_device_data_t *)calloc (1,
                                           sizeof (pocl_vulkan_device_data_t));
  dev->data = d;
  d->dev = dev;

  VkPhysicalDevice pd = pocl_vulkan_devices[j];

  uint32_t comp_queue_fam, comp_queue_count;
  VULKAN_CHECK (pocl_vulkan_get_best_compute_queue (pd, &comp_queue_fam,
                                                    &comp_queue_count));
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

  const char *requested_exts[4];
  uint32_t requested_ext_count = 0;

  uint32_t dev_ext_count = 0;
  VkExtensionProperties dev_exts[256];
  VULKAN_CHECK (
      vkEnumerateDeviceExtensionProperties (pd, NULL, &dev_ext_count, NULL));
  VULKAN_CHECK (vkEnumerateDeviceExtensionProperties (pd, NULL, &dev_ext_count,
                                                      dev_exts));
  assert (dev_ext_count < 256);

#ifdef VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD
  int have_amd_shader_count = 0;
#endif
  int have_needed_extensions = 0;
  for (i = 0; i < dev_ext_count; ++i)
    {
#ifdef VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD
      if (strncmp ("VK_AMD_shader_core_properties", dev_exts[i].extensionName,
                   VK_MAX_EXTENSION_NAME_SIZE)
          == 0)
        {
          ++have_amd_shader_count;
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
    }

  if (have_needed_extensions < 2)
    {
      POCL_MSG_ERR ("pocl-vulkan requires a device to support: "
                    "VK_KHR_variable_pointers + "
                    "VK_KHR_storage_buffer_storage_class + "
                    "VK_KHR_shader_non_semantic_info;\n"
                    "disabling device %s\n", dev->short_name);
      dev->available = CL_FALSE;
      return CL_SUCCESS;
    }
  else
    dev->available = CL_TRUE;

    /* get device properties */
#ifdef VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD
  if (have_amd_shader_count)
    {
      VkPhysicalDeviceProperties2 general_props;
      VkPhysicalDeviceShaderCorePropertiesAMD shader_core_properties;

      shader_core_properties.pNext = NULL;
      shader_core_properties.sType
          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD;

      general_props.pNext = &shader_core_properties;
      general_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;

      vkGetPhysicalDeviceProperties2 (pocl_vulkan_devices[j], &general_props);

      memcpy (&d->dev_props, &general_props.properties,
              sizeof (VkPhysicalDeviceProperties));

      dev->max_compute_units
          = shader_core_properties.shaderEngineCount
            * shader_core_properties.shaderArraysPerEngineCount
            * shader_core_properties.computeUnitsPerShaderArray;
    }
#else
  {
    vkGetPhysicalDeviceProperties (pd, &d->dev_props);
    dev->max_compute_units = 1;
  }
#endif

  /* TODO get this from Vulkan API */
  dev->max_clock_frequency = 1000;

  /* clspv:
    If the short/ushort types are used in the OpenCL C:
    The shaderInt16 field of VkPhysicalDeviceFeatures must be set to true.
          shaderFloat64                           = 1
          shaderInt64                             = 1
          shaderInt16                             = 0
  */
  VkPhysicalDeviceFeatures dev_features = { 0 };
  if (dev_features.shaderFloat64)
    {
      dev->extensions = "cl_khr_fp64";
      dev->double_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO
                              | CL_FP_ROUND_TO_INF | CL_FP_FMA | CL_FP_INF_NAN
                              | CL_FP_DENORM;
    }
  else
    {
      dev->extensions = "";
      dev->double_fp_config = 0;
    }
  dev->has_64bit_long = (dev_features.shaderInt64 > 0);

  /*
   If images are used in the OpenCL C:
   The shaderStorageImageReadWithoutFormat field of VkPhysicalDeviceFeatures
   must be set to true. The shaderStorageImageWriteWithoutFormat field of
   VkPhysicalDeviceFeatures must be set to true.
  */

  /* TODO: Get images working */
  dev->image_support = CL_FALSE;

  VkDeviceCreateInfo dev_cinfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                   0,
                                   0,
                                   1,
                                   &queue_fam_cinfo,
                                   0, /* deprecated */
                                   0, /* deprecated */
                                   requested_ext_count,
                                   requested_exts,
                                   &dev_features };

  /* create logical device */
  VULKAN_CHECK (
      vkCreateDevice (pocl_vulkan_devices[j], &dev_cinfo, 0, &d->device));

  dev->profile = "FULL_PROFILE";
  dev->vendor_id = d->dev_props.vendorID;

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
      dev->available = CL_FALSE;
      return CL_SUCCESS;
    }

  dev->execution_capabilities = CL_EXEC_KERNEL;
  dev->address_bits = 64;
  /* TODO: (cl_uint)d->dev_props.limits.minStorageBufferOffsetAlignment * 8; */
  dev->mem_base_addr_align = MAX_EXTENDED_ALIGNMENT;

#ifdef HAVE_CLSPV
  dev->compiler_available = CL_TRUE;
  dev->linker_available = CL_TRUE;
  dev->consumes_il_directly = CL_TRUE;
#else
  dev->compiler_available = CL_FALSE;
  dev->linker_available = CL_FALSE;
  /* TODO enable the following once the build callbacks
   * are fixed to extract kernel metadata from SPIR-V
   * directly instead of using clspv-reflection
   */
  dev->consumes_il_directly = CL_FALSE;
#endif

  dev->preferred_vector_width_char = 1;
  dev->preferred_vector_width_short = 1;
  dev->preferred_vector_width_int = 1;
  dev->preferred_vector_width_long = 1;
  dev->preferred_vector_width_float = 1;
  dev->preferred_vector_width_double = 1;
  dev->preferred_vector_width_half = 0;
  dev->native_vector_width_char = 1;
  dev->native_vector_width_short = 1;
  dev->native_vector_width_int = 1;
  dev->native_vector_width_long = 1;
  dev->native_vector_width_float = 1;
  dev->native_vector_width_double = 1;
  dev->native_vector_width_half = 0;

  dev->single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO
                          | CL_FP_ROUND_TO_INF | CL_FP_FMA | CL_FP_INF_NAN
                          | CL_FP_DENORM;

  dev->device_side_printf = 0;
  dev->printf_buffer_size = 0;

  dev->available = CL_TRUE;
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

  pocl_vulkan_setup_memory_types (dev, d, pd);

  dev->global_mem_size = d->device_mem_size;
  dev->global_mem_cacheline_size = HOST_CPU_CACHELINE_SIZE;
  dev->global_mem_cache_size = 32768; /* TODO we should detect somehow.. */
  dev->global_mem_cache_type = CL_READ_WRITE_CACHE;

  /* TODO VkPhysicalDeviceVulkan11Properties . maxMemoryAllocationSize */
  dev->max_mem_alloc_size = max (dev->global_mem_size / 2, 128 * 1024 * 1024);

  VkCommandPoolCreateInfo pool_cinfo;
  pool_cinfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_cinfo.pNext = NULL;
  pool_cinfo.queueFamilyIndex = comp_queue_fam;
  pool_cinfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT
                     | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VULKAN_CHECK (
      vkCreateCommandPool (d->device, &pool_cinfo, NULL, &d->command_pool));

  VkCommandBuffer tmp[2];
  VkCommandBufferAllocateInfo alloc_cinfo;
  alloc_cinfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_cinfo.pNext = NULL;
  alloc_cinfo.commandPool = d->command_pool;
  alloc_cinfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_cinfo.commandBufferCount = 2;
  VULKAN_CHECK (vkAllocateCommandBuffers (d->device, &alloc_cinfo, tmp));
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

  VULKAN_CHECK (vkCreateDescriptorPool ( d->device,
                                         &descriptor_pool_create_info,
                                         NULL, &d->buf_descriptor_pool));

  POCL_INIT_COND (d->wakeup_cond);

  POCL_FAST_INIT (d->wq_lock_fast);

  d->work_queue = NULL;

  pthread_create (&d->driver_pthread_id, NULL, pocl_vulkan_driver_pthread,
                  dev);

  return CL_SUCCESS;
}

#if 0
/* TODO finish implementation */
cl_int
pocl_vulkan_uninit (unsigned j, cl_device_id device)
{
  pocl_vulkan_device_data_t *d = (pocl_vulkan_device_data_t*)device->data;

  vkDestroyCommandPool (d->device, d->command_pool, NULL);

  vkDestroyDescriptorPool (d->device, d->buf_descriptor_pool, NULL);

  /* destroy logical device */
  vkDestroyDevice (d->device, NULL);

  return CL_SUCCESS;

  /* TODO this must be called after all devices !! */
  vkDestroyInstance(pocl_vulkan_instance, NULL);
}

cl_ulong
pocl_vulkan_get_timer_value(void *data)
{
  return pocl_gettimemono_ns();
}
#endif


int
pocl_vulkan_build_source (cl_program program, cl_uint device_i,
                          cl_uint num_input_headers,
                          const cl_program *input_headers,
                          const char **header_include_names, int link_program)
{
#ifdef HAVE_CLSPV
  assert (program->devices[device_i]->compiler_available == CL_TRUE);
  assert (program->devices[device_i]->linker_available == CL_TRUE);
  assert (program->source);
  size_t source_len = strlen (program->source);

  POCL_MSG_PRINT_VULKAN ("building with CLSPV from sources for device %d\n",
                         device_i);

  if (num_input_headers > 0)
    POCL_ABORT_UNIMPLEMENTED ("Vulkan compilation with headers\n");

  if (!link_program)
    POCL_ABORT_UNIMPLEMENTED ("Vulkan compilation without linking\n");

  char program_bc_path[POCL_FILENAME_LENGTH];
  pocl_cache_create_program_cachedir (program, device_i, program->source,
                                      source_len, program_bc_path);
  size_t len = strlen (program_bc_path);
  assert (len > 3);
  len -= 2;
  program_bc_path[len] = 0;
  char program_cl_path[POCL_FILENAME_LENGTH];
  strncpy (program_cl_path, program_bc_path, POCL_FILENAME_LENGTH);
  strcat (program_cl_path, "cl");
  char program_spv_path[POCL_FILENAME_LENGTH];
  strncpy (program_spv_path, program_bc_path, POCL_FILENAME_LENGTH);
  strcat (program_spv_path, "spv");
  char program_map_path[POCL_FILENAME_LENGTH];
  strncpy (program_map_path, program_bc_path, POCL_FILENAME_LENGTH);
  strcat (program_map_path, "map");

  pocl_write_file (program_cl_path, program->source, source_len, 0, 1);

  char *COMPILATION[1024]
      = { CLSPV, "-x=cl", "--spv-version=1.0", "-cl-kernel-arg-info",
          "--keep-unused-arguments", "--uniform-workgroup-size",
          /* "--pod-pushconstant",*/  /* TODO push constants should be faster */
          "--pod-ubo", "--cluster-pod-kernel-args", "-o", program_spv_path,
          program_cl_path, NULL };

  if (program->compiler_options)
    {
      size_t i = 0;
      for (i = 0; i < 1024; ++i)
        if (COMPILATION[i] == NULL)
          break;
      /* TODO mishandles quoted strings with spaces */
      char *token = NULL;
      char delim[] = { ' ', 0 };
      token = strtok (program->compiler_options, delim);
      while (i < 1024 && token != NULL)
        {
          COMPILATION[i] = strdup (token);
          token = strtok (NULL, delim);
          ++i;
        }
      COMPILATION[i] = NULL;
    }

  pocl_run_command (COMPILATION);

  POCL_RETURN_ERROR_ON (!pocl_exists (program_spv_path),
                        CL_BUILD_PROGRAM_FAILURE, "clspv compilation error");

  char *REFLECTION[] = { CLSPV_REFLECTION, program_spv_path, "-o",
                         program_map_path, NULL };

  pocl_run_command (REFLECTION);

  POCL_RETURN_ERROR_ON (!pocl_exists (program_map_path),
                        CL_BUILD_PROGRAM_FAILURE,
                        "clspv-reflection compilation error");

  uint64_t bin_size;
  char *binary;
  pocl_read_file (program_spv_path, &binary, &bin_size);
  program->binaries[device_i] = binary;
  program->binary_sizes[device_i] = bin_size;
  program->data[device_i] = strdup (program_map_path);
  return CL_SUCCESS;
#else
  return CL_BUILD_PROGRAM_FAILURE;
#endif
}

int
pocl_vulkan_supports_binary (cl_device_id device, const size_t length,
                             const char *binary)
{
/* TODO remove ifdef once the build callbacks
* are fixed to extract kernel metadata from SPIR-V directly
* instead of using clspv-reflection
*/
#ifdef HAVE_CLSPV
  return bitcode_is_spirv_execmodel_shader (binary, length);
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

      char program_bc_path[POCL_FILENAME_LENGTH];
      pocl_cache_program_bc_path (program_bc_path, program, device_i);

      size_t len = strlen (program_bc_path);
      assert (len > 3);
      len -= 2;
      program_bc_path[len] = 0;
      char program_spv_path[POCL_FILENAME_LENGTH];
      strncpy (program_spv_path, program_bc_path, POCL_FILENAME_LENGTH);
      strcat (program_spv_path, "spv");
      char program_map_path[POCL_FILENAME_LENGTH];
      strncpy (program_map_path, program_bc_path, POCL_FILENAME_LENGTH);
      strcat (program_map_path, "map");

      uint64_t bin_size;
      char *binary;
      pocl_read_file (program_spv_path, &binary, &bin_size);
      program->binaries[device_i] = binary;
      program->binary_sizes[device_i] = bin_size;
      program->data[device_i] = strdup (program_map_path);
      return CL_SUCCESS;
    }

#ifdef HAVE_CLSPV
  /* we have program->binaries[] which is SPIR-V */
  assert (program->binaries[device_i]);
  int is_spirv = bitcode_is_spirv_execmodel_shader(program->binaries[device_i], program->binary_sizes[device_i]);
  assert (is_spirv != 0);

  char program_bc_path[POCL_FILENAME_LENGTH];
  pocl_cache_create_program_cachedir (program, device_i,
                                      program->binaries[device_i],
                                      program->binary_sizes[device_i],
                                      program_bc_path);
  size_t len = strlen (program_bc_path);
  assert (len > 3);
  len -= 2;
  program_bc_path[len] = 0;
  char program_cl_path[POCL_FILENAME_LENGTH];
  strncpy (program_cl_path, program_bc_path, POCL_FILENAME_LENGTH);
  strcat (program_cl_path, "cl");
  char program_spv_path[POCL_FILENAME_LENGTH];
  strncpy (program_spv_path, program_bc_path, POCL_FILENAME_LENGTH);
  strcat (program_spv_path, "spv");
  char program_map_path[POCL_FILENAME_LENGTH];
  strncpy (program_map_path, program_bc_path, POCL_FILENAME_LENGTH);
  strcat (program_map_path, "map");

  if (!pocl_exists(program_spv_path))
    {
    pocl_write_file (program_spv_path, program->binaries[device_i], program->binary_sizes[device_i], 0, 1);
    POCL_RETURN_ERROR_ON (!pocl_exists (program_spv_path),
                          CL_BUILD_PROGRAM_FAILURE, "clspv compilation error");
    }

  if (!pocl_exists(program_map_path))
    {
      char *REFLECTION[] = { CLSPV "-reflection", program_spv_path, "-o",
                             program_map_path, NULL };

      pocl_run_command (REFLECTION);

      POCL_RETURN_ERROR_ON (!pocl_exists (program_map_path),
                            CL_BUILD_PROGRAM_FAILURE,
                            "clspv-reflection compilation error");
    }
  program->data[device_i] = strdup (program_map_path);

  return CL_SUCCESS;


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
  POCL_MEM_FREE (program->data[program_device_i]);
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
  /* TODO */
  p->build_hash = NULL;
  p->builtin_kernel = 0;
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
  token = NULL;

  char *arg_name = NULL;
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
          assert (errno == 0);
        }
      if (strcmp (tokens[j], "descriptorSet") == 0)
        {
          errno = 0;
          dSet = strtol (tokens[j + 1], NULL, 10);
          assert (errno == 0);
        }
      if (strcmp (tokens[j], "binding") == 0)
        {
          errno = 0;
          binding = strtol (tokens[j + 1], NULL, 10);
          assert (errno == 0);
        }
      if (strcmp (tokens[j], "offset") == 0)
        {
          errno = 0;
          offset = strtol (tokens[j + 1], NULL, 10);
          assert (errno == 0);
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
          else
            POCL_ABORT_UNIMPLEMENTED ("unknown arg Type\n");
        }
      if (strcmp (tokens[j], "argSize") == 0)
        {
          errno = 0;
          size = strtol (tokens[j + 1], NULL, 10);
          assert (errno == 0);
        }
      if (strcmp (tokens[j], "arrayElemSize") == 0)
        {
          errno = 0;
          elemSize = strtol (tokens[j + 1], NULL, 10);
          assert (errno == 0);
        }
      if (strcmp (tokens[j], "arrayNumElemSpecId") == 0)
        {
          errno = 0;
          specID = strtol (tokens[j + 1], NULL, 10);
          assert (errno == 0);
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
      assert (size > 0);
      p->arg_info[ord].type_size = size;
      pp->pods[pp->num_pods].binding = binding;
      pp->pods[pp->num_pods].dset = dSet;
      pp->pods[pp->num_pods].offset = offset;
      pp->pods[pp->num_pods].ord = ord;
      ++pp->num_pods;

      pp->num_pod_bytes += size;
    }

  /* TODO constants !!! */

  ++p->num_args;
  assert (p->num_args < MAX_ARGS);
}

int
pocl_vulkan_setup_metadata (cl_device_id device, cl_program program,
                            unsigned program_device_i)
{
  assert (program->data[program_device_i]);

  /* read map file from clspv-reflection */
  char *content;
  uint64_t content_size;
  int r = pocl_read_file (program->data[program_device_i], &content,
                          &content_size);
  if (r != 0)
    return 0;

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

  pocl_kernel_metadata_t kernel_meta_array[1024];
  pocl_kernel_metadata_t *p = NULL;
  unsigned num_kernels = 0;

  for (size_t i = 0; i < num_lines; ++i)
    {
      if (strncmp (lines[i], "kernel_decl,", 12) == 0)
        {
          p = &kernel_meta_array[num_kernels];
          parse_new_kernel (p, lines[i]);
          p->data = (void **)calloc (program->num_devices, sizeof (void *));
          p->data[program_device_i]
              = calloc (1, sizeof (pocl_vulkan_kernel_data_t));
          ++num_kernels;
        }
      if (strncmp (lines[i], "kernel,", 7) == 0)
        {
          assert (p != NULL);
          parse_arg_line (p, p->data[program_device_i], lines[i]);
        }
    }

  for (size_t i = 0; i < num_lines; ++i)
    free (lines[i]);

  program->num_kernels = num_kernels;
  program->kernel_meta
      = calloc (program->num_kernels, sizeof (pocl_kernel_metadata_t));
  memcpy (program->kernel_meta, kernel_meta_array,
          sizeof (pocl_kernel_metadata_t) * num_kernels);
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
  if (pocl_command_is_ready (node->event))
    {
      pocl_update_event_submitted (node->event);
      vulkan_push_command (cq->device, node);
    }
  POCL_UNLOCK_OBJ (node->event);
  return;
}

int
pocl_vulkan_init_queue (cl_device_id dev, cl_command_queue queue)
{
  queue->data
      = pocl_aligned_malloc (HOST_CPU_CACHELINE_SIZE, sizeof (pthread_cond_t));
  pthread_cond_t *cond = (pthread_cond_t *)queue->data;
  int r = pthread_cond_init (cond, NULL);
  assert (r == 0);
  return CL_SUCCESS;
}

int
pocl_vulkan_free_queue (cl_device_id dev, cl_command_queue queue)
{
  pthread_cond_t *cond = (pthread_cond_t *)queue->data;
  int r = pthread_cond_destroy (cond);
  assert (r == 0);
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
  int r = pthread_cond_broadcast (cq_cond);
  assert (r == 0);
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
          int r = pthread_cond_wait (cq_cond, &cq->pocl_lock);
          assert (r == 0);
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

  if (pocl_command_is_ready (node->event))
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

#define POCL_VK_FENCE_TIMEOUT (60ULL * 1000ULL * 1000ULL * 1000ULL)

static void submit_CB (pocl_vulkan_device_data_t *d, VkCommandBuffer *cmdbuf_p)
{
  VkFence fence;

  VkFenceCreateInfo fCreateInfo = {
    VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0 };

  VULKAN_CHECK (vkCreateFence (d->device, &fCreateInfo, NULL, &fence));

  d->submit_info.pCommandBuffers = cmdbuf_p;
  VULKAN_CHECK (vkQueueSubmit (d->compute_queue, 1, &d->submit_info, fence));

  VULKAN_CHECK (vkWaitForFences (d->device, 1, &fence, VK_TRUE, POCL_VK_FENCE_TIMEOUT));
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
      VULKAN_CHECK (vkResetCommandBuffer (cb, 0));
      VULKAN_CHECK (vkBeginCommandBuffer (cb, &d->cmd_buf_begin_info));
      VkBufferCopy copy;
      copy.srcOffset = offset2;
      copy.dstOffset = offset2;
      copy.size = size2;

      /* POCL_MSG_ERR ("DEV2HOST : %zu / %zu \n", offset, size); */
      vkCmdCopyBuffer (cb, memdata->device_buf, memdata->staging_buf, 1,
                       &copy);
      VULKAN_CHECK (vkEndCommandBuffer (cb));
      submit_CB (d, &cb);

      /* copy staging mem -> host_ptr */
      VkMappedMemoryRange mem_range
          = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL,
              memdata->staging_mem, offset2, size2 };
      /* TODO only if non-coherent */
      VULKAN_CHECK (vkInvalidateMappedMemoryRanges (d->device, 1, &mem_range));
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
      VULKAN_CHECK (vkFlushMappedMemoryRanges (d->device, 1, &mem_range));

      /* copy staging mem -> dev mem */
      VkCommandBuffer cb = d->command_buffer;
      VULKAN_CHECK (vkResetCommandBuffer (cb, 0));
      VULKAN_CHECK (vkBeginCommandBuffer (cb, &d->cmd_buf_begin_info));
      VkBufferCopy copy;
      copy.srcOffset = offset2;
      copy.dstOffset = offset2;
      copy.size = size2;

      vkCmdCopyBuffer (cb, memdata->staging_buf, memdata->device_buf, 1,
                       &copy);
      VULKAN_CHECK (vkEndCommandBuffer (cb));
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
  VULKAN_CHECK (vkResetCommandBuffer (cb, 0));
  VULKAN_CHECK (vkBeginCommandBuffer (cb, &d->cmd_buf_begin_info));
  VkBufferCopy copy;
  copy.srcOffset = src_offset;
  copy.dstOffset = dst_offset;
  copy.size = size;
  vkCmdCopyBuffer (cb, src->device_buf, dst->device_buf, 1, &copy);
  VULKAN_CHECK (vkEndCommandBuffer (cb));
  submit_CB (d, &cb);
}

/* TODO implement these callbacks */
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
  POCL_ABORT_UNIMPLEMENTED ("pocl_vulkan_copy_rect is not implemented\n");
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
  POCL_ABORT_UNIMPLEMENTED ("pocl_vulkan_read_rect is not implemented\n");
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
  POCL_ABORT_UNIMPLEMENTED ("pocl_vulkan_write_rect is not implemented\n");
}


void pocl_vulkan_memfill(void *data,
                        pocl_mem_identifier * dst_mem_id,
                        cl_mem dst_buf,
                        size_t size,
                        size_t offset,
                        const void *__restrict__  pattern,
                        size_t pattern_size)
{
  POCL_ABORT_UNIMPLEMENTED ("pocl_vulkan_memfill is not implemented\n");
}

cl_int
pocl_vulkan_map_mem (void *data, pocl_mem_identifier *src_mem_id,
                     cl_mem src_buf, mem_mapping_t *map)
{
  pocl_vulkan_mem_data_t *memdata
      = (pocl_vulkan_mem_data_t *)src_mem_id->mem_ptr;
  pocl_vulkan_device_data_t *d = (pocl_vulkan_device_data_t *)data;

  POCL_MSG_PRINT_VULKAN ("MAP MEM: %p FLAGS %zu\n", memdata, map->map_flags);

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
    pocl_vulkan_device_data_t *d, cl_device_id dev, _cl_command_node *cmd,
    VkShaderModule *compute_shader, VkDescriptorSet *ds,
    VkDescriptorSet *const_ds, VkDescriptorSetLayout *dsl,
    VkDescriptorSetLayout *const_dsl, VkSpecializationInfo *spec_info,
    uint32_t *spec_data, VkSpecializationMapEntry *entries,
    VkDescriptorSetLayoutBinding *bindings,
    VkDescriptorBufferInfo *descriptor_buffer_info,
    VkDescriptorSetLayoutBinding *const_binding,
    VkDescriptorBufferInfo *const_descriptor_buffer_info)
{
  _cl_command_run *co = &cmd->command.run;
  cl_kernel kernel = co->kernel;
  struct pocl_argument *pa = co->arguments;

  pocl_vulkan_kernel_data_t *pp = kernel->meta->data[cmd->program_device_i];
  assert (pp != NULL);

  if (pp->shader == NULL)
    {
      cl_program program = kernel->program;
      unsigned dev_i = cmd->program_device_i;
      assert (program->binaries[dev_i] != NULL);
      assert (program->binary_sizes[dev_i] > 0);

      VkShaderModuleCreateInfo shader_info;
      shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
      shader_info.pNext = NULL;
      shader_info.pCode = program->binaries[dev_i];
      shader_info.codeSize = program->binary_sizes[dev_i];
      shader_info.flags = 0;

      VULKAN_CHECK (
          vkCreateShaderModule (d->device, &shader_info, NULL, &pp->shader));
    }
  *compute_shader = pp->shader;

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

  uint32_t spec_entries = 3;
  uint32_t spec_offset = 12;

  uint64_t pod_offset = 0;
  uint64_t pod_entries = 0;

  uint32_t locals = 0;
  uint32_t pods = 0;
  uint32_t bufs = 0;

  unsigned current = 0;

  /****************************************************************************************/

  /* preallocate buffers if needed */
  VkMemoryRequirements mem_req;

  if (pp->kernarg_buf == NULL && pp->num_pods > 0)
    {
      assert (pp->num_pod_bytes > 0);

      size_t kernarg_aligned_size = pocl_align_value (pp->num_pod_bytes, PAGE_SIZE);

      VkBufferCreateInfo buffer_info = {
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, kernarg_aligned_size,

        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            | VK_BUFFER_USAGE_TRANSFER_DST_BIT,

        VK_SHARING_MODE_EXCLUSIVE, 1, &d->compute_queue_fam_index
      };

      chunk_info_t *chunk = alloc_buffer_from_region (&d->kernarg_region,
                                                      kernarg_aligned_size);
      assert (chunk);
      pp->kernarg_chunk = chunk;

      VULKAN_CHECK (
          vkCreateBuffer (d->device, &buffer_info, NULL, &pp->kernarg_buf));
      pp->kernarg_buf_offset = chunk->start_address;
      pp->kernarg_buf_size = kernarg_aligned_size;

      vkGetBufferMemoryRequirements (d->device, pp->kernarg_buf, &mem_req);
      assert (kernarg_aligned_size == mem_req.size);
      VULKAN_CHECK (vkBindBufferMemory (
          d->device, pp->kernarg_buf, d->kernarg_mem, pp->kernarg_buf_offset));
    }

  if (pp->constant_buf == NULL && pp->num_constant_bytes > 0)
    {
      size_t constants_aligned_size = pocl_align_value (pp->num_constant_bytes, PAGE_SIZE);

      VkBufferCreateInfo buffer_info
          = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
              NULL,
              0,
              constants_aligned_size,
              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                  | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                  | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
              VK_SHARING_MODE_EXCLUSIVE,
              1,
              &d->compute_queue_fam_index };

      chunk_info_t *chunk = alloc_buffer_from_region (&d->constant_region,
                                                      constants_aligned_size);
      assert (chunk);
      pp->constant_chunk = chunk;

      VULKAN_CHECK (
          vkCreateBuffer (d->device, &buffer_info, NULL, &pp->constant_buf));
      pp->constant_buf_offset = chunk->start_address;
      pp->constant_buf_size = constants_aligned_size;

      vkGetBufferMemoryRequirements (d->device, pp->constant_buf, &mem_req);
      assert (constants_aligned_size == mem_req.size);
      VULKAN_CHECK (vkBindBufferMemory (d->device, pp->constant_buf,
                                        d->constant_mem,
                                        pp->constant_buf_offset));

      /* copy data to contant mem */
      if (d->kernarg_is_mappable)
        {
          void *constant_pod_ptr;
          VULKAN_CHECK (
              vkMapMemory (d->device, d->constant_mem, pp->constant_buf_offset,
                           pp->constant_buf_size, 0, &constant_pod_ptr));
          memcpy (constant_pod_ptr, pp->constant_data, pp->num_constant_bytes);
          vkUnmapMemory (d->device, d->constant_mem);
        }
      else
        {
          memcpy (d->staging_mapped, pp->constant_data,
                  pp->num_constant_bytes);
          pocl_vulkan_enqueue_staging_buffer_copy (d, pp->constant_buf,
                                                   pp->constant_buf_size);
          VULKAN_CHECK (vkQueueWaitIdle (d->compute_queue));
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

  if (pp->num_pods > 0)
    {
      if (d->kernarg_is_mappable)
        {
          VULKAN_CHECK (vkMapMemory (
              d->device, d->kernarg_mem, pp->kernarg_buf_offset,
              pp->kernarg_buf_size, 0, (void **)(&kernarg_pod_ptr)));
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

          assert (pp->locals[locals].ord == i);
          uint32_t elems = pa[i].size / pp->locals[locals].elem_size;
          POCL_MSG_PRINT_VULKAN ("setting LOCAL argument %i TO"
                                 "SIZE %zu ELEMS: %u\n; ",
                                  i, pa[i].size, elems);
          assert (pp->locals[locals].spec_id == spec_entries);
          entries[spec_entries].constantID = spec_entries;
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
          bindings[current].binding = current;
          bindings[current].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          bindings[current].descriptorCount = 1;
          bindings[current].pImmutableSamplers = 0;
          bindings[current].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
          assert (current < 128);

          assert (pp->bufs[bufs].dset == 0);
          assert (pp->bufs[bufs].binding == current);
          assert (pp->bufs[bufs].ord == i);
          ++current;
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

          assert (pp->pods[pods].dset == 0);
          assert (pp->pods[pods].ord == i);
          assert (pp->pods[pods].offset == pod_offset);

          memcpy (kernarg_pod_ptr + pod_offset, pa[i].value, pa[i].size);
          pod_offset += pa[i].size;
          ++pod_entries;
          ++pods;
        }
    }

  if (pp->num_pods > 0)
    {
      if (d->kernarg_is_mappable)
        {
          vkUnmapMemory (d->device, d->kernarg_mem);
        }
      else
        {
          pocl_vulkan_enqueue_staging_buffer_copy (d, pp->kernarg_buf,
                                                   pp->kernarg_buf_size);
        }
    }

  /* PODs: setup descriptor & bindings for PODs; last binding in DS 0 */
  if (pp->num_pods > 0)
    {
      /* add the kernarg memory */
      descriptor_buffer_info[current].buffer
          = pp->kernarg_buf; /* the POD buffer */
      descriptor_buffer_info[current].offset = 0;
      descriptor_buffer_info[current].range = pp->kernarg_buf_size;

      assert (pp->pods[0].binding == current);
      bindings[current].binding = current;
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
  spec_info->pMapEntries = entries;
  spec_info->dataSize = spec_offset;
  spec_info->pData = spec_data;

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
  VULKAN_CHECK (
      vkCreateDescriptorSetLayout (d->device, &dslCreateInfo, NULL, dsl));

  VkDescriptorSetAllocateInfo descriptorSetallocate_info
      = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, 0,
          d->buf_descriptor_pool, 1, dsl };
  VULKAN_CHECK (
      vkAllocateDescriptorSets (d->device, &descriptorSetallocate_info, ds));

  /* cl_mem arguments */
  VkWriteDescriptorSet writeDescriptorSet
      = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          0,
          *ds, /* dstSet */
          0,   /* dstBinding */
          0,   /* dstArrayElement */
          (pp->num_pods > 0) ? current-1 : current, /* descriptorCount */
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, /* descriptorType */
          0,
          descriptor_buffer_info,
          0 };
  vkUpdateDescriptorSets (d->device, 1, &writeDescriptorSet, 0, 0);

  /* setup descriptor & bindings POD arguments in UBO */
  if (pp->num_pods > 0)
    {
      VkWriteDescriptorSet writeDescriptorSet2
          = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
              0,
              *ds, /* dstSet */
              current-1,   /* dstBinding */
              0,   /* dstArrayElement */
              1, /* descriptorCount */
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, /* descriptorType */
              0,
              &descriptor_buffer_info[current-1],
              0 };
      vkUpdateDescriptorSets (d->device, 1, &writeDescriptorSet2, 0, 0);
    }

  /* setup descriptor & bindings for constants   */
  if (pp->num_constant_bytes > 0)
    {
      /* add the constant memory */
      const_descriptor_buffer_info->buffer
          = pp->constant_buf; /* the constant buffer */
      const_descriptor_buffer_info->offset = 0;
      const_descriptor_buffer_info->range = VK_WHOLE_SIZE;

      const_binding->binding = current;
      const_binding->descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      const_binding->descriptorCount = 1;
      const_binding->pImmutableSamplers = 0;
      const_binding->stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

      VkDescriptorSetLayoutCreateInfo dslCreateInfo
          = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, 0, 0, 1,
              const_binding };
      VULKAN_CHECK (vkCreateDescriptorSetLayout (d->device, &dslCreateInfo,
                                                 NULL, const_dsl));

      VkDescriptorSetAllocateInfo descriptorSetallocate_info
          = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, 0,
              d->buf_descriptor_pool, 1, const_dsl };
      VULKAN_CHECK (vkAllocateDescriptorSets (
          d->device, &descriptorSetallocate_info, const_ds));

      VkWriteDescriptorSet writeDescriptorSet
          = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
              0,
              *const_ds,
              0,
              0,
              1,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              0,
              const_descriptor_buffer_info,
              0 };
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
  size_t wg_x = pc->num_groups[0];
  size_t wg_y = pc->num_groups[1];
  size_t wg_z = pc->num_groups[2];

  /* TODO we need working global offsets
   * before we can handle arbitrary group sizes */
  assert (wg_x < d->max_wg_count[0]);
  assert (wg_y < d->max_wg_count[1]);
  assert (wg_z < d->max_wg_count[2]);

  POCL_MSG_PRINT_VULKAN ("WG X %zu Y %zu Z %zu \n", wg_x, wg_y, wg_z);

  VkDescriptorSet descriptor_sets[2] = { NULL, NULL };
  VkDescriptorSetLayout descriptor_set_layouts[2];
  VkSpecializationInfo specInfo;
  uint32_t spec_data[128];
  VkSpecializationMapEntry entries[128];
  VkDescriptorSetLayoutBinding bindings[128];
  VkDescriptorBufferInfo descriptor_buffer_info[128];
  VkDescriptorSetLayoutBinding const_binding;
  VkDescriptorBufferInfo const_descriptor_buffer_info;
  VkPipelineLayout pipeline_layout;
  VkPipeline pipeline;
  VkShaderModule compute_shader = NULL;

  pocl_vulkan_setup_kernel_arguments (
      d, cmd->device, cmd, &compute_shader, descriptor_sets,
      descriptor_sets + 1, descriptor_set_layouts, descriptor_set_layouts + 1,
      &specInfo, spec_data, entries, bindings, descriptor_buffer_info,
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
  pipeline_layout_create_info.pPushConstantRanges = 0;
  pipeline_layout_create_info.pushConstantRangeCount = 0;
  if (descriptor_sets[1])
    pipeline_layout_create_info.setLayoutCount = 2;
  else
    pipeline_layout_create_info.setLayoutCount = 1;
  pipeline_layout_create_info.pSetLayouts = descriptor_set_layouts;
  VULKAN_CHECK (vkCreatePipelineLayout (
      d->device, &pipeline_layout_create_info, NULL, &pipeline_layout));

  VkComputePipelineCreateInfo pipeline_create_info;
  pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_create_info.pNext = NULL;
  pipeline_create_info.flags = 0;
  pipeline_create_info.stage = shader_stage_info;
  pipeline_create_info.layout = pipeline_layout;
  pipeline_create_info.basePipelineIndex = 0;
  pipeline_create_info.basePipelineHandle = 0;

  VULKAN_CHECK (vkCreateComputePipelines (
      d->device, VK_NULL_HANDLE, 1, &pipeline_create_info, NULL, &pipeline));

  VkCommandBuffer cb = d->command_buffer;
  VULKAN_CHECK (vkResetCommandBuffer (cb, 0));

  VkCommandBufferBeginInfo begin_info;
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.pNext = NULL;
  begin_info.flags
      = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      /* the buffer is only submitted and used once in this application. */

  begin_info.pInheritanceInfo = NULL;
  VULKAN_CHECK (
      vkBeginCommandBuffer (cb, &begin_info)); /* start recording commands. */

  vkCmdBindPipeline (cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets (cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout,
                           0, ((descriptor_sets[1] ? 2 : 1)), descriptor_sets,
                           0, NULL);

  vkCmdDispatch (cb, (uint32_t)wg_x, (uint32_t)wg_y, (uint32_t)wg_z);
  VULKAN_CHECK (vkEndCommandBuffer (cb)); /* end recording commands. */

  submit_CB (d, &cb);


  if (descriptor_sets[0]) {
    vkDestroyDescriptorSetLayout (d->device, descriptor_set_layouts[0], NULL);
    VULKAN_CHECK (vkFreeDescriptorSets (d->device, d->buf_descriptor_pool,
                                        1, descriptor_sets));
  }
  if (descriptor_sets[1]) {
    vkDestroyDescriptorSetLayout (d->device, descriptor_set_layouts[1], NULL);
    VULKAN_CHECK (vkFreeDescriptorSets (d->device, d->buf_descriptor_pool,
                                        1, descriptor_sets+1));
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

      assert (pocl_command_is_ready (cmd->event));
      assert (cmd->event->status == CL_SUBMITTED);

      pocl_exec_command (cmd);

      POCL_FAST_LOCK (d->wq_lock_fast);
    }

  if ((cmd == NULL) && (do_exit == 0))
    {
      pthread_cond_wait (&d->wakeup_cond, &d->wq_lock_fast);
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

  /* TODO free device data */

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

  VULKAN_CHECK (vkCreateBuffer (d->device, &buffer_info, NULL, &buffer));

  if (p->extra == 0)
    {
      vkGetBufferMemoryRequirements (d->device, buffer, mem_req);
      assert (mem_req->size > 0);
      p->extra = mem_req->size;
    }

  vkDestroyBuffer (d->device, buffer, NULL);

  return p->extra;
}

int
pocl_vulkan_alloc_mem_obj (cl_device_id device, cl_mem mem, void *host_ptr)
{
  pocl_vulkan_device_data_t *d = (pocl_vulkan_device_data_t *)device->data;
  pocl_mem_identifier *p = &mem->device_ptrs[device->global_mem_id];
  VkBuffer b;

  assert (p->mem_ptr == NULL);
  int ret = CL_MEM_OBJECT_ALLOCATION_FAILURE;

  /* TODO driver doesn't preallocate YET, but we should, when requested */
  if ((mem->flags & CL_MEM_ALLOC_HOST_PTR) && (mem->mem_host_ptr == NULL))
    goto ERROR;

  VkMemoryRequirements memReq;
  size_t actual_mem_size = pocl_vulkan_actual_memobj_size (d, mem, p, &memReq);
  assert (actual_mem_size > 0);

  /* POCL_MSG_WARN ("actual BUF size: %zu \n", actual_mem_size); */
  /* actual size already set up. */
  pocl_vulkan_mem_data_t *memdata
      = (pocl_vulkan_mem_data_t *)calloc (1, sizeof (pocl_vulkan_mem_data_t));
  VkDeviceMemory m;
  /* TODO host_ptr argument / CL_MEM_USE_HOST_PTR */
  void *vk_host_ptr = NULL;

  /* DEVICE MEM */
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

  VkMemoryAllocateInfo allocate_info
      = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, actual_mem_size,
          d->device_mem_type };

  VULKAN_CHECK (vkCreateBuffer (d->device, &buffer_info, NULL, &b));
  memdata->device_buf = b;
  VULKAN_CHECK (vkAllocateMemory (d->device, &allocate_info, NULL, &m));
  memdata->device_mem = m;
  vkGetBufferMemoryRequirements (d->device, memdata->device_buf, &memReq);
  assert (actual_mem_size == memReq.size);
  VULKAN_CHECK (vkBindBufferMemory (d->device, memdata->device_buf,
                                    memdata->device_mem, 0));

  /* STAGING MEM */
  if (d->needs_staging_mem)
    {
      if (mem->flags & (CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY))
        allocate_info.memoryTypeIndex = d->host_staging_write_type;
      else
        allocate_info.memoryTypeIndex = d->host_staging_read_type;

      VULKAN_CHECK (vkCreateBuffer (d->device, &buffer_info, NULL, &b));
      memdata->staging_buf = b;
      VULKAN_CHECK (vkAllocateMemory (d->device, &allocate_info, NULL, &m));
      memdata->staging_mem = m;
      vkGetBufferMemoryRequirements (d->device, memdata->staging_buf, &memReq);
      assert (actual_mem_size == memReq.size);
      VULKAN_CHECK (vkBindBufferMemory (d->device, memdata->staging_buf,
                                        memdata->staging_mem, 0));

      VULKAN_CHECK (vkMapMemory (d->device, memdata->staging_mem, 0,
                                 actual_mem_size, 0, &vk_host_ptr));
    }
  else
    {
      memdata->staging_buf = 0;
      memdata->staging_mem = 0;

      VULKAN_CHECK (vkMapMemory (d->device, memdata->device_mem, 0,
                                 actual_mem_size, 0, &vk_host_ptr));
    }

  p->mem_ptr = memdata;
  p->version = 0;
  p->extra_ptr = vk_host_ptr;
  p->extra = actual_mem_size;
  POCL_MSG_PRINT_MEMORY ("VULKAN DEVICE ALLOC PTR %p ALLOC %zu | "
                         "VULKAN DEV BUF %p | STA BUF %p | VK_HOST_PTR %p \n",
                         p->mem_ptr, p->extra, (void *)memdata->device_buf,
                         (void *)memdata->staging_buf, vk_host_ptr);

  ret = CL_SUCCESS;
ERROR:
  return ret;
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
      vkUnmapMemory (d->device, memdata->staging_mem);

      vkDestroyBuffer (d->device, memdata->staging_buf, NULL);
      vkFreeMemory (d->device, memdata->staging_mem, NULL);

      vkDestroyBuffer (d->device, memdata->device_buf, NULL);
      vkFreeMemory (d->device, memdata->device_mem, NULL);
    }
  else
    {
      vkUnmapMemory (d->device, memdata->device_mem);

      vkDestroyBuffer (d->device, memdata->device_buf, NULL);
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
