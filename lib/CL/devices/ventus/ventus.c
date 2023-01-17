/* ventus.c - a minimalistic single core pocl device driver layer implementation

   Copyright (c) 2011-2013 Universidad Rey Juan Carlos and
                 2011-2021 Pekka Jääskeläinen

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

#include "ventus.h"
#include "common.h"
#include "config.h"
#include "config2.h"
#include "cpuinfo.h"
#include "devices.h"
#include "pocl_local_size.h"
#include "pocl_util.h"
#include "topology/pocl_topology.h"
#include "utlist.h"

#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <utlist.h>

#include "pocl_cache.h"
#include "pocl_file_util.h"
#include "pocl_mem_management.h"
#include "pocl_timing.h"
#include "pocl_workgroup_func.h"

#include "common_driver.h"

#ifdef ENABLE_LLVM
#include "pocl_llvm.h"
#endif

// from driver/include/ventus.h
#if !defined(ENABLE_LLVM)
#include <ventus.h>
#endif

struct ventus_device_data_t {

#if !defined(ENABLE_LLVM)
  ventus_device_h ventus_device;
  size_t ventus_print_buf_d;
  ventus_buffer_h ventus_print_buf_h;
  uint32_t printf_buffer;
  uint32_t printf_buffer_position;
#endif

  /* List of commands ready to be executed */
  _cl_command_node *ready_list;
  /* List of commands not yet ready to be executed */
  _cl_command_node *command_list;
  /* Lock for command list related operations */
  pocl_lock_t cq_lock;

  /* Currently loaded kernel. */
  cl_kernel current_kernel;

  /* printf buffer */
  void *printf_buffer;
};

struct ventus_buffer_data_t {
#if !defined(ENABLE_LLVM)
  ventus_device_h ventus_device;
  ventus_buffer_h staging_buf;
#endif
  size_t dev_mem_addr;
};

struct kernel_context_t {
  uint32_t num_groups[3];
  uint32_t global_offset[3];
  uint32_t local_size[3];
  uint32_t printf_buffer;
  uint32_t printf_buffer_position;
  uint32_t printf_buffer_capacity;
  uint32_t work_dim;
};

static size_t ALIGNED_CTX_SIZE = 4 * ((sizeof(struct kernel_context_t) + 3) / 4);

// FIXME: Do not use hardcoded library search path!
static const char *ventus_final_ld_flags[] = {"-nodefaultlibs", "-L"CLANG_RESOURCE_DIR"/../../", "-lworkitem", NULL};

void
pocl_ventus_init_device_ops(struct pocl_device_ops *ops)
{
  ops->device_name = "ventus";

  ops->probe = pocl_ventus_probe;
  ops->uninit = pocl_ventus_uninit;
  ops->reinit = pocl_ventus_reinit;
  ops->init = pocl_ventus_init;

  ops->alloc_mem_obj = pocl_driver_alloc_mem_obj;
  ops->free = pocl_driver_free;

  ops->read = pocl_driver_read;
  ops->read_rect = pocl_driver_read_rect;
  ops->write = pocl_driver_write;
  ops->write_rect = pocl_driver_write_rect;
  ops->copy = pocl_driver_copy;
  ops->copy_with_size = pocl_driver_copy_with_size;
  ops->copy_rect = pocl_driver_copy_rect;
  ops->memfill = pocl_driver_memfill;
  ops->map_mem = pocl_driver_map_mem;
  ops->unmap_mem = pocl_driver_unmap_mem;
  ops->get_mapping_ptr = pocl_driver_get_mapping_ptr;
  ops->free_mapping_ptr = pocl_driver_free_mapping_ptr;

  ops->can_migrate_d2d = NULL;
  ops->migrate_d2d = NULL;

  ops->run = pocl_ventus_run;
  ops->run_native = pocl_ventus_run_native;

  ops->build_source = pocl_driver_build_source;
  ops->link_program = pocl_driver_link_program;
  ops->build_binary = pocl_driver_build_binary;
  ops->free_program = pocl_driver_free_program;
  ops->setup_metadata = pocl_driver_setup_metadata;
  ops->supports_binary = pocl_driver_supports_binary;
  ops->build_poclbinary = pocl_driver_build_poclbinary;
  ops->compile_kernel = pocl_ventus_compile_kernel;

  ops->join = pocl_ventus_join;
  ops->submit = pocl_ventus_submit;
  ops->broadcast = pocl_broadcast;
  ops->notify = pocl_ventus_notify;
  ops->flush = pocl_ventus_flush;
  ops->build_hash = pocl_ventus_build_hash;
  ops->compute_local_size = pocl_default_local_size_optimizer;

  ops->get_device_info_ext = NULL;

  ops->svm_free = pocl_ventus_svm_free;
  ops->svm_alloc = pocl_ventus_svm_alloc;
  /* no need to implement these two as they're noop
   * and pocl_exec_command takes care of it */
  ops->svm_map = NULL;
  ops->svm_unmap = NULL;
  ops->svm_copy = pocl_ventus_svm_copy;
  ops->svm_fill = pocl_driver_svm_fill;

  ops->create_kernel = NULL;
  ops->free_kernel = NULL;
  ops->create_sampler = NULL;
  ops->free_sampler = NULL;
  ops->copy_image_rect = pocl_ventus_copy_image_rect;
  ops->write_image_rect = pocl_ventus_write_image_rect;
  ops->read_image_rect = pocl_ventus_read_image_rect;
  ops->map_image = pocl_ventus_map_image;
  ops->unmap_image = pocl_ventus_unmap_image;
  ops->fill_image = pocl_ventus_fill_image;
}

char *
pocl_ventus_build_hash (cl_device_id device)
{
  char* res = (char *)calloc(1000, sizeof(char));
  snprintf(res, 1000, "THU-%s", device->llvm_cpu);
  return res;
}

unsigned int
pocl_ventus_probe(struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count(ops->device_name);

  /* No env specified, so pthread will be used instead of basic */
  if(env_count < 0)
    return 0;

  return env_count;
}

cl_int
pocl_ventus_init (unsigned j, cl_device_id dev, const char* parameters)
{
  struct ventus_device_data_t *d;
  cl_int ret = CL_SUCCESS;
  int err;

  d = (struct ventus_device_data_t *) calloc (1, sizeof (struct ventus_device_data_t));
  if (d == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  d->current_kernel = NULL;

  dev->data = d;

  SETUP_DEVICE_CL_VERSION(2, 0);
  dev->type = CL_DEVICE_TYPE_GPU;
  dev->long_name = "Ventus GPGPU device";
  dev->vendor = "THU";
  dev->vendor_id = 0x1234; // TODO: Update vendor id!
  dev->version = "2.0";
  dev->available = CL_TRUE;
  dev->compiler_available = CL_TRUE;
  dev->linker_available = CL_TRUE;
  dev->extensions = "";
  dev->profile = "FULL_PROFILE";
  dev->endian_little = CL_TRUE;

  dev->max_mem_alloc_size = 100 * 1024 * 1024;
  dev->mem_base_addr_align = 4;

  dev->max_constant_buffer_size = 32768;     // TODO: Update this to conformant to OCL 2.0
  dev->local_mem_size = 131072;     // TODO: Update this to conformant to OCL 2.0
  dev->global_mem_size = 1024 * 1024 * 1024; // 1G ram
  dev->global_mem_cache_type = CL_READ_WRITE_CACHE;
  dev->global_mem_cacheline_size = 64; // FIXME: Is this accurate?
  dev->global_mem_cache_size = 32768;  // FIXME: Is this accurate?
  dev->image_max_buffer_size = dev->max_mem_alloc_size / 16;

  dev->image2d_max_width = 1024; // TODO: Update
  dev->image3d_max_width = 1024; // TODO: Update

  dev->max_work_item_dimensions = 3;
  dev->max_work_group_size = 1024;
  dev->max_work_item_sizes[0] = 1024;
  dev->max_work_item_sizes[1] = 1024;
  dev->max_work_item_sizes[2] = 1024;
  dev->max_parameter_size = 64;
  dev->max_compute_units = 1;
  dev->max_clock_frequency = 600; // TODO: This is frequency in MHz
  dev->address_bits = 32;

  // Supports device side printf
  dev->device_side_printf = 1;
  dev->printf_buffer_size = PRINTF_BUFFER_SIZE;

  // Doesn't support partition
  dev->max_sub_devices = 1;
  dev->num_partition_properties = 1;
  dev->num_partition_types = 0;
  dev->partition_type = NULL;

  // Doesn't support SVM
  dev->svm_allocation_priority = 0;

  dev->final_linkage_flags = ventus_final_ld_flags;

  // TODO: Do we have builtin kernels for Ventus?

#ifdef ENABLE_LLVM
  dev->llvm_target_triplet = OCL_KERNEL_TARGET;
  dev->llvm_cpu = OCL_KERNEL_TARGET_CPU;
#else
  dev->llvm_target_triplet = "";
  dev->llvm_cpu = "";
#endif


#if (!defined(ENABLE_CONFORMANCE))
  /* full memory consistency model for atomic memory and fence operations
  https://www.khronos.org/registry/OpenCL/specs/3.0-unified/html/OpenCL_API.html#opencl-3.0-backwards-compatibility */
  dev->atomic_memory_capabilities = CL_DEVICE_ATOMIC_ORDER_RELAXED
                                       | CL_DEVICE_ATOMIC_ORDER_ACQ_REL
                                       | CL_DEVICE_ATOMIC_ORDER_SEQ_CST
                                       | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP
                                       | CL_DEVICE_ATOMIC_SCOPE_DEVICE
                                       | CL_DEVICE_ATOMIC_SCOPE_ALL_DEVICES;
  dev->atomic_fence_capabilities = CL_DEVICE_ATOMIC_ORDER_RELAXED
                                       | CL_DEVICE_ATOMIC_ORDER_ACQ_REL
                                       | CL_DEVICE_ATOMIC_ORDER_SEQ_CST
                                       | CL_DEVICE_ATOMIC_SCOPE_WORK_ITEM
                                       | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP
                                       | CL_DEVICE_ATOMIC_SCOPE_DEVICE;
#endif

  POCL_INIT_LOCK (d->cq_lock);

  assert (dev->printf_buffer_size > 0);
  d->printf_buffer = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT,
                                          dev->printf_buffer_size);
  assert (d->printf_buffer != NULL);

  return ret;
}

void
pocl_ventus_run (void *data, _cl_command_node *cmd)
{
  struct ventus_device_data_t *d;
  struct pocl_argument *al;
  size_t x, y, z;
  unsigned i;
  cl_kernel kernel = cmd->command.run.kernel;
  pocl_kernel_metadata_t *meta = kernel->meta;
  struct pocl_context *pc = &cmd->command.run.pc;

  if (pc->num_groups[0] == 0 || pc->num_groups[1] == 0 || pc->num_groups[2] == 0)
    return;

  assert (data != NULL);
  d = (struct ventus_device_data_t *) data;

  d->current_kernel = kernel;

  void **arguments = (void **)malloc (sizeof (void *)
                                      * (meta->num_args + meta->num_locals));

  /* Process the kernel arguments. Convert the opaque buffer
     pointers to real device pointers, allocate dynamic local
     memory buffers, etc. */
  for (i = 0; i < meta->num_args; ++i)
    {
      al = &(cmd->command.run.arguments[i]);
      if (ARG_IS_LOCAL (meta->arg_info[i]))
        {
          if (cmd->device->device_alloca_locals)
            {
              /* Local buffers are allocated in the device side work-group
                 launcher. Let's pass only the sizes of the local args in
                 the arg buffer. */
              assert (sizeof (size_t) == sizeof (void *));
              arguments[i] = (void *)al->size;
            }
          else
            {
              arguments[i] = malloc (sizeof (void *));
              *(void **)(arguments[i]) =
                pocl_aligned_malloc(MAX_EXTENDED_ALIGNMENT, al->size);
            }
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER)
        {
          /* It's legal to pass a NULL pointer to clSetKernelArguments. In
             that case we must pass the same NULL forward to the kernel.
             Otherwise, the user must have created a buffer with per device
             pointers stored in the cl_mem. */
          arguments[i] = malloc (sizeof (void *));
          if (al->value == NULL)
            {
              *(void **)arguments[i] = NULL;
            }
          else
            {
              void *ptr = NULL;
              if (al->is_svm)
                {
                  ptr = *(void **)al->value;
                }
              else
                {
                  cl_mem m = (*(cl_mem *)(al->value));
                  ptr = m->device_ptrs[cmd->device->global_mem_id].mem_ptr;
                }
              *(void **)arguments[i] = (char *)ptr + al->offset;
            }
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
        {
          dev_image_t di;
          pocl_fill_dev_image_t (&di, al, cmd->device);

          void *devptr = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT,
                                              sizeof (dev_image_t));
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = devptr;
          memcpy (devptr, &di, sizeof (dev_image_t));
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)
        {
          dev_sampler_t ds;
          pocl_fill_dev_sampler_t (&ds, al);
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = (void *)ds;
        }
      else
        {
          arguments[i] = al->value;
        }
    }

  if (!cmd->device->device_alloca_locals)
    for (i = 0; i < meta->num_locals; ++i)
      {
        size_t s = meta->local_sizes[i];
        size_t j = meta->num_args + i;
        arguments[j] = malloc (sizeof (void *));
        void *pp = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, s);
        *(void **)(arguments[j]) = pp;
      }

  pc->printf_buffer = d->printf_buffer;
  assert (pc->printf_buffer != NULL);
  pc->printf_buffer_capacity = cmd->device->printf_buffer_size;
  assert (pc->printf_buffer_capacity > 0);
  uint32_t position = 0;
  pc->printf_buffer_position = &position;

  unsigned rm = pocl_save_rm ();
  pocl_set_default_rm ();
  unsigned ftz = pocl_save_ftz ();
  pocl_set_ftz (kernel->program->flush_denorms);

  for (z = 0; z < pc->num_groups[2]; ++z)
    for (y = 0; y < pc->num_groups[1]; ++y)
      for (x = 0; x < pc->num_groups[0]; ++x)
        ((pocl_workgroup_func) cmd->command.run.wg)
	  ((uint8_t *)arguments, (uint8_t *)pc, x, y, z);

  pocl_restore_rm (rm);
  pocl_restore_ftz (ftz);

  if (position > 0)
    {
      write (STDOUT_FILENO, pc->printf_buffer, position);
      position = 0;
    }

  for (i = 0; i < meta->num_args; ++i)
    {
      if (ARG_IS_LOCAL (meta->arg_info[i]))
        {
          if (!cmd->device->device_alloca_locals)
            {
              POCL_MEM_FREE(*(void **)(arguments[i]));
              POCL_MEM_FREE(arguments[i]);
            }
          else
            {
              /* Device side local space allocation has deallocation via stack
                 unwind. */
            }
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE
               || meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)
        {
          if (meta->arg_info[i].type != POCL_ARG_TYPE_SAMPLER)
            POCL_MEM_FREE (*(void **)(arguments[i]));
          POCL_MEM_FREE(arguments[i]);
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER)
        {
          POCL_MEM_FREE(arguments[i]);
        }
    }

  if (!cmd->device->device_alloca_locals)
    for (i = 0; i < meta->num_locals; ++i)
      {
        POCL_MEM_FREE (*(void **)(arguments[meta->num_args + i]));
        POCL_MEM_FREE (arguments[meta->num_args + i]);
      }
  free(arguments);

  pocl_release_dlhandle_cache (cmd);
}

void
pocl_ventus_run_native (void *data, _cl_command_node *cmd)
{
  cl_event ev = cmd->sync.event.event;
  cl_device_id dev = cmd->device;
  size_t i;
  for (i = 0; i < ev->num_buffers; i++)
    {
      void *arg_loc = cmd->command.native.arg_locs[i];
      void *buf = ev->mem_objs[i]->device_ptrs[dev->global_mem_id].mem_ptr;
      if (dev->address_bits == 32)
        *((uint32_t *)arg_loc) = (uint32_t) (((uintptr_t)buf) & 0xFFFFFFFF);
      else
        *((uint64_t *)arg_loc) = (uint64_t) (uintptr_t)buf;
    }

  cmd->command.native.user_func(cmd->command.native.args);

  POCL_MEM_FREE (cmd->command.native.arg_locs);
}

cl_int
pocl_ventus_uninit (unsigned j, cl_device_id device)
{
  struct ventus_device_data_t *d = (struct ventus_device_data_t*)device->data;
  POCL_DESTROY_LOCK (d->cq_lock);
  pocl_aligned_free (d->printf_buffer);
  POCL_MEM_FREE(d);
  device->data = NULL;
  return CL_SUCCESS;
}

cl_int
pocl_ventus_reinit (unsigned j, cl_device_id device)
{
  struct ventus_device_data_t *d = (struct ventus_device_data_t *)calloc (1, sizeof (struct ventus_device_data_t));
  if (d == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  d->current_kernel = NULL;

  assert (device->printf_buffer_size > 0);
  d->printf_buffer = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT,
                                          device->printf_buffer_size);
  assert (d->printf_buffer != NULL);

  POCL_INIT_LOCK (d->cq_lock);
  device->data = d;
  return CL_SUCCESS;
}


static void ventus_command_scheduler (struct ventus_device_data_t *d)
{
  _cl_command_node *node;

  /* execute commands from ready list */
  while ((node = d->ready_list))
    {
      assert (pocl_command_is_ready(node->sync.event.event));
      assert (node->sync.event.event->status == CL_SUBMITTED);
      CDL_DELETE (d->ready_list, node);
      POCL_UNLOCK (d->cq_lock);
      pocl_exec_command (node);
      POCL_LOCK (d->cq_lock);
    }

  return;
}

void
pocl_ventus_submit (_cl_command_node *node, cl_command_queue cq)
{
  struct ventus_device_data_t *d = (struct ventus_device_data_t *)node->device->data;

  if (node != NULL && node->type == CL_COMMAND_NDRANGE_KERNEL)
    pocl_check_kernel_dlhandle_cache (node, 1, 1);

  node->ready = 1;
  POCL_LOCK (d->cq_lock);
  pocl_command_push(node, &d->ready_list, &d->command_list);

  POCL_UNLOCK_OBJ (node->sync.event.event);
  ventus_command_scheduler (d);
  POCL_UNLOCK (d->cq_lock);

  return;
}

void pocl_ventus_flush (cl_device_id device, cl_command_queue cq)
{
  struct ventus_device_data_t *d = (struct ventus_device_data_t *)device->data;

  POCL_LOCK (d->cq_lock);
  ventus_command_scheduler (d);
  POCL_UNLOCK (d->cq_lock);
}

void
pocl_ventus_join (cl_device_id device, cl_command_queue cq)
{
  struct ventus_device_data_t *d = (struct ventus_device_data_t *)device->data;

  POCL_LOCK (d->cq_lock);
  ventus_command_scheduler (d);
  POCL_UNLOCK (d->cq_lock);

  return;
}

void
pocl_ventus_notify (cl_device_id device, cl_event event, cl_event finished)
{
  struct ventus_device_data_t *d = (struct ventus_device_data_t *)device->data;
  _cl_command_node * volatile node = event->command;

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
          POCL_LOCK (d->cq_lock);
          CDL_DELETE (d->command_list, node);
          CDL_PREPEND (d->ready_list, node);
          ventus_command_scheduler (d);
          POCL_UNLOCK (d->cq_lock);
        }
      return;
    }
}

void
pocl_ventus_compile_kernel (_cl_command_node *cmd, cl_kernel kernel,
                           cl_device_id device, int specialize)
{
  if (cmd != NULL && cmd->type == CL_COMMAND_NDRANGE_KERNEL)
    pocl_check_kernel_dlhandle_cache (cmd, 0, specialize);
}

/*********************** IMAGES ********************************/

cl_int pocl_ventus_copy_image_rect( void *data,
                                   cl_mem src_image,
                                   cl_mem dst_image,
                                   pocl_mem_identifier *src_mem_id,
                                   pocl_mem_identifier *dst_mem_id,
                                   const size_t *src_origin,
                                   const size_t *dst_origin,
                                   const size_t *region)
{

  size_t px = src_image->image_elem_size * src_image->image_channels;
  const size_t adj_src_origin[3]
      = { src_origin[0] * px, src_origin[1], src_origin[2] };
  const size_t adj_dst_origin[3]
      = { dst_origin[0] * px, dst_origin[1], dst_origin[2] };
  const size_t adj_region[3] = { region[0] * px, region[1], region[2] };

  POCL_MSG_PRINT_MEMORY (
      " ventus COPY IMAGE RECT \n"
      "dst_image %p dst_mem_id %p \n"
      "src_image %p src_mem_id %p \n"
      "dst_origin [0,1,2] %zu %zu %zu \n"
      "src_origin [0,1,2] %zu %zu %zu \n"
      "region [0,1,2] %zu %zu %zu \n"
      "px %zu\n",
      dst_image, dst_mem_id,
      src_image, src_mem_id,
      dst_origin[0], dst_origin[1], dst_origin[2],
      src_origin[0], src_origin[1], src_origin[2],
      region[0], region[1], region[2],
      px);

  pocl_driver_copy_rect (
      data, dst_mem_id, NULL, src_mem_id, NULL, adj_dst_origin, adj_src_origin,
      adj_region, dst_image->image_row_pitch, dst_image->image_slice_pitch,
      src_image->image_row_pitch, src_image->image_slice_pitch);

  return CL_SUCCESS;
}

/* copies a region from host or device buffer to device image */
cl_int pocl_ventus_write_image_rect (  void *data,
                                      cl_mem dst_image,
                                      pocl_mem_identifier *dst_mem_id,
                                      const void *__restrict__ src_host_ptr,
                                      pocl_mem_identifier *src_mem_id,
                                      const size_t *origin,
                                      const size_t *region,
                                      size_t src_row_pitch,
                                      size_t src_slice_pitch,
                                      size_t src_offset)
{
  POCL_MSG_PRINT_MEMORY (
      "ventus WRITE IMAGE RECT \n"
      "dst_image %p dst_mem_id %p \n"
      "src_hostptr %p src_mem_id %p \n"
      "origin [0,1,2] %zu %zu %zu \n"
      "region [0,1,2] %zu %zu %zu \n"
      "row %zu slice %zu offset %zu \n",
      dst_image, dst_mem_id,
      src_host_ptr, src_mem_id,
      origin[0], origin[1], origin[2],
      region[0], region[1], region[2],
      src_row_pitch, src_slice_pitch, src_offset);

  const void *__restrict__ ptr
      = src_host_ptr ? src_host_ptr : src_mem_id->mem_ptr;
  ptr += src_offset;
  const size_t zero_origin[3] = { 0 };
  size_t px = dst_image->image_elem_size * dst_image->image_channels;
  if (src_row_pitch == 0)
    src_row_pitch = px * region[0];
  if (src_slice_pitch == 0)
    src_slice_pitch = src_row_pitch * region[1];

  const size_t adj_origin[3] = { origin[0] * px, origin[1], origin[2] };
  const size_t adj_region[3] = { region[0] * px, region[1], region[2] };

  pocl_driver_write_rect (data, ptr, dst_mem_id, NULL, adj_origin, zero_origin,
                          adj_region, dst_image->image_row_pitch,
                          dst_image->image_slice_pitch, src_row_pitch,
                          src_slice_pitch);
  return CL_SUCCESS;
}

/* copies a region from device image to host or device buffer */
cl_int pocl_ventus_read_image_rect(  void *data,
                                    cl_mem src_image,
                                    pocl_mem_identifier *src_mem_id,
                                    void *__restrict__ dst_host_ptr,
                                    pocl_mem_identifier *dst_mem_id,
                                    const size_t *origin,
                                    const size_t *region,
                                    size_t dst_row_pitch,
                                    size_t dst_slice_pitch,
                                    size_t dst_offset)
{
  POCL_MSG_PRINT_MEMORY (
      "ventus READ IMAGE RECT \n"
      "src_image %p src_mem_id %p \n"
      "dst_hostptr %p dst_mem_id %p \n"
      "origin [0,1,2] %zu %zu %zu \n"
      "region [0,1,2] %zu %zu %zu \n"
      "row %zu slice %zu offset %zu \n",
      src_image, src_mem_id,
      dst_host_ptr, dst_mem_id,
      origin[0], origin[1], origin[2],
      region[0], region[1], region[2],
      dst_row_pitch, dst_slice_pitch, dst_offset);

  void *__restrict__ ptr = dst_host_ptr ? dst_host_ptr : dst_mem_id->mem_ptr;
  ptr += dst_offset;
  const size_t zero_origin[3] = { 0 };
  size_t px = src_image->image_elem_size * src_image->image_channels;
  if (dst_row_pitch == 0)
    dst_row_pitch = px * region[0];
  if (dst_slice_pitch == 0)
    dst_slice_pitch = dst_row_pitch * region[1];
  const size_t adj_origin[3] = { origin[0] * px, origin[1], origin[2] };
  const size_t adj_region[3] = { region[0] * px, region[1], region[2] };

  pocl_driver_read_rect (data, ptr, src_mem_id, NULL, adj_origin, zero_origin,
                         adj_region, src_image->image_row_pitch,
                         src_image->image_slice_pitch, dst_row_pitch,
                         dst_slice_pitch);
  return CL_SUCCESS;
}


cl_int pocl_ventus_map_image (void *data,
                             pocl_mem_identifier *mem_id,
                             cl_mem src_image,
                             mem_mapping_t *map)
{
  assert (map->host_ptr != NULL);

  if (map->map_flags & CL_MAP_WRITE_INVALIDATE_REGION)
    return CL_SUCCESS;

  if (map->host_ptr != ((char *)mem_id->mem_ptr + map->offset))
    {
      pocl_ventus_read_image_rect (data, src_image, mem_id, map->host_ptr,
                                  NULL, map->origin, map->region,
                                  map->row_pitch, map->slice_pitch, 0);
    }
  return CL_SUCCESS;
}

cl_int pocl_ventus_unmap_image(void *data,
                              pocl_mem_identifier *mem_id,
                              cl_mem dst_image,
                              mem_mapping_t *map)
{
  if (map->map_flags == CL_MAP_READ)
    return CL_SUCCESS;

  if (map->host_ptr != ((char *)mem_id->mem_ptr + map->offset))
    {
      pocl_ventus_write_image_rect (data, dst_image, mem_id, map->host_ptr,
                                   NULL, map->origin, map->region,
                                   map->row_pitch, map->slice_pitch, 0);
    }
  return CL_SUCCESS;
}

cl_int
pocl_ventus_fill_image (void *data, cl_mem image,
                       pocl_mem_identifier *image_data, const size_t *origin,
                       const size_t *region, cl_uint4 orig_pixel,
                       pixel_t fill_pixel, size_t pixel_size)
{
   POCL_MSG_PRINT_MEMORY ("ventus / FILL IMAGE \n"
                          "image %p data %p \n"
                          "origin [0,1,2] %zu %zu %zu \n"
                          "region [0,1,2] %zu %zu %zu \n"
                          "pixel %p size %zu \n",
                          image, image_data,
                          origin[0], origin[1], origin[2],
                          region[0], region[1], region[2],
                          fill_pixel, pixel_size);

  size_t row_pitch = image->image_row_pitch;
  size_t slice_pitch = image->image_slice_pitch;
  char *__restrict const adjusted_device_ptr
      = (char *)image_data->mem_ptr
        + origin[0] * pixel_size
        + row_pitch * origin[1]
        + slice_pitch * origin[2];

  size_t i, j, k;

  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      for (i = 0; i < region[0]; ++i)
        memcpy (adjusted_device_ptr
                  + pixel_size * i
                  + row_pitch * j
                  + slice_pitch * k,
                fill_pixel,
                pixel_size);
  return CL_SUCCESS;
}

/***************************************************************************/
void
pocl_ventus_svm_free (cl_device_id dev, void *svm_ptr)
{
  /* TODO we should somehow figure out the size argument
   * and call pocl_free_global_mem */
  pocl_aligned_free (svm_ptr);
}

void *
pocl_ventus_svm_alloc (cl_device_id dev, cl_svm_mem_flags flags, size_t size)

{
  return pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, size);
}

void
pocl_ventus_svm_copy (cl_device_id dev, void *__restrict__ dst,
                     const void *__restrict__ src, size_t size)
{
  memcpy (dst, src, size);
}
