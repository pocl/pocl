/* basic.c - a minimalistic single core pocl device driver layer implementation

   Copyright (c) 2011-2013 Universidad Rey Juan Carlos and
                 2011-2021 Pekka Jääskeläinen
                 2023-2024 Pekka Jääskeläinen / Intel Finland Oy

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

#include "basic.h"
#include "common.h"
#include "config.h"
#include "config2.h"
#include "cpuinfo.h"
#include "devices.h"
#include "pocl_builtin_kernels.h"
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
#include "common_utils.h"

#ifdef ENABLE_LLVM
#include "pocl_llvm.h"
#endif

typedef struct
{
  /* List of commands ready to be executed */
  _cl_command_node *ready_list;
  /* List of commands not yet ready to be executed */
  _cl_command_node *command_list;
  /* Lock for command list related operations */
  pocl_lock_t cq_lock;

  /* printf buffer */
  void *printf_buffer;

  cl_bool available;
} pocl_basic_data_t;

typedef struct _pocl_basic_usm_allocation_t
{
  void *ptr;
  size_t size;
  cl_mem_alloc_flags_intel flags;
  unsigned alloc_type;

  struct _pocl_basic_usm_allocation_t *next, *prev;
} pocl_basic_usm_allocation_t;

void
pocl_basic_init_device_ops(struct pocl_device_ops *ops)
{
  ops->device_name = "cpu-minimal";

  ops->probe = pocl_basic_probe;
  ops->uninit = pocl_basic_uninit;
  ops->reinit = pocl_basic_reinit;
  ops->init = pocl_basic_init;

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

  ops->run = pocl_basic_run;
  ops->run_native = pocl_basic_run_native;

  ops->build_source = pocl_driver_build_source;
  ops->link_program = pocl_driver_link_program;
  ops->build_binary = pocl_driver_build_binary;
  ops->free_program = pocl_basic_free_program;
  ops->setup_metadata = pocl_driver_setup_metadata;
  ops->supports_binary = pocl_driver_supports_binary;
  ops->build_poclbinary = pocl_driver_build_poclbinary;
  ops->compile_kernel = pocl_basic_compile_kernel;
  ops->build_builtin = pocl_driver_build_opencl_builtins;
  ops->build_defined_builtin = pocl_cpu_build_defined_builtin;
  ops->supports_dbk = pocl_cpu_supports_dbk;

  ops->join = pocl_basic_join;
  ops->submit = pocl_basic_submit;
  ops->broadcast = pocl_broadcast;
  ops->notify = pocl_basic_notify;
  ops->flush = pocl_basic_flush;
  ops->build_hash = pocl_cpu_build_hash;
  ops->compute_local_size = pocl_default_local_size_optimizer;

  ops->get_device_info_ext = pocl_basic_get_device_info_ext;
  ops->get_subgroup_info_ext = pocl_basic_get_subgroup_info_ext;
  ops->set_kernel_exec_info_ext = pocl_basic_set_kernel_exec_info_ext;
  ops->get_synchronized_timestamps = pocl_driver_get_synchronized_timestamps;

  ops->svm_free = pocl_basic_svm_free;
  ops->svm_alloc = pocl_basic_svm_alloc;
  ops->usm_alloc = pocl_basic_usm_alloc;
  ops->usm_free = pocl_basic_usm_free;
  ops->usm_free_blocking = NULL;
  /* no need to implement these as they're noop
   * and pocl_exec_command takes care of it */
  ops->svm_map = NULL;
  ops->svm_unmap = NULL;
  ops->svm_advise = NULL;
  ops->svm_migrate = NULL;
  ops->svm_copy = pocl_driver_svm_copy;
  ops->svm_fill = pocl_driver_svm_fill;
  ops->svm_copy_rect = pocl_driver_svm_copy_rect;
  ops->svm_fill_rect = pocl_driver_svm_fill_rect;

  ops->create_kernel = pocl_basic_create_kernel;
  ops->free_kernel = pocl_basic_free_kernel;
  ops->create_sampler = NULL;
  ops->free_sampler = NULL;
  ops->copy_image_rect = pocl_basic_copy_image_rect;
  ops->write_image_rect = pocl_basic_write_image_rect;
  ops->read_image_rect = pocl_basic_read_image_rect;
  ops->map_image = pocl_basic_map_image;
  ops->unmap_image = pocl_basic_unmap_image;
  ops->fill_image = pocl_basic_fill_image;
}

unsigned int
pocl_basic_probe(struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count(ops->device_name);

  pocl_cpu_probe ();

  /* for backwards compatibility */
  if (env_count <= 0)
    env_count = pocl_device_get_env_count("basic");

  /* No env specified, so pthread will be used instead of basic */
  if(env_count < 0)
    return 0;

  return env_count;
}

cl_int
pocl_basic_init (unsigned j, cl_device_id device, const char* parameters)
{
  pocl_basic_data_t *d;
  cl_int ret = CL_SUCCESS;
  int err;
  static int first_basic_init = 1;

  if (first_basic_init)
    {
      pocl_init_dlhandle_cache();
      first_basic_init = 0;
    }

  d = (pocl_basic_data_t *)calloc (1, sizeof (pocl_basic_data_t));
  if (d == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  d->available = CL_TRUE;

  device->data = d;
  device->available = &d->available;

  ret = pocl_cpu_init_common (device);
  if (ret != CL_SUCCESS)
    return ret;

  POCL_INIT_LOCK (d->cq_lock);

  /* The cpu-minimal (also known as 'basic') driver represents only one
     "compute unit" as it doesn't exploit multiple hardware threads. Multiple
     basic devices can be still used for task level parallelism
     using multiple OpenCL devices in multiple client threads. */
  device->max_compute_units = 1;
  device->max_sub_devices = 0;
  device->num_partition_properties = 0;
  device->num_partition_types = 0;

  assert (device->printf_buffer_size > 0);
  d->printf_buffer
    = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, device->printf_buffer_size);
  assert (d->printf_buffer != NULL);

  return ret;
}

void
pocl_basic_run (void *data, _cl_command_node *cmd)
{
  pocl_basic_data_t *d = (pocl_basic_data_t *)data;
  struct pocl_argument *al = NULL;
  size_t x, y, z;
  unsigned i;
  cl_kernel kernel = cmd->command.run.kernel;
  cl_program program = kernel->program;
  pocl_kernel_metadata_t *meta = kernel->meta;
  struct pocl_context *pc = &cmd->command.run.pc;
  cl_uint dev_i = cmd->program_device_i;

  if (program->builtin_kernel_attributes)
    {
      assert (meta->builtin_kernel_id != 0);
      pocl_cpu_execute_dbk (program, kernel, meta, dev_i,
                            cmd->command.run.arguments);
      return;
    }

  pocl_driver_build_gvar_init_kernel (program, dev_i, cmd->device,
                                      pocl_cpu_gvar_init_callback);

  if (pc->num_groups[0] == 0 || pc->num_groups[1] == 0 || pc->num_groups[2] == 0)
    return;

  assert (data != NULL);

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
              if (al->is_raw_ptr)
                {
                  ptr = *(void **)al->value;
                }
              else
                {
                  cl_mem m = (*(cl_mem *)(al->value));
                  ptr = m->device_ptrs[cmd->device->global_mem_id].mem_ptr;
                }
              *(void **)arguments[i] = (char *)ptr;
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

  if (cmd->device->device_alloca_locals)
    {
      /* Local buffers are allocated in the device side work-group
         launcher. Let's pass only the sizes of the local args in
         the arg buffer. */
      for (i = 0; i < meta->num_locals; ++i)
        {
          assert (sizeof (size_t) == sizeof (void *));
          size_t s = meta->local_sizes[i];
          size_t j = meta->num_args + i;
          *(size_t *)(arguments[j]) = s;
        }
    }
  else
    {
      for (i = 0; i < meta->num_locals; ++i)
        {
          size_t s = meta->local_sizes[i];
          size_t j = meta->num_args + i;
          arguments[j] = malloc (sizeof (void *));
          void *pp = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, s);
          *(void **)(arguments[j]) = pp;
        }
    }

  pc->printf_buffer = d->printf_buffer;
  assert (pc->printf_buffer != NULL);
  uint32_t position = 0;
  pc->printf_buffer_position = &position;

  pc->printf_buffer_capacity = cmd->device->printf_buffer_size;
  assert (pc->printf_buffer_capacity > 0);

  pc->global_var_buffer = program->gvar_storage[dev_i];

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

#ifndef ENABLE_PRINTF_IMMEDIATE_FLUSH
  pocl_write_printf_buffer ((char *)d->printf_buffer, position);
#endif

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
  free (arguments);

  pocl_release_dlhandle_cache (cmd->command.run.device_data);
}

void
pocl_basic_run_native (void *data, _cl_command_node *cmd)
{
  cl_event ev = cmd->sync.event.event;
  cl_device_id dev = cmd->device;
  size_t i = 0;
  pocl_buffer_migration_info *mig = NULL;
  LL_FOREACH (cmd->migr_infos, mig)
  {
    void *arg_loc = cmd->command.native.arg_locs[i];
    void *buf_addr = mig->buffer->device_ptrs[dev->global_mem_id].mem_ptr;
    if (dev->address_bits == 32)
      *((uint32_t *)arg_loc) = (uint32_t)(((uintptr_t)buf_addr) & 0xFFFFFFFF);
    else
      *((uint64_t *)arg_loc) = (uint64_t)(uintptr_t)buf_addr;
    ++i;
    }

  cmd->command.native.user_func(cmd->command.native.args);

  POCL_MEM_FREE (cmd->command.native.arg_locs);
}

cl_int
pocl_basic_uninit (unsigned j, cl_device_id device)
{
  pocl_basic_data_t *d = (pocl_basic_data_t *)device->data;
  POCL_DESTROY_LOCK (d->cq_lock);
  pocl_aligned_free (d->printf_buffer);
  POCL_MEM_FREE(d);
  device->data = NULL;
  return CL_SUCCESS;
}

cl_int
pocl_basic_reinit (unsigned j, cl_device_id device, const char *parameters)
{
  pocl_basic_data_t *d
      = (pocl_basic_data_t *)calloc (1, sizeof (pocl_basic_data_t));
  if (d == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  assert (device->printf_buffer_size > 0);
  d->printf_buffer = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT,
                                          device->printf_buffer_size);
  assert (d->printf_buffer != NULL);

  POCL_INIT_LOCK (d->cq_lock);
  device->data = d;
  return CL_SUCCESS;
}

static void
basic_command_scheduler (pocl_basic_data_t *d)
{
  _cl_command_node *node;
  
  /* execute commands from ready list */
  while ((node = d->ready_list))
    {
      assert (pocl_command_is_ready (node->sync.event.event));
      assert (node->sync.event.event->status == CL_SUBMITTED);
      CDL_DELETE (d->ready_list, node);
      POCL_UNLOCK (d->cq_lock);
      pocl_exec_command (node);
      POCL_LOCK (d->cq_lock);
    }

  return;
}

void
pocl_basic_submit (_cl_command_node *node, cl_command_queue cq)
{
  pocl_basic_data_t *d = node->device->data;

  if (node != NULL && node->type == CL_COMMAND_NDRANGE_KERNEL)
    {
      cl_kernel kernel = node->command.run.kernel;
      cl_program program = kernel->program;
      if (!program->builtin_kernel_attributes)
        {
          void *handle
            = pocl_check_kernel_dlhandle_cache (node, CL_TRUE, CL_TRUE);
          if (handle == NULL)
            {
              pocl_update_event_running_unlocked (node->sync.event.event);
              POCL_UNLOCK_OBJ (node->sync.event.event);
              POCL_UPDATE_EVENT_FAILED (node->sync.event.event);
              return;
            }
          node->command.run.device_data = handle;
        }
    }

  node->ready = 1;
  POCL_LOCK (d->cq_lock);
  pocl_command_push(node, &d->ready_list, &d->command_list);

  POCL_UNLOCK_OBJ (node->sync.event.event);
  basic_command_scheduler (d);
  POCL_UNLOCK (d->cq_lock);

  return;
}

void pocl_basic_flush (cl_device_id device, cl_command_queue cq)
{
  pocl_basic_data_t *d = (pocl_basic_data_t *)device->data;

  POCL_LOCK (d->cq_lock);
  basic_command_scheduler (d);
  POCL_UNLOCK (d->cq_lock);
}

void
pocl_basic_join (cl_device_id device, cl_command_queue cq)
{
  pocl_basic_data_t *d = (pocl_basic_data_t *)device->data;

  POCL_LOCK (d->cq_lock);
  basic_command_scheduler (d);
  POCL_UNLOCK (d->cq_lock);

  return;
}

void
pocl_basic_notify (cl_device_id device, cl_event event, cl_event finished)
{
  pocl_basic_data_t *d = (pocl_basic_data_t *)device->data;
  _cl_command_node * volatile node = event->command;

  if (finished->status < CL_COMPLETE)
    {
      pocl_update_event_failed_locked (event);
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
          POCL_UNLOCK_OBJ (event);
          basic_command_scheduler (d);
          POCL_LOCK_OBJ (event);
          POCL_UNLOCK (d->cq_lock);
        }
      return;
    }
}

int
pocl_basic_compile_kernel (_cl_command_node *cmd,
                           cl_kernel kernel,
                           cl_device_id device,
                           int specialize)
{
  char *saved_name = NULL;
  if (cmd == NULL || cmd->type != CL_COMMAND_NDRANGE_KERNEL)
    return CL_INVALID_OPERATION;
  pocl_sanitize_builtin_kernel_name (kernel, &saved_name);
  void *handle = pocl_check_kernel_dlhandle_cache (cmd, CL_FALSE, specialize);
  pocl_restore_builtin_kernel_name (kernel, saved_name);
  return handle == NULL ? CL_COMPILE_PROGRAM_FAILURE : CL_SUCCESS;
}

int
pocl_basic_free_program (cl_device_id device, cl_program program,
                          unsigned dev_i)
{
  pocl_driver_free_program (device, program, dev_i);
  program->global_var_total_size[dev_i] = 0;
  pocl_aligned_free (program->gvar_storage[dev_i]);
  return 0;
}
/*********************** IMAGES ********************************/

cl_int pocl_basic_copy_image_rect( void *data,
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
      "CPU: COPY IMAGE RECT \n"
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
cl_int pocl_basic_write_image_rect (  void *data,
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
      "CPU: WRITE IMAGE RECT \n"
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
cl_int pocl_basic_read_image_rect(  void *data,
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
      "CPU: READ IMAGE RECT \n"
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


cl_int pocl_basic_map_image (void *data,
                             pocl_mem_identifier *mem_id,
                             cl_mem src_image,
                             mem_mapping_t *map)
{
  assert (map->host_ptr != NULL);

  if (map->map_flags & CL_MAP_WRITE_INVALIDATE_REGION)
    return CL_SUCCESS;

  if (map->host_ptr != ((char *)mem_id->mem_ptr + map->offset))
    {
      pocl_basic_read_image_rect (data, src_image, mem_id, map->host_ptr,
                                  NULL, map->origin, map->region,
                                  map->row_pitch, map->slice_pitch, 0);
    }
  return CL_SUCCESS;
}

cl_int pocl_basic_unmap_image(void *data,
                              pocl_mem_identifier *mem_id,
                              cl_mem dst_image,
                              mem_mapping_t *map)
{
  if (map->map_flags == CL_MAP_READ)
    return CL_SUCCESS;

  if (map->host_ptr != ((char *)mem_id->mem_ptr + map->offset))
    {
      pocl_basic_write_image_rect (data, dst_image, mem_id, map->host_ptr,
                                   NULL, map->origin, map->region,
                                   map->row_pitch, map->slice_pitch, 0);
    }
  return CL_SUCCESS;
}

cl_int
pocl_basic_fill_image (void *data, cl_mem image,
                       pocl_mem_identifier *image_data, const size_t *origin,
                       const size_t *region, cl_uint4 orig_pixel,
                       pixel_t fill_pixel, size_t pixel_size)
{
   POCL_MSG_PRINT_MEMORY ("CPU: FILL IMAGE \n"
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
pocl_basic_svm_free (cl_device_id dev, void *svm_ptr)
{
  /* TODO we should somehow figure out the size argument
   * and call pocl_free_global_mem */
  pocl_aligned_free (svm_ptr);
}

void *
pocl_basic_svm_alloc (cl_device_id dev, cl_svm_mem_flags flags, size_t size)

{
  return pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, size);
}

void *
pocl_basic_usm_alloc (cl_device_id dev, unsigned alloc_type,
                      cl_mem_alloc_flags_intel flags, size_t size,
                      cl_int *err_code)
{
  int errcode = CL_SUCCESS;
  void *ptr = NULL;

  ptr = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, size);

  if (err_code)
    *err_code = errcode;

  return ptr;
}

void
pocl_basic_usm_free (cl_device_id dev, void *usm_ptr)
{
  pocl_aligned_free (usm_ptr);
}

cl_int
pocl_basic_get_device_info_ext (cl_device_id device, cl_device_info param_name,
                            size_t param_value_size, void *param_value,
                            size_t *param_value_size_ret)
{
  switch (param_name)
    {
    case CL_DEVICE_SUB_GROUP_SIZES_INTEL:
      {
        /* We can basically support fixing any WG size with the CPU devices,
           but let's report something semi-sensible here for vectorization aid.
         */
        size_t sizes[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };
        POCL_RETURN_GETINFO_ARRAY (size_t, sizeof (sizes) / sizeof (size_t),
                                   sizes);
      }
    default:
      POCL_MSG_ERR ("Unknown param_name for get_device_info_ext: %u\n",
                    param_name);
      return CL_INVALID_VALUE;
    }
}

cl_int
pocl_basic_get_subgroup_info_ext (cl_device_id device,
                                  cl_kernel kernel,
                                  unsigned program_device_i,
                                  cl_kernel_sub_group_info param_name,
                                  size_t input_value_size,
                                  const void *input_value,
                                  size_t param_value_size,
                                  void *param_value,
                                  size_t *param_value_size_ret)
{
  switch (param_name)
    {
    case CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE:
      {

        /* For now assume SG == WG_x. */
        POCL_RETURN_GETINFO (size_t, ((size_t *)input_value)[0]);
      }
    case CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE:
      {
        /* For now assume SG == WG_x and thus we have WG_size_y*WG_size_z of
           them per WG. */
        POCL_RETURN_GETINFO (
          size_t,
          min (device->max_num_sub_groups,
               (input_value_size > sizeof (size_t) ? ((size_t *)input_value)[1]
                                                   : 1)
                 * (input_value_size > sizeof (size_t) * 2
                      ? ((size_t *)input_value)[2]
                      : 1)));
      }
    case CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT:
      {
        POCL_RETURN_ERROR_ON ((input_value == NULL), CL_INVALID_VALUE,
                              "SG size wish not given.");
        size_t n_wish = *(size_t *)input_value;
        /* For now assume SG == WG_x and the simplest way of looping only at
           y dimension. Use magic number 32 as the preferred SG size for now.
         */
        size_t nd[3];
        if (n_wish > device->max_num_sub_groups
            || (n_wish > 1 && param_value_size / sizeof (size_t) == 1))
          {
            nd[0] = nd[1] = nd[2] = 0;
            POCL_RETURN_GETINFO_ARRAY (size_t,
                                       param_value_size / sizeof (size_t), nd);
          }
        else
          {
            nd[0] = device->max_work_group_size / n_wish;
            nd[1] = n_wish;
            nd[2] = 1;
            POCL_RETURN_GETINFO_ARRAY (size_t,
                                       param_value_size / sizeof (size_t), nd);
          }
      }
    default:
      POCL_RETURN_ERROR_ON (1, CL_INVALID_VALUE, "Unknown param_name: %u\n",
                            param_name);
    }
}

cl_int
pocl_basic_set_kernel_exec_info_ext (cl_device_id dev,
                                     unsigned program_device_i,
                                     cl_kernel Kernel, cl_uint param_name,
                                     size_t param_value_size,
                                     const void *param_value)
{

  switch (param_name)
    {
    case CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM:
    case CL_KERNEL_EXEC_INFO_SVM_PTRS:
    case CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL:
    case CL_KERNEL_EXEC_INFO_DEVICE_PTRS_EXT:
    case CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL:
    case CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL:
    case CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL:
    return CL_SUCCESS;
    default:
      return CL_INVALID_VALUE;
    }
}

/**
 * Find the index in cl_program p's builtin_kernel_names of cl_kernel k.
 *
 * \return -1 if k is not a DBK of p and otherwise a valid index.
 */
static int
get_dbk_index (cl_program p, cl_kernel k)
{

  size_t dbk_index = SIZE_MAX;
  for (size_t i = 0; i < p->num_builtin_kernels; ++i)
    {
      if (strcmp (p->builtin_kernel_names[i], k->name) == 0)
        {
          dbk_index = i;
          return dbk_index;
        }
    }
  return dbk_index;
}

int
pocl_basic_create_kernel (cl_device_id device,
                          cl_program p,
                          cl_kernel k,
                          unsigned device_i)
{
  /* no dbks, nothing to do */
  if (p->num_builtin_kernels < 1)
    return CL_SUCCESS;

  int dbk_index = get_dbk_index (p, k);

  if (dbk_index < 0)
    return CL_INVALID_KERNEL_NAME;

  int status = CL_SUCCESS;
  BuiltinKernelId dbk_id = p->builtin_kernel_ids[dbk_index];
  switch (dbk_id)
    {
#ifdef HAVE_LIBXSMM
    case POCL_CDBI_DBK_EXP_GEMM:
    case POCL_CDBI_DBK_EXP_MATMUL:
      return status;
#endif
#ifdef HAVE_LIBJPEG_TURBO
    case POCL_CDBI_DBK_EXP_JPEG_ENCODE:
      {
        k->data[device_i] = pocl_cpu_init_dbk_khr_jpeg_encode (
          p->builtin_kernel_attributes[dbk_index], &status);
        return status;
      }
    case POCL_CDBI_DBK_EXP_JPEG_DECODE:
      {
        k->data[device_i] = pocl_cpu_init_dbk_khr_jpeg_decode (
          p->builtin_kernel_attributes[dbk_index], &status);
        return status;
      }
#endif
#ifdef HAVE_ONNXRT
    case POCL_CDBI_DBK_EXP_ONNX_INFERENCE:
      {
        status = pocl_create_ort_instance (
            p->builtin_kernel_attributes[dbk_index],
            (onnxrt_instance_t **)&k->data[device_i]);
        return status;
      }
#endif
    default:
      POCL_RETURN_ERROR_ON (1, CL_INVALID_DBK_ID,
                            "pocl_basic_create_kernel called with "
                            "unknown/unimplemented "
                            "DBK kernel.\n");
    }
}

int
pocl_basic_free_kernel (cl_device_id device,
                        cl_program p,
                        cl_kernel k,
                        unsigned device_i)
{
  /* no dbks, nothing to do */
  if (p->num_builtin_kernels < 1)
    return CL_SUCCESS;

  int dbk_index = get_dbk_index (p, k);

  if (dbk_index < 0)
    return CL_INVALID_KERNEL_NAME;

  int status = CL_SUCCESS;
  BuiltinKernelId dbk_id = p->builtin_kernel_ids[dbk_index];
  switch (dbk_id)
    {
#ifdef HAVE_LIBXSMM
    case POCL_CDBI_DBK_EXP_GEMM:
    case POCL_CDBI_DBK_EXP_MATMUL:
      return status;
#endif
#ifdef HAVE_LIBJPEG_TURBO
    case POCL_CDBI_DBK_EXP_JPEG_ENCODE:
      {
        status = pocl_cpu_destroy_dbk_khr_jpeg_encode (&(k->data[device_i]));
        return status;
      }
    case POCL_CDBI_DBK_EXP_JPEG_DECODE:
      {
        status = pocl_cpu_destroy_dbk_khr_jpeg_decode (&(k->data[device_i]));
        return status;
      }
#endif
#ifdef HAVE_ONNXRT
    case POCL_CDBI_DBK_EXP_ONNX_INFERENCE:
      {
        status = pocl_destroy_ort_instance (
            (onnxrt_instance_t **)&(k->data[device_i]));
        return status;
      }
#endif
    default:
      POCL_RETURN_ERROR_ON (1, CL_INVALID_DBK_ID,
                            "pocl_basic_free_kernel called with "
                            "unknown/unimplemented DBK kernel.\n");
    }
}
