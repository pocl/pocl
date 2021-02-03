/* basic.c - a minimalistic single core pocl device driver layer implementation

   Copyright (c) 2011-2013 Universidad Rey Juan Carlos and
                 2011-2020 Pekka Jääskeläinen

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
#include "pocl_local_size.h"
#include "pocl_util.h"
#include "topology/pocl_topology.h"
#include "utlist.h"

#include <assert.h>
#include <limits.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <utlist.h>

#include "pocl_cache.h"
#include "pocl_timing.h"
#include "pocl_file_util.h"
#include "pocl_workgroup_func.h"

#ifdef OCS_AVAILABLE
#include "pocl_llvm.h"
#endif

#ifndef HAVE_LIBDL
#error Basic driver requires DL library
#endif

struct data {
  /* Currently loaded kernel. */
  cl_kernel current_kernel;

  /* List of commands ready to be executed */
  _cl_command_node * volatile ready_list;
  /* List of commands not yet ready to be executed */
  _cl_command_node * volatile command_list;
  /* Lock for command list related operations */
  pocl_lock_t cq_lock;
  /* printf buffer */
  void *printf_buffer;
};

void
pocl_basic_init_device_ops(struct pocl_device_ops *ops)
{
  ops->device_name = "basic";

  ops->probe = pocl_basic_probe;
  ops->uninit = pocl_basic_uninit;
  ops->reinit = pocl_basic_reinit;
  ops->init = pocl_basic_init;
  ops->alloc_mem_obj = pocl_basic_alloc_mem_obj;
  ops->free = pocl_basic_free;
  ops->read = pocl_basic_read;
  ops->read_rect = pocl_basic_read_rect;
  ops->write = pocl_basic_write;
  ops->write_rect = pocl_basic_write_rect;
  ops->copy = pocl_basic_copy;
  ops->copy_rect = pocl_basic_copy_rect;
  ops->memfill = pocl_basic_memfill;
  ops->map_mem = pocl_basic_map_mem;
  ops->compile_kernel = pocl_basic_compile_kernel;
  ops->unmap_mem = pocl_basic_unmap_mem;
  ops->run = pocl_basic_run;
  ops->run_native = pocl_basic_run_native;
  ops->join = pocl_basic_join;
  ops->submit = pocl_basic_submit;
  ops->broadcast = pocl_broadcast;
  ops->notify = pocl_basic_notify;
  ops->flush = pocl_basic_flush;
  ops->build_hash = pocl_basic_build_hash;
  ops->compute_local_size = pocl_default_local_size_optimizer;

  ops->get_device_info_ext = NULL;

  ops->svm_free = pocl_basic_svm_free;
  ops->svm_alloc = pocl_basic_svm_alloc;
  /* no need to implement these two as they're noop
   * and pocl_exec_command takes care of it */
  ops->svm_map = NULL;
  ops->svm_unmap = NULL;
  ops->svm_copy = pocl_basic_svm_copy;
  ops->svm_fill = pocl_basic_svm_fill;

  ops->create_image = NULL;
  ops->free_image = NULL;
  ops->create_sampler = NULL;
  ops->free_sampler = NULL;
  ops->copy_image_rect = pocl_basic_copy_image_rect;
  ops->write_image_rect = pocl_basic_write_image_rect;
  ops->read_image_rect = pocl_basic_read_image_rect;
  ops->map_image = pocl_basic_map_image;
  ops->unmap_image = pocl_basic_unmap_image;
  ops->fill_image = pocl_basic_fill_image;
}

char *
pocl_basic_build_hash (cl_device_id device)
{
  char* res = calloc(1000, sizeof(char));
#ifdef KERNELLIB_HOST_DISTRO_VARIANTS
  char *name = get_llvm_cpu_name ();
  snprintf (res, 1000, "basic-%s-%s", HOST_DEVICE_BUILD_HASH, name);
  POCL_MEM_FREE (name);
#else
  snprintf (res, 1000, "basic-%s", HOST_DEVICE_BUILD_HASH);
#endif
  return res;
}

unsigned int
pocl_basic_probe(struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count(ops->device_name);

  /* No env specified, so pthread will be used instead of basic */
  if(env_count < 0)
    return 0;

  return env_count;
}

cl_int
pocl_basic_init (unsigned j, cl_device_id device, const char* parameters)
{
  struct data *d;
  cl_int ret = CL_SUCCESS;
  int err;
  static int first_basic_init = 1;

  if (first_basic_init)
    {
      POCL_MSG_WARN ("INIT dlcache DOTO delete\n");
      pocl_init_dlhandle_cache();
      first_basic_init = 0;
    }
  device->global_mem_id = 0;

  d = (struct data *) calloc (1, sizeof (struct data));
  if (d == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  d->current_kernel = NULL;
  device->data = d;

  pocl_init_default_device_infos (device);
  device->extensions = HOST_DEVICE_EXTENSIONS;

  /* hwloc probes OpenCL device info at its initialization in case
     the OpenCL extension is enabled. This causes to printout 
     an unimplemented property error because hwloc is used to
     initialize global_mem_size which it is not yet. Just put 
     a nonzero there for now. */
  device->global_mem_size = 1;
  err = pocl_topology_detect_device_info(device);
  if (err)
    ret = CL_INVALID_DEVICE;

  POCL_INIT_LOCK (d->cq_lock);

  assert (device->printf_buffer_size > 0);
  d->printf_buffer = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT,
                                          device->printf_buffer_size);
  assert (d->printf_buffer != NULL);

  pocl_cpuinfo_detect_device_info(device);
  pocl_set_buffer_image_limits(device);

  if (device->vendor_id == 0)
    device->vendor_id = CL_KHRONOS_VENDOR_ID_POCL;

  /* The basic driver represents only one "compute unit" as
     it doesn't exploit multiple hardware threads. Multiple
     basic devices can be still used for task level parallelism 
     using multiple OpenCL devices. */
  device->max_compute_units = 1;

  return ret;
}

cl_int
pocl_basic_alloc_mem_obj (cl_device_id device, cl_mem mem_obj, void* host_ptr)
{
  void *b = NULL;
  cl_mem_flags flags = mem_obj->flags;
  unsigned i;
  POCL_MSG_PRINT_MEMORY (" mem %p, dev %d\n", mem_obj, device->dev_id);
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
          POCL_MSG_PRINT_MEMORY (
              "mem %p dev %d, using already allocated mem\n", mem_obj,
              device->dev_id);
          return CL_SUCCESS;
        }
    }

  /* memory for this global memory is not yet allocated -> do it */
  if (flags & CL_MEM_USE_HOST_PTR)
    {
      // mem_host_ptr must be non-NULL
      assert(host_ptr != NULL);
      b = host_ptr;
    }
  else
    {
      POCL_MSG_PRINT_MEMORY ("!USE_HOST_PTR\n");
      b = pocl_aligned_malloc_global_mem (device, MAX_EXTENDED_ALIGNMENT,
                                          mem_obj->size);
      if (b==NULL)
        return CL_MEM_OBJECT_ALLOCATION_FAILURE;

      mem_obj->shared_mem_allocation_owner = device;
    }

  /* use this dev mem allocation as host ptr */
  if ((flags & CL_MEM_ALLOC_HOST_PTR) && (mem_obj->mem_host_ptr == NULL))
    mem_obj->mem_host_ptr = b;

  if (flags & CL_MEM_COPY_HOST_PTR)
    {
      POCL_MSG_PRINT_MEMORY ("COPY_HOST_PTR\n");
      // mem_host_ptr must be non-NULL
      assert(host_ptr != NULL);
      memcpy (b, host_ptr, mem_obj->size);
    }

  mem_obj->device_ptrs[device->dev_id].mem_ptr = b;

  return CL_SUCCESS;
}


void
pocl_basic_free (cl_device_id device, cl_mem memobj)
{
  cl_mem_flags flags = memobj->flags;

  /* allocation owner executes freeing */
  if (flags & CL_MEM_USE_HOST_PTR ||
      memobj->shared_mem_allocation_owner != device)
    return;

  void* ptr = memobj->device_ptrs[device->dev_id].mem_ptr;
  size_t size = memobj->size;

  if (memobj->flags | CL_MEM_ALLOC_HOST_PTR)
    memobj->mem_host_ptr = NULL;
  pocl_free_global_mem(device, ptr, size);
}

void
pocl_basic_read (void *data,
                 void *__restrict__ host_ptr,
                 pocl_mem_identifier * src_mem_id,
                 cl_mem src_buf,
                 size_t offset, size_t size)
{
  void *__restrict__ device_ptr = src_mem_id->mem_ptr;
  if (host_ptr == device_ptr)
    return;

  memcpy (host_ptr, (char *)device_ptr + offset, size);
}

void
pocl_basic_write (void *data,
                  const void *__restrict__  host_ptr,
                  pocl_mem_identifier * dst_mem_id,
                  cl_mem dst_buf,
                  size_t offset, size_t size)
{
  void *__restrict__ device_ptr = dst_mem_id->mem_ptr;
  if (host_ptr == device_ptr)
    return;

  memcpy ((char *)device_ptr + offset, host_ptr, size);
}

void
pocl_basic_run (void *data, _cl_command_node *cmd)
{
  struct data *d;
  struct pocl_argument *al;
  size_t x, y, z;
  unsigned i;
  cl_kernel kernel = cmd->command.run.kernel;
  pocl_kernel_metadata_t *meta = kernel->meta;
  struct pocl_context *pc = &cmd->command.run.pc;

  assert (data != NULL);
  d = (struct data *) data;

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
              cl_mem m = (*(cl_mem *)(al->value));
              void *ptr = m->device_ptrs[cmd->device->dev_id].mem_ptr;
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
pocl_basic_run_native (void *data, _cl_command_node *cmd)
{
  cmd->command.native.user_func(cmd->command.native.args);
}

void
pocl_basic_copy (void *data,
                 pocl_mem_identifier * dst_mem_id,
                 cl_mem dst_buf,
                 pocl_mem_identifier * src_mem_id,
                 cl_mem src_buf,
                 size_t dst_offset,
                 size_t src_offset,
                 size_t size)
{
  char *__restrict__ src_ptr = src_mem_id->mem_ptr;
  char *__restrict__ dst_ptr = dst_mem_id->mem_ptr;
  if ((src_ptr + src_offset) == (dst_ptr + dst_offset))
    return;

  memcpy (dst_ptr + dst_offset, src_ptr + src_offset, size);
}

void
pocl_basic_copy_rect (void *data,
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

  void *__restrict__ src_ptr = src_mem_id->mem_ptr;
  void *__restrict__ dst_ptr = dst_mem_id->mem_ptr;
  char const *__restrict const adjusted_src_ptr = 
    (char const*)src_ptr +
    src_origin[0] + src_row_pitch * src_origin[1] + src_slice_pitch * src_origin[2];
  char *__restrict__ const adjusted_dst_ptr = 
    (char*)dst_ptr +
    dst_origin[0] + dst_row_pitch * dst_origin[1] + dst_slice_pitch * dst_origin[2];

  POCL_MSG_PRINT_MEMORY (
      "BASIC COPY RECT \n"
      "SRC %p DST %p SIZE %zu\n"
      "src origin %u %u %u dst origin %u %u %u \n"
      "src_row_pitch %lu src_slice pitch %lu\n"
      "dst_row_pitch %lu dst_slice_pitch %lu\n"
      "reg[0] %lu reg[1] %lu reg[2] %lu\n",
      adjusted_src_ptr, adjusted_dst_ptr,
      region[0] * region[1] * region[2],
      (unsigned)src_origin[0], (unsigned)src_origin[1],
      (unsigned)src_origin[2], (unsigned)dst_origin[0],
      (unsigned)dst_origin[1], (unsigned)dst_origin[2],
      (unsigned long)src_row_pitch, (unsigned long)src_slice_pitch,
      (unsigned long)dst_row_pitch, (unsigned long)dst_slice_pitch,
      (unsigned long)region[0], (unsigned long)region[1], (unsigned long)region[2]);

  size_t j, k;

  /* TODO: handle overlaping regions */
  if ((src_row_pitch == dst_row_pitch && dst_row_pitch == region[0])
      && (src_slice_pitch == dst_slice_pitch && dst_slice_pitch == (region[1]*region[0])))
    {
      memcpy (adjusted_dst_ptr, adjusted_src_ptr,
              region[2] * region[1] * region[0]);
    }
  else
    {
      for (k = 0; k < region[2]; ++k)
        for (j = 0; j < region[1]; ++j)
          memcpy (adjusted_dst_ptr + dst_row_pitch * j + dst_slice_pitch * k,
                  adjusted_src_ptr + src_row_pitch * j + src_slice_pitch * k,
                  region[0]);
    }
}

void
pocl_basic_write_rect (void *data,
                       const void *__restrict__ const host_ptr,
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
  void *__restrict__ device_ptr = dst_mem_id->mem_ptr;

  char *__restrict const adjusted_device_ptr
      = (char *)device_ptr + buffer_origin[0]
        + buffer_row_pitch * buffer_origin[1]
        + buffer_slice_pitch * buffer_origin[2];
  char const *__restrict__ const adjusted_host_ptr = 
    (char const*)host_ptr +
    host_origin[0] + host_row_pitch * host_origin[1] + host_slice_pitch * host_origin[2];

  POCL_MSG_PRINT_MEMORY (
      "BASIC WRITE RECT \n"
      "SRC HOST %p DST DEV %p SIZE %zu\n"
      "borigin %u %u %u horigin %u %u %u \n"
      "row_pitch %lu slice pitch \n"
      "%lu host_row_pitch %lu host_slice_pitch %lu\n"
      "reg[0] %lu reg[1] %lu reg[2] %lu\n",
      adjusted_host_ptr, adjusted_device_ptr,
      region[0] * region[1] * region[2],
      (unsigned)buffer_origin[0], (unsigned)buffer_origin[1],
      (unsigned)buffer_origin[2], (unsigned)host_origin[0],
      (unsigned)host_origin[1], (unsigned)host_origin[2],
      (unsigned long)buffer_row_pitch, (unsigned long)buffer_slice_pitch,
      (unsigned long)host_row_pitch, (unsigned long)host_slice_pitch,
      (unsigned long)region[0], (unsigned long)region[1], (unsigned long)region[2]);

  size_t j, k;

  /* TODO: handle overlaping regions */
  if ((buffer_row_pitch == host_row_pitch
            && host_row_pitch == region[0])
      && (buffer_slice_pitch == host_slice_pitch
            && host_slice_pitch == (region[1]*region[0])))
    {
      memcpy (adjusted_device_ptr,
              adjusted_host_ptr,
              region[2] * region[1] * region[0]);
    }
  else
    {
      for (k = 0; k < region[2]; ++k)
        for (j = 0; j < region[1]; ++j)
          memcpy (adjusted_device_ptr
                    + buffer_row_pitch * j
                    + buffer_slice_pitch * k,
                  adjusted_host_ptr
                    + host_row_pitch * j
                    + host_slice_pitch * k,
                  region[0]);
    }
}

void
pocl_basic_read_rect (void *data,
                      void *__restrict__ const host_ptr,
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
  void *__restrict__ device_ptr = src_mem_id->mem_ptr;

  char const *__restrict const adjusted_device_ptr = 
    (char const*)device_ptr +
    buffer_origin[2] * buffer_slice_pitch + buffer_origin[1] * buffer_row_pitch + buffer_origin[0];
  char *__restrict__ const adjusted_host_ptr = 
    (char*)host_ptr +
    host_origin[2] * host_slice_pitch + host_origin[1] * host_row_pitch + host_origin[0];

  POCL_MSG_PRINT_MEMORY (
      "BASIC READ RECT \n"
      "SRC DEV %p DST HOST %p SIZE %zu\n"
      "borigin %u %u %u horigin %u %u %u row_pitch %lu slice pitch "
      "%lu host_row_pitch %lu host_slice_pitch %lu\n"
      "reg[0] %lu reg[1] %lu reg[2] %lu\n",
      adjusted_device_ptr, adjusted_host_ptr,
      region[0] * region[1] * region[2],
      (unsigned)buffer_origin[0], (unsigned)buffer_origin[1],
      (unsigned)buffer_origin[2], (unsigned)host_origin[0],
      (unsigned)host_origin[1], (unsigned)host_origin[2],
      (unsigned long)buffer_row_pitch, (unsigned long)buffer_slice_pitch,
      (unsigned long)host_row_pitch, (unsigned long)host_slice_pitch,
      (unsigned long)region[0], (unsigned long)region[1], (unsigned long)region[2]);

  size_t j, k;

  /* TODO: handle overlaping regions */
  if ((buffer_row_pitch == host_row_pitch
            && host_row_pitch == region[0])
      && (buffer_slice_pitch == host_slice_pitch
            && host_slice_pitch == (region[1]*region[0])))
    {
      memcpy (adjusted_host_ptr,
              adjusted_device_ptr,
              region[2] * region[1] * region[0]);
    }
  else
    {
      for (k = 0; k < region[2]; ++k)
        for (j = 0; j < region[1]; ++j)
          memcpy (adjusted_host_ptr
                    + host_row_pitch * j
                    + host_slice_pitch * k,
                  adjusted_device_ptr
                    + buffer_row_pitch * j
                    + buffer_slice_pitch * k,
                  region[0]);
    }
}

void
pocl_basic_memfill (void *data, pocl_mem_identifier *dst_mem_id,
                    cl_mem dst_buf, size_t size, size_t offset,
                    const void *__restrict__ pattern, size_t pattern_size)
{
  void *__restrict__ ptr = dst_mem_id->mem_ptr;
  size_t i;
  unsigned j;

  /* memfill size is in bytes, we wanto make it into elements */
  size /= pattern_size;
  offset /= pattern_size;

  switch (pattern_size)
    {
    case 1:
      {
      uint8_t * p = (uint8_t*)ptr + offset;
      for (i = 0; i < size; i++)
        p[i] = *(uint8_t*)pattern;
      }
      break;
    case 2:
      {
      uint16_t * p = (uint16_t*)ptr + offset;
      for (i = 0; i < size; i++)
        p[i] = *(uint16_t*)pattern;
      }
      break;
    case 4:
      {
      uint32_t * p = (uint32_t*)ptr + offset;
      for (i = 0; i < size; i++)
        p[i] = *(uint32_t*)pattern;
      }
      break;
    case 8:
      {
      uint64_t * p = (uint64_t*)ptr + offset;
      for (i = 0; i < size; i++)
        p[i] = *(uint64_t*)pattern;
      }
      break;
    case 16:
      {
      uint64_t * p = (uint64_t*)ptr + (offset << 1);
      for (i = 0; i < size; i++)
        for (j = 0; j < 2; j++)
          p[(i<<1) + j] = *((uint64_t*)pattern + j);
      }
      break;
    case 32:
      {
      uint64_t * p = (uint64_t*)ptr + (offset << 2);
      for (i = 0; i < size; i++)
        for (j = 0; j < 4; j++)
          p[(i<<2) + j] = *((uint64_t*)pattern + j);
      }
      break;
    case 64:
      {
      uint64_t * p = (uint64_t*)ptr + (offset << 3);
      for (i = 0; i < size; i++)
        for (j = 0; j < 8; j++)
          p[(i<<3) + j] = *((uint64_t*)pattern + j);
      }
      break;
    case 128:
      {
      uint64_t * p = (uint64_t*)ptr + (offset << 4);
      for (i = 0; i < size; i++)
        for (j = 0; j < 16; j++)
          p[(i<<4) + j] = *((uint64_t*)pattern + j);
      }
      break;
    default:
      assert (0 && "Invalid pattern size");
      break;
    }
}

cl_int
pocl_basic_map_mem (void *data,
                    pocl_mem_identifier * src_mem_id,
                    cl_mem src_buf,
                    mem_mapping_t *map)
{
  void *__restrict__ src_device_ptr = src_mem_id->mem_ptr;
  void *host_ptr = map->host_ptr;
  size_t offset = map->offset;
  size_t size = map->size;

  if (host_ptr == NULL)
    {
      map->host_ptr = ((char *)src_device_ptr + offset);
      return CL_SUCCESS;
    }

  if (map->map_flags & CL_MAP_WRITE_INVALIDATE_REGION)
    return CL_SUCCESS;

  /* If the buffer was allocated with CL_MEM_ALLOC_HOST_PTR |
   * CL_MEM_COPY_HOST_PTR,
   * the dst_host_ptr is not the same memory as pocl's device_ptrs[], and we
   * need to copy pocl's buffer content back to dst_host_ptr. */
  if (host_ptr != (src_device_ptr + offset))
    {
      POCL_MSG_PRINT_MEMORY ("device: MAP memcpy() "
                             "src_device_ptr %p + offset %zu"
                             "to dst_host_ptr %p\n",
                             src_device_ptr, offset, host_ptr);
      memcpy ((char *)host_ptr, (char *)src_device_ptr + offset, size);
    }
  return CL_SUCCESS;
}

cl_int
pocl_basic_unmap_mem(void *data,
                     pocl_mem_identifier * dst_mem_id,
                     cl_mem dst_buf,
                     mem_mapping_t *map)
{
  void *__restrict__ dst_device_ptr = dst_mem_id->mem_ptr;
  /* it could be CL_MAP_READ | CL_MAP_WRITE(..invalidate) which has to be
   * handled like a write */
  if (map->map_flags == CL_MAP_READ)
    return CL_SUCCESS;

  void *host_ptr = map->host_ptr;
  assert (host_ptr != NULL);
  size_t offset = map->offset;
  size_t size = map->size;

  /* If the buffer was allocated with CL_MEM_ALLOC_src_host_ptr |
   * CL_MEM_COPY_src_host_ptr,
   * the src_host_ptr is not the same memory as pocl's device_ptrs[], and we
   * need to copy src_host_ptr content back to  pocl's device_ptrs[]. */
  if (host_ptr != (dst_device_ptr + offset))
    {
      POCL_MSG_PRINT_MEMORY ("device: UNMAP memcpy() "
                             "host_ptr %p to buf_ptr %p + offset %zu\n",
                             host_ptr, dst_device_ptr, offset);
      memcpy ((char *)dst_device_ptr + offset, (char *)host_ptr, size);
    }
  return CL_SUCCESS;
}

cl_int
pocl_basic_uninit (unsigned j, cl_device_id device)
{
  struct data *d = (struct data*)device->data;
  POCL_DESTROY_LOCK (d->cq_lock);
  pocl_aligned_free (d->printf_buffer);
  POCL_MEM_FREE(d);
  device->data = NULL;
  return CL_SUCCESS;
}

cl_int
pocl_basic_reinit (unsigned j, cl_device_id device)
{
  struct data *d = (struct data *)calloc (1, sizeof (struct data));
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


static void basic_command_scheduler (struct data *d) 
{
  _cl_command_node *node;
  
  /* execute commands from ready list */
  while ((node = d->ready_list))
    {
      assert (pocl_command_is_ready(node->event));
      assert (node->event->status == CL_SUBMITTED);
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
  struct data *d = node->device->data;

  if (node != NULL && node->type == CL_COMMAND_NDRANGE_KERNEL)
    pocl_check_kernel_dlhandle_cache (node, 1, 1);

  node->ready = 1;
  POCL_LOCK (d->cq_lock);
  pocl_command_push(node, &d->ready_list, &d->command_list);

  POCL_UNLOCK_OBJ (node->event);
  basic_command_scheduler (d);
  POCL_UNLOCK (d->cq_lock);

  return;
}

void pocl_basic_flush (cl_device_id device, cl_command_queue cq)
{
  struct data *d = (struct data*)device->data;

  POCL_LOCK (d->cq_lock);
  basic_command_scheduler (d);
  POCL_UNLOCK (d->cq_lock);
}

void
pocl_basic_join (cl_device_id device, cl_command_queue cq)
{
  struct data *d = (struct data*)device->data;

  POCL_LOCK (d->cq_lock);
  basic_command_scheduler (d);
  POCL_UNLOCK (d->cq_lock);

  return;
}

void
pocl_basic_notify (cl_device_id device, cl_event event, cl_event finished)
{
  struct data *d = (struct data*)device->data;
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
          basic_command_scheduler (d);
          POCL_UNLOCK (d->cq_lock);
        }
      return;
    }
}

void
pocl_basic_compile_kernel (_cl_command_node *cmd, cl_kernel kernel,
                           cl_device_id device, int specialize)
{
  if (cmd != NULL && cmd->type == CL_COMMAND_NDRANGE_KERNEL)
    pocl_check_kernel_dlhandle_cache (cmd, 0, specialize);
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
      " BASIC COPY IMAGE RECT \n"
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

  pocl_basic_copy_rect(data,
                       dst_mem_id,
                       NULL,
                       src_mem_id,
                       NULL,
                       adj_dst_origin,
                       adj_src_origin,
                       adj_region,
                       dst_image->image_row_pitch,
                       dst_image->image_slice_pitch,
                       src_image->image_row_pitch,
                       src_image->image_slice_pitch);

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
      "BASIC WRITE IMAGE RECT \n"
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

  pocl_basic_write_rect (data,
                         ptr,
                         dst_mem_id,
                         NULL,
                         adj_origin,
                         zero_origin,
                         adj_region,
                         dst_image->image_row_pitch,
                         dst_image->image_slice_pitch,
                         src_row_pitch,
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
      "BASIC READ IMAGE RECT \n"
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

  pocl_basic_read_rect (data,
                        ptr,
                        src_mem_id,
                        NULL,
                        adj_origin,
                        zero_origin,
                        adj_region,
                        src_image->image_row_pitch,
                        src_image->image_slice_pitch,
                        dst_row_pitch,
                        dst_slice_pitch);
  return CL_SUCCESS;
}


cl_int pocl_basic_map_image (void *data,
                             pocl_mem_identifier *mem_id,
                             cl_mem src_image,
                             mem_mapping_t *map)
{
  if (map->host_ptr == NULL)
    {
      map->host_ptr = (char *)mem_id->mem_ptr + map->offset;
      return CL_SUCCESS;
    }

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
                       const size_t *region,
                       const void *__restrict__ fill_pixel, size_t pixel_size)
{
   POCL_MSG_PRINT_MEMORY ("BASIC / FILL IMAGE \n"
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

void
pocl_basic_svm_copy (cl_device_id dev, void *__restrict__ dst,
                     const void *__restrict__ src, size_t size)
{
  memcpy (dst, src, size);
}

void
pocl_basic_svm_fill (cl_device_id dev, void *__restrict__ svm_ptr, size_t size,
                     void *__restrict__ pattern, size_t pattern_size)
{
  pocl_mem_identifier temp;
  temp.mem_ptr = svm_ptr;
  pocl_basic_memfill (dev->data, &temp, NULL, size, 0, pattern, pattern_size);
}
