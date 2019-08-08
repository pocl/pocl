/* basic.c - a minimalistic pocl device driver layer implementation

   Copyright (c) 2011-2013 Universidad Rey Juan Carlos and
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

#include "config.h"
#include "config2.h"
#include "basic.h"
#include "cpuinfo.h"
#include "topology/pocl_topology.h"
#include "common.h"
#include "utlist.h"
#include "devices.h"
#include "pocl_util.h"

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
#include "pocl_runtime_config.h"

#ifdef ENABLE_LLVM
#include "pocl_llvm.h"
#endif

#ifndef HAVE_LIBDL
#error Basic driver requires DL library
#endif

/* default WG size in each dimension & total WG size.
 * this should be reasonable for CPU */
#define DEFAULT_WG_SIZE 4096

struct data {
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

static const cl_image_format supported_image_formats[] = {
    { CL_A, CL_SNORM_INT8 },
    { CL_A, CL_SNORM_INT16 },
    { CL_A, CL_UNORM_INT8 },
    { CL_A, CL_UNORM_INT16 },
    { CL_A, CL_SIGNED_INT8 },
    { CL_A, CL_SIGNED_INT16 }, 
    { CL_A, CL_SIGNED_INT32 },
    { CL_A, CL_UNSIGNED_INT8 }, 
    { CL_A, CL_UNSIGNED_INT16 },
    { CL_A, CL_UNSIGNED_INT32 }, 
    { CL_A, CL_FLOAT },
    { CL_RGBA, CL_SNORM_INT8 },
    { CL_RGBA, CL_SNORM_INT16 },
    { CL_RGBA, CL_UNORM_INT8 },
    { CL_RGBA, CL_UNORM_INT16 },
    { CL_RGBA, CL_SIGNED_INT8 },
    { CL_RGBA, CL_SIGNED_INT16 }, 
    { CL_RGBA, CL_SIGNED_INT32 },
    { CL_RGBA, CL_UNSIGNED_INT8 }, 
    { CL_RGBA, CL_UNSIGNED_INT16 },
    { CL_RGBA, CL_UNSIGNED_INT32 }, 
    { CL_RGBA, CL_HALF_FLOAT },
    { CL_RGBA, CL_FLOAT },
    { CL_ARGB, CL_SNORM_INT8 },
    { CL_ARGB, CL_UNORM_INT8 },
    { CL_ARGB, CL_SIGNED_INT8 },
    { CL_ARGB, CL_UNSIGNED_INT8 }, 
    { CL_BGRA, CL_SNORM_INT8 },
    { CL_BGRA, CL_UNORM_INT8 },
    { CL_BGRA, CL_SIGNED_INT8 },
    { CL_BGRA, CL_UNSIGNED_INT8 }
 };

static pocl_global_mem_t pocl_basic_global_memory;

void
pocl_basic_init_device_ops(struct pocl_device_ops *ops)
{
  ops->device_name = "basic";

  ops->probe = pocl_basic_probe;
  ops->uninit = pocl_basic_uninit;
  ops->reinit = pocl_basic_reinit;
  ops->init = pocl_basic_init;
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

  /* no need to implement these two as they're noop
   * and pocl_exec_command takes care of it */
  ops->svm_map = NULL;
  ops->svm_unmap = NULL;
  ops->svm_copy = pocl_basic_svm_copy;
  ops->svm_fill = pocl_basic_svm_fill;

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

static cl_device_partition_property basic_partition_properties[1] = { 0 };


static const char *final_ld_flags[] =
  {"-nostartfiles", HOST_LD_FLAGS_ARRAY, NULL};

void
pocl_init_cpu_device_infos (cl_device_id dev)
{
  size_t i;
  dev->type = CL_DEVICE_TYPE_CPU;
  dev->max_work_item_dimensions = 3;
  dev->final_linkage_flags = final_ld_flags;

  SETUP_DEVICE_CL_VERSION(HOST_DEVICE_CL_VERSION_MAJOR, HOST_DEVICE_CL_VERSION_MINOR)
  /*
    The hard restriction will be the context data which is
    stored in stack that can be as small as 8K in Linux.
    Thus, there should be enough work-items alive to fill up
    the SIMD lanes times the vector units, but not more than
    that to avoid stack overflow and cache trashing.
  */
  int max_wg
      = pocl_get_int_option ("POCL_MAX_WORK_GROUP_SIZE", DEFAULT_WG_SIZE);
  assert (max_wg > 0);
  max_wg = min (max_wg, DEFAULT_WG_SIZE);
  if (max_wg < 0)
    max_wg = DEFAULT_WG_SIZE;

  dev->max_work_item_sizes[0] = dev->max_work_item_sizes[1]
      = dev->max_work_item_sizes[2] = dev->max_work_group_size = max_wg;

  dev->preferred_wg_size_multiple = 8;
#ifdef ENABLE_LLVM
  cpu_setup_vector_widths (dev);
#else
  dev->preferred_vector_width_char = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_CHAR;
  dev->preferred_vector_width_short = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT;
  dev->preferred_vector_width_int = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_INT;
  dev->preferred_vector_width_long = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_LONG;
  dev->preferred_vector_width_float = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_FLOAT;
  /* TODO: figure out what the difference between preferred and native widths are */
  dev->native_vector_width_char = POCL_DEVICES_NATIVE_VECTOR_WIDTH_CHAR;
  dev->native_vector_width_short = POCL_DEVICES_NATIVE_VECTOR_WIDTH_SHORT;
  dev->native_vector_width_int = POCL_DEVICES_NATIVE_VECTOR_WIDTH_INT;
  dev->native_vector_width_long = POCL_DEVICES_NATIVE_VECTOR_WIDTH_LONG;
  dev->native_vector_width_float = POCL_DEVICES_NATIVE_VECTOR_WIDTH_FLOAT;

#ifdef _CL_DISABLE_DOUBLE
  dev->native_vector_width_double = 0;
  dev->preferred_vector_width_double = 0;
#else
  dev->native_vector_width_double = POCL_DEVICES_NATIVE_VECTOR_WIDTH_DOUBLE;
  dev->preferred_vector_width_double = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_DOUBLE;
#endif
#ifdef _CL_DISABLE_HALF
  dev->preferred_vector_width_half = 0;
  dev->native_vector_width_half = 0;
#else
  dev->preferred_vector_width_half = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_HALF;
  dev->native_vector_width_half = POCL_DEVICES_NATIVE_VECTOR_WIDTH_HALF;
#endif

#endif

  dev->grid_width_specialization_limit = USHRT_MAX;
  dev->address_bits = HOST_DEVICE_ADDRESS_BITS;
  dev->image_support = CL_TRUE;
  /* Use the minimum values until we get a more sensible upper limit from
     somewhere. */
  dev->max_read_image_args = dev->max_write_image_args
      = dev->max_read_write_image_args = 128;
  dev->image2d_max_width = dev->image2d_max_height = 8192;
  dev->image3d_max_width = dev->image3d_max_height = dev->image3d_max_depth = 2048;
  dev->max_samplers = 16;

  for (i = 0; i < NUM_OPENCL_IMAGE_TYPES; ++i)
    {
      dev->num_image_formats[i]
          = sizeof (supported_image_formats) / sizeof (cl_image_format);
      dev->image_formats[i] = supported_image_formats;
    }

  dev->image_max_buffer_size = 65536;
  dev->image_max_array_size = 2048;
  dev->max_constant_args = 8;
  dev->max_parameter_size = 1024;
  dev->min_data_type_align_size = MAX_EXTENDED_ALIGNMENT;
  dev->mem_base_addr_align = MAX_EXTENDED_ALIGNMENT;
  dev->half_fp_config = 0;
  dev->single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN;
#ifdef __x86_64__
  dev->single_fp_config |= (CL_FP_DENORM | CL_FP_ROUND_TO_INF
                            | CL_FP_ROUND_TO_ZERO
                            | CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT);
#ifdef ENABLE_LLVM
  if (cpu_has_fma())
    dev->single_fp_config |= CL_FP_FMA;
#endif
#endif

#ifdef _CL_DISABLE_DOUBLE
  dev->double_fp_config = 0;
#else
  /* TODO: all of these are the minimum mandated, but not all CPUs may actually
   * support all of them. */
  dev->double_fp_config = CL_FP_FMA | CL_FP_ROUND_TO_NEAREST
                          | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF
                          | CL_FP_INF_NAN | CL_FP_DENORM;
  /* this is a workaround for issue 28 in https://github.com/Oblomov/clinfo
   * https://github.com/Oblomov/clinfo/issues/28 */
  dev->double_fp_config |= CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT;
#endif

  dev->max_constant_buffer_size = 0;
  dev->max_constant_args = 8;
  dev->local_mem_type = CL_GLOBAL;
  dev->local_mem_size = 0;
  dev->error_correction_support = CL_FALSE;
  dev->host_unified_memory = CL_TRUE;

  dev->profiling_timer_resolution = pocl_timer_resolution;

  dev->endian_little = !(WORDS_BIGENDIAN);
  dev->available = CL_TRUE;
  dev->compiler_available = CL_TRUE;
  dev->spmd = CL_FALSE;
  dev->arg_buffer_launcher = CL_FALSE;
  dev->workgroup_pass = CL_TRUE;
  dev->execution_capabilities = CL_EXEC_KERNEL | CL_EXEC_NATIVE_KERNEL;
  dev->platform = 0;

  dev->parent_device = NULL;
  /* These two are only used for subdevices.
   * Each subdevice has these two setup when created.
   * The subdevice will then use these CUs:
   *  [start, start+1, ..., start+count-1]
   * this may not work with more complicated partitioning schemes,
   * but is good enough for now. */
  dev->core_start = 0;
  dev->core_count = 0;
  /* basic does not support partitioning */
  dev->max_sub_devices = 1;
  dev->num_partition_properties = 1;
  dev->partition_properties = basic_partition_properties;
  dev->num_partition_types = 0;
  dev->partition_type = NULL;

  dev->device_side_printf = 1;
  dev->printf_buffer_size = 16 * 1024 * 1024;

  dev->vendor = "pocl";
  dev->profile = "FULL_PROFILE";
  /* Note: The specification describes identifiers being delimited by
     only a single space character. Some programs that check the device's
     extension  string assume this rule. Future extension additions should
     ensure that there is no more than a single space between
     identifiers. */

  dev->global_as_id = dev->local_as_id = dev->constant_as_id = 0;

  /* OpenCL 2.0 properties */
  dev->svm_caps = CL_DEVICE_SVM_COARSE_GRAIN_BUFFER
                  | CL_DEVICE_SVM_FINE_GRAIN_BUFFER
                  | CL_DEVICE_SVM_ATOMICS;
  /* TODO these are minimums, figure out whats a reasonable value */
  dev->max_events = 1024;
  dev->max_queues = 1;
  dev->max_pipe_args = 16;
  dev->max_pipe_active_res = 1;
  dev->max_pipe_packet_size = 1024;
  dev->dev_queue_pref_size = 16 * 1024;
  dev->dev_queue_max_size = 256 * 1024;
  dev->on_dev_queue_props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
                               | CL_QUEUE_PROFILING_ENABLE;
  dev->on_host_queue_props = CL_QUEUE_PROFILING_ENABLE;
  dev->extensions = HOST_DEVICE_EXTENSIONS;
  dev->has_64bit_long = 1;
  dev->autolocals_to_args = 1;
  dev->device_alloca_locals = 0;

#ifdef ENABLE_LLVM

  dev->llvm_target_triplet = OCL_KERNEL_TARGET;
#ifdef HOST_CPU_FORCED
  dev->llvm_cpu = OCL_KERNEL_TARGET_CPU;
#else
  dev->llvm_cpu = get_llvm_cpu_name ();
#endif

  dev->spirv_version = "SPIR-V_1.2";
#else /* No compiler, no CPU info */
  dev->llvm_cpu = NULL;
  dev->llvm_target_triplet = "";
#endif

}

void
pocl_init_cpu_global_mem (cl_device_id device)
{

  if (pocl_basic_global_memory.size == 0)
    {
      pocl_basic_global_memory.currently_allocated = 0;
      pocl_basic_global_memory.is_svm = 1;
      pocl_basic_global_memory.max_ever_allocated = 0;
      pocl_basic_global_memory.svm_list = NULL;

      POCL_INIT_LOCK (pocl_basic_global_memory.lock);

      pocl_basic_global_memory.alloc_mem_obj = pocl_basic_alloc_mem_obj;
      pocl_basic_global_memory.free_mem_obj = pocl_basic_free_mem_obj;
      pocl_basic_global_memory.svm_alloc = pocl_basic_svm_alloc;
      pocl_basic_global_memory.svm_free = pocl_basic_svm_free;
      pocl_basic_global_memory.svm_pointer_to_clmem
          = pocl_basic_svm_pointer_to_clmem;

      pocl_topology_detect_globalmem_info (&pocl_basic_global_memory);
    }

  if (!pocl_svm_registered)
    {
      pocl_basic_global_memory.id = pocl_svm_id = pocl_num_global_mem++;
      pocl_svm_registered = 1;
    }

  device->global_mem_id = pocl_svm_id;
  device->global_memory = &pocl_basic_global_memory;
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
pocl_basic_init (unsigned j, cl_device_id device, const char *parameters)
{
  struct data *d;
  int err;

  pocl_init_dlhandle_cache ();

  pocl_init_cpu_global_mem (device);

  d = (struct data *) calloc (1, sizeof (struct data));
  if (d == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  d->current_kernel = NULL;
  device->data = d;

  pocl_init_cpu_device_infos (device);

  /* hwloc probes OpenCL device info at its initialization in case
     the OpenCL extension is enabled. This causes to printout 
     an unimplemented property error because hwloc is used to
     initialize global_mem_size which it is not yet. Just put 
     a nonzero there for now. */
  err = pocl_topology_detect_device_info(device);
  if (err)
    return CL_INVALID_DEVICE;

  POCL_INIT_LOCK (d->cq_lock);

  assert (device->printf_buffer_size > 0);
  d->printf_buffer = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT,
                                          device->printf_buffer_size);
  assert (d->printf_buffer != NULL);

  pocl_cpuinfo_detect_device_info(device);
  pocl_set_buffer_image_limits(device);

  /* in case hwloc doesn't provide a PCI ID, let's generate
     a vendor id that hopefully is unique across vendors. */
  const char *magic = "pocl";
  if (device->vendor_id == 0)
    device->vendor_id =
      magic[0] | magic[1] << 8 | magic[2] << 16 | magic[3] << 24;

  device->vendor_id += j;

  /* The basic driver represents only one "compute unit" as
     it doesn't exploit multiple hardware threads. Multiple
     basic devices can be still used for task level parallelism 
     using multiple OpenCL devices. */
  device->max_compute_units = 1;

  return CL_SUCCESS;
}

int
pocl_basic_alloc_mem_obj (pocl_global_mem_t *gmem, cl_mem mem,
                          pocl_mem_identifier *p, void *host_ptr)
{
  assert (p->mem_ptr == NULL);
  int ret = CL_MEM_OBJECT_ALLOCATION_FAILURE;

  POCL_LOCK (gmem->lock);
  p->device_size = mem->size;
  if (!pocl_global_mem_can_allocate (gmem, p))
    goto ERROR;

  /* In case USE_HOST_PTR is true, don't allocate anything */
  if (mem->flags & CL_MEM_USE_HOST_PTR)
    {
      assert (mem->mem_host_ptr != NULL);
      p->mem_ptr = mem->mem_host_ptr;
      POCL_MSG_PRINT_MEMORY ("Basic device USE_HOST_PTR %p / size %zu \n",
                             p->mem_ptr, mem->size);
      p->host_ptr = NULL;
      goto END;
    }

  /* ... else allocate. */
  p->mem_ptr = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, p->device_size);
  if (p->mem_ptr == NULL)
    goto ERROR;

  p->host_ptr = p->mem_ptr;
  if (mem->mem_host_ptr == NULL)
    mem->mem_host_ptr = p->mem_ptr;

  if (mem->flags & CL_MEM_COPY_HOST_PTR)
    {
      assert (host_ptr != NULL);
      memcpy (p->mem_ptr, host_ptr, mem->size);
    }

  POCL_MSG_PRINT_MEMORY ("Basic device ALLOC %p / size %zu \n", p->mem_ptr,
                         mem->size);
END:
  pocl_global_mem_allocated (gmem, p);
  ret = CL_SUCCESS;
ERROR:
  POCL_UNLOCK (gmem->lock);
  return ret;
}

int
pocl_basic_free_mem_obj (pocl_global_mem_t *gmem, cl_mem mem,
                         pocl_mem_identifier *p)
{
  int ret = CL_SUCCESS;
  POCL_LOCK (gmem->lock);

  /* if we didn't allocate anything, return */
  if (p->host_ptr == NULL)
    {
      p->mem_ptr = NULL;
      goto END;
    }

  if (p->host_ptr == mem->mem_host_ptr)
    mem->mem_host_ptr = NULL;

  pocl_aligned_free (p->mem_ptr);
  p->mem_ptr = NULL;
  p->host_ptr = NULL;

END:
  pocl_global_mem_freed (gmem, p);
  POCL_UNLOCK (gmem->lock);
  return ret;
}

void
pocl_basic_read (void *data, void *__restrict__ host_ptr,
                 pocl_mem_identifier *src_mem_id, cl_mem src_buf,
                 size_t offset, size_t size)
{
  void *__restrict__ device_ptr = src_mem_id->mem_ptr;
  if (host_ptr == device_ptr)
    return;

  memcpy (host_ptr, (char *)device_ptr + offset, size);
}

void
pocl_basic_write (void *data, const void *__restrict__ host_ptr,
                  pocl_mem_identifier *dst_mem_id, cl_mem dst_buf,
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
              void *ptr = m->gmem_ptrs[cmd->device->global_mem_id].mem_ptr;
              *(void **)arguments[i] = (char *)ptr + al->offset;
            }
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
        {
          dev_image_t di;
          fill_dev_image_t (&di, al, cmd->device);

          void *devptr = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT,
                                              sizeof (dev_image_t));
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = devptr;
          memcpy (devptr, &di, sizeof (dev_image_t));
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)
        {
          dev_sampler_t ds;
          fill_dev_sampler_t(&ds, al);
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
  cl_event ev = cmd->event;
  cl_device_id dev = cmd->device;
  size_t i;
  for (i = 0; i < ev->num_buffers; i++)
    {
      void *arg_loc = cmd->command.native.arg_locs[i];
      void *buf = ev->mem_objs[i]->gmem_ptrs[dev->global_mem_id].mem_ptr;
      if (dev->address_bits == 32)
        *((uint32_t *)arg_loc) = (uint32_t) (((uintptr_t)buf) & 0xFFFFFFFF);
      else
        *((uint64_t *)arg_loc) = (uint64_t) (uintptr_t)buf;
    }

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
pocl_basic_map_mem (void *data, pocl_mem_identifier *src_mem_id,
                    cl_mem src_buf, mem_mapping_t *map)
{
  void *__restrict__ src_device_ptr = src_mem_id->mem_ptr;
  void *host_ptr = map->host_ptr;
  size_t offset = map->offset;
  size_t size = map->size;

  if (map->map_flags & CL_MAP_WRITE_INVALIDATE_REGION)
    return CL_SUCCESS;

  // TODO can this actually happen now ?
  /* If the buffer was allocated with CL_MEM_ALLOC_HOST_PTR |
   * CL_MEM_COPY_HOST_PTR,
   * the dst_host_ptr is not the same memory as pocl's gmem_ptrs[], and we
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

  // TODO can this actually happen now ?
  /* If the buffer was allocated with CL_MEM_ALLOC_src_host_ptr |
   * CL_MEM_COPY_src_host_ptr,
   * the src_host_ptr is not the same memory as pocl's gmem_ptrs[], and we
   * need to copy src_host_ptr content back to  pocl's gmem_ptrs[]. */
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

  node->was_submitted = 1;
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

  if (!node->was_submitted)
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

int
pocl_basic_svm_free (pocl_global_mem_t *gmem, void *svm_ptr)
{
  pocl_gmem_allocation_t *found = NULL, *tmp;
  POCL_LOCK (gmem->lock);
  DL_FOREACH (gmem->svm_list, tmp)
  {
    if (tmp->svm->mem_host_ptr == svm_ptr)
      {
        found = tmp;
        break;
      }
  }

  if (found)
    {
      DL_DELETE (gmem->svm_list, found);
      assert (gmem->currently_allocated >= found->size);
      gmem->currently_allocated -= found->size;
      POCL_MEM_FREE (found->svm->gmem_ptrs);
      POCL_MEM_FREE (found->svm->mem_host_ptr);
      POCL_MEM_FREE (found->svm);
      POCL_MEM_FREE (found);
    }
  POCL_UNLOCK (gmem->lock);
  return found ? CL_SUCCESS : CL_FAILED;
}

void *
pocl_basic_svm_alloc (pocl_global_mem_t *gmem, cl_svm_mem_flags flags,
                      size_t size)
{
  void *retval = NULL;

  POCL_LOCK (gmem->lock);
  if ((gmem->currently_allocated + size) > gmem->size)
    goto ERROR;

  retval = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, size);
  if (retval)
    {
      gmem->currently_allocated += size;
      if (gmem->currently_allocated > gmem->max_ever_allocated)
        gmem->max_ever_allocated = gmem->currently_allocated;

      pocl_gmem_allocation_t *all
          = calloc (1, sizeof (pocl_gmem_allocation_t));
      /* create a hidden cl_mem that'll be useful for clSetKernelArg */
      all->svm = calloc (1, sizeof (struct _cl_mem));
      POCL_INIT_OBJECT (all->svm);
      all->svm->gmem_ptrs
          = calloc (pocl_num_global_mem, sizeof (pocl_mem_identifier));
      all->svm->size = all->size = size;
      all->svm->mem_host_ptr = retval;
      all->svm->gmem_ptrs[pocl_svm_id].mem_ptr = retval;
      all->svm->gmem_ptrs[pocl_svm_id].device_size = size;

      DL_PREPEND (gmem->svm_list, all);
    }

ERROR:
  POCL_UNLOCK (gmem->lock);
  return retval;
}

cl_mem
pocl_basic_svm_pointer_to_clmem (pocl_global_mem_t *gmem, const void *ptr,
                                 size_t *offset)
{
  pocl_gmem_allocation_t *tmp, *found = NULL;
  POCL_LOCK (gmem->lock);
  DL_FOREACH (gmem->svm_list, tmp)
  {
    const void *end = (const char *)tmp->svm->mem_host_ptr + tmp->size;
    if ((ptr >= tmp->svm->mem_host_ptr) && (ptr < end))
      {
        found = tmp;
        *offset = (const char *)ptr - (const char *)tmp->svm->mem_host_ptr;
        break;
      }
  }

  POCL_UNLOCK (gmem->lock);
  return found ? found->svm : NULL;
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
