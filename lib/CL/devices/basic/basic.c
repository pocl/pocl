/* basic.c - a minimalistic pocl device driver layer implementation

   Copyright (c) 2011-2013 Universidad Rey Juan Carlos and
                 2011-2014 Pekka Jääskeläinen / Tampere University of Technology
   
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

#include "config.h"
#include "basic.h"
#include "cpuinfo.h"
#include "topology/pocl_topology.h"
#include "common.h"
#include "utlist.h"
#include "devices.h"
#include "pocl_util.h"

#include <pthread.h>
#include <utlist.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#include "pocl_cache.h"
#include "pocl_timing.h"
#include "pocl_file_util.h"

#ifdef OCS_AVAILABLE
#include "pocl_llvm.h"
#endif

#define max(a,b) (((a) > (b)) ? (a) : (b))

#define COMMAND_LENGTH 2048
#define WORKGROUP_STRING_LENGTH 1024

struct data {
  /* Currently loaded kernel. */
  cl_kernel current_kernel;
  /* Loaded kernel dynamic library handle. */
  lt_dlhandle current_dlhandle;
  
  /* List of commands ready to be executed */
  _cl_command_node * volatile ready_list;
  /* List of commands not yet ready to be executed */
  _cl_command_node * volatile command_list;
  pocl_lock_t cq_lock;      /* Lock for command list related operations */
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


void
pocl_basic_init_device_ops(struct pocl_device_ops *ops)
{
  ops->device_name = "basic";

  ops->init_device_infos = pocl_basic_init_device_infos;
  ops->probe = pocl_basic_probe;
  ops->uninit = pocl_basic_uninit;
  ops->init = pocl_basic_init;
  ops->alloc_mem_obj = pocl_basic_alloc_mem_obj;
  ops->free = pocl_basic_free;
  ops->free_ptr = pocl_basic_free_ptr;
  ops->read = pocl_basic_read;
  ops->read_rect = pocl_basic_read_rect;
  ops->write = pocl_basic_write;
  ops->write_rect = pocl_basic_write_rect;
  ops->copy = pocl_basic_copy;
  ops->copy_rect = pocl_basic_copy_rect;
  ops->fill_rect = pocl_basic_fill_rect;
  ops->memfill = pocl_basic_memfill;
  ops->map_mem = pocl_basic_map_mem;
  ops->compile_kernel = pocl_basic_compile_kernel;
  ops->unmap_mem = pocl_basic_unmap_mem;
  ops->run = pocl_basic_run;
  ops->run_native = pocl_basic_run_native;
  ops->get_timer_value = pocl_basic_get_timer_value;
  ops->get_supported_image_formats = pocl_basic_get_supported_image_formats;
  ops->join = pocl_basic_join;
  ops->submit = pocl_basic_submit;
  ops->broadcast = pocl_basic_broadcast;
  ops->notify = pocl_basic_notify;
  ops->flush = pocl_basic_flush;
  ops->build_hash = pocl_basic_build_hash;
}

char *
pocl_basic_build_hash (cl_device_id device)
{
  char* res = calloc(1000, sizeof(char));
  snprintf(res, 1000, "basic-%s", HOST_DEVICE_BUILD_HASH);
  return res;
}

static cl_device_partition_property basic_partition_properties[1] = { 0 };

void
pocl_basic_init_device_infos(unsigned j, struct _cl_device_id* dev)
{
  dev->type = CL_DEVICE_TYPE_CPU;
  dev->vendor_id = 0;
  dev->max_compute_units = 0;
  dev->max_work_item_dimensions = 3;

  SETUP_DEVICE_CL_VERSION(HOST_DEVICE_CL_VERSION_MAJOR, HOST_DEVICE_CL_VERSION_MINOR)
  /*
    The hard restriction will be the context data which is
    stored in stack that can be as small as 8K in Linux.
    Thus, there should be enough work-items alive to fill up
    the SIMD lanes times the vector units, but not more than
    that to avoid stack overflow and cache trashing.
  */
  dev->max_work_item_sizes[0] = dev->max_work_item_sizes[1] =
	  dev->max_work_item_sizes[2] = dev->max_work_group_size = 1024*4;

  dev->preferred_wg_size_multiple = 8;
  dev->preferred_vector_width_char = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_CHAR;
  dev->preferred_vector_width_short = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT;
  dev->preferred_vector_width_int = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_INT;
  dev->preferred_vector_width_long = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_LONG;
  dev->preferred_vector_width_float = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_FLOAT;
  dev->preferred_vector_width_double = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_DOUBLE;
  dev->preferred_vector_width_half = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_HALF;
  /* TODO: figure out what the difference between preferred and native widths are */
  dev->native_vector_width_char = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_CHAR;
  dev->native_vector_width_short = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT;
  dev->native_vector_width_int = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_INT;
  dev->native_vector_width_long = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_LONG;
  dev->native_vector_width_float = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_FLOAT;
  dev->native_vector_width_double = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_DOUBLE;
  dev->native_vector_width_half = POCL_DEVICES_PREFERRED_VECTOR_WIDTH_HALF;
  dev->max_clock_frequency = 0;
  dev->address_bits = POCL_DEVICE_ADDRESS_BITS;
  dev->image_support = CL_TRUE;
  /* Use the minimum values until we get a more sensible upper limit from
     somewhere. */
  dev->max_read_image_args = dev->max_write_image_args = dev->max_read_write_image_args = 128;
  dev->image2d_max_width = dev->image2d_max_height = 8192;
  dev->image3d_max_width = dev->image3d_max_height = dev->image3d_max_depth = 2048;
  dev->image_max_buffer_size = 65536;
  dev->image_max_array_size = 2048;
  dev->max_samplers = 16;
  dev->max_constant_args = 8;

  dev->max_mem_alloc_size = 0;
  dev->max_parameter_size = 1024;
  dev->min_data_type_align_size = MAX_EXTENDED_ALIGNMENT; // this is in bytes
  dev->mem_base_addr_align = MAX_EXTENDED_ALIGNMENT*8; // this is in bits
  dev->half_fp_config = 0;
  dev->single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN;
#ifdef __x86_64__
  dev->single_fp_config |= (CL_FP_DENORM | CL_FP_ROUND_TO_INF | CL_FP_ROUND_TO_ZERO);
  if (cpu_has_fma())
    dev->single_fp_config |= CL_FP_FMA;
#endif
  dev->double_fp_config = CL_FP_FMA | CL_FP_ROUND_TO_NEAREST
                          | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF
                          | CL_FP_INF_NAN | CL_FP_DENORM;
  dev->global_mem_cache_type = CL_NONE;
  dev->global_mem_cacheline_size = 0;
  dev->global_mem_cache_size = 0;
  dev->global_mem_size = 0;
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
  dev->workgroup_pass = CL_TRUE;
  dev->execution_capabilities = CL_EXEC_KERNEL | CL_EXEC_NATIVE_KERNEL;
  dev->platform = 0;

  dev->parent_device = NULL;
  // basic does not support partitioning
  dev->max_sub_devices = 1;
  dev->num_partition_properties = 1;
  dev->partition_properties = basic_partition_properties;
  dev->num_partition_types = 0;
  dev->partition_type = NULL;

  /* printf buffer size is meaningless for pocl, so just set it to
   * the minimum value required by the spec
   */
  dev->printf_buffer_size = 1024*1024 ;
  dev->vendor = "pocl";
  dev->profile = "FULL_PROFILE";
  /* Note: The specification describes identifiers being delimited by
     only a single space character. Some programs that check the device's
     extension  string assume this rule. Future extension additions should
     ensure that there is no more than a single space between
     identifiers. */

  dev->global_as_id = dev->local_as_id = dev->constant_as_id = 0;

  dev->should_allocate_svm = 0;
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

#ifdef OCS_AVAILABLE

  dev->llvm_target_triplet = OCL_KERNEL_TARGET;
  dev->llvm_cpu = get_cpu_name();

#else
  dev->llvm_cpu = NULL;
  dev->llvm_target_triplet = "";
#endif

  dev->has_64bit_long = 1;
  dev->autolocals_to_args = 1;
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
      pocl_init_dlhandle_cache();
      first_basic_init = 0;
    }
  device->global_mem_id = 0;

  d = (struct data *) calloc (1, sizeof (struct data));
  if (d == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  d->current_kernel = NULL;
  d->current_dlhandle = 0;
  device->data = d;
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

  if(device->llvm_cpu && (!strcmp(device->llvm_cpu, "(unknown)")))
    device->llvm_cpu = NULL;

  // work-around LLVM bug where sizeof(long)=4
  #ifdef _CL_DISABLE_LONG
  device->has_64bit_long=0;
  #endif

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
      b = pocl_memalign_alloc_global_mem (device, MAX_EXTENDED_ALIGNMENT,
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

  /* aloocation owner executes freeing */
  if (flags & CL_MEM_USE_HOST_PTR ||
      memobj->shared_mem_allocation_owner != device)
    return;

  void* ptr = memobj->device_ptrs[device->dev_id].mem_ptr;
  size_t size = memobj->size;

  if (memobj->flags | CL_MEM_ALLOC_HOST_PTR)
    memobj->mem_host_ptr = NULL;
  pocl_free_global_mem(device, ptr, size);
}

void pocl_basic_free_ptr (cl_device_id device, void* mem_ptr)
{
  /* TODO we should somehow figure out the size argument
   * and call pocl_free_global_mem */
  POCL_MEM_FREE(mem_ptr);
}

void
pocl_basic_read (void *data, void *host_ptr, const void *device_ptr, 
                 size_t offset, size_t cb)
{
  if (host_ptr == device_ptr)
    return;

  memcpy (host_ptr, (char*)device_ptr + offset, cb);
}

void
pocl_basic_write (void *data, const void *host_ptr, void *device_ptr, 
                  size_t offset, size_t cb)
{
  if (host_ptr == device_ptr)
    return;

  memcpy ((char*)device_ptr + offset, host_ptr, cb);
}


void
pocl_basic_run 
(void *data, 
 _cl_command_node* cmd)
{
  struct data *d;
  struct pocl_argument *al;
  size_t x, y, z;
  unsigned i;
  cl_kernel kernel = cmd->command.run.kernel;
  struct pocl_context *pc = &cmd->command.run.pc;

  assert (data != NULL);
  d = (struct data *) data;

  d->current_kernel = kernel;

  void **arguments = (void**)malloc(
      sizeof(void*) * (kernel->num_args + kernel->num_locals)
    );

  /* Process the kernel arguments. Convert the opaque buffer
     pointers to real device pointers, allocate dynamic local 
     memory buffers, etc. */
  for (i = 0; i < kernel->num_args; ++i)
    {
      al = &(cmd->command.run.arguments[i]);
      if (kernel->arg_info[i].is_local)
        {
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = pocl_memalign_alloc(MAX_EXTENDED_ALIGNMENT, al->size);
        }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_POINTER)
        {
          /* It's legal to pass a NULL pointer to clSetKernelArguments. In 
             that case we must pass the same NULL forward to the kernel.
             Otherwise, the user must have created a buffer with per device
             pointers stored in the cl_mem. */
          if (al->value == NULL)
            {
              arguments[i] = malloc (sizeof (void *));
              *(void **)arguments[i] = NULL;
            }
          else
            arguments[i] = &((*(cl_mem *) (al->value))->device_ptrs[cmd->device->dev_id].mem_ptr);
        }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
        {
          dev_image_t di;
          fill_dev_image_t (&di, al, cmd->device);

          void* devptr = pocl_memalign_alloc(MAX_EXTENDED_ALIGNMENT,  sizeof(dev_image_t));
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = devptr; 
          pocl_basic_write (data, &di, devptr, 0, sizeof(dev_image_t));
        }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)
        {
          dev_sampler_t ds;
          fill_dev_sampler_t(&ds, al);
          
          void* devptr = pocl_memalign_alloc(MAX_EXTENDED_ALIGNMENT, sizeof(dev_sampler_t));
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = devptr;
          pocl_basic_write (data, &ds, devptr, 0, sizeof(dev_sampler_t));
        }
      else
        {
          arguments[i] = al->value;
        }
    }
  for (i = kernel->num_args;
       i < kernel->num_args + kernel->num_locals;
       ++i)
    {
      al = &(cmd->command.run.arguments[i]);
      arguments[i] = malloc (sizeof (void *));
      *(void **)(arguments[i]) = pocl_memalign_alloc(MAX_EXTENDED_ALIGNMENT, al->size);
    }

  pc->local_size[0] = cmd->command.run.local_x;
  pc->local_size[1] = cmd->command.run.local_y;
  pc->local_size[2] = cmd->command.run.local_z;

  unsigned rm = pocl_save_rm ();
  pocl_set_default_rm ();
  unsigned ftz = pocl_save_ftz ();
  pocl_set_ftz (kernel->program->flush_denorms);

  for (z = 0; z < pc->num_groups[2]; ++z)
    {
      for (y = 0; y < pc->num_groups[1]; ++y)
        {
          for (x = 0; x < pc->num_groups[0]; ++x)
            {
              pc->group_id[0] = x;
              pc->group_id[1] = y;
              pc->group_id[2] = z;

              cmd->command.run.wg (arguments, pc);

            }
        }
    }

  pocl_restore_rm (rm);
  pocl_restore_ftz (ftz);

  for (i = 0; i < kernel->num_args; ++i)
    {
      if (kernel->arg_info[i].is_local)
        {
          POCL_MEM_FREE(*(void **)(arguments[i]));
          POCL_MEM_FREE(arguments[i]);
        }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_IMAGE ||
                kernel->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)
        {
          POCL_MEM_FREE(*(void **)(arguments[i]));
          POCL_MEM_FREE(arguments[i]);
        }
      else if (kernel->arg_info[i].type == POCL_ARG_TYPE_POINTER && *(void**)arguments[i] == NULL)
        {
          POCL_MEM_FREE(arguments[i]);
        }
    }
  for (i = kernel->num_args;
       i < kernel->num_args + kernel->num_locals;
       ++i)
    {
      POCL_MEM_FREE(*(void **)(arguments[i]));
      POCL_MEM_FREE(arguments[i]);
    }
  free(arguments);
}

void
pocl_basic_run_native 
(void *data, 
 _cl_command_node* cmd)
{
  cmd->command.native.user_func(cmd->command.native.args);
}

void
pocl_basic_copy (void *data, const void *src_ptr, size_t src_offset, 
                 void *__restrict__ dst_ptr, size_t dst_offset, size_t cb)
{
  if (src_ptr == dst_ptr)
    return;
  
  memcpy ((char*)dst_ptr + dst_offset, (char*)src_ptr + src_offset, cb);
}

void
pocl_basic_copy_rect (void *data,
                      const void *__restrict const src_ptr,
                      void *__restrict__ const dst_ptr,
                      const size_t *__restrict__ const src_origin,
                      const size_t *__restrict__ const dst_origin, 
                      const size_t *__restrict__ const region,
                      size_t const src_row_pitch,
                      size_t const src_slice_pitch,
                      size_t const dst_row_pitch,
                      size_t const dst_slice_pitch)
{
  char const *__restrict const adjusted_src_ptr = 
    (char const*)src_ptr +
    src_origin[0] + src_row_pitch * src_origin[1] + src_slice_pitch * src_origin[2];
  char *__restrict__ const adjusted_dst_ptr = 
    (char*)dst_ptr +
    dst_origin[0] + dst_row_pitch * dst_origin[1] + dst_slice_pitch * dst_origin[2];
  
  size_t j, k;

  /* TODO: handle overlaping regions */
  
  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      memcpy (adjusted_dst_ptr + dst_row_pitch * j + dst_slice_pitch * k,
              adjusted_src_ptr + src_row_pitch * j + src_slice_pitch * k,
              region[0]);
}

void
pocl_basic_write_rect (void *data,
                       const void *__restrict__ const host_ptr,
                       void *__restrict__ const device_ptr,
                       const size_t *__restrict__ const buffer_origin,
                       const size_t *__restrict__ const host_origin, 
                       const size_t *__restrict__ const region,
                       size_t const buffer_row_pitch,
                       size_t const buffer_slice_pitch,
                       size_t const host_row_pitch,
                       size_t const host_slice_pitch)
{
  char *__restrict const adjusted_device_ptr = 
    (char*)device_ptr +
    buffer_origin[0] + buffer_row_pitch * buffer_origin[1] + buffer_slice_pitch * buffer_origin[2];
  char const *__restrict__ const adjusted_host_ptr = 
    (char const*)host_ptr +
    host_origin[0] + host_row_pitch * host_origin[1] + host_slice_pitch * host_origin[2];
  
  size_t j, k;

  /* TODO: handle overlaping regions */
  
  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      memcpy (adjusted_device_ptr + buffer_row_pitch * j + buffer_slice_pitch * k,
              adjusted_host_ptr + host_row_pitch * j + host_slice_pitch * k,
              region[0]);
}

void
pocl_basic_read_rect (void *data,
                      void *__restrict__ const host_ptr,
                      void *__restrict__ const device_ptr,
                      const size_t *__restrict__ const buffer_origin,
                      const size_t *__restrict__ const host_origin, 
                      const size_t *__restrict__ const region,
                      size_t const buffer_row_pitch,
                      size_t const buffer_slice_pitch,
                      size_t const host_row_pitch,
                      size_t const host_slice_pitch)
{
  char const *__restrict const adjusted_device_ptr = 
    (char const*)device_ptr +
    buffer_origin[2] * buffer_slice_pitch + buffer_origin[1] * buffer_row_pitch + buffer_origin[0];
  char *__restrict__ const adjusted_host_ptr = 
    (char*)host_ptr +
    host_origin[2] * host_slice_pitch + host_origin[1] * host_row_pitch + host_origin[0];
  
  size_t j, k;
  
  /* TODO: handle overlaping regions */
  
  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      memcpy (adjusted_host_ptr + host_row_pitch * j + host_slice_pitch * k,
              adjusted_device_ptr + buffer_row_pitch * j + buffer_slice_pitch * k,
              region[0]);
}

/* origin and region must be in original shape unlike in copy/read/write_rect()
 */
void
pocl_basic_fill_rect (void *data,
                      void *__restrict__ const device_ptr,
                      const size_t *__restrict__ const buffer_origin,
                      const size_t *__restrict__ const region,
                      size_t const buffer_row_pitch,
                      size_t const buffer_slice_pitch,
                      void *fill_pixel,
                      size_t pixel_size)                    
{
  char *__restrict const adjusted_device_ptr = (char*)device_ptr 
    + buffer_origin[0] * pixel_size 
    + buffer_row_pitch * buffer_origin[1] 
    + buffer_slice_pitch * buffer_origin[2];
    
  size_t i, j, k;

  for (k = 0; k < region[2]; ++k)
    for (j = 0; j < region[1]; ++j)
      for (i = 0; i < region[0]; ++i)
        memcpy (adjusted_device_ptr + pixel_size * i 
                + buffer_row_pitch * j 
                + buffer_slice_pitch * k, fill_pixel, pixel_size);
}

void pocl_basic_memfill(void *ptr,
                        size_t size,
                        size_t offset,
                        const void* pattern,
                        size_t pattern_size)
{
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

void *
pocl_basic_map_mem (void *data, void *buf_ptr, 
                      size_t offset, size_t size,
                      void *host_ptr) 
{
  /* If the buffer was allocated with CL_MEM_ALLOC_HOST_PTR |
   * CL_MEM_COPY_HOST_PTR,
   * the host_ptr is not the same memory as pocl's device_ptrs[], and we need
   * to copy pocl's buffer content back to host_ptr. */
  if ((host_ptr != NULL) && (host_ptr != (buf_ptr + offset)))
    {
      POCL_MSG_PRINT_MEMORY ("device: MAP memcpy() "
                             "buf_ptr %p + offset %zu to host_ptr %p\n",
                             buf_ptr, offset, host_ptr);
      memcpy ((char *)host_ptr, (char *)buf_ptr + offset, size);
    }
  return (char*)buf_ptr + offset;
}

void* pocl_basic_unmap_mem(void *data, void *host_ptr,
                           void *device_start_ptr,
                           size_t offset, size_t size)
{
  /* If the buffer was allocated with CL_MEM_ALLOC_HOST_PTR |
   * CL_MEM_COPY_HOST_PTR,
   * the host_ptr is not the same memory as pocl's device_ptrs[], and we need
   * to copy host_ptr content back to  pocl's device_ptrs[]. */
  if ((host_ptr != NULL) && (host_ptr != (device_start_ptr + offset)))
    {
      POCL_MSG_PRINT_MEMORY ("device: UNMAP memcpy() "
                             "host_ptr %p to buf_ptr %p + offset %zu\n",
                             host_ptr, device_start_ptr, offset);
      memcpy ((char *)device_start_ptr + offset, (char *)host_ptr, size);
    }
  return (char *)host_ptr;
}


void
pocl_basic_uninit (cl_device_id device)
{
  struct data *d = (struct data*)device->data;
  POCL_MEM_FREE(d);
  device->data = NULL;
}

cl_ulong
pocl_basic_get_timer_value (void *data) 
{
  return pocl_gettimemono_ns();
}

cl_int 
pocl_basic_get_supported_image_formats (cl_mem_flags flags,
                                        const cl_image_format **image_formats,
                                        cl_uint *num_img_formats)
{
    if (num_img_formats == NULL || image_formats == NULL)
      return CL_INVALID_VALUE;
  
    *num_img_formats = sizeof(supported_image_formats)/sizeof(cl_image_format);
    *image_formats = supported_image_formats;
    
    return CL_SUCCESS; 
}

static void basic_command_scheduler (struct data *d) 
{
  _cl_command_node *node;
  
  /* execute commands from ready list */
  while ((node = d->ready_list))
    {
      assert (pocl_command_is_ready(node->event));
      CDL_DELETE (d->ready_list, node);

      
      pthread_mutex_unlock (&d->cq_lock);
      assert (node->event->status == CL_SUBMITTED);
      pocl_exec_command(node);
      pthread_mutex_lock (&d->cq_lock);
    }
    
  return;
}

void
pocl_basic_submit (_cl_command_node *node, cl_command_queue cq)
{
  struct data *d = node->device->data;
  cl_event *event = &(node->event);
  
  node->device->ops->compile_kernel (node, NULL, NULL);
  POCL_LOCK (d->cq_lock);
  POCL_UPDATE_EVENT_SUBMITTED(event);
  pocl_command_push(node, &d->ready_list, &d->command_list);
  
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
/*
static void
pocl_basic_push_command (_cl_command_node *node)
{
  struct data *d = (struct data*)node->device->data;

  pocl_command_push(node, &d->ready_list, &d->command_list);

}
*/
void
pocl_basic_join(cl_device_id device, cl_command_queue cq)
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
  
  POCL_LOCK_OBJ (event);
  if (!(node->ready) && pocl_command_is_ready(node->event))
    {
      node->ready = 1;
      POCL_UNLOCK_OBJ (event);
      if (node->event->status == CL_SUBMITTED)
        {
          POCL_LOCK (d->cq_lock);
          CDL_DELETE (d->command_list, node);
          CDL_PREPEND (d->ready_list, node);
          basic_command_scheduler (d);
          POCL_UNLOCK (d->cq_lock);
        }
      return;
    }
  POCL_UNLOCK_OBJ (event);
}

void
pocl_basic_broadcast (cl_event event)
{
  pocl_broadcast (event);
}

void
pocl_basic_compile_kernel (_cl_command_node *cmd, cl_kernel kernel, cl_device_id device)
{
  if (cmd != NULL && cmd->type == CL_COMMAND_NDRANGE_KERNEL)
    pocl_check_dlhandle_cache (cmd);
}
