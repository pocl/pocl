/* pocl-metal.m - driver for Apple GPUs using the Metal API

   Copyright (c) 2026 PoCL developers

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

/* Overview:

   - Kernels are compiled by PoCL's Clang/LLVM for the spir64 target and
     linked with the Metal kernel library (lib/kernel/metal). After the
     usual SPMD PoCL passes, pocl-air-gen.cc lowers the module to AIR and
     packages it as a Metal library, which is loaded with
     -[MTLDevice newLibraryWithData:error:]. No Xcode / Metal toolchain is
     needed at runtime.

   - Buffers are MTLBuffers in shared storage. Apple GPUs have unified
     memory, so the host accesses the buffer contents directly, and the
     memory commands (read, write, copy, fill, map) are memcpy's done by the
     common driver code.

   - Commands are executed in order by a worker thread per device; a kernel
     launch waits for its command buffer to complete before the next command
     runs.
*/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "pocl-metal.h"
#include "pocl-air-gen.h"

#include "common.h"
#include "common_driver.h"
#include "config.h"
#include "devices.h"
#include "pocl_cache.h"
#include "pocl_file_util.h"
#include "pocl_llvm.h"
#include "pocl_local_size.h"
#include "pocl_timing.h"
#include "pocl_util.h"
#include "utlist.h"

#include <assert.h>
#include <pthread.h>
#include <string.h>

typedef struct
{
  void *device; /* id<MTLDevice>, retained */
  void *queue;  /* id<MTLCommandQueue>, retained */
  pocl_metal_air_target_t target;

  /* commands ready to be executed by the worker thread */
  _cl_command_node *ready_list;
  pocl_lock_t cq_lock;
  pocl_cond_t work_cond;
  pthread_t worker;
  int exit_worker;
  /* serializes AIR generation and library creation */
  pocl_lock_t compile_lock;
  /* serializes kernel launches, which share the printf buffer */
  pocl_lock_t run_lock;
  void *printf_buffer; /* id<MTLBuffer>, retained; allocated on first use */

  cl_bool available;
} pocl_metal_device_data_t;

typedef struct
{
  void *library; /* id<MTLLibrary>, retained */
  unsigned local_region_size;
  void *gvar_buffer; /* id<MTLBuffer>, retained; program-scope variables */
} pocl_metal_program_data_t;

typedef struct
{
  void *pipeline; /* id<MTLComputePipelineState>, retained */
  unsigned refcount;
} pocl_metal_kernel_data_t;

static int metal_start_worker (pocl_metal_device_data_t *d);
static void metal_stop_worker (pocl_metal_device_data_t *d);
static void metal_run_locked (pocl_metal_device_data_t *d,
                              _cl_command_node *cmd,
                              pocl_metal_kernel_data_t *kdata,
                              pocl_metal_program_data_t *pdata);
static void pocl_metal_compute_local_size (
    cl_device_id dev, cl_kernel kernel, unsigned device_i,
    size_t max_group_size, size_t global_x, size_t global_y, size_t global_z,
    size_t *local_x, size_t *local_y, size_t *local_z);

/* The devices found by the probe, in the order they're initialized. */
static NSArray<id<MTLDevice>> *metal_devices = nil;

static id<MTLDevice>
metal_device (pocl_metal_device_data_t *d)
{
  return (__bridge id<MTLDevice>)d->device;
}

void
pocl_metal_init_device_ops (struct pocl_device_ops *ops)
{
  ops->device_name = "metal";

  ops->probe = pocl_metal_probe;
  ops->init = pocl_metal_init;
  ops->uninit = pocl_metal_uninit;
  ops->reinit = NULL;
  ops->build_hash = pocl_metal_build_hash;

  ops->alloc_mem_obj = pocl_metal_alloc_mem_obj;
  ops->free = pocl_metal_free;

  /* the buffer contents are host-accessible */
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

  ops->run = pocl_metal_run;
  ops->run_native = NULL;

  ops->build_source = pocl_driver_build_source;
  ops->link_program = pocl_driver_link_program;
  ops->build_binary = pocl_driver_build_binary;
  ops->count_kernels = pocl_driver_count_kernels;
  ops->setup_metadata = pocl_driver_setup_metadata;
  ops->supports_binary = pocl_driver_supports_binary;
  ops->build_poclbinary = pocl_driver_build_poclbinary;
  ops->post_build_program = pocl_metal_post_build_program;
  ops->free_program = pocl_metal_free_program;
  ops->build_builtin = NULL;

  ops->compile_kernel = pocl_metal_compile_kernel;
  ops->create_kernel = pocl_metal_create_kernel;
  ops->free_kernel = pocl_metal_free_kernel;
  ops->compute_local_size = pocl_metal_compute_local_size;

  ops->submit = pocl_metal_submit;
  ops->join = pocl_metal_join;
  ops->flush = pocl_metal_flush;
  ops->notify = pocl_metal_notify;
  ops->broadcast = pocl_broadcast;
  ops->wait_event = pocl_metal_wait_event;
  ops->update_event = pocl_metal_update_event;
  ops->free_event_data = pocl_metal_free_event_data;
  ops->notify_event_finished = pocl_metal_notify_event_finished;
  ops->notify_cmdq_finished = pocl_metal_notify_cmdq_finished;
  ops->init_queue = pocl_metal_init_queue;
  ops->free_queue = pocl_metal_free_queue;

  ops->get_synchronized_timestamps = pocl_driver_get_synchronized_timestamps;
}

unsigned int
pocl_metal_probe (struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count (ops->device_name);

  @autoreleasepool
    {
      if (metal_devices == nil)
        {
          NSArray<id<MTLDevice>> *all = MTLCopyAllDevices ();
          /* Only Apple-family GPUs with unified memory: the driver relies on
             host-accessible shared buffers, and on the Apple GPU AIR
             conventions. */
          NSMutableArray<id<MTLDevice>> *usable = [NSMutableArray array];
          for (id<MTLDevice> dev in all)
            if ([dev supportsFamily:MTLGPUFamilyApple7]
                && dev.hasUnifiedMemory)
              [usable addObject:dev];
          metal_devices = [usable copy];
        }
    }

  int probe_count = (int)metal_devices.count;
  /* enabled by default, like the other GPU drivers */
  if (env_count < 0)
    return probe_count;
  if (env_count > probe_count)
    POCL_ABORT ("[Metal] %d devices requested, but only %d are available\n",
                env_count, probe_count);
  return env_count;
}

char *
pocl_metal_build_hash (cl_device_id device)
{
  pocl_metal_device_data_t *d = (pocl_metal_device_data_t *)device->data;
  char *res = calloc (128, 1);
  snprintf (res, 128, "pocl-metal-v4-air%u.%u-metal%u.%u", d->target.air_major,
            d->target.air_minor, d->target.metal_major,
            d->target.metal_minor);
  return res;
}

cl_int
pocl_metal_init (unsigned j, cl_device_id dev, const char *parameters)
{
  assert (dev->data == NULL);
  if (j >= metal_devices.count)
    return CL_INVALID_DEVICE;

  pocl_metal_device_data_t *d = calloc (1, sizeof (pocl_metal_device_data_t));
  if (d == NULL)
    return CL_OUT_OF_HOST_MEMORY;
  dev->data = d;

  id<MTLDevice> mdev = metal_devices[j];
  d->device = (__bridge_retained void *)mdev;
  d->queue = (__bridge_retained void *)[mdev newCommandQueue];
  if (d->queue == NULL)
    {
      POCL_MSG_ERR ("[Metal] could not create a command queue\n");
      return CL_INVALID_DEVICE;
    }
  POCL_INIT_LOCK (d->cq_lock);
  POCL_INIT_LOCK (d->compile_lock);
  POCL_INIT_LOCK (d->run_lock);
  if (metal_start_worker (d) != 0)
    {
      POCL_MSG_ERR ("[Metal] could not start the worker thread\n");
      return CL_INVALID_DEVICE;
    }
  d->available = CL_TRUE;
  dev->available = &d->available;

  /* Generate AIR 2.7 / MSL 3.2 / metallib 1.2.8, supported since macOS 15
     (and read by newer releases). */
  d->target.air_major = 2;
  d->target.air_minor = 7;
  d->target.metal_major = 3;
  d->target.metal_minor = 2;
  d->target.macos_major = 15;
  d->target.macos_minor = 0;
  d->target.local_mem_size = (unsigned)mdev.maxThreadgroupMemoryLength;

  pocl_init_default_device_infos (dev, METAL_DEVICE_EXTENSIONS);
  SETUP_DEVICE_CL_VERSION (dev, METAL_DEVICE_CL_VERSION_MAJOR,
                           METAL_DEVICE_CL_VERSION_MINOR);

  dev->type = CL_DEVICE_TYPE_GPU;
  dev->vendor = "Apple";
  dev->vendor_id = 0x106b;
  dev->long_name = dev->short_name = strdup (mdev.name.UTF8String);

  /* Clang target & kernel library: spir64 shares AIR's address space
     numbering, the driver lowers to AIR after PoCL's passes */
  dev->llvm_target_triplet = "spir64-unknown-unknown";
  dev->llvm_cpu = NULL;
  dev->kernellib_name = "kernel-metal-air64";
  dev->kernellib_fallback_name = NULL;
  dev->kernellib_subdir = "metal";
  dev->llvm_fp_contract_mode = "on";

  dev->spmd = CL_TRUE;
  dev->run_workgroup_pass = CL_FALSE;
  dev->execution_capabilities = CL_EXEC_KERNEL;
  dev->global_as_id = 1;
  dev->local_as_id = 3;
  dev->constant_as_id = 2;
  dev->generic_as_support = CL_FALSE;
  dev->run_program_scope_variables_pass = CL_FALSE;
  /* static __local variables stay threadgroup globals in AIR */
  dev->autolocals_to_args = POCL_AUTOLOCALS_TO_ARGS_NEVER;
  /* printf() stores to a buffer that the host decodes after the launch */
  dev->device_side_printf = 1;
  dev->has_64bit_long = 1;
  dev->address_bits = 64;
  dev->endian_little = CL_TRUE;
  dev->image_support = CL_FALSE;
  dev->non_uniform_work_group_support = CL_FALSE;

  MTLSize max_tg = mdev.maxThreadsPerThreadgroup;
  dev->max_work_item_dimensions = 3;
  dev->max_work_item_sizes[0] = max_tg.width;
  dev->max_work_item_sizes[1] = max_tg.height;
  dev->max_work_item_sizes[2] = max_tg.depth;
  dev->max_work_group_size = max_tg.width;
  /* the SIMD-group width of all Apple GPUs */
  dev->preferred_wg_size_multiple = 32;

  dev->preferred_vector_width_char = dev->native_vector_width_char = 1;
  dev->preferred_vector_width_short = dev->native_vector_width_short = 1;
  dev->preferred_vector_width_int = dev->native_vector_width_int = 1;
  dev->preferred_vector_width_long = dev->native_vector_width_long = 1;
  dev->preferred_vector_width_float = dev->native_vector_width_float = 1;
  dev->preferred_vector_width_double = dev->native_vector_width_double = 0;
  if (strstr (METAL_DEVICE_EXTENSIONS, "cl_khr_fp16") != NULL)
    dev->preferred_vector_width_half = dev->native_vector_width_half = 1;
  else
    dev->preferred_vector_width_half = dev->native_vector_width_half = 0;

  dev->local_mem_type = CL_LOCAL;
  dev->local_mem_size = mdev.maxThreadgroupMemoryLength;
  dev->global_mem_size = mdev.recommendedMaxWorkingSetSize;
  dev->max_mem_alloc_size = mdev.maxBufferLength;
  if (dev->max_mem_alloc_size > dev->global_mem_size)
    dev->max_mem_alloc_size = dev->global_mem_size;
  dev->max_constant_buffer_size = 64 * 1024;
  dev->max_constant_args = 8;
  dev->global_mem_cache_type = CL_READ_WRITE_CACHE;
  dev->global_mem_cacheline_size = 128;
  dev->host_unified_memory = CL_TRUE;
  dev->error_correction_support = CL_FALSE;
  /* setBytes is limited to 4 KiB per argument */
  dev->max_parameter_size = 4096;
  dev->mem_base_addr_align = 256 * 8; /* in bits */
  dev->min_data_type_align_size = 128;
  /* TODO: query the GPU core count from the IORegistry */
  dev->max_compute_units = 8;
  dev->max_clock_frequency = 1000;

  /* Apple GPUs flush float denormals, so no CL_FP_DENORM. Precise division
     and sqrt (without fast math) are correctly rounded. */
  dev->single_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN | CL_FP_FMA
                          | CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT;
  dev->double_fp_config = 0;
  if (strstr (METAL_DEVICE_EXTENSIONS, "cl_khr_fp16") != NULL)
    dev->half_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN | CL_FP_FMA;
  else
    dev->half_fp_config = 0;

  /* OpenCL 3.0 properties; the minimum mandated */
  dev->atomic_memory_capabilities
      = CL_DEVICE_ATOMIC_ORDER_RELAXED | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP;
  dev->atomic_fence_capabilities = CL_DEVICE_ATOMIC_ORDER_RELAXED
                                   | CL_DEVICE_ATOMIC_ORDER_ACQ_REL
                                   | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP;
  dev->svm_caps = 0;
  dev->svm_allocation_priority = 0;

  dev->features = METAL_DEVICE_FEATURES_30;
  /* program-scope variables live in a per-program buffer (pocl-air-gen.cc) */
  dev->global_var_max_size = 64 * 1024;
  dev->global_var_pref_size = 64 * 1024;
  pocl_setup_opencl_c_with_version (dev);
  pocl_setup_features_with_version (dev);
  pocl_setup_extensions_with_version (dev);
  pocl_setup_builtin_kernels_with_version (dev);
  pocl_setup_ils_with_version (dev);

  POCL_MSG_PRINT_METAL ("initialized device %u: %s, %zu bytes of threadgroup "
                        "memory\n",
                        j, dev->long_name, (size_t)dev->local_mem_size);
  return CL_SUCCESS;
}

cl_int
pocl_metal_uninit (unsigned j, cl_device_id dev)
{
  pocl_metal_device_data_t *d = (pocl_metal_device_data_t *)dev->data;
  if (d == NULL)
    return CL_SUCCESS;
  metal_stop_worker (d);
  if (d->printf_buffer)
    CFRelease (d->printf_buffer);
  if (d->queue)
    CFRelease (d->queue);
  if (d->device)
    CFRelease (d->device);
  POCL_DESTROY_LOCK (d->cq_lock);
  POCL_DESTROY_LOCK (d->compile_lock);
  POCL_DESTROY_LOCK (d->run_lock);
  free ((char *)dev->long_name);
  dev->long_name = dev->short_name = NULL;
  POCL_MEM_FREE (d);
  dev->data = NULL;
  return CL_SUCCESS;
}

/***************************************************************************/
/* memory */

cl_int
pocl_metal_alloc_mem_obj (cl_device_id device, cl_mem mem, void *host_ptr)
{
  pocl_metal_device_data_t *d = (pocl_metal_device_data_t *)device->data;
  pocl_mem_identifier *p = &mem->device_ptrs[device->global_mem_id];

  if (mem->is_image)
    return CL_IMAGE_FORMAT_NOT_SUPPORTED;

  @autoreleasepool
    {
      id<MTLBuffer> buf = [metal_device (d)
          newBufferWithLength:mem->size
                      options:MTLResourceStorageModeShared];
      if (buf == nil)
        return CL_MEM_OBJECT_ALLOCATION_FAILURE;
      p->extra_ptr = (__bridge_retained void *)buf;
      p->mem_ptr = buf.contents;
      p->device_addr = (void *)buf.gpuAddress;
      /* the contents are migrated by the runtime from mem_host_ptr */
      p->version = 0;
    }

  POCL_MSG_PRINT_MEMORY ("[Metal] ALLOC %p / size %zu\n", p->mem_ptr,
                         mem->size);
  return CL_SUCCESS;
}

void
pocl_metal_free (cl_device_id device, cl_mem mem)
{
  pocl_mem_identifier *p = &mem->device_ptrs[device->global_mem_id];
  if (p->extra_ptr)
    CFRelease (p->extra_ptr);
  p->extra_ptr = NULL;
  p->mem_ptr = NULL;
  p->device_addr = NULL;
  p->version = 0;
}

/***************************************************************************/
/* programs & kernels */

static int
metal_create_library (pocl_metal_device_data_t *d,
                      pocl_metal_program_data_t *pdata, const char *bytes,
                      size_t size, char **error_log)
{
  @autoreleasepool
    {
      dispatch_data_t data = dispatch_data_create (
          bytes, size, NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
      NSError *error = nil;
      id<MTLLibrary> lib = [metal_device (d) newLibraryWithData:data
                                                          error:&error];
      if (lib == nil)
        {
          const char *msg
              = error ? error.localizedDescription.UTF8String : "unknown";
          *error_log = strdup (msg);
          return -1;
        }
      pdata->library = (__bridge_retained void *)lib;
    }
  return 0;
}

int
pocl_metal_post_build_program (cl_program program, cl_uint device_i)
{
  cl_device_id device = program->devices[device_i];
  pocl_metal_device_data_t *d = (pocl_metal_device_data_t *)device->data;

  if (program->llvm_irs[device_i] == NULL)
    {
      POCL_MSG_ERR ("[Metal] the program has no LLVM IR\n");
      return CL_BUILD_PROGRAM_FAILURE;
    }

  pocl_metal_program_data_t *pdata
      = calloc (1, sizeof (pocl_metal_program_data_t));
  if (pdata == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  /* nothing to run in a program without kernels */
  if (program->num_kernels == 0)
    {
      program->data[device_i] = pdata;
      return CL_SUCCESS;
    }

  POCL_LOCK (d->compile_lock);

  /* The cached library starts with a header of three uint32_t's: the size
     of the __local argument region, the size of the program-scope variable
     storage and the number of relocations in it, followed by the initial
     contents of that storage, the relocations, and the metallib. */
  char cache_path[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_program_bc_path (cache_path, program, device_i);
  strncat (cache_path, ".metallib",
           POCL_MAX_PATHNAME_LENGTH - strlen (cache_path) - 1);

  char *content = NULL;
  uint64_t content_size = 0;
  char *error_log = NULL;
  int err = CL_SUCCESS;
  uint32_t header[3];

  if (pocl_exists (cache_path)
      && pocl_read_file (cache_path, &content, &content_size) == 0
      && content_size > sizeof (header))
    {
      POCL_MSG_PRINT_METAL ("using cached Metal library %s\n", cache_path);
    }
  else
    {
      POCL_MEM_FREE (content);
      int r = pocl_llvm_run_passes_on_program (program, device_i);
      if (r != CL_SUCCESS)
        {
          err = CL_BUILD_PROGRAM_FAILURE;
          goto out;
        }

      char *lib = NULL, *gvar_init = NULL;
      pocl_metal_gvar_reloc_t *relocs = NULL;
      size_t lib_size = 0;
      unsigned region = 0, gvar_size = 0, num_relocs = 0;
      r = pocl_metal_generate_metallib (program, device_i, &d->target, &lib,
                                        &lib_size, &region, &gvar_size,
                                        &gvar_init, &relocs, &num_relocs,
                                        &error_log);
      if (r != 0)
        {
          err = CL_BUILD_PROGRAM_FAILURE;
          goto out;
        }

      header[0] = region;
      header[1] = gvar_size;
      header[2] = num_relocs;
      size_t relocs_size = num_relocs * sizeof (pocl_metal_gvar_reloc_t);
      content_size = sizeof (header) + gvar_size + relocs_size + lib_size;
      content = malloc (content_size);
      memcpy (content, header, sizeof (header));
      if (gvar_size)
        memcpy (content + sizeof (header), gvar_init, gvar_size);
      if (relocs_size)
        memcpy (content + sizeof (header) + gvar_size, relocs, relocs_size);
      memcpy (content + sizeof (header) + gvar_size + relocs_size, lib,
              lib_size);
      free (lib);
      free (gvar_init);
      free (relocs);
      pocl_write_file (cache_path, content, content_size, 0);
    }

  memcpy (header, content, sizeof (header));
  size_t relocs_size = header[2] * sizeof (pocl_metal_gvar_reloc_t);
  if (content_size < sizeof (header) + header[1] + relocs_size)
    {
      error_log = strdup ("corrupt cached Metal library");
      pocl_remove (cache_path);
      err = CL_BUILD_PROGRAM_FAILURE;
      goto out;
    }
  pdata->local_region_size = header[0];
  if (header[1] > 0)
    {
      id<MTLBuffer> gvars = [metal_device (d)
          newBufferWithBytes:content + sizeof (header)
                      length:header[1]
                     options:MTLResourceStorageModeShared];
      if (gvars == nil)
        {
          error_log = strdup ("could not allocate the program-scope "
                              "variable storage");
          err = CL_OUT_OF_RESOURCES;
          goto out;
        }
      /* pointers between program-scope variables */
      const pocl_metal_gvar_reloc_t *relocs
          = (const pocl_metal_gvar_reloc_t *)(content + sizeof (header)
                                              + header[1]);
      for (unsigned i = 0; i < header[2]; ++i)
        {
          pocl_metal_gvar_reloc_t rel;
          memcpy (&rel, relocs + i, sizeof (rel));
          uint64_t addr = gvars.gpuAddress + rel.target;
          if (rel.offset + sizeof (addr) <= header[1])
            memcpy ((char *)gvars.contents + rel.offset, &addr,
                    sizeof (addr));
        }
      pdata->gvar_buffer = (__bridge_retained void *)gvars;
    }
  size_t lib_offset = sizeof (header) + header[1] + relocs_size;
  if (metal_create_library (d, pdata, content + lib_offset,
                            content_size - lib_offset, &error_log)
      != 0)
    {
      /* don't keep a library Metal can't load */
      pocl_remove (cache_path);
      err = CL_BUILD_PROGRAM_FAILURE;
      goto out;
    }

out:
  POCL_UNLOCK (d->compile_lock);
  POCL_MEM_FREE (content);

  if (err != CL_SUCCESS)
    {
      const char *msg = error_log ? error_log : "unknown error";
      POCL_MSG_ERR ("[Metal] failed to build the program: %s\n", msg);
      size_t len = strlen (msg) + 64;
      char *log = malloc (len);
      snprintf (log, len, "Metal: failed to build the program: %s\n", msg);
      pocl_append_to_buildlog (program, device_i, log, strlen (log));
      free (error_log);
      if (pdata->gvar_buffer)
        CFRelease (pdata->gvar_buffer);
      free (pdata);
      return err;
    }

  program->data[device_i] = pdata;
  return CL_SUCCESS;
}

int
pocl_metal_free_program (cl_device_id device, cl_program program,
                         unsigned program_device_i)
{
  pocl_driver_free_program (device, program, program_device_i);
  pocl_metal_program_data_t *pdata
      = (pocl_metal_program_data_t *)program->data[program_device_i];
  if (pdata == NULL)
    return CL_SUCCESS;
  if (pdata->library)
    CFRelease (pdata->library);
  if (pdata->gvar_buffer)
    CFRelease (pdata->gvar_buffer);
  POCL_MEM_FREE (program->data[program_device_i]);
  return CL_SUCCESS;
}

int
pocl_metal_compile_kernel (_cl_command_node *cmd, cl_kernel kernel,
                           cl_device_id device, int specialize)
{
  return CL_SUCCESS;
}

int
pocl_metal_create_kernel (cl_device_id device, cl_program program,
                          cl_kernel kernel, unsigned device_i)
{
  pocl_metal_device_data_t *d = (pocl_metal_device_data_t *)device->data;
  pocl_kernel_metadata_t *meta = kernel->meta;
  pocl_metal_kernel_data_t *kdata
      = (pocl_metal_kernel_data_t *)meta->devices[device_i].driver_data;
  if (kdata != NULL)
    {
      ++kdata->refcount;
      return CL_SUCCESS;
    }

  pocl_metal_program_data_t *pdata
      = (pocl_metal_program_data_t *)program->data[device_i];
  POCL_RETURN_ERROR_ON ((pdata == NULL || pdata->library == NULL),
                        CL_INVALID_PROGRAM_EXECUTABLE,
                        "[Metal] the program was not built for the device\n");

  @autoreleasepool
    {
      id<MTLLibrary> lib = (__bridge id<MTLLibrary>)pdata->library;
      id<MTLFunction> fn =
          [lib newFunctionWithName:[NSString stringWithUTF8String:kernel->name]];
      POCL_RETURN_ERROR_ON ((fn == nil), CL_INVALID_KERNEL_NAME,
                            "[Metal] kernel %s not found in the library\n",
                            kernel->name);
      NSError *error = nil;
      id<MTLComputePipelineState> pso =
          [metal_device (d) newComputePipelineStateWithFunction:fn
                                                          error:&error];
      POCL_RETURN_ERROR_ON (
          (pso == nil), CL_OUT_OF_RESOURCES,
          "[Metal] could not create a pipeline for kernel %s: %s\n",
          kernel->name,
          error ? error.localizedDescription.UTF8String : "unknown error");

      kdata = calloc (1, sizeof (pocl_metal_kernel_data_t));
      kdata->pipeline = (__bridge_retained void *)pso;
      kdata->refcount = 1;
      meta->devices[device_i].driver_data = kdata;
      meta->devices[device_i].max_workgroup_size
          = pso.maxTotalThreadsPerThreadgroup;
      meta->devices[device_i].preferred_wg_multiple
          = pso.threadExecutionWidth;
      meta->devices[device_i].local_mem_size
          = pso.staticThreadgroupMemoryLength;
    }
  return CL_SUCCESS;
}

int
pocl_metal_free_kernel (cl_device_id device, cl_program program,
                        cl_kernel kernel, unsigned device_i)
{
  pocl_kernel_metadata_t *meta = kernel->meta;
  pocl_metal_kernel_data_t *kdata
      = (pocl_metal_kernel_data_t *)meta->devices[device_i].driver_data;
  if (kdata == NULL)
    return CL_SUCCESS;
  if (--kdata->refcount == 0)
    {
      if (kdata->pipeline)
        CFRelease (kdata->pipeline);
      POCL_MEM_FREE (meta->devices[device_i].driver_data);
    }
  return CL_SUCCESS;
}

static void
pocl_metal_compute_local_size (cl_device_id dev, cl_kernel kernel,
                               unsigned device_i, size_t max_group_size,
                               size_t global_x, size_t global_y,
                               size_t global_z, size_t *local_x,
                               size_t *local_y, size_t *local_z)
{
  pocl_metal_kernel_data_t *kdata
      = (pocl_metal_kernel_data_t *)kernel->meta->devices[device_i]
            .driver_data;
  if (kdata != NULL)
    {
      id<MTLComputePipelineState> pso
          = (__bridge id<MTLComputePipelineState>)kdata->pipeline;
      if (pso.maxTotalThreadsPerThreadgroup < max_group_size)
        max_group_size = pso.maxTotalThreadsPerThreadgroup;
    }
  pocl_default_local_size_optimizer (dev, kernel, device_i, max_group_size,
                                     global_x, global_y, global_z, local_x,
                                     local_y, local_z);
}

/***************************************************************************/
/* kernel execution */

void
pocl_metal_run (void *data, _cl_command_node *cmd)
{
  pocl_metal_device_data_t *d = (pocl_metal_device_data_t *)data;
  _cl_command_run *run = &cmd->command.run;
  cl_kernel kernel = run->kernel;
  pocl_kernel_metadata_t *meta = kernel->meta;
  struct pocl_context *pc = &run->pc;
  unsigned dev_i = cmd->program_device_i;

  if (pc->num_groups[0] == 0 || pc->num_groups[1] == 0
      || pc->num_groups[2] == 0)
    return;

  pocl_metal_kernel_data_t *kdata
      = (pocl_metal_kernel_data_t *)meta->devices[dev_i].driver_data;
  pocl_metal_program_data_t *pdata
      = (pocl_metal_program_data_t *)kernel->program->data[dev_i];
  if (kdata == NULL || pdata == NULL)
    {
      POCL_MSG_ERR ("[Metal] kernel %s was not built for the device\n",
                    kernel->name);
      return;
    }

  POCL_LOCK (d->run_lock);
  metal_run_locked (d, cmd, kdata, pdata);
  POCL_UNLOCK (d->run_lock);
}

static void
metal_run_locked (pocl_metal_device_data_t *d, _cl_command_node *cmd,
                  pocl_metal_kernel_data_t *kdata,
                  pocl_metal_program_data_t *pdata)
{
  cl_device_id dev = cmd->device;
  _cl_command_run *run = &cmd->command.run;
  cl_kernel kernel = run->kernel;
  pocl_kernel_metadata_t *meta = kernel->meta;
  struct pocl_context *pc = &run->pc;

  @autoreleasepool
    {
      if (d->printf_buffer == NULL)
        {
          id<MTLBuffer> pb = [metal_device (d)
              newBufferWithLength:POCL_METAL_PRINTF_HEADER_SIZE
                                  + dev->printf_buffer_size
                          options:MTLResourceStorageModeShared];
          if (pb == nil)
            {
              POCL_MSG_ERR ("[Metal] could not allocate the printf buffer\n");
              return;
            }
          d->printf_buffer = (__bridge_retained void *)pb;
        }
      id<MTLBuffer> printf_buf = (__bridge id<MTLBuffer>)d->printf_buffer;
      uint32_t *printf_header = (uint32_t *)printf_buf.contents;
      printf_header[0] = 0;
      printf_header[1] = (uint32_t)dev->printf_buffer_size;

      id<MTLComputePipelineState> pso
          = (__bridge id<MTLComputePipelineState>)kdata->pipeline;
      size_t wg_size = pc->local_size[0] * pc->local_size[1]
                       * pc->local_size[2];
      if (wg_size > pso.maxTotalThreadsPerThreadgroup)
        {
          POCL_MSG_ERR ("[Metal] kernel %s: work-group size %zu exceeds the "
                        "pipeline maximum %zu\n",
                        kernel->name, wg_size,
                        (size_t)pso.maxTotalThreadsPerThreadgroup);
          return;
        }

      id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)d->queue;
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];

      uint32_t local_offset = 0;
      uint32_t null_args = 0;
      for (unsigned i = 0; i < meta->num_args; ++i)
        {
          pocl_argument *arg = &run->arguments[i];
          pocl_argument_info *ai = &meta->arg_info[i];
          switch (ai->type)
            {
            case POCL_ARG_TYPE_NONE:
              [enc setBytes:arg->value length:arg->size atIndex:i];
              break;
            case POCL_ARG_TYPE_POINTER:
              if (ai->address_qualifier == CL_KERNEL_ARG_ADDRESS_LOCAL)
                {
                  uint32_t off = local_offset;
                  local_offset += (uint32_t)arg->size;
                  local_offset = (local_offset + POCL_METAL_LOCAL_ARG_ALIGN - 1)
                                 & ~(uint32_t)(POCL_METAL_LOCAL_ARG_ALIGN - 1);
                  [enc setBytes:&off length:sizeof (off) atIndex:i];
                }
              else if (arg->is_raw_ptr)
                {
                  POCL_MSG_ERR ("[Metal] SVM / USM pointers are not "
                                "supported\n");
                  [enc endEncoding];
                  return;
                }
              else if (arg->value == NULL || *(cl_mem *)arg->value == NULL)
                {
                  null_args |= 1u << i;
                  [enc setBuffer:nil offset:0 atIndex:i];
                }
              else
                {
                  cl_mem mem = *(cl_mem *)arg->value;
                  size_t offset = arg->offset;
                  /* sub-buffers share the parent's MTLBuffer */
                  if (mem->parent != NULL)
                    {
                      offset += mem->origin;
                      mem = mem->parent;
                    }
                  pocl_mem_identifier *p
                      = &mem->device_ptrs[dev->global_mem_id];
                  id<MTLBuffer> buf = (__bridge id<MTLBuffer>)p->extra_ptr;
                  [enc setBuffer:buf offset:offset atIndex:i];
                }
              break;
            default:
              POCL_MSG_ERR ("[Metal] unsupported argument type for "
                            "argument %u of kernel %s\n",
                            i, kernel->name);
              [enc endEncoding];
              return;
            }
        }

      if (local_offset > pdata->local_region_size)
        {
          POCL_MSG_ERR ("[Metal] kernel %s: the __local arguments need %u "
                        "bytes, but only %u are available\n",
                        kernel->name, local_offset, pdata->local_region_size);
          [enc endEncoding];
          return;
        }

      pocl_metal_launch_params_t lp;
      lp.work_dim = pc->work_dim;
      lp.null_args = null_args;
      for (unsigned i = 0; i < 3; ++i)
        lp.global_offset[i] = (unsigned)pc->global_offset[i];
      [enc setBytes:&lp length:sizeof (lp) atIndex:meta->num_args];
      [enc setBuffer:printf_buf offset:0 atIndex:meta->num_args + 1];
      [enc setBuffer:(__bridge id<MTLBuffer>)pdata->gvar_buffer
              offset:0
             atIndex:meta->num_args + 2];

      MTLSize grid = MTLSizeMake (pc->num_groups[0] * pc->local_size[0],
                                  pc->num_groups[1] * pc->local_size[1],
                                  pc->num_groups[2] * pc->local_size[2]);
      MTLSize tg = MTLSizeMake (pc->local_size[0], pc->local_size[1],
                                pc->local_size[2]);
      [enc dispatchThreads:grid threadsPerThreadgroup:tg];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];

      if (cb.status == MTLCommandBufferStatusError)
        POCL_MSG_ERR ("[Metal] kernel %s failed: %s\n", kernel->name,
                      cb.error ? cb.error.localizedDescription.UTF8String
                               : "unknown error");
      else
        POCL_MSG_PRINT_METAL ("kernel %s: grid %zu x %zu x %zu, threadgroup "
                              "%zu x %zu x %zu, GPU time %.3f ms\n",
                              kernel->name, (size_t)grid.width,
                              (size_t)grid.height, (size_t)grid.depth,
                              (size_t)tg.width, (size_t)tg.height,
                              (size_t)tg.depth,
                              (cb.GPUEndTime - cb.GPUStartTime) * 1e3);

      uint32_t printf_bytes = printf_header[0];
      if (printf_bytes > printf_header[1])
        printf_bytes = printf_header[1];
      if (printf_bytes > 0)
        pocl_write_printf_buffer (
            (char *)printf_buf.contents + POCL_METAL_PRINTF_HEADER_SIZE,
            printf_bytes);
    }
}

/***************************************************************************/
/* command scheduling: commands that are ready to run are executed in order
   by a worker thread per device, so that enqueueing doesn't block (like the
   pthread driver) */

typedef struct
{
  pocl_cond_t cq_cond;
} metal_queue_data_t;

typedef struct
{
  pocl_cond_t event_cond;
} metal_event_data_t;

/* called with the event locked */
static void
metal_push_command (pocl_metal_device_data_t *d, _cl_command_node *node)
{
  POCL_LOCK (d->cq_lock);
  DL_APPEND (d->ready_list, node);
  POCL_SIGNAL_COND (d->work_cond);
  POCL_UNLOCK (d->cq_lock);
}

static void *
metal_worker (void *arg)
{
  pocl_metal_device_data_t *d = (pocl_metal_device_data_t *)arg;
  POCL_LOCK (d->cq_lock);
  while (1)
    {
      while (d->ready_list == NULL && !d->exit_worker)
        POCL_WAIT_COND (d->work_cond, d->cq_lock);
      if (d->ready_list == NULL && d->exit_worker)
        break;
      _cl_command_node *node = d->ready_list;
      DL_DELETE (d->ready_list, node);
      POCL_UNLOCK (d->cq_lock);
      @autoreleasepool
        {
          pocl_exec_command (node);
        }
      POCL_LOCK (d->cq_lock);
    }
  POCL_UNLOCK (d->cq_lock);
  return NULL;
}

static int
metal_start_worker (pocl_metal_device_data_t *d)
{
  POCL_INIT_COND (d->work_cond);
  d->exit_worker = 0;
  return pthread_create (&d->worker, NULL, metal_worker, d);
}

static void
metal_stop_worker (pocl_metal_device_data_t *d)
{
  POCL_LOCK (d->cq_lock);
  d->exit_worker = 1;
  POCL_SIGNAL_COND (d->work_cond);
  POCL_UNLOCK (d->cq_lock);
  pthread_join (d->worker, NULL);
  POCL_DESTROY_COND (d->work_cond);
}

void
pocl_metal_submit (_cl_command_node *node, cl_command_queue cq)
{
  pocl_metal_device_data_t *d = (pocl_metal_device_data_t *)node->device->data;

  node->state = POCL_COMMAND_READY;

  if (node->sync.event.event->failed_dependency)
    {
      POCL_UNLOCK_OBJ (node->sync.event.event);
      pocl_update_event_failed (CL_FAILED, NULL, 0, node->sync.event.event,
                                NULL);
      return;
    }

  if (pocl_command_is_ready (node->sync.event.event))
    {
      pocl_update_event_submitted (node->sync.event.event);
      metal_push_command (d, node);
    }
  POCL_UNLOCK_OBJ (node->sync.event.event);
}

void
pocl_metal_flush (cl_device_id device, cl_command_queue cq)
{
}

void
pocl_metal_join (cl_device_id device, cl_command_queue cq)
{
  metal_queue_data_t *qdata = (metal_queue_data_t *)cq->data;
  POCL_LOCK_OBJ (cq);
  while (cq->command_count != 0)
    POCL_WAIT_COND (qdata->cq_cond, cq->pocl_lock);
  POCL_UNLOCK_OBJ (cq);
}

void
pocl_metal_notify (cl_device_id device, cl_event event, cl_event finished)
{
  pocl_metal_device_data_t *d = (pocl_metal_device_data_t *)device->data;
  _cl_command_node *node = event->command;

  if (finished->status < CL_COMPLETE || event->failed_dependency)
    {
      pocl_unlock_events_inorder (event, finished);
      pocl_update_event_failed (CL_FAILED, NULL, 0, event, NULL);
      pocl_lock_events_inorder (finished, event);
      return;
    }

  if (node->state != POCL_COMMAND_READY)
    return;

  if (pocl_command_is_ready (event) && event->status == CL_QUEUED)
    {
      pocl_update_event_submitted (event);
      metal_push_command (d, node);
    }
}

void
pocl_metal_notify_cmdq_finished (cl_command_queue cq)
{
  /* called with the queue locked */
  metal_queue_data_t *qdata = (metal_queue_data_t *)cq->data;
  POCL_BROADCAST_COND (qdata->cq_cond);
}

void
pocl_metal_notify_event_finished (cl_event event)
{
  metal_event_data_t *e = (metal_event_data_t *)event->data;
  POCL_BROADCAST_COND (e->event_cond);
}

void
pocl_metal_update_event (cl_device_id device, cl_event event)
{
  if (event->data == NULL && event->status == CL_QUEUED)
    {
      metal_event_data_t *e = calloc (1, sizeof (metal_event_data_t));
      POCL_INIT_COND (e->event_cond);
      event->data = e;
    }
}

void
pocl_metal_wait_event (cl_device_id device, cl_event event)
{
  metal_event_data_t *e = (metal_event_data_t *)event->data;
  POCL_LOCK_OBJ (event);
  while (event->status > CL_COMPLETE)
    POCL_WAIT_COND (e->event_cond, event->pocl_lock);
  POCL_UNLOCK_OBJ (event);
}

void
pocl_metal_free_event_data (cl_event event)
{
  metal_event_data_t *e = (metal_event_data_t *)event->data;
  if (e == NULL)
    return;
  POCL_DESTROY_COND (e->event_cond);
  POCL_MEM_FREE (event->data);
}

int
pocl_metal_init_queue (cl_device_id device, cl_command_queue queue)
{
  metal_queue_data_t *qdata = calloc (1, sizeof (metal_queue_data_t));
  if (qdata == NULL)
    return CL_OUT_OF_HOST_MEMORY;
  POCL_INIT_COND (qdata->cq_cond);
  queue->data = qdata;
  return CL_SUCCESS;
}

int
pocl_metal_free_queue (cl_device_id device, cl_command_queue queue)
{
  metal_queue_data_t *qdata = (metal_queue_data_t *)queue->data;
  if (qdata == NULL)
    return CL_SUCCESS;
  POCL_DESTROY_COND (qdata->cq_cond);
  POCL_MEM_FREE (queue->data);
  return CL_SUCCESS;
}
