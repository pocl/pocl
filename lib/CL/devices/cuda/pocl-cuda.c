/* pocl-cuda.c - driver for CUDA devices

   Copyright (c) 2016-2017 James Price / University of Bristol

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal
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
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "config.h"

#include "common.h"
#include "devices.h"
#include "pocl-cuda.h"
#include "pocl-ptx-gen.h"
#include "pocl_cache.h"
#include "pocl_file_util.h"
#include "pocl_llvm.h"
#include "pocl_mem_management.h"
#include "pocl_runtime_config.h"
#include "pocl_util.h"

#include <string.h>

#include <cuda.h>

typedef struct pocl_cuda_device_data_s
{
  CUdevice device;
  CUcontext context;
  CUevent epoch_event;
  cl_ulong epoch;
  char libdevice[PATH_MAX];
  pocl_lock_t compile_lock;
} pocl_cuda_device_data_t;

typedef struct pocl_cuda_kernel_data_s
{
  CUmodule module;
  CUmodule module_offsets;
  CUfunction kernel;
  CUfunction kernel_offsets;
  size_t *alignments;
} pocl_cuda_kernel_data_t;

typedef struct pocl_cuda_event_data_s
{
  CUevent start;
  CUevent end;
  cl_int *ext_event_flag;
  unsigned num_ext_events;
} pocl_cuda_event_data_t;

extern unsigned int pocl_num_devices;

static void
pocl_cuda_abort_on_error (CUresult result, unsigned line, const char *func,
                          const char *code, const char *api)
{
  if (result != CUDA_SUCCESS)
    {
      const char *err_name;
      const char *err_string;
      cuGetErrorName (result, &err_name);
      cuGetErrorString (result, &err_string);
      POCL_MSG_PRINT2 (CUDA, func, line, "Error during %s\n", api);
      POCL_ABORT ("%s: %s\n", err_name, err_string);
    }
}

static int
pocl_cuda_error (CUresult result, unsigned line, const char *func,
                          const char *code, const char *api)
{
  int err = (result != CUDA_SUCCESS);
  if (err)
    {
      const char *err_name;
      const char *err_string;
      cuGetErrorName (result, &err_name);
      cuGetErrorString (result, &err_string);
      POCL_MSG_ERR ("CUDA error during %s. %s: %s\n", api, err_name, err_string);
    }
  return err;
}

#define CUDA_CHECK(result, api)                                               \
  pocl_cuda_abort_on_error (result, __LINE__, __FUNCTION__, #result, api)

#define CUDA_CHECK_ERROR(result, api)                                         \
  pocl_cuda_error (result, __LINE__, __FUNCTION__, #result, api)

void
pocl_cuda_init_device_ops (struct pocl_device_ops *ops)
{
  pocl_basic_init_device_ops (ops);

  ops->device_name = "CUDA";
  ops->init_device_infos = pocl_cuda_init_device_infos;
  ops->build_hash = pocl_cuda_build_hash;
  ops->probe = pocl_cuda_probe;
  ops->uninit = pocl_cuda_uninit;
  ops->init = pocl_cuda_init;
  ops->init_queue = pocl_cuda_init_queue;
  ops->free_queue = pocl_cuda_free_queue;
  ops->alloc_mem_obj = pocl_cuda_alloc_mem_obj;
  ops->free = pocl_cuda_free;
  ops->compile_kernel = pocl_cuda_compile_kernel;
  ops->map_mem = pocl_cuda_map_mem;
  ops->submit = pocl_cuda_submit;
  ops->notify = pocl_cuda_notify;
  ops->broadcast = pocl_cuda_broadcast;
  ops->wait_event = pocl_cuda_wait_event;
  ops->free_event_data = pocl_cuda_free_event_data;
  ops->join = pocl_cuda_join;
  ops->flush = pocl_cuda_flush;

  ops->read = NULL;
  ops->read_rect = NULL;
  ops->write = NULL;
  ops->write_rect = NULL;
  ops->copy = NULL;
  ops->copy_rect = NULL;
  ops->unmap_mem = NULL;
  ops->run = NULL;

  /* TODO: implement remaining ops functions: */
  /* get_timer_value */
  /* update_event */
}

cl_int
pocl_cuda_init (unsigned j, cl_device_id dev, const char *parameters)
{
  CUresult result;
  int ret = CL_SUCCESS;

  if (dev->data)
    return ret;

  pocl_cuda_device_data_t *data = calloc (1, sizeof (pocl_cuda_device_data_t));
  result = cuDeviceGet (&data->device, j);
  if (CUDA_CHECK_ERROR (result, "cuDeviceGet"))
    ret = CL_INVALID_DEVICE;

  // Get specific device name
  dev->long_name = dev->short_name = calloc (256, sizeof (char));

  if (ret != CL_INVALID_DEVICE)
    cuDeviceGetName (dev->long_name, 256, data->device);
  else
    snprintf (dev->long_name, 255, "Unavailable CUDA device #%d", j);

  SETUP_DEVICE_CL_VERSION (CUDA_DEVICE_CL_VERSION_MAJOR,
                           CUDA_DEVICE_CL_VERSION_MINOR);

  // Get other device properties
  if (ret != CL_INVALID_DEVICE)
    {
      cuDeviceGetAttribute ((int *)&dev->max_work_group_size,
                            CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                            data->device);
      cuDeviceGetAttribute ((int *)(dev->max_work_item_sizes + 0),
                            CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, data->device);
      cuDeviceGetAttribute ((int *)(dev->max_work_item_sizes + 1),
                            CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, data->device);
      cuDeviceGetAttribute ((int *)(dev->max_work_item_sizes + 2),
                            CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, data->device);
      cuDeviceGetAttribute (
          (int *)&dev->local_mem_size,
          CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, data->device);
      cuDeviceGetAttribute ((int *)&dev->max_compute_units,
                            CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                            data->device);
      cuDeviceGetAttribute ((int *)&dev->error_correction_support,
                            CU_DEVICE_ATTRIBUTE_ECC_ENABLED, data->device);
      cuDeviceGetAttribute ((int *)&dev->host_unified_memory,
                            CU_DEVICE_ATTRIBUTE_INTEGRATED, data->device);
      cuDeviceGetAttribute ((int *)&dev->max_constant_buffer_size,
                            CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
                            data->device);
      cuDeviceGetAttribute ((int *)&dev->max_clock_frequency,
                            CU_DEVICE_ATTRIBUTE_CLOCK_RATE, data->device);
      dev->max_clock_frequency /= 1000;
    }

  dev->preferred_wg_size_multiple = 32;
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
  dev->double_fp_config = CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO
                          | CL_FP_ROUND_TO_INF | CL_FP_FMA | CL_FP_INF_NAN
                          | CL_FP_DENORM;

  dev->local_mem_type = CL_LOCAL;
  dev->host_unified_memory = 0;

  // Get GPU architecture name
  int sm_maj = 0, sm_min = 0;
  if (ret != CL_INVALID_DEVICE)
    {
      cuDeviceGetAttribute (&sm_maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                            data->device);
      cuDeviceGetAttribute (&sm_min, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                            data->device);
    }
  char *gpu_arch = calloc (16, sizeof (char));
  snprintf (gpu_arch, 16, "sm_%d%d", sm_maj, sm_min);
  dev->llvm_cpu = pocl_get_string_option ("POCL_CUDA_GPU_ARCH", gpu_arch);
  POCL_MSG_PRINT_INFO ("[CUDA] GPU architecture = %s\n", dev->llvm_cpu);

  // Find libdevice library
  if (findLibDevice (data->libdevice, dev->llvm_cpu))
    {
      if (ret != CL_INVALID_DEVICE)
        {
          POCL_MSG_ERR ("[CUDA] failed to find libdevice library\n");
          dev->compiler_available = 0;
        }
    }

  // Create context
  if (ret != CL_INVALID_DEVICE)
    {
      result = cuCtxCreate (&data->context, CU_CTX_MAP_HOST, data->device);
      if (CUDA_CHECK_ERROR (result, "cuCtxCreate"))
        ret = CL_INVALID_DEVICE;
    }

  // Create epoch event for timing info
  if (ret != CL_INVALID_DEVICE)
    {
      result = cuEventCreate (&data->epoch_event, CU_EVENT_DEFAULT);
      CUDA_CHECK_ERROR (result, "cuEventCreate");

      data->epoch = dev->ops->get_timer_value (dev->data);

      result = cuEventRecord (data->epoch_event, 0);
      result = cuEventSynchronize (data->epoch_event);
      if (CUDA_CHECK_ERROR (result, "cuEventSynchronize"))
        ret = CL_INVALID_DEVICE;
    }

  // Get global memory size
  size_t memfree = 0, memtotal = 0;
  if (ret != CL_INVALID_DEVICE)
    result = cuMemGetInfo (&memfree, &memtotal);
  dev->max_mem_alloc_size = max (memtotal / 4, 128 * 1024 * 1024);
  dev->global_mem_size = memtotal;

  dev->data = data;

  POCL_INIT_LOCK (data->compile_lock);
  return ret;
}

cl_int
pocl_cuda_init_queue (cl_command_queue queue)
{
  cuCtxSetCurrent (((pocl_cuda_device_data_t *)queue->device->data)->context);

  queue->data = calloc (1, sizeof (CUstream));
  CUresult result = cuStreamCreate (queue->data, CU_STREAM_NON_BLOCKING);
  if (CUDA_CHECK_ERROR (result, "cuStreamCreate"))
    return CL_OUT_OF_RESOURCES;

  return CL_SUCCESS;
}

void
pocl_cuda_free_queue (cl_command_queue queue)
{
  cuCtxSetCurrent (((pocl_cuda_device_data_t *)queue->device->data)->context);
  cuStreamDestroy (*(CUstream *)queue->data);
}

char *
pocl_cuda_build_hash (cl_device_id device)
{
  char *res = calloc (1000, sizeof (char));
  snprintf (res, 1000, "CUDA-%s", device->llvm_cpu);
  return res;
}

void
pocl_cuda_init_device_infos (unsigned j, struct _cl_device_id *dev)
{
  pocl_basic_init_device_infos (j, dev);

  dev->type = CL_DEVICE_TYPE_GPU;
  dev->address_bits = (sizeof (void *) * 8);
  dev->llvm_target_triplet = (sizeof (void *) == 8) ? "nvptx64" : "nvptx";
  dev->spmd = CL_TRUE;
  dev->workgroup_pass = CL_FALSE;
  dev->execution_capabilities = CL_EXEC_KERNEL;

  dev->global_as_id = 1;
  dev->local_as_id = 3;
  dev->constant_as_id = 1;

  // TODO: Get images working
  dev->image_support = CL_FALSE;
  // always enable 64bit ints
  dev->has_64bit_long = 1;
}

unsigned int
pocl_cuda_probe (struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count (ops->device_name);

  int probe_count = 0;
  CUresult ret = cuInit (0);
  if (ret == CUDA_SUCCESS)
    {
      ret = cuDeviceGetCount (&probe_count);
      if (ret != CUDA_SUCCESS)
        probe_count = 0;
    }

  /* If the user requested a specific number of CUDA devices,
   * pretend we only have that many, if we can. If they requested
   * more than there are, abort informing the user of the issue.
   */
  if (env_count >= 0)
    {
      if (env_count > probe_count)
        POCL_ABORT ("[CUDA] %d devices requested, but only %d are available\n",
          env_count, probe_count);
      probe_count = env_count;
    }

  return probe_count;
}

void
pocl_cuda_uninit (cl_device_id device)
{
  pocl_cuda_device_data_t *data = device->data;

  if (device->available)
      cuCtxDestroy (data->context);

  POCL_MEM_FREE (data);
  device->data = NULL;

  POCL_MEM_FREE (device->long_name);
}

cl_int
pocl_cuda_alloc_mem_obj (cl_device_id device, cl_mem mem_obj, void *host_ptr)
{
  cuCtxSetCurrent (((pocl_cuda_device_data_t *)device->data)->context);

  CUresult result;
  void *b = NULL;

  /* if memory for this global memory is not yet allocated -> do it */
  if (mem_obj->device_ptrs[device->global_mem_id].mem_ptr == NULL)
    {
      cl_mem_flags flags = mem_obj->flags;

      if (flags & CL_MEM_USE_HOST_PTR)
        {
#if defined __arm__
          // cuMemHostRegister is not supported on ARN
          // Allocate device memory and perform explicit copies
          // before and after running a kernel
          result = cuMemAlloc ((CUdeviceptr *)&b, mem_obj->size);
          CUDA_CHECK (result, "cuMemAlloc");
#else
          result = cuMemHostRegister (host_ptr, mem_obj->size,
                                      CU_MEMHOSTREGISTER_DEVICEMAP);
          if (result != CUDA_SUCCESS
              && result != CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED)
            CUDA_CHECK (result, "cuMemHostRegister");
          result = cuMemHostGetDevicePointer ((CUdeviceptr *)&b, host_ptr, 0);
          CUDA_CHECK (result, "cuMemHostGetDevicePointer");
#endif
        }
      else if (flags & CL_MEM_ALLOC_HOST_PTR)
        {
          result = cuMemHostAlloc (&mem_obj->mem_host_ptr, mem_obj->size,
                                   CU_MEMHOSTREGISTER_DEVICEMAP);
          CUDA_CHECK (result, "cuMemHostAlloc");
          result = cuMemHostGetDevicePointer ((CUdeviceptr *)&b,
                                              mem_obj->mem_host_ptr, 0);
          CUDA_CHECK (result, "cuMemHostGetDevicePointer");
        }
      else
        {
          result = cuMemAlloc ((CUdeviceptr *)&b, mem_obj->size);
          if (result != CUDA_SUCCESS)
            {
              const char *err;
              cuGetErrorName (result, &err);
              POCL_MSG_PRINT2 (CUDA, __FUNCTION__, __LINE__,
                               "-> Failed to allocate memory: %s\n", err);
              return CL_MEM_OBJECT_ALLOCATION_FAILURE;
            }
        }

      if (flags & CL_MEM_COPY_HOST_PTR)
        {
          result = cuMemcpyHtoD ((CUdeviceptr)b, host_ptr, mem_obj->size);
          CUDA_CHECK (result, "cuMemcpyHtoD");

          result = cuStreamSynchronize (0);
          CUDA_CHECK (result, "cuStreamSynchronize");
        }

      mem_obj->device_ptrs[device->global_mem_id].mem_ptr = b;
      mem_obj->device_ptrs[device->global_mem_id].global_mem_id
          = device->global_mem_id;
    }

  /* copy already allocated global mem info to devices own slot */
  mem_obj->device_ptrs[device->dev_id]
      = mem_obj->device_ptrs[device->global_mem_id];

  return CL_SUCCESS;
}

void
pocl_cuda_free (cl_device_id device, cl_mem mem_obj)
{
  cuCtxSetCurrent (((pocl_cuda_device_data_t *)device->data)->context);

  if (mem_obj->flags & CL_MEM_ALLOC_HOST_PTR)
    {
      cuMemFreeHost (mem_obj->mem_host_ptr);
      mem_obj->mem_host_ptr = NULL;
    }
  else if (mem_obj->flags & CL_MEM_USE_HOST_PTR)
    {
      cuMemHostUnregister (mem_obj->mem_host_ptr);
      mem_obj->mem_host_ptr = NULL;
    }
  else
    {
      void *ptr = mem_obj->device_ptrs[device->dev_id].mem_ptr;
      cuMemFree ((CUdeviceptr)ptr);
    }
}

void
pocl_cuda_submit_read (CUstream stream, void *host_ptr, const void *device_ptr,
                       size_t offset, size_t cb)
{
  CUresult result = cuMemcpyDtoHAsync (
      host_ptr, (CUdeviceptr) (device_ptr + offset), cb, stream);
  CUDA_CHECK (result, "cuMemcpyDtoHAsync");
}

void
pocl_cuda_submit_write (CUstream stream, const void *host_ptr,
                        void *device_ptr, size_t offset, size_t cb)
{
  CUresult result = cuMemcpyHtoDAsync ((CUdeviceptr) (device_ptr + offset),
                                       host_ptr, cb, stream);
  CUDA_CHECK (result, "cuMemcpyHtoDAsync");
}

void
pocl_cuda_submit_copy (CUstream stream, const void *src_ptr, size_t src_offset,
                       void *__restrict__ dst_ptr, size_t dst_offset,
                       size_t cb)
{
  if (src_ptr == dst_ptr)
    return;

  CUresult result
      = cuMemcpyDtoDAsync ((CUdeviceptr) (dst_ptr + dst_offset),
                           (CUdeviceptr) (src_ptr + src_offset), cb, stream);
  CUDA_CHECK (result, "cuMemcpyDtoDAsync");
}

void
pocl_cuda_submit_read_rect (CUstream stream, void *__restrict__ const host_ptr,
                            void *__restrict__ const device_ptr,
                            const size_t *__restrict__ const buffer_origin,
                            const size_t *__restrict__ const host_origin,
                            const size_t *__restrict__ const region,
                            size_t const buffer_row_pitch,
                            size_t const buffer_slice_pitch,
                            size_t const host_row_pitch,
                            size_t const host_slice_pitch)
{
  CUDA_MEMCPY3D params = { 0 };

  params.WidthInBytes = region[0];
  params.Height = region[1];
  params.Depth = region[2];

  params.dstMemoryType = CU_MEMORYTYPE_HOST;
  params.dstHost = host_ptr;
  params.dstXInBytes = host_origin[0];
  params.dstY = host_origin[1];
  params.dstZ = host_origin[2];
  params.dstPitch = host_row_pitch;
  params.dstHeight = host_slice_pitch / host_row_pitch;

  params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  params.srcDevice = (CUdeviceptr)device_ptr;
  params.srcXInBytes = buffer_origin[0];
  params.srcY = buffer_origin[1];
  params.srcZ = buffer_origin[2];
  params.srcPitch = buffer_row_pitch;
  params.srcHeight = buffer_slice_pitch / buffer_row_pitch;

  CUresult result = cuMemcpy3DAsync (&params, stream);
  CUDA_CHECK (result, "cuMemcpy3DAsync");
}

void
pocl_cuda_submit_write_rect (CUstream stream,
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
  CUDA_MEMCPY3D params = { 0 };

  params.WidthInBytes = region[0];
  params.Height = region[1];
  params.Depth = region[2];

  params.srcMemoryType = CU_MEMORYTYPE_HOST;
  params.srcHost = host_ptr;
  params.srcXInBytes = host_origin[0];
  params.srcY = host_origin[1];
  params.srcZ = host_origin[2];
  params.srcPitch = host_row_pitch;
  params.srcHeight = host_slice_pitch / host_row_pitch;

  params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  params.dstDevice = (CUdeviceptr)device_ptr;
  params.dstXInBytes = buffer_origin[0];
  params.dstY = buffer_origin[1];
  params.dstZ = buffer_origin[2];
  params.dstPitch = buffer_row_pitch;
  params.dstHeight = buffer_slice_pitch / buffer_row_pitch;

  CUresult result = cuMemcpy3DAsync (&params, stream);
  CUDA_CHECK (result, "cuMemcpy3DAsync");
}

void
pocl_cuda_submit_copy_rect (CUstream stream,
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
  CUDA_MEMCPY3D params = { 0 };

  params.WidthInBytes = region[0];
  params.Height = region[1];
  params.Depth = region[2];

  params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  params.srcDevice = (CUdeviceptr)src_ptr;
  params.srcXInBytes = src_origin[0];
  params.srcY = src_origin[1];
  params.srcZ = src_origin[2];
  params.srcPitch = src_row_pitch;
  params.srcHeight = src_slice_pitch / src_row_pitch;

  params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  params.dstDevice = (CUdeviceptr)dst_ptr;
  params.dstXInBytes = dst_origin[0];
  params.dstY = dst_origin[1];
  params.dstZ = dst_origin[2];
  params.dstPitch = dst_row_pitch;
  params.dstHeight = dst_slice_pitch / dst_row_pitch;

  CUresult result = cuMemcpy3DAsync (&params, stream);
  CUDA_CHECK (result, "cuMemcpy3DAsync");
}

void *
pocl_cuda_map_mem (void *data, void *buf_ptr, size_t offset, size_t size,
                   void *host_ptr)
{
  assert (host_ptr == NULL);

  return malloc (size);
}

void
pocl_cuda_submit_map_mem (CUstream stream, void *buf_ptr, size_t offset,
                          size_t size, void *host_ptr)
{
  assert (host_ptr != NULL);

  // TODO: Map instead of copy?
  // TODO: don't copy if mapped as CL_MAP_WRITE_INVALIDATE_REGION
  CUresult result = cuMemcpyDtoHAsync (
      host_ptr, (CUdeviceptr) (buf_ptr + offset), size, stream);
  CUDA_CHECK (result, "cuMemcpyDtoHAsync");
}

void *
pocl_cuda_submit_unmap_mem (CUstream stream, void *host_ptr,
                            void *device_start_ptr, size_t offset, size_t size)
{
  if (host_ptr)
    {
      // TODO: Only copy back if mapped for writing
      CUresult result = cuMemcpyHtoDAsync (
          (CUdeviceptr) (device_start_ptr + offset), host_ptr, size, stream);
      CUDA_CHECK (result, "cuMemcpyHtoDAsync");
    }
  return NULL;
}

static pocl_cuda_kernel_data_t *
load_or_generate_kernel (cl_kernel kernel, cl_device_id device,
                         int has_offsets)
{
  CUresult result;

  // Check if we already have a compiled kernel function
  pocl_cuda_kernel_data_t *kdata = (pocl_cuda_kernel_data_t *)kernel->data;
  if (kdata)
    {
      kdata += device->dev_id;
      if ((has_offsets && kdata->kernel_offsets)
          || (!has_offsets && kdata->kernel))
        return kdata;
    }
  else
    {
      // TODO: when can we release this?
      kernel->data
          = calloc (pocl_num_devices, sizeof (pocl_cuda_kernel_data_t));
      kdata = kernel->data + device->dev_id;
    }

  pocl_cuda_device_data_t *ddata = (pocl_cuda_device_data_t *)device->data;
  cuCtxSetCurrent (ddata->context);

  POCL_LOCK(ddata->compile_lock);

  // Generate the parallel bitcode file linked with the kernel library
  int error = pocl_llvm_generate_workgroup_function (device, kernel, 0, 0, 0);
  if (error)
    {
      POCL_MSG_PRINT_GENERAL ("pocl_llvm_generate_workgroup_function() failed"
                              " for kernel %s\n", kernel->name);
      assert (error == 0);
    }

  char bc_filename[POCL_FILENAME_LENGTH];
  unsigned device_i = pocl_cl_device_to_index (kernel->program, device);
  pocl_cache_work_group_function_path (bc_filename, kernel->program, device_i,
                                       kernel, 0, 0, 0);

  char ptx_filename[POCL_FILENAME_LENGTH];
  strcpy (ptx_filename, bc_filename);
  if (has_offsets)
    strncat (ptx_filename, ".offsets", POCL_FILENAME_LENGTH - 1);
  strncat (ptx_filename, ".ptx", POCL_FILENAME_LENGTH - 1);

  if (!pocl_exists (ptx_filename))
    {
      // Generate PTX from LLVM bitcode
      if (pocl_ptx_gen (bc_filename, ptx_filename, kernel->name,
                        device->llvm_cpu,
                        ((pocl_cuda_device_data_t *)device->data)->libdevice,
                        has_offsets))
        POCL_ABORT ("pocl-cuda: failed to generate PTX\n");
    }

  // Load PTX module
  // TODO: When can we unload the module?
  CUmodule module;
  result = cuModuleLoad (&module, ptx_filename);
  CUDA_CHECK (result, "cuModuleLoad");

  // Get kernel function
  CUfunction function;
  result = cuModuleGetFunction (&function, module, kernel->name);
  CUDA_CHECK (result, "cuModuleGetFunction");

  // Get pointer aligment
  if (!kdata->alignments)
    {
      kdata->alignments = calloc (kernel->num_args + kernel->num_locals + 4,
                                  sizeof (size_t));
      pocl_cuda_get_ptr_arg_alignment (bc_filename, kernel->name,
                                       kdata->alignments);
    }

  if (has_offsets)
    {
      kdata->module_offsets = module;
      kdata->kernel_offsets = function;
    }
  else
    {
      kdata->module = module;
      kdata->kernel = function;
    }

  POCL_UNLOCK(ddata->compile_lock);

  return kdata;
}

void
pocl_cuda_compile_kernel (_cl_command_node *cmd, cl_kernel kernel,
                          cl_device_id device)
{
  load_or_generate_kernel (kernel, device, 0);
}

void
pocl_cuda_submit_kernel (CUstream stream, _cl_command_run run,
                         cl_device_id device, cl_event event)
{
  cl_kernel kernel = run.kernel;
  pocl_argument *arguments = run.arguments;
  struct pocl_context pc = run.pc;

  // Check if we need to handle global work offsets
  int has_offsets = 0;
  if (pc.global_offset[0] || pc.global_offset[1] || pc.global_offset[2])
    has_offsets = 1;

  // Get kernel function
  pocl_cuda_kernel_data_t *kdata
      = load_or_generate_kernel (kernel, device, has_offsets);
  CUmodule module = has_offsets ? kdata->module_offsets : kdata->module;
  CUfunction function = has_offsets ? kdata->kernel_offsets : kdata->kernel;

  // Prepare kernel arguments
  void *null = NULL;
  unsigned sharedMemBytes = 0;
  void *params[kernel->num_args + kernel->num_locals + 4];
  unsigned sharedMemOffsets[kernel->num_args + kernel->num_locals];
  unsigned constantMemBytes = 0;
  unsigned constantMemOffsets[kernel->num_args];
  unsigned globalOffsets[3];

  // Get handle to constant memory buffer
  size_t constant_mem_size;
  CUdeviceptr constant_mem_base = 0;
  cuModuleGetGlobal (&constant_mem_base, &constant_mem_size, module,
                     "_constant_memory_region_");

  CUresult result;
  unsigned i;
  for (i = 0; i < kernel->num_args; i++)
    {
      pocl_argument_type type = kernel->arg_info[i].type;
      switch (type)
        {
        case POCL_ARG_TYPE_NONE:
          params[i] = arguments[i].value;
          break;
        case POCL_ARG_TYPE_POINTER:
          {
            if (kernel->arg_info[i].is_local)
              {
                size_t size = arguments[i].size;
                size_t align = kdata->alignments[i];

                // Pad offset to align memory
                if (sharedMemBytes % align)
                  sharedMemBytes += align - (sharedMemBytes % align);

                sharedMemOffsets[i] = sharedMemBytes;
                params[i] = sharedMemOffsets + i;

                sharedMemBytes += size;
              }
            else if (kernel->arg_info[i].address_qualifier
                     == CL_KERNEL_ARG_ADDRESS_CONSTANT)
              {
                assert (constant_mem_base);

                // Get device pointer
                cl_mem mem = *(void **)arguments[i].value;
                CUdeviceptr src
                    = (CUdeviceptr)mem->device_ptrs[device->dev_id].mem_ptr;

                size_t align = kdata->alignments[i];
                if (constantMemBytes % align)
                  {
                    constantMemBytes += align - (constantMemBytes % align);
                  }

                // Copy to constant buffer at current offset
                result
                    = cuMemcpyDtoDAsync (constant_mem_base + constantMemBytes,
                                         src, mem->size, stream);
                CUDA_CHECK (result, "cuMemcpyDtoDAsync");

                constantMemOffsets[i] = constantMemBytes;
                params[i] = constantMemOffsets + i;

                constantMemBytes += mem->size;
              }
            else
              {
                if (arguments[i].value)
                  {
                    cl_mem mem = *(void **)arguments[i].value;
                    params[i] = &mem->device_ptrs[device->dev_id].mem_ptr;

#if defined __arm__
                    // On ARM with USE_HOST_PTR, perform explicit copy to
                    // device
                    if (mem->flags & CL_MEM_USE_HOST_PTR)
                      {
                        cuMemcpyHtoD (*(CUdeviceptr *)(params[i]),
                                      mem->mem_host_ptr, mem->size);
                        cuStreamSynchronize (0);
                      }
#endif
                  }
                else
                  {
                    params[i] = &null;
                  }
              }
            break;
          }
        case POCL_ARG_TYPE_IMAGE:
        case POCL_ARG_TYPE_SAMPLER:
          POCL_ABORT ("Unhandled argument type for CUDA");
          break;
        }
    }

  if (constantMemBytes > constant_mem_size)
    POCL_ABORT ("[CUDA] Total constant buffer size %u exceeds %lu allocated",
                constantMemBytes, constant_mem_size);

  unsigned arg_index = kernel->num_args;

  // Deal with automatic local allocations
  // TODO: Would be better to remove arguments and make these static GEPs
  for (i = 0; i < kernel->num_locals; ++i, ++arg_index)
    {
      size_t size = arguments[arg_index].size;
      size_t align = kdata->alignments[arg_index];

      // Pad offset to align memory
      if (sharedMemBytes % align)
        sharedMemBytes += align - (sharedMemBytes % align);

      sharedMemOffsets[arg_index] = sharedMemBytes;
      sharedMemBytes += size;
      params[arg_index] = sharedMemOffsets + arg_index;
    }

  // Add global work dimensionality
  params[arg_index++] = &pc.work_dim;

  // Add global offsets if necessary
  if (has_offsets)
    {
      globalOffsets[0] = pc.global_offset[0];
      globalOffsets[1] = pc.global_offset[1];
      globalOffsets[2] = pc.global_offset[2];
      params[arg_index++] = globalOffsets + 0;
      params[arg_index++] = globalOffsets + 1;
      params[arg_index++] = globalOffsets + 2;
    }

  // Launch kernel
  result = cuLaunchKernel (function, pc.num_groups[0], pc.num_groups[1],
                           pc.num_groups[2], run.local_x, run.local_y,
                           run.local_z, sharedMemBytes, stream, params, NULL);
  CUDA_CHECK (result, "cuLaunchKernel");
}

void
pocl_cuda_submit (_cl_command_node *node, cl_command_queue cq)
{
  cuCtxSetCurrent (((pocl_cuda_device_data_t *)cq->device->data)->context);

  CUresult result;
  CUstream stream = *(CUstream *)cq->data;

  POCL_LOCK_OBJ (node->event);

  pocl_cuda_event_data_t *event_data
      = (pocl_cuda_event_data_t *)calloc (1, sizeof (pocl_cuda_event_data_t));
  node->event->data = event_data;

  // Process event dependencies
  event_node *dep = NULL;
  LL_FOREACH (node->event->wait_list, dep)
    {
      POname (clRetainEvent) (dep->event);

      // Add CUDA event dependency
      if (dep->event->command_type != CL_COMMAND_USER
          && dep->event->queue->device->ops == cq->device->ops)
        {
          // TODO: Don't bother if queue is the same
          pocl_cuda_event_data_t *dep_data
              = (pocl_cuda_event_data_t *)dep->event->data;
          result = cuStreamWaitEvent (stream, dep_data->end, 0);
          CUDA_CHECK (result, "cuStreamWaitEvent");
        }
      else
        event_data->num_ext_events++;
    }

  // Wait on flag for external events
  if (event_data->num_ext_events)
    {
      CUdeviceptr dev_ext_event_flag;
      result = cuMemHostAlloc ((void **)&event_data->ext_event_flag, 4,
                               CU_MEMHOSTALLOC_DEVICEMAP);
      CUDA_CHECK (result, "cuMemAllocHost");

      *event_data->ext_event_flag = 0;

      result = cuMemHostGetDevicePointer (&dev_ext_event_flag,
                                          event_data->ext_event_flag, 0);
      CUDA_CHECK (result, "cuMemHostGetDevicePointer");
      result = cuStreamWaitValue32 (stream, dev_ext_event_flag, 1,
                                    CU_STREAM_WAIT_VALUE_GEQ);
      CUDA_CHECK (result, "cuStreamWaitValue32");
    }

  // Create and record event for command start if profiling enabled
  if (cq->properties & CL_QUEUE_PROFILING_ENABLE)
    {
      result = cuEventCreate (&event_data->start, CU_EVENT_DEFAULT);
      CUDA_CHECK (result, "cuEventCreate");
      result = cuEventRecord (event_data->start, stream);
      CUDA_CHECK (result, "cuEventRecord");
    }

  POCL_UPDATE_EVENT_SUBMITTED (&node->event);

  POCL_UNLOCK_OBJ (node->event);

  switch (node->type)
    {
    case CL_COMMAND_READ_BUFFER:
      pocl_cuda_submit_read (stream, node->command.read.host_ptr,
                             node->command.read.device_ptr,
                             node->command.read.offset, node->command.read.cb);
      break;
    case CL_COMMAND_WRITE_BUFFER:
      pocl_cuda_submit_write (
          stream, node->command.write.host_ptr, node->command.write.device_ptr,
          node->command.write.offset, node->command.write.cb);
      break;
    case CL_COMMAND_COPY_BUFFER:
      {
        cl_device_id src_dev = node->command.copy.src_dev;
        cl_mem src_buf = node->command.copy.src_buffer;
        cl_device_id dst_dev = node->command.copy.dst_dev;
        cl_mem dst_buf = node->command.copy.dst_buffer;
        if (!src_dev)
          src_dev = dst_dev;
        pocl_cuda_submit_copy (
            stream, src_buf->device_ptrs[src_dev->dev_id].mem_ptr,
            node->command.copy.src_offset,
            dst_buf->device_ptrs[dst_dev->dev_id].mem_ptr,
            node->command.copy.dst_offset, node->command.copy.cb);
        break;
      }
    case CL_COMMAND_READ_BUFFER_RECT:
      pocl_cuda_submit_read_rect (
          stream, node->command.read_image.host_ptr,
          node->command.read_image.device_ptr, node->command.read_image.origin,
          node->command.read_image.h_origin, node->command.read_image.region,
          node->command.read_image.b_rowpitch,
          node->command.read_image.b_slicepitch,
          node->command.read_image.h_rowpitch,
          node->command.read_image.h_slicepitch);
      break;
    case CL_COMMAND_WRITE_BUFFER_RECT:
      pocl_cuda_submit_write_rect (stream, node->command.write_image.host_ptr,
                                   node->command.write_image.device_ptr,
                                   node->command.write_image.origin,
                                   node->command.write_image.h_origin,
                                   node->command.write_image.region,
                                   node->command.write_image.b_rowpitch,
                                   node->command.write_image.b_slicepitch,
                                   node->command.write_image.h_rowpitch,
                                   node->command.write_image.h_slicepitch);
      break;
    case CL_COMMAND_COPY_BUFFER_RECT:
      {
        cl_device_id src_dev = node->command.copy_image.src_device;
        cl_mem src_buf = node->command.copy_image.src_buffer;
        cl_device_id dst_dev = node->command.copy_image.dst_device;
        cl_mem dst_buf = node->command.copy_image.dst_buffer;
        if (!src_dev)
          src_dev = dst_dev;
        pocl_cuda_submit_copy_rect (
            stream, src_buf->device_ptrs[src_dev->dev_id].mem_ptr,
            dst_buf->device_ptrs[dst_dev->dev_id].mem_ptr,
            node->command.copy_image.src_origin,
            node->command.copy_image.dst_origin,
            node->command.copy_image.region,
            node->command.copy_image.src_rowpitch,
            node->command.copy_image.src_slicepitch,
            node->command.copy_image.dst_rowpitch,
            node->command.copy_image.dst_slicepitch);
        break;
      }
    case CL_COMMAND_MAP_BUFFER:
      {
        cl_device_id device = node->device;
        cl_mem buffer = node->command.map.buffer;
        if (!(buffer->flags & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR)))
          pocl_cuda_submit_map_mem (
              stream, buffer->device_ptrs[device->dev_id].mem_ptr,
              node->command.map.mapping->offset,
              node->command.map.mapping->size,
              node->command.map.mapping->host_ptr);
        POCL_LOCK_OBJ (buffer);
        buffer->map_count++;
        POCL_UNLOCK_OBJ (buffer);
        break;
      }
    case CL_COMMAND_UNMAP_MEM_OBJECT:
      {
        cl_device_id device = node->device;
        cl_mem buffer = node->command.unmap.memobj;
        pocl_cuda_submit_unmap_mem (
            stream, node->command.unmap.mapping->host_ptr,
            buffer->device_ptrs[device->dev_id].mem_ptr,
            node->command.unmap.mapping->offset,
            node->command.unmap.mapping->size);
        break;
      }
    case CL_COMMAND_NDRANGE_KERNEL:
      pocl_cuda_submit_kernel (stream, node->command.run, node->device,
                               node->event);
      break;

    case CL_COMMAND_MARKER:
    case CL_COMMAND_BARRIER:
      break;

    case CL_COMMAND_FILL_BUFFER:
    case CL_COMMAND_MIGRATE_MEM_OBJECTS:
    case CL_COMMAND_READ_IMAGE:
    case CL_COMMAND_WRITE_IMAGE:
    case CL_COMMAND_COPY_IMAGE:
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
    case CL_COMMAND_FILL_IMAGE:
    case CL_COMMAND_MAP_IMAGE:
    case CL_COMMAND_NATIVE_KERNEL:
    case CL_COMMAND_SVM_FREE:
    case CL_COMMAND_SVM_MAP:
    case CL_COMMAND_SVM_UNMAP:
    case CL_COMMAND_SVM_MEMCPY:
    case CL_COMMAND_SVM_MEMFILL:
    default:
      POCL_ABORT_UNIMPLEMENTED ("Command type for CUDA devices");
      break;
    }

  // Create and record event for command end
  if (cq->properties & CL_QUEUE_PROFILING_ENABLE)
    result = cuEventCreate (&event_data->end, CU_EVENT_DEFAULT);
  else
    result = cuEventCreate (&event_data->end, CU_EVENT_DISABLE_TIMING);
  CUDA_CHECK (result, "cuEventCreate");
  result = cuEventRecord (event_data->end, stream);
  CUDA_CHECK (result, "cuEventRecord");
}

void
pocl_cuda_notify (cl_device_id device, cl_event event)
{
  pocl_cuda_event_data_t *event_data = (pocl_cuda_event_data_t *)event->data;

  assert (event_data);
  assert (event_data->num_ext_events > 0);
  assert (event_data->ext_event_flag);

  // Decrement external event counter
  // Trigger flag if none left
  if (!--event_data->num_ext_events)
    *event_data->ext_event_flag = 1;
}

void
pocl_cuda_broadcast (cl_event event)
{
  // TODO: call notify for each non-CUDA event in notify_list
}

void
pocl_cuda_flush (cl_device_id device, cl_command_queue cq)
{
  // TODO: Something here?
}

void
pocl_cuda_wait_event (cl_device_id device, cl_event event)
{
  if (!event || event->status <= CL_COMPLETE)
    return;
  if (event->command_type == CL_COMMAND_USER)
    {
      // TODO: Can get rid of this loop if we handle via CUDA event?
      while (event->status > CL_COMPLETE)
        {
        };
      return;
    }

  // Recursively wait for dependencies
  int dep_failed = 0;
  event_node *dep = NULL;
  LL_FOREACH (event->wait_list, dep)
    {
      // TODO: Use device->ops->wait_event for non-CUDA events
      pocl_cuda_wait_event (event->queue->device, dep->event);

      if (dep->event->status < 0)
        dep_failed = 1;
    }

  // Clean up wait_list nodes
  dep = event->wait_list;
  while (dep)
    {
      event_node *next = dep->next;
      POname(clReleaseEvent) (dep->event);
      pocl_mem_manager_free_event_node (dep);
      dep = next;
    }

  if (event->status <= CL_COMPLETE)
    return;

  if (dep_failed)
    {
      pocl_mem_manager_free_command (event->command);
      event->status = -1;
      return;
    }

  if (event->command_type == CL_COMMAND_MARKER
      || event->command_type == CL_COMMAND_BARRIER)
    {
      // TODO: set start/end time to CUDA event times
      pocl_mem_manager_free_command (event->command);
      POCL_UPDATE_EVENT_RUNNING (&event);
      POCL_UPDATE_EVENT_COMPLETE (&event);
      return;
    }

  // Wait for command to finish
  cuCtxSetCurrent (((pocl_cuda_device_data_t *)device->data)->context);
  pocl_cuda_event_data_t *event_data = ((pocl_cuda_event_data_t *)event->data);
  CUresult result = cuEventSynchronize (event_data->end);
  CUDA_CHECK (result, "cuEventSynchronize");

  if (event->command_type == CL_COMMAND_UNMAP_MEM_OBJECT)
    {
      cl_mem buffer = event->command->command.unmap.memobj;
      mem_mapping_t *mapping = event->command->command.unmap.mapping;
      if (mapping->host_ptr
          && !(buffer->flags & (CL_MEM_USE_HOST_PTR | CL_MEM_ALLOC_HOST_PTR)))
        free (mapping->host_ptr);

      POCL_LOCK_OBJ (buffer);
      DL_DELETE (buffer->mappings, mapping);
      buffer->map_count--;
      POCL_UNLOCK_OBJ (buffer);
    }

  if (event->command_type == CL_COMMAND_NDRANGE_KERNEL
      || event->command_type == CL_COMMAND_TASK)
    {
#if defined __arm__
      // On ARM with USE_HOST_PTR, perform explict copies back from device
      cl_kernel kernel = event->command.run.kernel;
      pocl_argument *arguments = event->command.run.arguments;
      unsigned i;
      for (i = 0; i < kernel->num_args; i++)
        {
          if (kernel->arg_info[i].type == POCL_ARG_TYPE_POINTER)
            {
              if (!kernel->arg_info[i].is_local && arguments[i].value)
                {
                  cl_mem mem = *(void **)arguments[i].value;
                  if (mem->flags & CL_MEM_USE_HOST_PTR)
                    {
                      CUdeviceptr ptr
                          = (CUdeviceptr)mem->device_ptrs[device->dev_id]
                                .mem_ptr;
                      cuMemcpyDtoH (mem->mem_host_ptr, ptr, mem->size);
                      cuStreamSynchronize (0);
                    }
                }
            }
        }
#endif

      pocl_ndrange_node_cleanup (event->command);
    }
  else
    {
      pocl_mem_manager_free_command (event->command);
    }

  // Retain this event temporarily, otherwise the call to
  // POCL_UPDATE_EVENT_COMPLETE might release the event before we've
  // dealt with the timing.
  // TODO: Maybe better to handle update_event so we don't have to do this?
  POname (clRetainEvent) (event);

  POCL_UPDATE_EVENT_RUNNING (&event);
  POCL_UPDATE_EVENT_COMPLETE (&event);

  // Update timing info with CUDA event timers if profiling enabled
  if (event->queue->properties & CL_QUEUE_PROFILING_ENABLE)
    {
      // CUDA doesn't provide a way to get event timestamps directly,
      // only the elapsed time between two events. We use the elapsed
      // time from the epoch event enqueued on device creation to get
      // the actual timestamps.
      //
      // Since the CUDA timer resolution is lower than the host timer,
      // this can sometimes result in the start time being before the
      // submit time, so we use max() to ensure the timestamps are
      // sane.

      float diff;
      cl_ulong epoch = ((pocl_cuda_device_data_t *)device->data)->epoch;

      result = cuEventElapsedTime (
          &diff, ((pocl_cuda_device_data_t *)device->data)->epoch_event,
          event_data->start);
      event->time_start = max (event->time_submit + 1, epoch + diff * 1e6);

      result = cuEventElapsedTime (
          &diff, ((pocl_cuda_device_data_t *)device->data)->epoch_event,
          event_data->end);
      event->time_end = max (event->time_start + 1, epoch + diff * 1e6);
    }

  POname (clReleaseEvent) (event);
}

void
pocl_cuda_free_event_data (cl_event event)
{
  if (event->data)
    {
      pocl_cuda_event_data_t *event_data
          = (pocl_cuda_event_data_t *)event->data;

      if (event->queue->properties & CL_QUEUE_PROFILING_ENABLE)
        cuEventDestroy (event_data->start);
      cuEventDestroy (event_data->end);
      if (event_data->ext_event_flag)
        {
          CUresult result = cuMemFreeHost (event_data->ext_event_flag);
          CUDA_CHECK (result, "cuMemFreeHost");
        }
      free (event->data);
    }
}

void
pocl_cuda_join (cl_device_id device, cl_command_queue cq)
{
  pocl_cuda_wait_event (device, cq->last_event.event);
}
