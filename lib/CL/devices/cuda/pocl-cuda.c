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

#include "builtin_kernels.hh"
#include "common.h"
#include "common_driver.h"
#include "devices.h"
#include "pocl-cuda.h"
#include "pocl-ptx-gen.h"
#include "pocl.h"
#include "pocl_cache.h"
#include "pocl_file_util.h"
#include "pocl_llvm.h"
#include "pocl_mem_management.h"
#include "pocl_runtime_config.h"
#include "pocl_timing.h"
#include "pocl_util.h"

#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef ENABLE_CUDNN
#include <cudnn.h>
#define CUDNN_CALL(f)                                                         \
  {                                                                           \
    cudnnStatus_t err = (f);                                                  \
    if (err != CUDNN_STATUS_SUCCESS)                                          \
      {                                                                       \
        POCL_ABORT ("  CUDNN Error occurred: %d", err);                       \
      }                                                                       \
  }

cudnnHandle_t cudnn;
#endif // ENABLE_CUDNN

#define CUDA_CALL(f)                                                          \
  {                                                                           \
    cudaError_t err = (f);                                                    \
    if (err != cudaSuccess)                                                   \
      {                                                                       \
        POCL_ABORT ("  Error occurred: %d", err);                             \
      }                                                                       \
  }

#define CUDA_BUILTIN_KERNELS 6
static const char *cuda_builtin_kernels[CUDA_BUILTIN_KERNELS]
    = { "pocl.mul.i32",
        "pocl.add.i32",
        "pocl.dnn.conv2d_int8_relu",
        "pocl.sgemm.local.f32",
        "pocl.sgemm.tensor.f16f16f32",
        "pocl.sgemm_ab.tensor.f16f16f32" };

#define OPENCL_BUILTIN_KERNELS 5
static const char *opencl_builtin_kernels[OPENCL_BUILTIN_KERNELS] = {
  "pocl.abs.f32",
  // from common builtin kernels:
  "pocl.add.i8",
  "org.khronos.openvx.scale_image.nn.u8",
  "org.khronos.openvx.scale_image.bl.u8",
  "org.khronos.openvx.tensor_convert_depth.wrap.u8.f32",
};

#ifdef ENABLE_CUDNN
#define CUDNN_BUILTIN_KERNELS 1
static const char *cudnn_builtin_kernels[CUDNN_BUILTIN_KERNELS]
    = { "pocl.dnn.conv2d.nchw.f32" };
#else
#define CUDNN_BUILTIN_KERNELS 0
#endif

/* The frexp functions are wrappers required because the __nv_frexp and the
 * LLVM's frexp intrinsics have different signatures. */
static const char *cuda_native_device_aux_funcs[] =
  {"frexpf_f32_i32", "frexp_f64_i32", NULL};


void pocl_cuda_svm_copy_async (CUstream, void *restrict, const void *restrict,
                               size_t);

typedef struct pocl_cuda_queue_data_s
{
  CUstream stream;
  int use_threads;
  pthread_t submit_thread;
  pthread_t finalize_thread;
  pthread_mutex_t lock;
  pthread_cond_t pending_cond;
  pthread_cond_t running_cond;
  _cl_command_node *volatile pending_queue;
  _cl_command_node *volatile running_queue;
  cl_command_queue queue;
} pocl_cuda_queue_data_t;

typedef struct pocl_cuda_kernel_data_s
{
  CUfunction kernel;
  CUfunction kernel_offsets;
  size_t *alignments;
  size_t refcount;
} pocl_cuda_kernel_data_t;

typedef struct pocl_cuda_program_data_s
{
  CUmodule module;
  CUmodule module_offsets;
  CUdeviceptr constant_mem_base;
  CUdeviceptr constant_mem_base_offsets;
  size_t constant_mem_size;
  size_t constant_mem_size_offsets;
  void *align_map;
  void *align_map_offsets;
} pocl_cuda_program_data_t;

typedef struct pocl_cuda_event_data_s
{
  CUevent start;
  CUevent end;
  volatile int events_ready;
  cl_int *ext_event_flag;
  pthread_cond_t event_cond;
  volatile unsigned num_ext_events;
} pocl_cuda_event_data_t;

typedef struct pocl_cuda_device_data_s
{
  CUdevice device;
  CUcontext context;
  CUevent epoch_event;
  cl_ulong epoch;
  char libdevice[PATH_MAX];
  pocl_lock_t compile_lock;
  int supports_cu_mem_host_register;
  int supports_managed_memory;
  int sm_maj, sm_min, warp_size;
  cl_bool available;

  pocl_cuda_kernel_data_t cuda_builtin_kernels_data[CUDA_BUILTIN_KERNELS];
  pocl_cuda_kernel_data_t cudnn_builtin_kernels_data[CUDNN_BUILTIN_KERNELS];
  pocl_cuda_program_data_t cuda_builtin_kernels_program;
  pocl_cuda_program_data_t cudnn_builtin_kernels_program;
  int cuda_builtin_kernels_built;

} pocl_cuda_device_data_t;

extern unsigned int pocl_num_devices;

void *pocl_cuda_submit_thread (void *);
void *pocl_cuda_finalize_thread (void *);

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

#define CUDA_CHECK_ABORT(result, api)                                         \
  pocl_cuda_abort_on_error (result, __LINE__, __FUNCTION__, #result, api)

#define CUDA_CHECK_ERROR(result, api)                                         \
  pocl_cuda_error (result, __LINE__, __FUNCTION__, #result, api)


void
pocl_cuda_init_device_ops (struct pocl_device_ops *ops)
{
  ops->device_name = "cuda";
  ops->build_hash = pocl_cuda_build_hash;
  ops->probe = pocl_cuda_probe;
  ops->uninit = pocl_cuda_uninit;
  ops->reinit = NULL;
  ops->init = pocl_cuda_init;
  ops->init_queue = pocl_cuda_init_queue;
  ops->free_queue = pocl_cuda_free_queue;

  ops->alloc_mem_obj = pocl_cuda_alloc_mem_obj;
  ops->free = pocl_cuda_free;

  ops->submit = pocl_cuda_submit;
  ops->notify = pocl_cuda_notify;
  ops->broadcast = pocl_broadcast;
  ops->wait_event = pocl_cuda_wait_event;
  ops->update_event = pocl_cuda_update_event;
  ops->free_event_data = pocl_cuda_free_event_data;
  ops->join = pocl_cuda_join;
  ops->flush = pocl_cuda_flush;
  ops->init_build = pocl_cuda_init_build;
  // TODO
  ops->notify_event_finished = pocl_cuda_notify_event_finished;

  ops->get_device_info_ext = pocl_cuda_get_device_info_ext;
  ops->get_mem_info_ext = NULL; // pocl_cuda_get_mem_info_ext;
  ops->set_kernel_exec_info_ext = pocl_cuda_set_kernel_exec_info_ext;

  ops->build_source = pocl_driver_build_source;
  ops->link_program = pocl_driver_link_program;
  ops->build_binary = pocl_driver_build_binary;
  ops->setup_metadata = pocl_driver_setup_metadata;
  ops->supports_binary = pocl_driver_supports_binary;
  ops->build_poclbinary = pocl_driver_build_poclbinary;

  ops->post_build_program = pocl_cuda_post_build_program;
  ops->free_program = pocl_cuda_free_program;
  ops->build_builtin = pocl_cuda_build_builtin;

  ops->compile_kernel = pocl_cuda_compile_kernel;
  ops->create_kernel = pocl_cuda_create_kernel;
  ops->free_kernel = pocl_cuda_free_kernel;

  // TODO
  ops->get_mapping_ptr = pocl_driver_get_mapping_ptr;
  ops->free_mapping_ptr = pocl_driver_free_mapping_ptr;

  ops->can_migrate_d2d = pocl_cuda_can_migrate_d2d;
  ops->migrate_d2d = NULL;
  ops->read = NULL;
  ops->read_rect = NULL;
  ops->write = NULL;
  ops->write_rect = NULL;
  ops->copy = NULL;
  ops->copy_rect = NULL;
  ops->map_mem = NULL;
  ops->unmap_mem = NULL;
  ops->run = NULL;

  ops->svm_alloc = pocl_cuda_svm_alloc;
  ops->svm_free = pocl_cuda_svm_free;
  /* No need to implement these two as they are no-ops
   * and pocl_exec_command takes care of them. */
  ops->svm_map = NULL;
  ops->svm_unmap = NULL;
  ops->svm_copy = pocl_cuda_svm_copy;
  ops->svm_fill = pocl_cuda_svm_fill;
}

cl_int
pocl_cuda_init (unsigned j, cl_device_id dev, const char *parameters)
{
  CUresult result;
  int ret = CL_SUCCESS;

  assert (dev->data == NULL);

  pocl_init_default_device_infos (dev);
  dev->extensions = CUDA_DEVICE_EXTENSIONS;

  dev->vendor = "NVIDIA Corporation";
  dev->vendor_id = 0x10de; /* the PCIID for NVIDIA */

  dev->type = CL_DEVICE_TYPE_GPU;
  dev->address_bits = (sizeof (void *) * 8);

  dev->llvm_target_triplet = (sizeof (void *) == 8) ? "nvptx64" : "nvptx";
  dev->kernellib_name = (sizeof (void *) == 8) ? "kernel-nvptx64" : "kernel-nvptx";
  dev->kernellib_fallback_name = NULL;
  dev->kernellib_subdir = "cuda";
  dev->llvm_fp_contract_mode = "fast";

  dev->spmd = CL_TRUE;
  dev->run_workgroup_pass = CL_FALSE;
  dev->execution_capabilities = CL_EXEC_KERNEL;

  dev->global_as_id = 1;
  dev->local_as_id = 3;
  dev->constant_as_id = 1;
  dev->device_aux_functions = cuda_native_device_aux_funcs;

  /* TODO: Get images working */
  dev->image_support = CL_FALSE;

  dev->autolocals_to_args
      = POCL_AUTOLOCALS_TO_ARGS_ONLY_IF_DYNAMIC_LOCALS_PRESENT;

  dev->has_64bit_long = 1;

  pocl_cuda_device_data_t *data = calloc (1, sizeof (pocl_cuda_device_data_t));
  dev->data = data;

  result = cuDeviceGet (&data->device, j);
  if (CUDA_CHECK_ERROR (result, "cuDeviceGet"))
    ret = CL_INVALID_DEVICE;

  /* Get specific device name */
  {
     char *name = calloc (256, sizeof (char));

     if (ret != CL_INVALID_DEVICE)
       cuDeviceGetName (name, 256, data->device);
     else
       snprintf (name, 255, "Unavailable CUDA device #%d", j);
     dev->long_name = dev->short_name = name;
  }

  SETUP_DEVICE_CL_VERSION (CUDA_DEVICE_CL_VERSION_MAJOR,
                           CUDA_DEVICE_CL_VERSION_MINOR);

  /* Get other device properties */
  if (ret != CL_INVALID_DEVICE)
    {
      /* CUDA device attributes (as fetched by cuDeviceGetAttribute) are always (unsigned)
       * integers, where the OpenCL counterparts are of a variety of (other) integer types.
       * Fetch the values in an unsigned int and copy it over.
       * We also OR all return values of cuDeviceGetAttribute, and at the end we will check
       * if it's not CL_SUCCESS. We miss the exact line that failed this way, but it's
       * faster than checking after each attribute fetch.
       */
      int value = 0;
#define GET_CU_PROP(key, target) do { \
  result |= cuDeviceGetAttribute (&value, CU_DEVICE_ATTRIBUTE_##key, data->device); \
  target = value; \
} while (0)

      GET_CU_PROP (MAX_THREADS_PER_BLOCK, dev->max_work_group_size);
      GET_CU_PROP (MAX_BLOCK_DIM_X, dev->max_work_item_sizes[0]);
      GET_CU_PROP (MAX_BLOCK_DIM_Y, dev->max_work_item_sizes[1]);
      GET_CU_PROP (MAX_BLOCK_DIM_Z, dev->max_work_item_sizes[2]);
      GET_CU_PROP (MAX_SHARED_MEMORY_PER_BLOCK, dev->local_mem_size);
      GET_CU_PROP (MULTIPROCESSOR_COUNT, dev->max_compute_units);
      GET_CU_PROP (ECC_ENABLED, dev->error_correction_support);
      GET_CU_PROP (INTEGRATED, dev->host_unified_memory);
      GET_CU_PROP (TOTAL_CONSTANT_MEMORY, dev->max_constant_buffer_size);
      GET_CU_PROP (CLOCK_RATE, dev->max_clock_frequency);
      dev->max_clock_frequency /= 1000;
      GET_CU_PROP (TEXTURE_ALIGNMENT, dev->mem_base_addr_align);
      GET_CU_PROP (INTEGRATED, dev->host_unified_memory);
      data->supports_managed_memory = 0;
      GET_CU_PROP (MANAGED_MEMORY, data->supports_managed_memory);
    }
  if (CUDA_CHECK_ERROR (result, "cuDeviceGetAttribute"))
    ret = CL_INVALID_DEVICE;

  if (ret != CL_INVALID_DEVICE)
    {
      int driver_version = 0;
      result = cuDriverGetVersion(&driver_version);
      if (CUDA_CHECK_ERROR (result, "cuDriverGetVersion"))
	{
          ret = CL_INVALID_DEVICE;
	}
      else
	{
#if CUDA_VERSION >= 11010
          if (driver_version >= 11010)
            {
              int value;
              result = cuDeviceGetAttribute (&value,
                CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED, data->device);
              data->supports_cu_mem_host_register = value;
              if (CUDA_CHECK_ERROR (result, "cuDeviceGetAttribute"))
                ret = CL_INVALID_DEVICE;
            } else {
#else
            {
#endif
#if defined(__aarch64__) || defined(__arm__)
              // For cuda < 11.1, we don't know if the device supports cuMemHostRegister
              // or not. Let's assume that it doesn't in ARM devices.
              // This gives a false negative for Jetson Xavier, but it is the best we could do.
              data->supports_cu_mem_host_register = pocl_get_bool_option ("POCL_CUDA_SUPPORTS_CU_MEM_HOST_REGISTER", 0);
#else
              data->supports_cu_mem_host_register = pocl_get_bool_option ("POCL_CUDA_SUPPORTS_CU_MEM_HOST_REGISTER", 1);
#endif
            }
        }
    }

  data->available = CL_TRUE;
  dev->available = &data->available;

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

  if (strstr (CUDA_DEVICE_EXTENSIONS, "cl_ext_float_atomics")
      != NULL) {
    dev->single_fp_atomic_caps = dev->double_fp_atomic_caps =
      CL_DEVICE_GLOBAL_FP_ATOMIC_LOAD_STORE_EXT |
      CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT |
      CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT |
      CL_DEVICE_LOCAL_FP_ATOMIC_LOAD_STORE_EXT |
      CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT |
      CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT;
  }


  dev->local_mem_type = CL_LOCAL;

#ifdef ENABLE_SPIRV
  dev->supported_spir_v_versions = "SPIR-V_1.2";
#else
  dev->supported_spir_v_versions = "";
#endif

  /* Get GPU architecture name */
  int sm_maj = 0, sm_min = 0, warp_size = 32;
  if (ret != CL_INVALID_DEVICE)
    {
      cuDeviceGetAttribute (&sm_maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                            data->device);
      cuDeviceGetAttribute (&sm_min, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                            data->device);
    }
  cuDeviceGetAttribute (&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                        data->device);

  char *gpu_arch = calloc (16, sizeof (char));
  snprintf (gpu_arch, 16, "sm_%d%d", sm_maj, sm_min);
  dev->llvm_cpu = pocl_get_string_option ("POCL_CUDA_GPU_ARCH", gpu_arch);
  POCL_MSG_PRINT_INFO ("[CUDA] GPU architecture = %s\n", dev->llvm_cpu);
  data->sm_maj = sm_maj;
  data->sm_min = sm_min;
  data->warp_size = warp_size;

  /* Find libdevice library */
  if (findLibDevice (data->libdevice, dev->llvm_cpu))
    {
      if (ret != CL_INVALID_DEVICE)
        {
          POCL_MSG_ERR ("[CUDA] failed to find libdevice library\n");
          dev->compiler_available = dev->linker_available = 0;
        }
    }

  /* setup builtin kernels */
  if (ret != CL_INVALID_DEVICE && dev->compiler_available)
    {
      dev->num_builtin_kernels = CUDA_BUILTIN_KERNELS;
      dev->builtins_sources_path = "builtins.cl";
      if (sm_maj < 7)
        {
          dev->num_builtin_kernels
              -= 2; // last two kernels require tensor cores
        }
      dev->builtin_kernel_list = (char *)malloc (1024);
      dev->builtin_kernel_list[0] = 0;
      for (unsigned i = 0; i < dev->num_builtin_kernels; ++i)
        {
          if (i > 0)
            strcat (dev->builtin_kernel_list, ";");
          strcat (dev->builtin_kernel_list, cuda_builtin_kernels[i]);
        }
      dev->num_builtin_kernels += OPENCL_BUILTIN_KERNELS;
      for (unsigned i = 0; i < OPENCL_BUILTIN_KERNELS; ++i)
        {
          strcat (dev->builtin_kernel_list, ";");
          strcat (dev->builtin_kernel_list, opencl_builtin_kernels[i]);
        }
#ifdef ENABLE_CUDNN
      dev->num_builtin_kernels += CUDNN_BUILTIN_KERNELS;
      for (unsigned i = 0; i < CUDNN_BUILTIN_KERNELS; ++i)
        {
          strcat(dev->builtin_kernel_list, ";");
          strcat(dev->builtin_kernel_list, cudnn_builtin_kernels[i]);
        }
#endif //ENABLE_CUDNN

    }

  dev->device_side_printf = 0;

  /* Create context */
  if (ret != CL_INVALID_DEVICE)
    {
      result = cuCtxCreate (&data->context, CU_CTX_MAP_HOST, data->device);
      if (CUDA_CHECK_ERROR (result, "cuCtxCreate"))
        ret = CL_INVALID_DEVICE;
    }

  /* Create epoch event for timing info */
  if (ret != CL_INVALID_DEVICE)
    {
      result = cuEventCreate (&data->epoch_event, CU_EVENT_DEFAULT);
      CUDA_CHECK_ERROR (result, "cuEventCreate");

      data->epoch = pocl_gettimemono_ns ();

      result = cuEventRecord (data->epoch_event, 0);
      result = cuEventSynchronize (data->epoch_event);
      if (CUDA_CHECK_ERROR (result, "cuEventSynchronize"))
        ret = CL_INVALID_DEVICE;
    }

  /* Get global memory size */
  size_t memfree = 0, memtotal = 0;
  if (ret != CL_INVALID_DEVICE)
    result = cuMemGetInfo (&memfree, &memtotal);
  dev->max_mem_alloc_size = max (memtotal / 4, 128 * 1024 * 1024);
  dev->global_mem_size = memtotal;

  dev->svm_allocation_priority = 2;
  dev->svm_caps = CL_DEVICE_SVM_COARSE_GRAIN_BUFFER;
  if (data->supports_managed_memory) {
    dev->svm_allocation_priority = 2;

    /* OpenCL 2.0 properties */
    dev->svm_caps = CL_DEVICE_SVM_COARSE_GRAIN_BUFFER
                    | CL_DEVICE_SVM_FINE_GRAIN_BUFFER;
//                    | CL_DEVICE_SVM_ATOMICS;
  }

  /* OpenCL 3.0 properties */
  /* Minimum mandated capability */
  dev->atomic_memory_capabilities
      = CL_DEVICE_ATOMIC_ORDER_RELAXED | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP;
  dev->atomic_fence_capabilities = CL_DEVICE_ATOMIC_ORDER_RELAXED
                                   | CL_DEVICE_ATOMIC_ORDER_ACQ_REL
                                   | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP;

  dev->sub_group_independent_forward_progress =
      (data->sm_maj >= 7) ? CL_TRUE : CL_FALSE;

  /* Just an arbitrary number here based on assumption of SG size 32. */
  dev->max_num_sub_groups = dev->max_work_group_size / 32;

  // All devices starting from Compute Capability 2.0 have this limit;
  // See e.g.
  // https://forums.developer.nvidia.com/t/max-size-of-cuda-arguments/50218
  dev->max_parameter_size = 4352;

#if (CUDA_DEVICE_CL_VERSION_MAJOR >= 3)
  dev->features = CUDA_DEVICE_FEATURES_30;
  /* this is not enabled because we're compiling program.bc and
   * handing it over to CUDA; we're not running workgroup function
   * generation. Therefore CUDA takes care of program-scope variables. */
  dev->run_program_scope_variables_pass = CL_FALSE;
  dev->generic_as_support = CL_TRUE;

  pocl_setup_opencl_c_with_version (dev, CL_TRUE);
  pocl_setup_features_with_version (dev);
#else
  pocl_setup_opencl_c_with_version (dev, CL_FALSE);
#endif

  pocl_setup_extensions_with_version (dev);

  pocl_setup_builtin_kernels_with_version (dev);

  pocl_setup_ils_with_version (dev);

#ifdef ENABLE_CUDNN
  CUDNN_CALL (cudnnCreate (&cudnn));
#endif

  POCL_INIT_LOCK (data->compile_lock);
  return ret;
}

cl_int
pocl_cuda_init_queue (cl_device_id device, cl_command_queue queue)
{
  cuCtxSetCurrent (((pocl_cuda_device_data_t *)queue->device->data)->context);

  pocl_cuda_queue_data_t *queue_data
      = calloc (1, sizeof (pocl_cuda_queue_data_t));
  queue->data = queue_data;
  queue_data->queue = queue;

  CUresult result
      = cuStreamCreate (&queue_data->stream, CU_STREAM_NON_BLOCKING);
  if (CUDA_CHECK_ERROR (result, "cuStreamCreate"))
    return CL_OUT_OF_RESOURCES;

  queue_data->use_threads
      = !pocl_get_bool_option ("POCL_CUDA_DISABLE_QUEUE_THREADS", 1);

  if (queue_data->use_threads)
    {
      PTHREAD_CHECK (pthread_mutex_init (&queue_data->lock, NULL));
      PTHREAD_CHECK (pthread_cond_init (&queue_data->pending_cond, NULL));
      PTHREAD_CHECK (pthread_cond_init (&queue_data->running_cond, NULL));
      int err = pthread_create (&queue_data->submit_thread, NULL,
                                pocl_cuda_submit_thread, queue_data);
      if (err)
        {
          POCL_MSG_ERR ("[CUDA] Error creating submit thread: %d\n", err);
          return CL_OUT_OF_RESOURCES;
        }

      err = pthread_create (&queue_data->finalize_thread, NULL,
                            pocl_cuda_finalize_thread, queue_data);
      if (err)
        {
          POCL_MSG_ERR ("[CUDA] Error creating finalize thread: %d\n", err);
          return CL_OUT_OF_RESOURCES;
        }
    }

  return CL_SUCCESS;
}

int
pocl_cuda_free_queue (cl_device_id device, cl_command_queue queue)
{
  pocl_cuda_queue_data_t *queue_data = (pocl_cuda_queue_data_t *)queue->data;

  cuCtxSetCurrent (((pocl_cuda_device_data_t *)queue->device->data)->context);
  cuStreamDestroy (queue_data->stream);

  assert (queue_data->pending_queue == NULL);
  assert (queue_data->running_queue == NULL);

  /* Kill queue threads */
  if (queue_data->use_threads)
    {
      PTHREAD_CHECK (pthread_mutex_lock (&queue_data->lock));
      queue_data->queue = NULL;
      PTHREAD_CHECK (pthread_cond_signal (&queue_data->pending_cond));
      PTHREAD_CHECK (pthread_cond_signal (&queue_data->running_cond));
      PTHREAD_CHECK (pthread_mutex_unlock (&queue_data->lock));
      PTHREAD_CHECK (pthread_join (queue_data->submit_thread, NULL));
      PTHREAD_CHECK (pthread_join (queue_data->finalize_thread, NULL));
    }
  return CL_SUCCESS;
}

char *
pocl_cuda_build_hash (cl_device_id device)
{
  char *res = calloc (1000, sizeof (char));
  snprintf (res, 1000, "CUDA-%s", device->llvm_cpu);
  return res;
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

cl_int
pocl_cuda_uninit (unsigned j, cl_device_id device)
{
  pocl_cuda_device_data_t *data = device->data;

  if (*(device->available) == CL_TRUE)
    {
      cuEventDestroy (data->epoch_event);
      cuCtxDestroy (data->context);
    }

  POCL_MEM_FREE (data);
  device->data = NULL;

  char *name = (char*)device->long_name;
  POCL_MEM_FREE (name);
  device->long_name = device->short_name = NULL;

#ifdef ENABLE_CUDNN
  CUDNN_CALL (cudnnDestroy (cudnn));
#endif

  return CL_SUCCESS;
}

cl_int
pocl_cuda_alloc_mem_obj (cl_device_id device, cl_mem mem, void *host_ptr)
{
  cuCtxSetCurrent (((pocl_cuda_device_data_t *)device->data)->context);
  pocl_mem_identifier *p = &mem->device_ptrs[device->global_mem_id];
  int err = CL_MEM_OBJECT_ALLOCATION_FAILURE;

  CUresult result;
  void *b = NULL;

  p->extra_ptr = NULL;
  p->mem_ptr = NULL;
  p->version = 0;
  cl_mem_flags flags = mem->flags;

  if (flags & CL_MEM_USE_HOST_PTR)
    {
      if (!((pocl_cuda_device_data_t *)device->data)->supports_cu_mem_host_register)
        {
          /* cuMemHostRegister is not supported on some ARM devices like the Nano, but supported on Xavier.
           * Allocate device memory and perform explicit copies
           * before and after running a kernel */
          result = cuMemAlloc ((CUdeviceptr *)&b, mem->size);
          if (CUDA_CHECK_ERROR (result, "cuMemAlloc"))
            return CL_MEM_OBJECT_ALLOCATION_FAILURE;
        }
      else
        {
          POCL_RETURN_ERROR_ON ((pocl_alloc_or_retain_mem_host_ptr (mem) != 0),
                                CL_OUT_OF_HOST_MEMORY,
                                "Cannot allocate backing memory!\n");

          result = cuMemHostRegister (mem->mem_host_ptr, mem->size,
                                      CU_MEMHOSTREGISTER_DEVICEMAP);
          if (result != CUDA_SUCCESS
              && result != CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED)
            {
              CUDA_CHECK_ERROR (result, "cuMemHostRegister");
              return CL_MEM_OBJECT_ALLOCATION_FAILURE;
            }
          result = cuMemHostGetDevicePointer ((CUdeviceptr *)&b, mem->mem_host_ptr,
                                              0);
          if (CUDA_CHECK_ERROR (result, "cuMemHostGetDevicePointer"))
            return CL_MEM_OBJECT_ALLOCATION_FAILURE;

          /* TODO can we assume cuMemHostRegister copies
           * the content of host memory to the device ? for now, lets not */
          p->version = 0;
        }
    }
  /* preallocate host visible memory */
  else if ((flags & CL_MEM_ALLOC_HOST_PTR) && (mem->mem_host_ptr == NULL))
    {
      result = cuMemHostAlloc (&p->extra_ptr, mem->size,
                               CU_MEMHOSTREGISTER_DEVICEMAP);
      if (CUDA_CHECK_ERROR (result, "cuMemHostAlloc"))
        {
          p->extra_ptr = NULL;
          return CL_MEM_OBJECT_ALLOCATION_FAILURE;
        }

      result = cuMemHostGetDevicePointer ((CUdeviceptr *)&b, p->extra_ptr, 0);
      if (CUDA_CHECK_ERROR (result, "cuMemHostGetDevicePointer"))
        {
          cuMemFreeHost (p->extra_ptr);
          p->extra_ptr = NULL;
          return CL_MEM_OBJECT_ALLOCATION_FAILURE;
        }

      mem->mem_host_ptr = p->extra_ptr;
      mem->mem_host_ptr_refcount = 1;
      mem->mem_host_ptr_version = 0;

      if (flags & CL_MEM_COPY_HOST_PTR)
        {
          result = cuMemcpyHtoD ((CUdeviceptr)b, host_ptr, mem->size);
          if (CUDA_CHECK_ERROR (result, "cuMemcpyHtoD"))
            {
              cuMemFreeHost (p->extra_ptr);
              p->extra_ptr = NULL;
              return CL_MEM_OBJECT_ALLOCATION_FAILURE;
            }

          result = cuStreamSynchronize (0);
          if (CUDA_CHECK_ERROR (result, "cuStreamSynchronize"))
            {
              cuMemFreeHost (p->extra_ptr);
              p->extra_ptr = NULL;
              return CL_MEM_OBJECT_ALLOCATION_FAILURE;
            }

          mem->mem_host_ptr_version = 1;
          mem->latest_version = 1;
          p->version = 1;
        }
    }
  else
    {
      result = cuMemAlloc ((CUdeviceptr *)&b, mem->size);
      if (CUDA_CHECK_ERROR (result, "cuMemAlloc"))
        {
          return CL_MEM_OBJECT_ALLOCATION_FAILURE;
        }
    }

  p->mem_ptr = b;
  return CL_SUCCESS;
}

void
pocl_cuda_free (cl_device_id device, cl_mem mem_obj)
{
  cuCtxSetCurrent (((pocl_cuda_device_data_t *)device->data)->context);
  pocl_mem_identifier *p = &mem_obj->device_ptrs[device->global_mem_id];

  if (mem_obj->flags & CL_MEM_USE_HOST_PTR)
    {
      if (((pocl_cuda_device_data_t *)device->data)->supports_cu_mem_host_register)
        {
          assert (p->extra_ptr == NULL);
          cuMemHostUnregister (mem_obj->mem_host_ptr);
        }
      else
        {
          cuMemFree ((CUdeviceptr)p->mem_ptr);
        }
    }
  else if (p->extra_ptr)
    {
      mem_obj->mem_host_ptr = NULL;
      mem_obj->mem_host_ptr_refcount = 0;
      mem_obj->mem_host_ptr_version = 0;
      cuMemFreeHost (p->extra_ptr);
      p->extra_ptr = NULL;
    }
  else
    {
      assert (p->extra_ptr == NULL);
      assert (p->mem_ptr != NULL);
      cuMemFree ((CUdeviceptr)p->mem_ptr);
    }
  p->mem_ptr = NULL;
  p->version = 0;
}

int
pocl_cuda_can_migrate_d2d (cl_device_id dest, cl_device_id source)
{
  assert (dest != source);
  if (strcmp (dest->ops->device_name, source->ops->device_name) == 0)
    {
      int possible;
      pocl_cuda_device_data_t *src_dev
          = (pocl_cuda_device_data_t *)source->data;
      pocl_cuda_device_data_t *dst_dev = (pocl_cuda_device_data_t *)dest->data;
      cuDeviceCanAccessPeer (&possible, src_dev->device, dst_dev->device);
      POCL_MSG_PRINT_CUDA ("cuDeviceCanAccessPeer %p and %p -> %s\n", source,
                           dest, possible ? "yes" : "no");
      if (possible)
        {
          cuCtxSetCurrent (dst_dev->context);
          CUresult res = cuCtxEnablePeerAccess (src_dev->context, 0);
          POCL_MSG_PRINT_CUDA ("cuCtxEnablePeerAccess from %p to %p : %u\n",
                               source, dest, res);
          if (res == CUDA_SUCCESS
              || res == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
            return 1;
        }
    }

  return 0;
}

void
pocl_cuda_submit_read (CUstream stream, void *host_ptr, const void *device_ptr,
                       size_t offset, size_t cb)
{
  POCL_MSG_PRINT_CUDA ("cuMemcpyDtoHAsync %p -> %p / %zu B \n", device_ptr, host_ptr, cb);
  CUresult result = cuMemcpyDtoHAsync (
      host_ptr, (CUdeviceptr) (device_ptr + offset), cb, stream);
  CUDA_CHECK_ABORT (result, "cuMemcpyDtoHAsync");
}

void
pocl_cuda_submit_memfill (CUstream stream, void *mem_ptr, size_t size_in_bytes,
                          size_t offset, const void *pattern,
                          size_t pattern_size)
{
  CUresult result;
  switch (pattern_size)
    {
    case 1:
      result
          = cuMemsetD8Async ((CUdeviceptr) (((char *)mem_ptr) + offset),
                             *(unsigned char *)pattern, size_in_bytes, stream);
      break;
    case 2:
      result = cuMemsetD16Async ((CUdeviceptr) (((char *)mem_ptr) + offset),
                                 *(unsigned short *)pattern, size_in_bytes / 2,
                                 stream);
      break;
    case 4:
      result = cuMemsetD32Async ((CUdeviceptr) (((char *)mem_ptr) + offset),
                                 *(unsigned int *)pattern, size_in_bytes / 4,
                                 stream);
      break;
    case 8:
    case 16:
    case 32:
    case 64:
    case 128:
      POCL_ABORT_UNIMPLEMENTED ("fill_kernel with pattern_size >=8");
    default:
      POCL_ABORT ("unrecognized pattern_size");
    }
  CUDA_CHECK_ABORT (result, "cuMemset*Async");
}

void
pocl_cuda_submit_write (CUstream stream, const void *host_ptr,
                        void *device_ptr, size_t offset, size_t cb)
{
  POCL_MSG_PRINT_CUDA ("cuMemcpyHtoDAsync %p -> %p / %zu B \n", host_ptr, device_ptr, cb);
  CUresult result = cuMemcpyHtoDAsync ((CUdeviceptr) (device_ptr + offset),
                                       host_ptr, cb, stream);
  CUDA_CHECK_ABORT (result, "cuMemcpyHtoDAsync");
}

void
pocl_cuda_submit_copy (CUstream stream, void*__restrict__ src_mem_ptr,
                       size_t src_offset,  void *__restrict__ dst_mem_ptr,
                       size_t dst_offset, size_t cb)
{
  void *src_ptr = src_mem_ptr + src_offset;
  void *dst_ptr = dst_mem_ptr + dst_offset;

  if (src_ptr == dst_ptr)
    return;

  CUresult result;
  POCL_MSG_PRINT_CUDA ("cuMemcpyDtoDAsync %p -> %p / %zu B \n", src_ptr, dst_ptr, cb);
  result = cuMemcpyDtoDAsync ((CUdeviceptr)dst_ptr, (CUdeviceptr)src_ptr,
                                cb, stream);
  CUDA_CHECK_ABORT (result, "cuMemcpyDtoDAsync");
}

void
pocl_cuda_submit_copy_p2p (CUstream stream, cl_device_id src_device,
                           void *__restrict__ src_mem_ptr, size_t src_offset,
                           cl_device_id dst_device,
                           void *__restrict__ dst_mem_ptr, size_t dst_offset,
                           size_t cb)
{
  void *src_ptr = src_mem_ptr + src_offset;
  void *dst_ptr = dst_mem_ptr + dst_offset;

  CUresult result;
  POCL_MSG_PRINT_CUDA ("cuMemcpyPeerAsync %p -> %p / %zu B \n", src_ptr,
                       dst_ptr, cb);
  result = cuMemcpyPeerAsync (
      (CUdeviceptr)dst_ptr,
      ((pocl_cuda_device_data_t *)dst_device->data)->context,
      (CUdeviceptr)src_ptr,
      ((pocl_cuda_device_data_t *)dst_device->data)->context, cb, stream);
  CUDA_CHECK_ABORT (result, "cuMemcpyPeerAsync");
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

  POCL_MSG_PRINT_CUDA ("cuMemcpy3D / READ_RECT %p -> %p \n", device_ptr, host_ptr);

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
  CUDA_CHECK_ABORT (result, "cuMemcpy3DAsync");
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

  POCL_MSG_PRINT_CUDA ("cuMemcpy3D / WRITE_RECT %p -> %p \n", host_ptr, device_ptr);

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
  CUDA_CHECK_ABORT (result, "cuMemcpy3DAsync");
}

void
pocl_cuda_submit_copy_rect (CUstream stream,
                            cl_device_id dev,
                            void* src_ptr,
                            void* dst_ptr,
                            const size_t *__restrict__ const src_origin,
                            const size_t *__restrict__ const dst_origin,
                            const size_t *__restrict__ const region,
                            size_t const src_row_pitch,
                            size_t const src_slice_pitch,
                            size_t const dst_row_pitch,
                            size_t const dst_slice_pitch)
{
  CUDA_MEMCPY3D params = { 0 };

  POCL_MSG_PRINT_CUDA ("cuMemcpy3D / COPY_RECT %p -> %p \n", src_ptr, dst_ptr);

  params.WidthInBytes = region[0];
  params.Height = region[1];
  params.Depth = region[2];

  params.srcDevice = (CUdeviceptr)src_ptr;
  params.srcXInBytes = src_origin[0];
  params.srcY = src_origin[1];
  params.srcZ = src_origin[2];
  params.srcPitch = src_row_pitch;
  params.srcHeight = src_slice_pitch / src_row_pitch;

  params.dstDevice = (CUdeviceptr)dst_ptr;
  params.dstXInBytes = dst_origin[0];
  params.dstY = dst_origin[1];
  params.dstZ = dst_origin[2];
  params.dstPitch = dst_row_pitch;
  params.dstHeight = dst_slice_pitch / dst_row_pitch;

  params.srcMemoryType = params.dstMemoryType = CU_MEMORYTYPE_DEVICE;

  CUresult result = cuMemcpy3DAsync (&params, stream);
  CUDA_CHECK_ABORT (result, "cuMemcpy3DAsync");
}

void
pocl_cuda_submit_map_mem (CUstream stream, cl_mem mem,
                          pocl_mem_identifier *p,
                          size_t offset, size_t size, void *host_ptr)
{
  assert (host_ptr != NULL);

  if ((mem->flags & CL_MEM_USE_HOST_PTR)
      || (p->extra_ptr))
    return;

  POCL_MSG_PRINT_CUDA ("cuMemcpyDtoHAsync %p / %zu B \n", host_ptr, size);

  void *buf_ptr = p->mem_ptr;

  CUresult result = cuMemcpyDtoHAsync (
      host_ptr, (CUdeviceptr) (buf_ptr + offset), size, stream);
  CUDA_CHECK_ABORT (result, "cuMemcpyDtoHAsync");
}

void *
pocl_cuda_submit_unmap_mem (CUstream stream, pocl_mem_identifier *dst_mem_id,
                            size_t offset, size_t size, void *host_ptr,
                            cl_map_flags map_flags)
{
  /* Only copy back if mapped for writing */
  if (map_flags == CL_MAP_READ)
    return NULL;

  if (host_ptr)
    {
      CUresult result = cuMemcpyHtoDAsync (
          (CUdeviceptr) (dst_mem_id->mem_ptr + offset), host_ptr, size, stream);
      CUDA_CHECK_ABORT (result, "cuMemcpyHtoDAsync");
    }
  return NULL;
}

// https://docs.nvidia.com/cuda/ptx-compiler-api/index.html#basic-usage

/* build a program with builtin kernels. */

static int
pocl_cuda_build_cuda_builtins (cl_program program, cl_uint device_i)
{
  POCL_MSG_PRINT_CUDA ("preparing CUDA builtin kernels\n");
  cl_device_id dev = program->devices[device_i];
  pocl_cuda_device_data_t *ddata = (pocl_cuda_device_data_t *)dev->data;
  /* these only need to be built once per device */
  if (ddata->cuda_builtin_kernels_built)
    {
      POCL_MSG_PRINT_CUDA ("CUDA builtin kernels already built\n");
      return 0;
    }

  int have_sm70 = (ddata->sm_maj >= 7);

  uint64_t builtins_file_len = 0;
  char *builtins_file = NULL;
  char builtin_path[POCL_MAX_PATHNAME_LENGTH];

  char filename[64];
  filename[0] = '/';
  pocl_str_tolower (filename + 1, dev->ops->device_name);
  strcat (filename, "/");
  if (have_sm70)
    strcat (filename, "builtins_sm70.ptx");
  else
    strcat (filename, "builtins_sm50.ptx");

  pocl_get_srcdir_or_datadir (builtin_path, "/lib/CL/devices", "", filename);

  if (pocl_read_file (builtin_path, &builtins_file, &builtins_file_len) < 0)
    {
      POCL_MSG_ERR ("can't read cuda builtins from file %s\n", builtin_path);
      return -1;
    }

  CUresult res;
  CUfunction ff;
  CUmodule mod;
  res = cuModuleLoadData (&mod, builtins_file);
  if (CUDA_CHECK_ERROR (res, "cuModuleLoadData CudaBuiltinKernelsPTX"))
    return -1;

  memset (&ddata->cuda_builtin_kernels_program, 0,
          sizeof (pocl_cuda_program_data_t));
  ddata->cuda_builtin_kernels_program.module = mod;
  ddata->cuda_builtin_kernels_program.module_offsets = mod; // TODO fix this

  static size_t cuda_builtin_kernel_zero_alignments[20] = { 0 };

  res = cuModuleGetFunction (&ff, mod, "pocl_mul_i32");
  if (CUDA_CHECK_ERROR (res, "cuModuleGetFunction  pocl_mul_i32"))
    return -1;
  ddata->cuda_builtin_kernels_data[0].kernel = ff;
  ddata->cuda_builtin_kernels_data[0].kernel_offsets = ff; // TODO fix this
  ddata->cuda_builtin_kernels_data[0].alignments
      = cuda_builtin_kernel_zero_alignments;
  ddata->cuda_builtin_kernels_data[0].refcount = 1;

  res = cuModuleGetFunction (&ff, mod, "pocl_add_i32");
  if (CUDA_CHECK_ERROR (res, "cuModuleGetFunction  pocl_add_i32"))
    return -1;
  ddata->cuda_builtin_kernels_data[1].kernel = ff;
  ddata->cuda_builtin_kernels_data[1].kernel_offsets = ff; // TODO fix this
  ddata->cuda_builtin_kernels_data[1].alignments
      = cuda_builtin_kernel_zero_alignments;
  ddata->cuda_builtin_kernels_data[1].refcount = 1;

  res = cuModuleGetFunction (&ff, mod, "pocl_dnn_conv2d_int8_relu");
  if (CUDA_CHECK_ERROR (res, "cuModuleGetFunction  pocl_dnn_conv2d_int8_relu"))
    return -1;
  ddata->cuda_builtin_kernels_data[2].kernel = ff;
  ddata->cuda_builtin_kernels_data[2].kernel_offsets = ff; // TODO fix this
  ddata->cuda_builtin_kernels_data[2].alignments
      = cuda_builtin_kernel_zero_alignments;
  ddata->cuda_builtin_kernels_data[2].refcount = 1;

  res = cuModuleGetFunction (&ff, mod, "pocl_sgemm_local_f32");
  if (CUDA_CHECK_ERROR (res, "cuModuleGetFunction  pocl_sgemm_local_f32"))
    return -1;
  ddata->cuda_builtin_kernels_data[3].kernel = ff;
  ddata->cuda_builtin_kernels_data[3].kernel_offsets = ff; // TODO fix this
  // 3 pointers, 3 unsigned
  static size_t sgemm_local_alignments[] = {0, 0, 0, 4, 4, 4,};
  ddata->cuda_builtin_kernels_data[3].alignments = sgemm_local_alignments;
  ddata->cuda_builtin_kernels_data[3].refcount = 1;

  if (have_sm70)
  {
    res = cuModuleGetFunction (&ff, mod, "pocl_sgemm_tensor_f16f16f32");
    if (CUDA_CHECK_ERROR (res,
                          "cuModuleGetFunction  pocl_sgemm_tensor_f16f16f32"))
      return -1;
    ddata->cuda_builtin_kernels_data[4].kernel = ff;
    ddata->cuda_builtin_kernels_data[4].kernel_offsets = ff; // TODO fix this
    // 3 pointers, 3 unsigned
    static size_t sgemm_tensor_alignments[] = { 0, 0, 0, 4, 4, 4 };
    ddata->cuda_builtin_kernels_data[4].alignments = sgemm_tensor_alignments;
    ddata->cuda_builtin_kernels_data[4].refcount = 1;

    res = cuModuleGetFunction (&ff, mod, "pocl_sgemm_scale_tensor_f16f16f32");
    if (CUDA_CHECK_ERROR (
            res, "cuModuleGetFunction  pocl_sgemm_scale_tensor_f16f16f32"))
      return -1;
    ddata->cuda_builtin_kernels_data[5].kernel = ff;
    ddata->cuda_builtin_kernels_data[5].kernel_offsets = ff; // TODO fix this
    // 3 pointers, 3 unsigned, 2 floats
    static size_t sgemm_tensor_scale_alignments[] = { 0, 0, 0, 4, 4, 4, 4, 4 };
    ddata->cuda_builtin_kernels_data[5].alignments
        = sgemm_tensor_scale_alignments;
    ddata->cuda_builtin_kernels_data[5].refcount = 1;
  }
  ddata->cuda_builtin_kernels_built = CL_TRUE;
  return 0;
}


int
pocl_cuda_build_builtin (cl_program program, cl_uint device_i)
{
  if (pocl_cuda_build_cuda_builtins (program, device_i) != 0)
    {
      POCL_MSG_ERR ("pocl-cuda: failed to build CUDA builtin kernels\n");
      return -1;
    }

  if (pocl_driver_build_opencl_builtins (program, device_i) != 0)
    {
      POCL_MSG_ERR ("pocl-cuda: failed to build OpenCL builtin kernels\n");
      return -1;
    }

  return 0;
}

static int
pocl_cuda_build_ptx (void *llvm_ir, char *out_ptx, CUmodule *out_module,
                     cl_device_id device, pocl_cuda_device_data_t *ddata,
                     int use_offsets, CUdeviceptr *constant_mem_base,
                     size_t *constant_mem_size, void **alignments)
{
  assert (llvm_ir);
  assert (out_ptx);
  assert (out_module);
  CUresult result;

  /* Generate PTX from LLVM bitcode */
  if (!pocl_exists (out_ptx))
    {
      int r = pocl_ptx_gen (llvm_ir, out_ptx, device->llvm_cpu,
                            ddata->libdevice, use_offsets, alignments);
      POCL_RETURN_ERROR_ON ((r != 0), CL_BUILD_PROGRAM_FAILURE,
                            "pocl-cuda: failed to generate PTX file %s\n",
                            out_ptx);
    }
  else
    {
      int r = pocl_cuda_create_alignments (llvm_ir, alignments);
      POCL_RETURN_ERROR_ON (
          (r != 0), CL_BUILD_PROGRAM_FAILURE,
          "pocl-cuda: failed to create Alignments from BC\n");
    }

#ifdef POCL_DEBUG_MESSAGES
  if (!(pocl_debug_messages_filter & POCL_DEBUG_FLAG_CUDA))
    {
      result = cuModuleLoad (out_module, out_ptx);
      POCL_RETURN_ERROR_ON ((result != CUDA_SUCCESS), CL_BUILD_PROGRAM_FAILURE,
                            "cuModuleLoad PTX failed\n");
    }
  else
    {
#endif
      uintptr_t log_size = 1 << 12;
      char *log = (char *)malloc (log_size);
      CUjit_option opt[]
          = { CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES };
      char *content = NULL;
      uint64_t content_size = 0;

      pocl_read_file (out_ptx, &content, &content_size);
      POCL_RETURN_ERROR_ON ((content_size == 0), CL_BUILD_PROGRAM_FAILURE,
                            "failed to read PTX file: %s\n", out_ptx);
      void *val[] = { log, (void *)log_size };
      result = cuModuleLoadDataEx (out_module, content,
                                   sizeof (opt) / sizeof (opt[0]), opt, val);

      uintptr_t out_size = (uintptr_t)val[1];

      if (out_size > 0 || result != CUDA_SUCCESS)
        POCL_MSG_PRINT_CUDA ("cuModuleLoadDataEx(%s) log: %s\n", out_ptx, log);

      free (content);
      free (log);

      POCL_RETURN_ERROR_ON ((result != CUDA_SUCCESS), CL_BUILD_PROGRAM_FAILURE,
                            "cuModuleLoadDataEx failed\n");
#ifdef POCL_DEBUG_MESSAGES
    }
#endif

  /* Get handle to constant memory buffer */
  // this call might fail actually
  cuModuleGetGlobal (constant_mem_base, constant_mem_size, *out_module,
                     "_constant_memory_region_");

  return CL_SUCCESS;
}

int
pocl_cuda_post_build_program (cl_program program, cl_uint device_i)
{
  int result;
  cl_device_id device = program->devices[device_i];
  pocl_cuda_device_data_t *ddata = (pocl_cuda_device_data_t *)device->data;
  cuCtxSetCurrent (ddata->context);

  POCL_LOCK (ddata->compile_lock);
  char program_bc[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_program_bc_path (program_bc, program, device_i);

  char noofs_ptx_filename[POCL_MAX_PATHNAME_LENGTH];
  strcpy (noofs_ptx_filename, program_bc);
  strncat (noofs_ptx_filename, ".noOfs.ptx", POCL_MAX_PATHNAME_LENGTH - 1);

  char ofs_ptx_filename[POCL_MAX_PATHNAME_LENGTH];
  strcpy (ofs_ptx_filename, program_bc);
  strncat (ofs_ptx_filename, ".Ofs.ptx", POCL_MAX_PATHNAME_LENGTH - 1);

  /* Load PTX module */
  size_t constant_mem_size = 0;
  void *align_map = NULL;
  CUdeviceptr constant_mem_base;
  CUmodule noofs_module;
  CUmodule ofs_module;

  assert (program->llvm_irs[device_i]);
  result = pocl_llvm_run_passes_on_program (program, device_i);
  assert (result == CL_SUCCESS);

  /*
    pocl_cuda_build_ptx(program_bc, noofs_ptx_filename, &noofs_module, device,
    ddata, 0, &constant_mem_base, &constant_mem_size); pdata->module =
    noofs_module; pdata->constant_mem_base = constant_mem_base;
    pdata->constant_mem_size = constant_mem_size;
  */

  result = pocl_cuda_build_ptx (
      program->llvm_irs[device_i], ofs_ptx_filename, &ofs_module, device,
      ddata, 1, &constant_mem_base, &constant_mem_size, &align_map);
  if (result == CL_SUCCESS)
    {
      pocl_cuda_program_data_t *pdata = (pocl_cuda_program_data_t *)calloc (
          1, sizeof (pocl_cuda_program_data_t));
      POCL_RETURN_ERROR_COND ((pdata == NULL), CL_OUT_OF_HOST_MEMORY);
      pdata->module_offsets = ofs_module;
      pdata->constant_mem_base_offsets = constant_mem_base;
      pdata->constant_mem_size_offsets = constant_mem_size;
      pdata->align_map_offsets = align_map;
      program->data[device_i] = pdata;
    }
  POCL_UNLOCK (ddata->compile_lock);
  return result;
}

int
pocl_cuda_free_program (cl_device_id device, cl_program program,
                        unsigned device_i)
{
  CUresult result;
  pocl_cuda_device_data_t *ddata = (pocl_cuda_device_data_t *)device->data;
  if (program->data[device_i] == NULL)
    return CL_SUCCESS;

  pocl_cuda_program_data_t *pdata
      = (pocl_cuda_program_data_t *)program->data[device_i];
  /*
      if (pdata->module)
      {
        result = cuModuleUnload(pdata->module);
        CUDA_CHECK (result, "cuModuleUnload");
        pdata->module = NULL;
      }
  */
  if (pdata->module_offsets)
    {
      result = cuModuleUnload (pdata->module_offsets);
      if (result != CUDA_SUCCESS)
        {
          POCL_MSG_ERR ("cuModuleUnload failed\n");
        }
      pdata->module_offsets = NULL;
    }
  if (pdata->align_map_offsets)
    {
      pocl_cuda_destroy_alignments (program->llvm_irs[device_i],
                                    pdata->align_map_offsets);
      pdata->align_map_offsets = NULL;
    }
  pocl_driver_free_program (device, program, device_i);
  POCL_MEM_FREE (program->data[device_i]);
  return CL_SUCCESS;
}

void
pocl_cuda_compile_kernel (_cl_command_node *cmd, cl_kernel kernel,
                          cl_device_id device, int specialize)
{
  return;
}

int
pocl_cuda_create_kernel (cl_device_id device, cl_program program,
                         cl_kernel kernel, unsigned device_i)
{
  CUresult result;
  pocl_kernel_metadata_t *meta = kernel->meta;
  pocl_cuda_program_data_t *pdata
      = (pocl_cuda_program_data_t *)program->data[device_i];
  assert (pdata != NULL);
  pocl_cuda_kernel_data_t *kdata
      = (pocl_cuda_kernel_data_t *)meta->data[device_i];
  if (kdata != NULL)
    {
      ++kdata->refcount;
      return CL_SUCCESS;
    }

  kdata = meta->data[device_i]
      = (void *)calloc (1, sizeof (pocl_cuda_kernel_data_t));

  /* Get kernel function */
  CUfunction function = NULL;
  /*
    result = cuModuleGetFunction (&function, pdata->module, kernel->name);
    CUDA_CHECK (result, "cuModuleGetFunction");
    kdata->kernel = function;
  */
  assert (pdata->module_offsets);
  result
      = cuModuleGetFunction (&function, pdata->module_offsets, kernel->name);
  if (result != CUDA_SUCCESS)
    {
      POCL_MSG_ERR ("pocl_cuda_create_kernel: cuModuleGetFunction() failed\n");
      POCL_MEM_FREE (meta->data[device_i]);
      return CL_OUT_OF_RESOURCES;
    }

  kdata->kernel_offsets = function;
  /* Get pointer alignment */
  kdata->alignments
      = calloc (meta->num_args + meta->num_locals + 4, sizeof (size_t));
  pocl_cuda_get_ptr_arg_alignment (program->llvm_irs[device_i], kernel->name,
                                   kdata->alignments,
                                   pdata->align_map_offsets);
  ++kdata->refcount;
  return CL_SUCCESS;
}

int
pocl_cuda_free_kernel (cl_device_id device, cl_program program,
                       cl_kernel kernel, unsigned device_i)
{
  pocl_kernel_metadata_t *meta = kernel->meta;
  pocl_cuda_kernel_data_t *kdata
      = (pocl_cuda_kernel_data_t *)meta->data[device_i];
  if (kdata != NULL)
    {
      --kdata->refcount;
      if (kdata->refcount == 0)
        {
          POCL_MEM_FREE (kdata);
          meta->data[device_i] = NULL;
        }
    }
  return CL_SUCCESS;
}

#ifdef ENABLE_CUDNN
void
submit_cudnn_kernel(CUstream stream, _cl_command_node *cmd,
                    cl_device_id device, cl_event event)
{
  
  _cl_command_run run = cmd->command.run;
  cl_kernel kernel = run.kernel;
  cl_program prog = kernel->program;
  pocl_argument *arguments = run.arguments;
  struct pocl_context pc = run.pc;
  pocl_kernel_metadata_t *meta = kernel->meta;

  // Input, weight and output buffers come from pocl's buffer management
  cl_mem mem = *(void **)arguments[0].value;
  float* in_data = (float*)(mem->device_ptrs[device->global_mem_id].mem_ptr + arguments[0].offset);
  // cl_float16* in_data = (float*)(mem->device_ptrs[device->global_mem_id].mem_ptr + arguments[0].offset);

  mem = *(void **)arguments[1].value;
  float* filt_data = (float*)(mem->device_ptrs[device->global_mem_id].mem_ptr + arguments[1].offset);
  // cl_float16* filt_data = (float*)(mem->device_ptrs[device->global_mem_id].mem_ptr + arguments[1].offset);

  mem = *(void **)arguments[2].value;
  float* out_data = (float*)(mem->device_ptrs[device->global_mem_id].mem_ptr + arguments[2].offset);
  // cl_float16* out_data = (float*)(mem->device_ptrs[device->global_mem_id].mem_ptr + arguments[2].offset);

  // All the other convolution dimensions are passed as arguments.
  int in_n      = *(int*)(arguments[3].value);
  int in_c      = *(int*)(arguments[4].value);
  int in_h      = *(int*)(arguments[5].value);
  int in_w      = *(int*)(arguments[6].value);

  int filt_k = *(int*)(arguments[7].value);
  int filt_c = *(int*)(arguments[8].value);
  int filt_h = *(int*)(arguments[9].value);
  int filt_w = *(int*)(arguments[10].value);
  
  int str_h = *(int*)(arguments[11].value);
  int str_w = *(int*)(arguments[12].value);
  int dil_h = *(int*)(arguments[13].value);
  int dil_w = *(int*)(arguments[14].value);
  int pad_h = *(int*)(arguments[15].value);
  int pad_w = *(int*)(arguments[16].value);
  
  int groups = *(int*)(arguments[17].value);

  float alpha = *(float*)(arguments[18].value);
  float beta  = *(float*)(arguments[19].value);

  /*POCL_MSG_PRINT_INFO("ARGS:%zx,%zx,%zx in:%i,%i,%i,%i filt %i,%i,%i,%i strdilpad %i,%i,%i,%i,%i,%i\n",
  in_data, filt_data, out_data, in_n, in_c, in_h, in_w, filt_k, filt_c, filt_h, filt_w,
   str_h, str_w, dil_h, dil_w,pad_h,pad_w);*/

  cudnnTensorDescriptor_t in_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, /*CUDNN_DATA_HALF*/
        in_n, in_c, in_h, in_w));

  cudnnFilterDescriptor_t filt_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(
        filt_desc, CUDNN_DATA_FLOAT /*CUDNN_DATA_HALF*/, CUDNN_TENSOR_NCHW,
        filt_k, filt_c, filt_h, filt_w));

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w, str_h, str_w, dil_h, dil_w,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT /*CUDNN_DATA_HALF*/));

  //For depth-wise convolutions
  CUDNN_CALL(cudnnSetConvolutionGroupCount(conv_desc, groups));

  // output
  int out_n,out_c,out_h,out_w;
  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc,
        &out_n, &out_c, &out_h, &out_w));

  cudnnTensorDescriptor_t out_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,/*CUDNN_DATA_HALF*/
        out_n, out_c, out_h, out_w));

  // algorithm
  cudnnConvolutionFwdAlgoPerf_t algos;
  int return_count = 0;
  /*CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn,
        in_desc, filt_desc, conv_desc, out_desc,
        1, &return_count, &algos));

  cudnnConvolutionFwdAlgo_t algo = algos.algo;*/
  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM; // try IMPLICIT_PRECOMP_GEMM if tensor cores not working
  POCL_MSG_PRINT_INFO("CuDNN Picking ALGO %d",algo);

  // CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH)); // this may not be necessary

  // workspace
  size_t ws_size; 
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));
  float *ws_data;
  CUDA_CALL(cudaMalloc(&ws_data, ws_size));

  CUDNN_CALL(cudnnConvolutionForward(
      cudnn,
      &alpha, in_desc, in_data, filt_desc, filt_data,
      conv_desc, algo, ws_data, ws_size,
      &beta, out_desc, out_data));

 // Not sure if needed
  CUDA_CALL (cudaDeviceSynchronize ());

  // finalizing
  CUDA_CALL(cudaFree(ws_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
}

#endif //ENABLE_CUDNN

void
pocl_cuda_submit_kernel (CUstream stream, _cl_command_node *cmd,
                         cl_device_id device, cl_event event)
{
  _cl_command_run run = cmd->command.run;
  cl_kernel kernel = run.kernel;
  cl_program prog = kernel->program;
  pocl_argument *arguments = run.arguments;
  struct pocl_context pc = run.pc;
  pocl_kernel_metadata_t *meta = kernel->meta;
  CUdeviceptr constant_mem_base;
  size_t constant_mem_size;
  pocl_cuda_device_data_t *ddata = (pocl_cuda_device_data_t *)device->data;

  if (pc.num_groups[0] == 0 || pc.num_groups[1] == 0 || pc.num_groups[2] == 0)
    return;

  /* Check if we need to handle global work offsets */
  int has_offsets =
    (pc.global_offset[0] || pc.global_offset[1] || pc.global_offset[2]);

  CUmodule module = NULL;
  CUfunction function = NULL;
  pocl_cuda_kernel_data_t *kdata = NULL;
  char *saved_kernel_name = NULL;

  /* Get kernel function */
  if (prog->num_builtin_kernels > 0)
    {
      assert (0 && "builtin kernels unsupported");
      /* CUDA builtins */
      for (size_t i = 0; i < CUDA_BUILTIN_KERNELS; ++i)
        {
          if (strcmp (kernel->name, cuda_builtin_kernels[i]) == 0)
            {
              module = ddata->cuda_builtin_kernels_program.module;
              kdata = &ddata->cuda_builtin_kernels_data[i];
              function = kdata->kernel;
              break;
            }
        }
#ifdef ENABLE_CUDNN
      /* CUDNN builtins */
      for (size_t i = 0; i < CUDNN_BUILTIN_KERNELS; ++i)
        {
          if (strcmp(kernel->name, cudnn_builtin_kernels[i]) == 0)
            {
              submit_cudnn_kernel(stream, cmd, device, event);
              return;
            }
        }
#endif //ENABLE_CUDNN
    }

  if (kdata == NULL)
    {
      /* handle OpenCL builtins */
      for (size_t i = 0; i < OPENCL_BUILTIN_KERNELS; ++i)
        {
          if (strcmp (kernel->name, opencl_builtin_kernels[i]) == 0)
            {
              pocl_sanitize_builtin_kernel_name (kernel, &saved_kernel_name);
              break;
            }
        }
      int has_offsets1 = 1;
      kdata = (pocl_cuda_kernel_data_t *)meta->data[cmd->program_device_i];
      pocl_cuda_program_data_t *pdata
          = (pocl_cuda_program_data_t *)prog->data[cmd->program_device_i];
      module = has_offsets1 ? pdata->module_offsets : pdata->module;
      function = has_offsets1 ? kdata->kernel_offsets : kdata->kernel;
      constant_mem_base = has_offsets1 ? pdata->constant_mem_base_offsets
                                       : pdata->constant_mem_base;
      constant_mem_size = has_offsets1 ? pdata->constant_mem_size_offsets
                                       : pdata->constant_mem_size;
      if (saved_kernel_name)
        pocl_restore_builtin_kernel_name (kernel, saved_kernel_name);
    }

  assert (kdata);

  /* Prepare kernel arguments */
  void *null = NULL;
  unsigned sharedMemBytes = 0;
  void *params[meta->num_args + meta->num_locals + 4];
  unsigned sharedMemOffsets[meta->num_args + meta->num_locals];
  unsigned constantMemBytes = 0;
  unsigned constantMemOffsets[meta->num_args];
  unsigned globalOffsets[3];

  CUresult result;
  unsigned i;
  for (i = 0; i < meta->num_args; i++)
    {
      pocl_argument_type type = meta->arg_info[i].type;
      switch (type)
        {
        case POCL_ARG_TYPE_NONE:
          params[i] = arguments[i].value;
          break;
        case POCL_ARG_TYPE_POINTER:
          {
            if (ARG_IS_LOCAL (meta->arg_info[i]))
              {
                size_t size = arguments[i].size;
                size_t align = kdata->alignments[i];
                if (align < 1)
                  align = 1;

                /* Pad offset to align memory */
                if (sharedMemBytes % align)
                  sharedMemBytes += align - (sharedMemBytes % align);

                sharedMemOffsets[i] = sharedMemBytes;
                params[i] = sharedMemOffsets + i;

                sharedMemBytes += size;
              }
            else if (arguments[i].is_svm == 1)
              {
                params[i] = arguments[i].value;
              }
            else if (meta->arg_info[i].address_qualifier
                     == CL_KERNEL_ARG_ADDRESS_CONSTANT)
              {
                assert (constant_mem_base);
                assert (arguments[i].is_svm == 0);

                /* Get device pointer */
                cl_mem mem = *(void **)arguments[i].value;
                CUdeviceptr src
                    = (CUdeviceptr)mem->device_ptrs[device->global_mem_id].mem_ptr
                      + arguments[i].offset;

                size_t align = kdata->alignments[i];
                assert (align && "Zero alignment for pointer argument!");
                if (constantMemBytes % align)
                  {
                    constantMemBytes += align - (constantMemBytes % align);
                  }

                /* Copy to constant buffer at current offset */
                result
                    = cuMemcpyDtoDAsync (constant_mem_base + constantMemBytes,
                                         src, mem->size, stream);
                CUDA_CHECK_ABORT (result, "cuMemcpyDtoDAsync");

                constantMemOffsets[i] = constantMemBytes;
                params[i] = constantMemOffsets + i;

                constantMemBytes += mem->size;
              }
            else
              {
                assert (arguments[i].is_svm == 0);
                if (arguments[i].value)
                  {
                    cl_mem mem = *(void **)arguments[i].value;
                    params[i] = &mem->device_ptrs[device->global_mem_id].mem_ptr
                                + arguments[i].offset;

                    /* On ARM with USE_HOST_PTR, perform explicit copy to
                     * device */
                    if ((mem->flags & CL_MEM_USE_HOST_PTR) &&
                        !((pocl_cuda_device_data_t *)device->data)->supports_cu_mem_host_register)
                      {
                        cuMemcpyHtoD (*(CUdeviceptr *)(params[i]),
                                      mem->mem_host_ptr, mem->size);
                        cuStreamSynchronize (0);
                      }
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
          POCL_ABORT ("Unhandled argument type for CUDA\n");
          break;
        }
    }

  if (constantMemBytes > constant_mem_size)
    POCL_ABORT ("[CUDA] Total constant buffer size %u exceeds %lu allocated\n",
                constantMemBytes, constant_mem_size);

  unsigned arg_index = meta->num_args;

  if (sharedMemBytes != 0)
    {
      /* Deal with automatic local allocations if there are local function args
       */
      /* TODO: Would be better to remove arguments and make these static GEPs
       */
      for (i = 0; i < meta->num_locals; ++i, ++arg_index)
        {
          size_t size = meta->local_sizes[i];
          size_t align = kdata->alignments[arg_index];
          if (align < 1)
            align = 1;

          /* Pad offset to align memory */
          if (sharedMemBytes % align)
            sharedMemBytes += align - (sharedMemBytes % align);

          sharedMemOffsets[arg_index] = sharedMemBytes;
          sharedMemBytes += size;
          params[arg_index] = sharedMemOffsets + arg_index;
        }
    }

  /* Add global work dimensionality */
  params[arg_index++] = &pc.work_dim;

  /* Add global offsets if necessary */
  if (has_offsets)
    {
      globalOffsets[0] = pc.global_offset[0];
      globalOffsets[1] = pc.global_offset[1];
      globalOffsets[2] = pc.global_offset[2];
      params[arg_index++] = globalOffsets + 0;
      params[arg_index++] = globalOffsets + 1;
      params[arg_index++] = globalOffsets + 2;
    }

  /* Launch kernel */
  result = cuLaunchKernel (function, pc.num_groups[0], pc.num_groups[1],
                           pc.num_groups[2], pc.local_size[0],
                           pc.local_size[1], pc.local_size[2], sharedMemBytes,
                           stream, params, NULL);
  CUDA_CHECK_ABORT (result, "cuLaunchKernel");
}

void
pocl_cuda_submit_node (_cl_command_node *node, cl_command_queue cq, int locked)
{
  CUresult result;
  CUstream stream = ((pocl_cuda_queue_data_t *)cq->data)->stream;

  if (!locked)
    POCL_LOCK_OBJ (node->sync.event.event);

  pocl_cuda_event_data_t *event_data
      = (pocl_cuda_event_data_t *)node->sync.event.event->data;

  /* Process event dependencies */
  event_node *dep = NULL;
  LL_FOREACH (node->sync.event.event->wait_list, dep)
  {
    /* If it is in the process of completing, just skip it */
    if (dep->event->status <= CL_COMPLETE)
      continue;

    /* Add CUDA event dependency */
    if (dep->event->command_type != CL_COMMAND_USER
        && dep->event->queue->device->ops == cq->device->ops)
      {
        /* Block stream on event, but only for different queues */
        if (dep->event->queue != node->sync.event.event->queue)
          {
            pocl_cuda_event_data_t *dep_data
                = (pocl_cuda_event_data_t *)dep->event->data;

            /* Wait until dependency has finished being submitted */
            while (!dep_data->events_ready)
              ;

            result = cuStreamWaitEvent (stream, dep_data->end, 0);
            CUDA_CHECK_ABORT (result, "cuStreamWaitEvent");
          }
      }
    else
      {
        if (!((pocl_cuda_queue_data_t *)cq->data)->use_threads)
          POCL_ABORT (
              "Can't handle non-CUDA dependencies without queue threads\n");

        event_data->num_ext_events++;
      }
    }

  /* Wait on flag for external events */
  if (event_data->num_ext_events)
    {
      CUdeviceptr dev_ext_event_flag;
      result = cuMemHostAlloc ((void **)&event_data->ext_event_flag, 4,
                               CU_MEMHOSTALLOC_DEVICEMAP);
      CUDA_CHECK_ABORT (result, "cuMemAllocHost");

      *event_data->ext_event_flag = 0;

      result = cuMemHostGetDevicePointer (&dev_ext_event_flag,
                                           event_data->ext_event_flag, 0);
      CUDA_CHECK_ABORT (result, "cuMemHostGetDevicePointer");
      result = cuStreamWaitValue32 (stream, dev_ext_event_flag, 1,
                                    CU_STREAM_WAIT_VALUE_GEQ);
      CUDA_CHECK_ABORT (result, "cuStreamWaitValue32");
    }

  /* Create and record event for command start if profiling enabled */
  if (cq->properties & CL_QUEUE_PROFILING_ENABLE)
    {
      result = cuEventCreate (&event_data->start, CU_EVENT_DEFAULT);
      CUDA_CHECK_ABORT (result, "cuEventCreate");
      result = cuEventRecord (event_data->start, stream);
      CUDA_CHECK_ABORT (result, "cuEventRecord");
    }

  pocl_update_event_submitted (node->sync.event.event);

  POCL_UNLOCK_OBJ (node->sync.event.event);

  cl_event event = node->sync.event.event;
  cl_device_id dev = node->device;
  _cl_command_t *cmd = &node->command;

  switch (node->type)
    {
    case CL_COMMAND_READ_BUFFER:
      pocl_cuda_submit_read (
          stream, cmd->read.dst_host_ptr, cmd->read.src_mem_id->mem_ptr,
          node->command.read.offset, node->command.read.size);
      break;
    case CL_COMMAND_WRITE_BUFFER:
      pocl_cuda_submit_write (
          stream, cmd->write.src_host_ptr, cmd->write.dst_mem_id->mem_ptr,
          node->command.write.offset, node->command.write.size);
      break;
    case CL_COMMAND_COPY_BUFFER:
      {
        pocl_cuda_submit_copy (
            stream, cmd->copy.src_mem_id->mem_ptr, cmd->copy.src_offset,
            cmd->copy.dst_mem_id->mem_ptr, cmd->copy.dst_offset, cmd->copy.size);
        break;
      }
    case CL_COMMAND_READ_BUFFER_RECT:
      pocl_cuda_submit_read_rect (
          stream,
          cmd->read_rect.dst_host_ptr,
          cmd->read_rect.src_mem_id->mem_ptr,
          cmd->read_rect.buffer_origin,
          cmd->read_rect.host_origin,
          cmd->read_rect.region,
          cmd->read_rect.buffer_row_pitch,
          cmd->read_rect.buffer_slice_pitch,
          cmd->read_rect.host_row_pitch,
          cmd->read_rect.host_slice_pitch);
      break;
    case CL_COMMAND_WRITE_BUFFER_RECT:
      pocl_cuda_submit_write_rect (
          stream,
          cmd->write_rect.src_host_ptr,
          cmd->write_rect.dst_mem_id->mem_ptr,
          cmd->write_rect.buffer_origin,
          cmd->write_rect.host_origin,
          cmd->write_rect.region,
          cmd->read_rect.buffer_row_pitch,
          cmd->read_rect.buffer_slice_pitch,
          cmd->read_rect.host_row_pitch,
          cmd->read_rect.host_slice_pitch);
      break;
    case CL_COMMAND_COPY_BUFFER_RECT:
      {
        pocl_cuda_submit_copy_rect (
          stream, dev,
          cmd->copy_rect.src_mem_id->mem_ptr,
          cmd->copy_rect.dst_mem_id->mem_ptr,
          cmd->copy_rect.src_origin,
          cmd->copy_rect.dst_origin,
          cmd->copy_rect.region,
          cmd->copy_rect.src_row_pitch,
          cmd->copy_rect.src_slice_pitch,
          cmd->copy_rect.dst_row_pitch,
          cmd->copy_rect.dst_slice_pitch);
        break;
      }
    case CL_COMMAND_MAP_BUFFER:
      {
        cl_mem buffer = event->mem_objs[0];
        pocl_cuda_submit_map_mem (
            stream, buffer, cmd->map.mem_id, cmd->map.mapping->offset,
            cmd->map.mapping->size, cmd->map.mapping->host_ptr);
        break;
      }
    case CL_COMMAND_UNMAP_MEM_OBJECT:
      {
        cl_mem buffer = event->mem_objs[0];
        assert (buffer->is_image == CL_FALSE);
        pocl_cuda_submit_unmap_mem (
            stream,
            cmd->unmap.mem_id,
            cmd->unmap.mapping->offset,
            cmd->unmap.mapping->size,
            cmd->unmap.mapping->host_ptr,
            cmd->unmap.mapping->map_flags);
        break;
      }
    case CL_COMMAND_NDRANGE_KERNEL:
      pocl_cuda_submit_kernel (stream, node, node->device,
                               node->sync.event.event);
      break;

    case CL_COMMAND_MIGRATE_MEM_OBJECTS:
      switch (cmd->migrate.type)
        {
        case ENQUEUE_MIGRATE_TYPE_D2H:
          {
            cl_mem mem = event->mem_objs[0];
            pocl_cuda_submit_read (stream, mem->mem_host_ptr,
                                   cmd->migrate.mem_id->mem_ptr, 0, mem->size);
            break;
          }
        case ENQUEUE_MIGRATE_TYPE_H2D:
          {
            cl_mem mem = event->mem_objs[0];
            pocl_cuda_submit_write (stream, mem->mem_host_ptr,
                                    cmd->migrate.mem_id->mem_ptr, 0,
                                    mem->size);
            break;
          }
        case ENQUEUE_MIGRATE_TYPE_D2D:
          {
            cl_mem mem = event->mem_objs[0];
            pocl_cuda_submit_copy_p2p (
                stream, cmd->migrate.src_device, cmd->migrate.src_id->mem_ptr,
                0, cq->device, cmd->migrate.dst_id->mem_ptr, 0, mem->size);
            break;
          }
        case ENQUEUE_MIGRATE_TYPE_NOP:
          {
            break;
          }
        }
      break;

    case CL_COMMAND_MARKER:
    case CL_COMMAND_BARRIER:
      break;

    case CL_COMMAND_FILL_BUFFER:
      pocl_cuda_submit_memfill (stream, cmd->memfill.dst_mem_id->mem_ptr,
                                cmd->memfill.size, cmd->memfill.offset,
                                cmd->memfill.pattern,
                                cmd->memfill.pattern_size);
      break;
    case CL_COMMAND_SVM_MAP:
    case CL_COMMAND_SVM_UNMAP:
      /* empty */
      break;

    case CL_COMMAND_SVM_MEMCPY:
      pocl_cuda_svm_copy_async (stream, cmd->svm_memcpy.dst,
                                cmd->svm_memcpy.src, cmd->svm_memcpy.size);
      break;
    case CL_COMMAND_SVM_MEMFILL:
      pocl_cuda_submit_memfill (stream, cmd->svm_fill.svm_ptr,
                                cmd->svm_fill.size, 0, cmd->svm_fill.pattern,
                                cmd->svm_fill.pattern_size);
      break;
    case CL_COMMAND_SVM_FREE:
      if (cmd->svm_free.pfn_free_func)
        {
          cmd->svm_free.pfn_free_func (
              cmd->svm_free.queue, cmd->svm_free.num_svm_pointers,
              cmd->svm_free.svm_pointers, cmd->svm_free.data);
        }
      else
        {
          int i;
          for (i = 0; i < cmd->svm_free.num_svm_pointers; i++)
            {
              void *ptr = cmd->svm_free.svm_pointers[i];
              POCL_LOCK_OBJ (event->context);
              pocl_svm_ptr *tmp = NULL, *item = NULL;
              DL_FOREACH_SAFE (event->context->svm_ptrs, item, tmp)
              {
                if (item->svm_ptr == ptr)
                  {
                    DL_DELETE (event->context->svm_ptrs, item);
                    break;
                  }
              }
              POCL_UNLOCK_OBJ (event->context);
              assert (item);
              POCL_MEM_FREE (item);
              // Leads to 'undefined symbol: POclReleaseContext'
              // POname (clReleaseContext) (event->context);

              dev->ops->svm_free (dev, ptr);
            }
        }
      break;
    case CL_COMMAND_READ_IMAGE:
    case CL_COMMAND_WRITE_IMAGE:
    case CL_COMMAND_COPY_IMAGE:
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
    case CL_COMMAND_FILL_IMAGE:
    case CL_COMMAND_MAP_IMAGE:
    case CL_COMMAND_NATIVE_KERNEL:
    default:
      POCL_ABORT_UNIMPLEMENTED (pocl_command_to_str (node->type));
      break;
    }

  /* Create and record event for command end */
  if (cq->properties & CL_QUEUE_PROFILING_ENABLE)
    result = cuEventCreate (&event_data->end, CU_EVENT_DEFAULT);
  else
    result = cuEventCreate (&event_data->end, CU_EVENT_DISABLE_TIMING);
  CUDA_CHECK_ABORT (result, "cuEventCreate");
  result = cuEventRecord (event_data->end, stream);
  CUDA_CHECK_ABORT (result, "cuEventRecord");

  event_data->events_ready = 1;
}

void
pocl_cuda_submit (_cl_command_node *node, cl_command_queue cq)
{
  /* Allocate CUDA event data */
  pocl_cuda_event_data_t *p
      = (pocl_cuda_event_data_t *)calloc (1, sizeof (pocl_cuda_event_data_t));
  node->sync.event.event->data = p;

  if (((pocl_cuda_queue_data_t *)cq->data)->use_threads)
    {

      PTHREAD_CHECK (pthread_cond_init (&p->event_cond, NULL));
      /* Add command to work queue */
      POCL_UNLOCK_OBJ (node->sync.event.event);
      pocl_cuda_queue_data_t *queue_data = (pocl_cuda_queue_data_t *)cq->data;
      PTHREAD_CHECK (pthread_mutex_lock (&queue_data->lock));
      DL_APPEND (queue_data->pending_queue, node);
      PTHREAD_CHECK (pthread_cond_signal (&queue_data->pending_cond));
      PTHREAD_CHECK (pthread_mutex_unlock (&queue_data->lock));
    }
  else
    {
      /* Submit command in this thread */
      cuCtxSetCurrent (((pocl_cuda_device_data_t *)cq->device->data)->context);
      pocl_cuda_submit_node (node, cq, 1);
    }
}

void
pocl_cuda_notify (cl_device_id device, cl_event event, cl_event finished)
{
  /* Ignore CUDA device events, we've already handled these dependencies */
  if (finished->queue && finished->queue->device->ops == device->ops)
    return;

  if (event->status == CL_QUEUED)
    return;

  pocl_cuda_event_data_t *event_data = (pocl_cuda_event_data_t *)event->data;

  assert (event_data);
  assert (event_data->num_ext_events > 0);
  assert (event_data->ext_event_flag);

  /* If dependency failed, so should we */
  /* TODO: This isn't true if this is an implicit dependency */
  if (finished->status < 0)
    event->status = -1;

  /* Decrement external event counter */
  /* Trigger flag if none left */
  if (!--event_data->num_ext_events)
    *event_data->ext_event_flag = 1;
}

void
pocl_cuda_flush (cl_device_id device, cl_command_queue cq)
{
  /* TODO: Something here? */
}

void
pocl_cuda_finalize_command (cl_device_id device, cl_event event)
{
  CUresult result;
  pocl_cuda_event_data_t *event_data = (pocl_cuda_event_data_t *)event->data;

  /* Wait for command to finish */
  cuCtxSetCurrent (((pocl_cuda_device_data_t *)device->data)->context);
  result = cuEventSynchronize (event_data->end);
  CUDA_CHECK_ABORT (result, "cuEventSynchronize");

  if (event->command_type == CL_COMMAND_NDRANGE_KERNEL
      || event->command_type == CL_COMMAND_TASK)
    {
      if (!((pocl_cuda_device_data_t *)device->data)->supports_cu_mem_host_register)
        {
          /* On ARM with USE_HOST_PTR, perform explict copies back from device */
          cl_kernel kernel = event->command->command.run.kernel;
          pocl_argument *arguments = event->command->command.run.arguments;
          unsigned i;
          pocl_kernel_metadata_t *meta = kernel->meta;
          for (i = 0; i < meta->num_args; i++)
            {
              if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER)
                {
                  if (!ARG_IS_LOCAL (meta->arg_info[i]) && arguments[i].value)
                    {
                      cl_mem mem = *(void **)arguments[i].value;
                      if (mem->flags & CL_MEM_USE_HOST_PTR)
                        {
                          CUdeviceptr ptr
                              = (CUdeviceptr)mem->device_ptrs[device->global_mem_id]
                                    .mem_ptr;
                          cuMemcpyDtoH (mem->mem_host_ptr, ptr, mem->size);
                          cuStreamSynchronize (0);
                        }
                    }
                }
            }
        }
    }

  /* Handle failed events */

  pocl_update_event_running (event);
  if (event->status < 0)
    pocl_update_event_failed (event);
  else
    POCL_UPDATE_EVENT_COMPLETE_MSG (event, "CUDA event");
}

void
pocl_cuda_update_event (cl_device_id device, cl_event event)
{
  if ((event->status == CL_COMPLETE)
      && (event->queue->properties & CL_QUEUE_PROFILING_ENABLE))
    {
      /* Update timing info with CUDA event timers if profiling enabled */
      /* CUDA doesn't provide a way to get event timestamps directly,
       * only the elapsed time between two events. We use the elapsed
       * time from the epoch event enqueued on device creation to get
       * the actual timestamps.
       * More specifically, we measure the time between the epoch event
       * and the start event. This results in the start time. Then we
       * take the elapsed time between the start and end event to
       * compute the end time. Measuring the end time relative to the
       * epoch event may result in unprecise measurements due to the
       * usage of float by CUDA. */

      float diff;
      CUresult result, result2;
      pocl_cuda_event_data_t *event_data
          = (pocl_cuda_event_data_t *)event->data;
      cl_ulong epoch = ((pocl_cuda_device_data_t *)device->data)->epoch;

      result = cuEventElapsedTime (
          &diff, ((pocl_cuda_device_data_t *)device->data)->epoch_event,
          event_data->start);
      CUDA_CHECK_ERROR (result, "cuEventElapsedTime");
      event->time_start = epoch + (cl_ulong)(diff * 1e6);

      result2 = cuEventElapsedTime (&diff, event_data->start, event_data->end);
      CUDA_CHECK_ERROR (result2, "cuEventElapsedTime");
      event->time_end = event->time_start + (cl_ulong)(diff * 1e6);

      if (result != CUDA_SUCCESS || result2 != CUDA_SUCCESS)
        {
          event->time_start = 0;
          event->time_end = 0;
        }
    }
}

void
pocl_cuda_wait_event_recurse (cl_device_id device, cl_event event)
{
  while (event->wait_list)
    pocl_cuda_wait_event_recurse (device, event->wait_list->event);

  if (event->status > CL_COMPLETE)
    pocl_cuda_finalize_command (device, event);
}

void
pocl_cuda_notify_event_finished (cl_event event)
{
  pocl_cuda_event_data_t *e_d = (pocl_cuda_event_data_t *)event->data;

  if (((pocl_cuda_queue_data_t *)event->queue->data)->use_threads)
    PTHREAD_CHECK (pthread_cond_broadcast (&e_d->event_cond));
}

void
pocl_cuda_wait_event (cl_device_id device, cl_event event)
{
  pocl_cuda_event_data_t *e_d = (pocl_cuda_event_data_t *)event->data;

  if (((pocl_cuda_queue_data_t *)event->queue->data)->use_threads)
    {
      /* Wait until background thread marks command as complete */
      POCL_LOCK_OBJ (event);
      while (event->status > CL_COMPLETE)
        {
          PTHREAD_CHECK (
              pthread_cond_wait (&e_d->event_cond, &event->pocl_lock));
        }
      POCL_UNLOCK_OBJ (event);
    }
  else
    {
      /* Recursively finalize commands in this thread */
      pocl_cuda_wait_event_recurse (device, event);
    }
}

void
pocl_cuda_free_event_data (cl_event event)
{
  if (event->data)
    {
      pocl_cuda_event_data_t *event_data
          = (pocl_cuda_event_data_t *)event->data;
      PTHREAD_CHECK (pthread_cond_destroy (&event_data->event_cond));
      if (event->queue->properties & CL_QUEUE_PROFILING_ENABLE)
        cuEventDestroy (event_data->start);
      cuEventDestroy (event_data->end);
      if (event_data->ext_event_flag)
        {
          CUresult result = cuMemFreeHost (event_data->ext_event_flag);
          CUDA_CHECK_ERROR (result, "pocl_cuda_free_event_data cuMemFreeHost");
        }
      free (event->data);
    }
}

void
pocl_cuda_join (cl_device_id device, cl_command_queue cq)
{
  /* Grab event at end of queue */
  POCL_LOCK_OBJ (cq);
  cl_event event = cq->last_event.event;
  if (!event)
    {
      POCL_UNLOCK_OBJ (cq);
      return;
    }
  POname (clRetainEvent) (event);
  POCL_UNLOCK_OBJ (cq);

  pocl_cuda_wait_event (device, event);

  POname (clReleaseEvent) (event);
}

void *
pocl_cuda_submit_thread (void *data)
{
  pocl_cuda_queue_data_t *queue_data = (pocl_cuda_queue_data_t *)data;

  cl_command_queue queue = queue_data->queue;
  if (queue)
    cuCtxSetCurrent (
        ((pocl_cuda_device_data_t *)queue->device->data)->context);
  else
    /* This queue has already been released */
    return NULL;

  while (1)
    {
      /* Attempt to get next command from work queue */
      _cl_command_node *node = NULL;
      PTHREAD_CHECK (pthread_mutex_lock (&queue_data->lock));
      if (!queue_data->queue)
        {
          PTHREAD_CHECK (pthread_mutex_unlock (&queue_data->lock));
          break;
        }
      if (!queue_data->pending_queue)
        {
          PTHREAD_CHECK (pthread_cond_wait (&queue_data->pending_cond,
                                            &queue_data->lock));
        }
      if (queue_data->pending_queue)
        {
          node = queue_data->pending_queue;
          DL_DELETE (queue_data->pending_queue, node);
        }
      PTHREAD_CHECK (pthread_mutex_unlock (&queue_data->lock));

      /* Submit command, if we found one */
      if (node)
        {
          pocl_cuda_submit_node (node, queue_data->queue, 0);

          /* Add command to running queue */
          PTHREAD_CHECK (pthread_mutex_lock (&queue_data->lock));
          DL_APPEND (queue_data->running_queue, node);
          PTHREAD_CHECK (pthread_cond_signal (&queue_data->running_cond));
          PTHREAD_CHECK (pthread_mutex_unlock (&queue_data->lock));
        }
    }

  return NULL;
}

void *
pocl_cuda_finalize_thread (void *data)
{
  pocl_cuda_queue_data_t *queue_data = (pocl_cuda_queue_data_t *)data;

  cl_command_queue queue = queue_data->queue;
  if (queue)
    cuCtxSetCurrent (
        ((pocl_cuda_device_data_t *)queue->device->data)->context);
  else
    /* This queue has already been released */
    return NULL;

  while (1)
    {
      /* Attempt to get next node from running queue */
      _cl_command_node *node = NULL;
      PTHREAD_CHECK (pthread_mutex_lock (&queue_data->lock));
      if (!queue_data->queue)
        {
          PTHREAD_CHECK (pthread_mutex_unlock (&queue_data->lock));
          break;
        }
      if (!queue_data->running_queue)
        {
          PTHREAD_CHECK (pthread_cond_wait (&queue_data->running_cond,
                                            &queue_data->lock));
        }
      if (queue_data->running_queue)
        {
          node = queue_data->running_queue;
          DL_DELETE (queue_data->running_queue, node);
        }
      PTHREAD_CHECK (pthread_mutex_unlock (&queue_data->lock));

      /* Wait for command to finish, if we found one */
      if (node)
        pocl_cuda_finalize_command (queue->device, node->sync.event.event);
    }

  return NULL;
}

char* pocl_cuda_init_build(void *data)
{
    return strdup("-mllvm --nvptx-short-ptr");
}

/****** SVM callbacks *****/

void *
pocl_cuda_svm_alloc (cl_device_id dev, cl_svm_mem_flags flags, size_t size)
{
  POCL_MSG_PRINT_CUDA ("SVM cuMemAllocManaged %lu\n", size);
  if ((flags & CL_MEM_SVM_FINE_GRAIN_BUFFER)
      && ((dev->svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) == 0))
    {
      POCL_MSG_ERR (
          "This device does not support SVM fine-grained buffers.\n");
      return NULL;
    }

  CUdeviceptr dptr;
  CUresult res;
  res = cuMemAllocManaged (&dptr, size, CU_MEM_ATTACH_GLOBAL);
  CUDA_CHECK_ERROR (res, "pocl_cuda_svm_alloc cuMemAllocManaged");
  if (res != CUDA_SUCCESS)
    dptr = 0;
  return (void *)dptr;
}

void
pocl_cuda_svm_free (cl_device_id dev, void *svm_ptr)
{
  POCL_MSG_PRINT_CUDA ("SVM cuMemFree %p\n", svm_ptr);
  CUresult res;
  res = cuMemFree ((CUdeviceptr)svm_ptr);
  CUDA_CHECK_ERROR (res, "pocl_cuda_svm_free cuMemFree");
}

void
pocl_cuda_svm_copy (cl_device_id dev, void *__restrict__ dst,
                    const void *__restrict__ src, size_t size)
{
  POCL_MSG_PRINT_CUDA ("SVM cuMemcpy %p -> %p, %lu bytes\n", src, dst, size);
  CUresult res;
  res = cuMemcpy ((CUdeviceptr)dst, (CUdeviceptr)src, size);
  CUDA_CHECK_ABORT (res, "pocl_cuda_svm_copy cuMemcpy");
}

void
pocl_cuda_svm_copy_async (CUstream stream, void *__restrict__ dst,
                          const void *__restrict__ src, size_t size)
{
  POCL_MSG_PRINT_CUDA ("SVM cuMemcpyAsync %p -> %p, %lu bytes\n", src, dst,
                       size);

  CUresult res;
  res = cuMemcpyAsync ((CUdeviceptr)dst, (CUdeviceptr)src, size, stream);
  CUDA_CHECK_ABORT (res, "cuMemcpyAsync");
}

void
pocl_cuda_svm_fill (cl_device_id dev, void *__restrict__ svm_ptr, size_t size,
                    void *__restrict__ pattern, size_t pattern_size)
{
  POCL_MSG_PRINT_CUDA ("SVM MEMFILL %p \n", svm_ptr);

  pocl_cuda_submit_memfill (0, svm_ptr, size, 0, pattern, pattern_size);
}


cl_int
pocl_cuda_set_kernel_exec_info_ext (cl_device_id dev,
                                     unsigned program_device_i,
                                     cl_kernel Kernel, cl_uint param_name,
                                     size_t param_value_size,
                                     const void *param_value)
{
  pocl_cuda_device_data_t *data = (pocl_cuda_device_data_t *)dev->data;
  switch (param_name)
    {
    case CL_KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM:
    case CL_KERNEL_EXEC_INFO_SVM_PTRS:
    case CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL:
      return data->supports_managed_memory ? CL_SUCCESS : CL_INVALID_OPERATION;
    default:
      POCL_MSG_ERR (
          "CUDA: clSetKernelExecInfo with parameter %u not implemented\n",
          param_name);
      return CL_INVALID_OPERATION;
    }
}

cl_int
pocl_cuda_get_device_info_ext (cl_device_id device, cl_device_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret)
{
  pocl_cuda_device_data_t *data = (pocl_cuda_device_data_t *)device->data;
  CUdevice cudaDev = data->device;
  int value;
  CUresult res;

  switch (param_name)
    {
    case CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV:
      res = cuDeviceGetAttribute (
          &value, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cudaDev);
      CUDA_CHECK_ABORT (res, "cuDeviceGetAttribute");
      POCL_RETURN_GETINFO (cl_uint, value);
    case CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV:
      res = cuDeviceGetAttribute (
          &value, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cudaDev);
      CUDA_CHECK_ABORT (res, "cuDeviceGetAttribute");
      POCL_RETURN_GETINFO (cl_uint, value);
    case CL_DEVICE_REGISTERS_PER_BLOCK_NV:
      res = cuDeviceGetAttribute (
          &value, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, cudaDev);
      CUDA_CHECK_ABORT (res, "cuDeviceGetAttribute");
      POCL_RETURN_GETINFO (cl_uint, value);
    case CL_DEVICE_WARP_SIZE_NV:
      res = cuDeviceGetAttribute (&value, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                                  cudaDev);
      CUDA_CHECK_ABORT (res, "cuDeviceGetAttribute");
      POCL_RETURN_GETINFO (cl_uint, value);
    case CL_DEVICE_GPU_OVERLAP_NV:
      res = cuDeviceGetAttribute (&value, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP,
                                  cudaDev);
      CUDA_CHECK_ABORT (res, "cuDeviceGetAttribute");
      POCL_RETURN_GETINFO (cl_bool, value);
    case CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV:
      res = cuDeviceGetAttribute (
          &value, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, cudaDev);
      CUDA_CHECK_ABORT (res, "cuDeviceGetAttribute");
      POCL_RETURN_GETINFO (cl_bool, value);
    case CL_DEVICE_INTEGRATED_MEMORY_NV:
      res = cuDeviceGetAttribute (&value, CU_DEVICE_ATTRIBUTE_INTEGRATED,
                                  cudaDev);
      CUDA_CHECK_ABORT (res, "cuDeviceGetAttribute");
      POCL_RETURN_GETINFO (cl_bool, value);
    case CL_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT_NV:
      res = cuDeviceGetAttribute (
          &value, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, cudaDev);
      CUDA_CHECK_ABORT (res, "cuDeviceGetAttribute");
      POCL_RETURN_GETINFO (cl_uint, value);
    case CL_DEVICE_PCI_BUS_ID_NV:
      res = cuDeviceGetAttribute (&value, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                                  cudaDev);
      CUDA_CHECK_ABORT (res, "cuDeviceGetAttribute");
      POCL_RETURN_GETINFO (cl_uint, value);
    case CL_DEVICE_PCI_SLOT_ID_NV:
      res = cuDeviceGetAttribute (&value, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
                                  cudaDev);
      CUDA_CHECK_ABORT (res, "cuDeviceGetAttribute");
      POCL_RETURN_GETINFO (cl_uint, value);
    case CL_DEVICE_PCI_DOMAIN_ID_NV:
      res = cuDeviceGetAttribute (&value, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
                                  cudaDev);
      CUDA_CHECK_ABORT (res, "cuDeviceGetAttribute");
      POCL_RETURN_GETINFO (cl_uint, value);

    case CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL:
    case CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL:
    case CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL:
    case CL_DEVICE_CROSS_DEVICE_SHARED_MEM_CAPABILITIES_INTEL:
    case CL_DEVICE_SHARED_SYSTEM_MEM_CAPABILITIES_INTEL:
      return CL_INVALID_VALUE;

    case CL_DEVICE_SUB_GROUP_SIZES_INTEL:
      {
        size_t sizes[] = { data->warp_size };
        POCL_RETURN_GETINFO_ARRAY (size_t, sizeof (sizes) / sizeof (size_t),
                                   sizes);
      }

    default:
      POCL_MSG_ERR ("Unknown param_name for get_device_info_ext: %u\n",
                    param_name);
      return CL_INVALID_VALUE;
    }
}
