/* pocl-cuda.c - driver for CUDA devices

   Copyright (c) 2016 James Price / University of Bristol

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

#include "pocl-cuda.h"
#include "pocl-ptx-gen.h"
#include "common.h"
#include "devices.h"
#include "pocl_cache.h"
#include "pocl_file_util.h"

#include <string.h>

#include <cuda.h>

typedef struct pocl_cuda_device_data_s {
  CUdevice device;
  CUcontext context;
} pocl_cuda_device_data_t;

static void pocl_cuda_abort_on_error(CUresult result,
                                     unsigned line,
                                     const char* func,
                                     const char* code,
                                     const char* api)
{
  if (result != CUDA_SUCCESS)
  {
    const char *err_name;
    const char *err_string;
    cuGetErrorName(result, &err_name);
    cuGetErrorString(result, &err_string);
    POCL_MSG_PRINT2(func, line, "Error during %s\n", api);
    POCL_ABORT("%s: %s\n", err_name, err_string);
  }
}

#define CUDA_CHECK(result, api) \
  pocl_cuda_abort_on_error(result, __LINE__, __FUNCTION__, #result, api);


void
pocl_cuda_init_device_ops(struct pocl_device_ops *ops)
{
  pocl_basic_init_device_ops (ops);

  ops->device_name = "CUDA";
  ops->init_device_infos = pocl_cuda_init_device_infos;
  ops->probe = pocl_cuda_probe;
  ops->uninit = pocl_cuda_uninit;
  ops->init = pocl_cuda_init;
  ops->alloc_mem_obj = pocl_cuda_alloc_mem_obj;
  ops->free = pocl_cuda_free;
  ops->compile_submitted_kernels = pocl_cuda_compile_submitted_kernels;
  ops->run = pocl_cuda_run;
  ops->read = pocl_cuda_read;
  //ops->read_rect = pocl_basic_read_rect;
  ops->write = pocl_cuda_write;
  //ops->write_rect = pocl_basic_write_rect;
  //ops->copy = pocl_cuda_copy;
  //ops->copy_rect = pocl_basic_copy_rect;
  //ops->get_timer_value = pocl_cuda_get_timer_value;
}

void
pocl_cuda_init(cl_device_id device, const char* parameters)
{
  CUresult result;

  result = cuInit(0);
  CUDA_CHECK(result, "cuInit");

  if (device->data)
    return;

  pocl_cuda_device_data_t *data = malloc(sizeof(pocl_cuda_device_data_t));
  result = cuDeviceGet(&data->device, 0);
  CUDA_CHECK(result, "cuDeviceGet");

  // Get specific device name
  device->long_name = device->short_name = malloc(256*sizeof(char));
  cuDeviceGetName(device->long_name, 256, data->device);

  // Get other device properties
  cuDeviceGetAttribute((int*)&device->max_work_group_size,
                       CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                       data->device);
  cuDeviceGetAttribute((int*)(device->max_work_item_sizes+0),
                       CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                       data->device);
  cuDeviceGetAttribute((int*)(device->max_work_item_sizes+1),
                       CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                       data->device);
  cuDeviceGetAttribute((int*)(device->max_work_item_sizes+2),
                       CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
                       data->device);
  cuDeviceGetAttribute((int*)&device->local_mem_size,
                       CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
                       data->device);
  cuDeviceGetAttribute((int*)&device->max_compute_units,
                       CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                       data->device);
  cuDeviceGetAttribute((int*)&device->max_clock_frequency,
                       CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                       data->device);
  cuDeviceGetAttribute((int*)&device->error_correction_support,
                       CU_DEVICE_ATTRIBUTE_ECC_ENABLED,
                       data->device);
  cuDeviceGetAttribute((int*)&device->host_unified_memory,
                       CU_DEVICE_ATTRIBUTE_INTEGRATED,
                       data->device);

  device->preferred_vector_width_char   = 1;
  device->preferred_vector_width_short  = 1;
  device->preferred_vector_width_int    = 1;
  device->preferred_vector_width_long   = 1;
  device->preferred_vector_width_float  = 1;
  device->preferred_vector_width_double = 1;
  device->preferred_vector_width_half   = 0;
  device->native_vector_width_char      = 1;
  device->native_vector_width_short     = 1;
  device->native_vector_width_int       = 1;
  device->native_vector_width_long      = 1;
  device->native_vector_width_float     = 1;
  device->native_vector_width_double    = 1;
  device->native_vector_width_half      = 0;

  // TODO: Actual maximum size
  device->max_mem_alloc_size = 1024*1024*1024;
  device->global_mem_size    = 1024*1024*1024;

  device->local_mem_type = CL_LOCAL;
  device->host_unified_memory = 0;

  // Get GPU architecture name
  int sm_maj, sm_min;
  cuDeviceGetAttribute(&sm_maj,
                       CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                       data->device);
  cuDeviceGetAttribute(&sm_min,
                       CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                       data->device);
  char *gpu_arch = malloc(16*sizeof(char));
  snprintf(gpu_arch, 16, "sm_%d%d", sm_maj, sm_min);
  device->llvm_cpu = gpu_arch;
  POCL_MSG_PRINT_INFO("[CUDA] GPU architecture = %s\n", gpu_arch);

  // Create context
  result = cuCtxCreate(&data->context, 0, data->device);
  CUDA_CHECK(result, "cuCtxCreate");

  device->data = data;
}

void
pocl_cuda_init_device_infos(struct _cl_device_id* dev)
{
  pocl_basic_init_device_infos(dev);

  dev->type = CL_DEVICE_TYPE_GPU;
  dev->llvm_target_triplet = "nvptx64";
  dev->spmd = CL_TRUE;

  // TODO: Get images working
  dev->image_support = CL_FALSE;
}

unsigned int
pocl_cuda_probe(struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count(ops->device_name);

  // TODO: Check how many CUDA device available (if any)

  if(env_count < 0)
    return 1;

  return env_count;
}

void
pocl_cuda_uninit(cl_device_id device)
{
  pocl_cuda_device_data_t *data = device->data;

  cuCtxDestroy(data->context);

  POCL_MEM_FREE(data);
  device->data = NULL;

  POCL_MEM_FREE(device->long_name);
}

cl_int pocl_cuda_alloc_mem_obj(cl_device_id device, cl_mem mem_obj)
{
  void *b = NULL;

  /* if memory for this global memory is not yet allocated -> do it */
  if (mem_obj->device_ptrs[device->global_mem_id].mem_ptr == NULL)
  {
    // TODO: Deal with mem flags
    CUresult result = cuMemAlloc((CUdeviceptr*)&b, mem_obj->size);
    if (result != CUDA_SUCCESS)
    {
      const char *err;
      cuGetErrorName(result, &err);
      POCL_MSG_PRINT2(__FUNCTION__, __LINE__,
                      "-> Failed to allocate memory: %s\n", err);
      return CL_MEM_OBJECT_ALLOCATION_FAILURE;
    }

    mem_obj->device_ptrs[device->global_mem_id].mem_ptr = b;
    mem_obj->device_ptrs[device->global_mem_id].global_mem_id =
      device->global_mem_id;
  }

  /* copy already allocated global mem info to devices own slot */
  mem_obj->device_ptrs[device->dev_id] =
    mem_obj->device_ptrs[device->global_mem_id];

  return CL_SUCCESS;
}

void pocl_cuda_free(cl_device_id device, cl_mem mem_obj)
{
  void* ptr = mem_obj->device_ptrs[device->dev_id].mem_ptr;
  cuMemFree((CUdeviceptr)ptr);
}

void
pocl_cuda_read(void *data, void *host_ptr, const void *device_ptr,
               size_t offset, size_t cb)
{
  cuMemcpyDtoH(host_ptr, (CUdeviceptr)(device_ptr+offset), cb);
}

void
pocl_cuda_write(void *data, const void *host_ptr, void *device_ptr,
                size_t offset, size_t cb)
{
  cuMemcpyHtoD((CUdeviceptr)(device_ptr+offset), host_ptr, cb);
}

void
pocl_cuda_compile_submitted_kernels(_cl_command_node *cmd)
{
  CUresult result;

  if (cmd->type != CL_COMMAND_NDRANGE_KERNEL)
    return;

  cl_kernel kernel = cmd->command.run.kernel;

  // Check if we already have a compiled kernel function
  if (kernel->data)
    return;

  char bc_filename[POCL_FILENAME_LENGTH];
  snprintf(bc_filename, POCL_FILENAME_LENGTH, "%s%s",
           cmd->command.run.tmp_dir, POCL_PARALLEL_BC_FILENAME);

  char ptx_filename[POCL_FILENAME_LENGTH];
  snprintf(ptx_filename, POCL_FILENAME_LENGTH, "%s/program.ptx",
           cmd->command.run.tmp_dir);

  // Generate PTX from LLVM bitcode
  // TODO: Load from cache if present
  if (pocl_ptx_gen(bc_filename, ptx_filename, cmd->device->llvm_cpu))
    POCL_ABORT("pocl-cuda: failed to generate PTX\n");

  // Load PTX module
  // TODO: When can we unload the module?
  CUmodule module;
  result = cuModuleLoad(&module, ptx_filename);
  CUDA_CHECK(result, "cuModuleLoad");

  // Get kernel function
  CUfunction function;
  result = cuModuleGetFunction(&function, module, kernel->name);
  CUDA_CHECK(result, "cuModuleGetFunction");

  kernel->data = function;
}

void
pocl_cuda_run(void *dptr, _cl_command_node* cmd)
{
  CUresult result;

  cl_device_id device = cmd->device;
  CUfunction function = cmd->command.run.kernel->data;

  // Prepare kernel arguments
  unsigned sharedMemBytes = 0;
  cl_uint nargs = cmd->command.run.kernel->num_args;
  void *params[nargs];
  unsigned sharedMemOffsets[nargs];
  for (unsigned i = 0; i < nargs; i++)
  {
    pocl_argument_type type = cmd->command.run.kernel->arg_info[i].type;
    switch (type)
    {
    case POCL_ARG_TYPE_NONE:
      params[i] = cmd->command.run.kernel->dyn_arguments[i].value;
      break;
    case POCL_ARG_TYPE_POINTER:
    {
      if (cmd->command.run.kernel->arg_info[i].is_local)
      {
        sharedMemOffsets[i] = sharedMemBytes;
        sharedMemBytes += cmd->command.run.kernel->dyn_arguments[i].size;
        params[i] = sharedMemOffsets+i;
      }
      else
      {
        cl_mem mem = *(void**)cmd->command.run.kernel->dyn_arguments[i].value;
        params[i] = &mem->device_ptrs[device->dev_id].mem_ptr;
      }
      break;
    }
    case POCL_ARG_TYPE_IMAGE:
    case POCL_ARG_TYPE_SAMPLER:
      POCL_ABORT("Unhandled argument type for CUDA");
      break;
    }
  }

  // TODO: Deal with static local memory allocations

  // Launch kernel
  struct pocl_context pc = cmd->command.run.pc;
  result = cuLaunchKernel(
    function,
    pc.num_groups[0],
    pc.num_groups[1],
    pc.num_groups[2],
    cmd->command.run.local_x,
    cmd->command.run.local_y,
    cmd->command.run.local_z,
    sharedMemBytes, NULL, params, NULL);
  CUDA_CHECK(result, "cuLaunchKernel");

  // TODO: We don't really want to sync here
  result = cuStreamSynchronize(0);
  CUDA_CHECK(result, "cuStreamSynchronize");
}
