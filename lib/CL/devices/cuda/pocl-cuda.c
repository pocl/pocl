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
  cuDeviceGetAttribute((int*)&device->max_compute_units,
                       CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                       data->device);

  // TODO: Actual maximum size
  device->max_mem_alloc_size = 1024*1024*1024;

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
  dev->llvm_cpu = NULL;
  dev->spmd = CL_TRUE;
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

  char bc_filename[POCL_FILENAME_LENGTH];
  snprintf(bc_filename, POCL_FILENAME_LENGTH, "%s%s",
           cmd->command.run.tmp_dir, POCL_PARALLEL_BC_FILENAME);

  char ptx_filename[POCL_FILENAME_LENGTH];
  snprintf(ptx_filename, POCL_FILENAME_LENGTH, "%s/program.ptx",
           cmd->command.run.tmp_dir);

  // Generate PTX from LLVM bitcode
  if (pocl_ptx_gen(bc_filename, ptx_filename))
    POCL_ABORT("pocl-cuda: failed to generate PTX\n");

  // Load PTX module
  CUmodule module;
  result = cuModuleLoad(&module, ptx_filename);
  CUDA_CHECK(result, "cuModuleLoad");

  cmd->command.run.data = module;
}

void
pocl_cuda_run(void *dptr, _cl_command_node* cmd)
{
  CUresult result;
  CUmodule module = cmd->command.run.data;
  cl_device_id device = cmd->device;

  // Get kernel function
  CUfunction function;
  result = cuModuleGetFunction(&function, module,
                               cmd->command.run.kernel->name);
  CUDA_CHECK(result, "cuModuleGetFunction");

  // Prepare kernel arguments
  cl_uint nargs = cmd->command.run.kernel->num_args;
  void *params[nargs];
  for (unsigned i = 0; i < nargs; i++)
  {
    cl_mem mem = *(void**)cmd->command.run.kernel->dyn_arguments[i].value;
    params[i] = &mem->device_ptrs[device->dev_id].mem_ptr;
  }

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
    0, NULL, params, NULL);
  CUDA_CHECK(result, "cuLaunchKernel");
}
